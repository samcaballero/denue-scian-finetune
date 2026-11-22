"""Generador de los kernels de Kaggle del clasificador (Fase 2, pasos 6b y 9).

La CLI de Kaggle 2.2.5 solo sube el archivo `code_file` del kernel (más su metadata), así que
los módulos de `src/` que usa cada kernel van incrustados en ese archivo, sin copiarlos a mano:

- el texto exacto de cada módulo, con los saltos de línea normalizados a "\\n" y el SHA-256 de
  cada uno (igual en Windows y en Linux);
- los SHA-256 esperados de los archivos de datos, leídos de `data/manifest.json` (sección
  `kaggle_datasets`), nunca copiados a mano;
- en los kernels de corridas del clasificador, la configuración de la corrida (`CORRIDA`);
- un arranque que escribe los módulos en /tmp/denue_src, verifica sus hashes y llama a la
  función `main()` del módulo del kernel.

Cada kernel tiene una especificación (`EspecKernel`) en `KERNELS`:
- `humo-6b`: la prueba de punta a punta (paso 6b), que ya corrió en Kaggle; se genera idéntica
  byte a byte, y sus módulos (`src/humo_kaggle.py` y los que incrusta) no se modifican;
- `06b-inicial`: la primera corrida real del clasificador con Qwen3-0.6B (paso 9).
Un kernel ya lanzado no se regenera con otro contenido. Las corridas siguientes (barrido, 4B)
serán especificaciones nuevas, armadas con `corrida()` sobre `CORRIDA_POR_OMISION`.

También genera `kernel-metadata.json` (privado, GPU T4, internet y el dataset como única
fuente), en ASCII.

Candado de cada kernel: no puede contener los textos "test.jsonl" ni "hard.jsonl", ni el SHA-256
de test o del hard set, ni ningún SHA-256 del manifest salvo los de sus propios archivos de datos.

Uso (sin --kernel, recorre todos los kernels):
  .venv/Scripts/python -m src.construir_kernel [--kernel CLAVE]              # resumen y comparación
  .venv/Scripts/python -m src.construir_kernel [--kernel CLAVE] --escribir   # escribe la carpeta del kernel
  .venv/Scripts/python -m src.construir_kernel [--kernel CLAVE] --verificar  # falla si lo escrito no es lo generado
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pprint
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.clasificador_kaggle import ARCHIVO_TRAIN as ARCHIVO_TRAIN_FINAL
from src.clasificador_kaggle import ARCHIVO_VAL as ARCHIVO_VAL_CLASIFICADOR
from src.clasificador_kaggle import validar_corrida
from src.denue import REPO_RAIZ
from src.humo_kaggle import ARCHIVO_TRAIN, ARCHIVO_VAL, LORA, MACHINE_SHAPE, MODELO_06B

ARCHIVO_METADATA = "kernel-metadata.json"  # nombre que usa el código de la CLI de Kaggle 2.2.5
DIR_EXTRACCION = "/tmp/denue_src"
SECCION = "kaggle_datasets"
ENTRADA_TEST = "clasificador_test"  # de esta entrada solo se lee `sha256`, para el candado
ENTRADA_HARD = "clasificador_hard"  # ídem
TEXTOS_PROHIBIDOS = ("test.jsonl", "hard.jsonl")

# Largo máximo de secuencia: estándar de todas las corridas del clasificador (entrenamiento,
# barrido, 4B e inferencia). Lo aprobó Samuel en el paso 9: el train final tiene nombres de hasta
# 151 bytes, y con 256 fichas ninguna fila se trunca ni se excluye.
LARGO_MAXIMO_CLASIFICADOR = 256

# Valores por omisión de las corridas del clasificador: la configuración inicial de la guía.
CORRIDA_POR_OMISION = {
    "max_seq_length": LARGO_MAXIMO_CLASIFICADOR,
    "lora": {
        "r": LORA["r"],
        "lora_alpha": LORA["lora_alpha"],
        "lora_dropout": LORA["lora_dropout"],
        "bias": LORA["bias"],
        "target_modules": list(LORA["target_modules"]),
        "use_gradient_checkpointing": LORA["use_gradient_checkpointing"],
        "random_state": LORA["random_state"],
    },
    "gradient_accumulation_steps": 1,
    "epocas": 1,
    "optim": "adamw_8bit",
    "lr_scheduler_type": "linear",
    "fraccion_warmup": 0.03,
    "eval_cada_pasos": 500,
    "lote_eval": 32,
    "semilla": 42,
}


class ErrorKernel(ValueError):
    """El paquete no se puede generar o no cumple el candado."""


def corrida(nombre: str, descripcion: str, modelo: Mapping[str, str], lote: int, learning_rate: float, **cambios) -> dict:
    """Corrida del clasificador: los valores por omisión más lo propio de la corrida, en orden fijo."""
    base = copy.deepcopy(CORRIDA_POR_OMISION)
    desconocidas = sorted(set(cambios) - set(base))
    if desconocidas:
        raise ErrorKernel(f"Claves desconocidas en la corrida: {desconocidas}.")
    base.update(copy.deepcopy(cambios))
    res = {
        "nombre": nombre,
        "descripcion": descripcion,
        "modelo": {"repo": modelo["repo"], "revision": modelo["revision"]},
        "max_seq_length": base["max_seq_length"],
        "lora": base["lora"],
        "lote": lote,
        "gradient_accumulation_steps": base["gradient_accumulation_steps"],
        "learning_rate": learning_rate,
        "epocas": base["epocas"],
        "optim": base["optim"],
        "lr_scheduler_type": base["lr_scheduler_type"],
        "fraccion_warmup": base["fraccion_warmup"],
        "eval_cada_pasos": base["eval_cada_pasos"],
        "lote_eval": base["lote_eval"],
        "semilla": base["semilla"],
    }
    try:
        return validar_corrida(res)
    except ValueError as e:
        raise ErrorKernel(str(e)) from e


@dataclass(frozen=True)
class EspecKernel:
    """Todo lo que distingue a un kernel: carpeta, id, módulos, datos y configuración."""

    clave: str
    carpeta_relativa: str
    archivo_kernel: str
    id_kernel: str
    titulo: str  # su slug debe ser el del id
    entrada: str  # clave en kaggle_datasets del manifest
    archivos_datos: tuple[str, ...]
    fuentes: tuple[str, ...]
    modulo: str  # módulo cuyo main() llama el kernel
    resumen: str  # primera línea del docstring del kernel
    comentario_esperados: str
    corrida: Mapping | None = field(default=None)

    @property
    def carpeta(self) -> Path:
        return REPO_RAIZ / self.carpeta_relativa


HUMO = EspecKernel(
    clave="humo-6b",
    carpeta_relativa="notebooks/kaggle-humo",
    archivo_kernel="humo_6b.py",
    id_kernel="samcaballero/denue-scian-humo-6b",
    titulo="DENUE SCIAN humo 6b",
    entrada="denue_scian_humo",
    archivos_datos=(ARCHIVO_TRAIN, ARCHIVO_VAL),
    fuentes=("src/__init__.py", "src/normalizacion.py", "src/prompt_clasificador.py", "src/humo_kaggle.py"),
    modulo="src.humo_kaggle",
    resumen="Kernel de Kaggle: prueba de punta a punta del clasificador (Fase 2, paso 6b).",
    comentario_esperados="SHA-256 de las muestras del dataset privado",
)

CLASIFICADOR_06B_INICIAL = EspecKernel(
    clave="06b-inicial",
    carpeta_relativa="notebooks/kaggle-06b-inicial",
    archivo_kernel="clasificador_06b_inicial.py",
    id_kernel="samcaballero/denue-scian-06b-inicial",
    titulo="DENUE SCIAN 06b inicial",
    entrada="denue_scian_clasificador",
    archivos_datos=(ARCHIVO_TRAIN_FINAL, ARCHIVO_VAL_CLASIFICADOR),
    fuentes=(
        "src/__init__.py",
        "src/normalizacion.py",
        "src/prompt_clasificador.py",
        "src/humo_kaggle.py",
        "src/clasificador_kaggle.py",
    ),
    modulo="src.clasificador_kaggle",
    resumen="Kernel de Kaggle: corrida 06b-inicial del clasificador con Qwen3-0.6B (Fase 2, paso 9).",
    comentario_esperados="SHA-256 de los archivos del dataset privado",
    corrida=corrida(
        "06b-inicial",
        "Primera corrida real del clasificador (Fase 2, paso 9): Qwen3-0.6B con la configuración inicial de la guía.",
        MODELO_06B,
        lote=32,
        learning_rate=2e-4,
    ),
)

KERNELS: dict[str, EspecKernel] = {e.clave: e for e in (HUMO, CLASIFICADOR_06B_INICIAL)}

# Nombres de antes de la generalización; apuntan a la prueba (los usa src/evaluar_humo.py).
CARPETA_RELATIVA = HUMO.carpeta_relativa
CARPETA = HUMO.carpeta
ARCHIVO_KERNEL = HUMO.archivo_kernel
FUENTES = HUMO.fuentes
ID_KERNEL = HUMO.id_kernel
TITULO = HUMO.titulo
ENTRADA = HUMO.entrada
ARCHIVOS_DATOS = HUMO.archivos_datos


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def normalizar_saltos(datos: bytes) -> bytes:
    """CRLF → LF: con core.autocrlf=true, una copia nueva en Windows puede tener CRLF."""
    return datos.replace(b"\r\n", b"\n")


def leer_fuente(ruta: Path) -> str:
    """Texto UTF-8 de un módulo con saltos "\\n"; falla si queda algún "\\r" suelto."""
    datos = normalizar_saltos(Path(ruta).read_bytes())
    if b"\r" in datos:
        raise ErrorKernel(f"{ruta} tiene un \\r suelto.")
    if datos.startswith(b"\xef\xbb\xbf"):
        raise ErrorKernel(f"{ruta} tiene BOM.")
    return datos.decode("utf-8")


def leer_fuentes(raiz: Path = REPO_RAIZ, espec: EspecKernel = HUMO) -> dict[str, str]:
    return {f: leer_fuente(Path(raiz) / f) for f in espec.fuentes}


def esperados(manifest: Mapping, espec: EspecKernel = HUMO) -> dict[str, str]:
    """SHA-256 de los archivos de datos del kernel, tal como están en el manifest."""
    archivos = manifest[SECCION][espec.entrada]["archivos"]
    res = {a: archivos[a]["sha256"] for a in espec.archivos_datos}
    if not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in res.values()):
        raise ErrorKernel("Los SHA-256 de los datos en el manifest no tienen el formato esperado.")
    return res


def hashes_del_manifest(obj) -> set[str]:
    """Todos los valores de claves `sha256` o terminadas en `_sha256` del manifest."""
    res: set[str] = set()
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(v, str) and (k == "sha256" or str(k).endswith("_sha256")):
                res.add(v)
            else:
                res |= hashes_del_manifest(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            res |= hashes_del_manifest(v)
    return res


def _literal(texto: str, sangria: str) -> str:
    """Literal de Python que da exactamente `texto`: un fragmento por línea, concatenados."""
    trozos = [t for t in re.split(r"(?<=\n)", texto) if t]
    if not trozos:
        return '""'
    return "(\n" + "".join(f"{sangria}    {t!r}\n" for t in trozos) + f"{sangria})"


def _dict_literal(d: Mapping[str, str]) -> str:
    return "{\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in d.items()) + "}"


def _valor_literal(obj) -> str:
    """Literal de Python (determinista y legible) de la configuración de una corrida."""
    return pprint.pformat(obj, indent=1, width=120, sort_dicts=False)


def generar_kernel(textos: Mapping[str, str], hashes_datos: Mapping[str, str], espec: EspecKernel = HUMO) -> str:
    """Texto del kernel autocontenido (un script de Python)."""
    sha_fuentes = {f: sha256_bytes(t.encode("utf-8")) for f, t in textos.items()}
    fuentes = "{\n" + "".join(f"    {f!r}: {_literal(t, '    ')},\n" for f, t in textos.items()) + "}"
    modulos = ", ".join(f"`{f}`" for f in textos)
    if espec.corrida is None:
        nota, bloque_corrida, argumentos = "", "", "ESPERADOS, T_ARRANQUE, SHA256_FUENTES"
    else:
        nota = " con la\nconfiguración de la corrida (CORRIDA)"
        bloque_corrida = (
            f'\n# Configuración de la corrida "{espec.clave}" (src/construir_kernel.py).\n'
            f"CORRIDA = {_valor_literal(dict(espec.corrida))}\n"
        )
        argumentos = "ESPERADOS, T_ARRANQUE, SHA256_FUENTES, CORRIDA"
    return f'''# -*- coding: utf-8 -*-
"""{espec.resumen}

ARCHIVO GENERADO por src/construir_kernel.py; no se edita a mano. Lleva incrustado el texto
exacto de {modulos}, con su SHA-256, y los SHA-256 esperados de los datos, leídos de
data/manifest.json. Al correr, escribe esos módulos en {DIR_EXTRACCION}, verifica sus hashes y
llama a {espec.modulo}.main(){nota}.
"""

import hashlib
import sys
import time
from pathlib import Path

T_ARRANQUE = time.time()

# {espec.comentario_esperados} (data/manifest.json, {SECCION}.{espec.entrada}).
ESPERADOS = {_dict_literal(hashes_datos)}
{bloque_corrida}
# SHA-256 del texto UTF-8 (saltos "\\n") de cada módulo incrustado.
SHA256_FUENTES = {_dict_literal(sha_fuentes)}

FUENTES = {fuentes}


def extraer(destino):
    """Escribe los módulos incrustados en `destino` y verifica el SHA-256 de cada uno."""
    destino = Path(destino)
    if set(FUENTES) != set(SHA256_FUENTES):
        raise SystemExit("Las fuentes incrustadas no coinciden con sus hashes; el kernel está dañado.")
    for ruta, texto in FUENTES.items():
        datos = texto.encode("utf-8")
        if hashlib.sha256(datos).hexdigest() != SHA256_FUENTES[ruta]:
            raise SystemExit(f"El SHA-256 de {{ruta}} no coincide; el kernel está dañado.")
        archivo = destino / ruta
        archivo.parent.mkdir(parents=True, exist_ok=True)
        archivo.write_bytes(datos)
    return destino


def arrancar(destino={DIR_EXTRACCION!r}):
    raiz = extraer(destino)
    sys.path.insert(0, str(raiz))
    from {espec.modulo} import main

    main({argumentos})


if __name__ == "__main__":
    arrancar()
'''


def slug(titulo: str) -> str:
    """Slug como el de Kaggle para títulos ASCII: minúsculas y guiones entre palabras."""
    return re.sub(r"[^a-z0-9]+", "-", titulo.lower()).strip("-")


def metadata_kernel(manifest: Mapping, espec: EspecKernel = HUMO) -> bytes:
    """kernel-metadata.json en ASCII: privado, GPU T4 (machine_shape), internet y el dataset como fuente."""
    duenio, _, nombre = espec.id_kernel.partition("/")
    if not duenio or "/" in nombre or slug(espec.titulo) != nombre:
        raise ErrorKernel("El slug del título debe ser el del id del kernel.")
    meta = {
        "id": espec.id_kernel,
        "title": espec.titulo,
        "code_file": espec.archivo_kernel,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [manifest[SECCION][espec.entrada]["id_kaggle"]],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image_pinning_type": "original",
    }
    return (json.dumps(meta, ensure_ascii=True, indent=2) + "\n").encode("ascii")


def candado(kernel: str, manifest: Mapping, espec: EspecKernel = HUMO) -> None:
    """Sin test.jsonl ni hard.jsonl, sin sus SHA-256 y con solo los SHA-256 de los datos del kernel."""
    for texto in TEXTOS_PROHIBIDOS:
        if texto in kernel:
            raise ErrorKernel(f"El kernel menciona {texto}.")
    permitidos = set(esperados(manifest, espec).values())
    presentes = {h for h in hashes_del_manifest(manifest) if h in kernel}
    reservados = {manifest["splits"][e]["sha256"] for e in (ENTRADA_TEST, ENTRADA_HARD) if e in manifest["splits"]}
    if any(h in kernel for h in reservados) or presentes - permitidos:
        raise ErrorKernel("El kernel lleva un SHA-256 del manifest que no es de sus datos.")
    if presentes != permitidos:
        raise ErrorKernel("El kernel no lleva los SHA-256 de sus archivos de datos.")


def generar(manifest: Mapping, raiz: Path = REPO_RAIZ, espec: EspecKernel = HUMO) -> dict[str, bytes]:
    """Bytes de los dos archivos de la carpeta del kernel."""
    if espec.corrida is not None:
        try:
            validar_corrida(espec.corrida)
        except ValueError as e:
            raise ErrorKernel(str(e)) from e
    kernel = generar_kernel(leer_fuentes(raiz, espec), esperados(manifest, espec), espec)
    candado(kernel, manifest, espec)
    compile(kernel, espec.archivo_kernel, "exec")
    return {espec.archivo_kernel: kernel.encode("utf-8"), ARCHIVO_METADATA: metadata_kernel(manifest, espec)}


def diferencias(generados: Mapping[str, bytes], carpeta: Path) -> list[str]:
    """Archivos de la carpeta que faltan o no son lo generado (con los saltos normalizados)."""
    res = []
    for nombre, datos in sorted(generados.items()):
        ruta = Path(carpeta) / nombre
        if not ruta.is_file() or normalizar_saltos(ruta.read_bytes()) != datos:
            res.append(nombre)
    return res


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.scian import leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kernel", choices=list(KERNELS), help="un solo kernel (sin esta opción, todos)")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--escribir", action="store_true", help="escribe el kernel y su metadata")
    modo.add_argument("--verificar", action="store_true", help="falla si lo escrito no es lo que se generaría")
    args = parser.parse_args(argv)

    manifest = leer_manifest()
    especs = [KERNELS[args.kernel]] if args.kernel else list(KERNELS.values())
    pendientes_total = []
    for espec in especs:
        print(f"== {espec.clave} ({espec.id_kernel}) ==")
        try:
            generados = generar(manifest, espec=espec)
        except ErrorKernel as e:
            sys.exit(f"Me detengo ({espec.clave}): {e}")
        print("Módulos incrustados (SHA-256 del texto con saltos \\n):")
        for f, t in leer_fuentes(espec=espec).items():
            print(f"  {f}: {sha256_bytes(t.encode('utf-8'))}")
        print("SHA-256 esperados de los datos (data/manifest.json):")
        for a, h in esperados(manifest, espec).items():
            print(f"  {a}: {h}")
        if espec.corrida is not None:
            c = espec.corrida
            print(
                f"Corrida: {c['modelo']['repo']}, lote {c['lote']}, tasa {c['learning_rate']}, "
                f"{c['epocas']} época(s), largo máximo {c['max_seq_length']}, LoRA r={c['lora']['r']}"
            )
        for nombre, datos in generados.items():
            print(f"{espec.carpeta_relativa}/{nombre}: {len(datos):,} bytes, SHA-256 {sha256_bytes(datos)}")
        print("Candado: sin test.jsonl ni hard.jsonl y sin otros SHA-256 del manifest.")

        if args.escribir:
            espec.carpeta.mkdir(parents=True, exist_ok=True)
            for nombre, datos in generados.items():
                (espec.carpeta / nombre).write_bytes(datos)
            print(f"Escritos en {espec.carpeta}.")
        pendientes = diferencias(generados, espec.carpeta)
        if pendientes:
            print(f"Distintos de lo generado (o faltan): {pendientes}.")
            pendientes_total.append((espec.clave, pendientes))
        else:
            print(f"{espec.carpeta_relativa}: los archivos escritos son exactamente lo generado.")
    if pendientes_total and args.verificar:
        sys.exit(f"Me detengo: {pendientes_total}.")


if __name__ == "__main__":
    main()
