"""Generador del kernel de Kaggle de la prueba de punta a punta (Fase 2, paso 6b).

La CLI de Kaggle 2.2.5 solo sube el archivo `code_file` del kernel (más su metadata), así que
los módulos de `src/` que usa el kernel van incrustados en ese archivo, sin copiarlos a mano:

- el texto exacto de `src/__init__.py`, `src/normalizacion.py`, `src/prompt_clasificador.py` y
  `src/humo_kaggle.py`, con los saltos de línea normalizados a "\\n" y el SHA-256 de cada uno
  (igual en Windows y en Linux);
- los SHA-256 esperados de `humo_train.jsonl` y `humo_val.jsonl`, leídos de
  `data/manifest.json` (sección `kaggle_datasets.denue_scian_humo`), nunca copiados a mano;
- un arranque que escribe los módulos en /tmp/denue_src, verifica sus hashes y llama a
  `src.humo_kaggle.main()`.

También genera `kernel-metadata.json` (privado, GPU T4, internet y el dataset como fuente), en
ASCII.

Candado: el paquete no puede contener la cadena "test.jsonl" ni ningún SHA-256 del manifest
salvo los de las dos muestras.

Uso:
  .venv/Scripts/python -m src.construir_kernel              # resumen y comparación con lo escrito
  .venv/Scripts/python -m src.construir_kernel --escribir   # escribe notebooks/kaggle-humo/
  .venv/Scripts/python -m src.construir_kernel --verificar  # falla si lo escrito no es lo que se generaría
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from src.denue import REPO_RAIZ
from src.humo_kaggle import ARCHIVO_TRAIN, ARCHIVO_VAL, MACHINE_SHAPE

CARPETA_RELATIVA = "notebooks/kaggle-humo"
CARPETA = REPO_RAIZ / CARPETA_RELATIVA
ARCHIVO_KERNEL = "humo_6b.py"
ARCHIVO_METADATA = "kernel-metadata.json"  # nombre que usa el código de la CLI de Kaggle 2.2.5
FUENTES = ("src/__init__.py", "src/normalizacion.py", "src/prompt_clasificador.py", "src/humo_kaggle.py")
DIR_EXTRACCION = "/tmp/denue_src"

ID_KERNEL = "samcaballero/denue-scian-humo-6b"
TITULO = "DENUE SCIAN humo 6b"  # su slug debe ser el del id
SECCION = "kaggle_datasets"
ENTRADA = "denue_scian_humo"
ARCHIVOS_DATOS = (ARCHIVO_TRAIN, ARCHIVO_VAL)
ENTRADA_TEST = "clasificador_test"  # de esta entrada solo se lee `sha256`, para el candado


class ErrorKernel(ValueError):
    """El paquete no se puede generar o no cumple el candado."""


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


def leer_fuentes(raiz: Path = REPO_RAIZ) -> dict[str, str]:
    return {f: leer_fuente(Path(raiz) / f) for f in FUENTES}


def esperados(manifest: Mapping) -> dict[str, str]:
    """SHA-256 de las dos muestras, tal como están en el manifest."""
    archivos = manifest[SECCION][ENTRADA]["archivos"]
    res = {a: archivos[a]["sha256"] for a in ARCHIVOS_DATOS}
    if not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in res.values()):
        raise ErrorKernel("Los SHA-256 de las muestras en el manifest no tienen el formato esperado.")
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


def generar_kernel(textos: Mapping[str, str], hashes_datos: Mapping[str, str]) -> str:
    """Texto del kernel autocontenido (un script de Python)."""
    sha_fuentes = {f: sha256_bytes(t.encode("utf-8")) for f, t in textos.items()}
    fuentes = "{\n" + "".join(f"    {f!r}: {_literal(t, '    ')},\n" for f, t in textos.items()) + "}"
    modulos = ", ".join(f"`{f}`" for f in textos)
    return f'''# -*- coding: utf-8 -*-
"""Kernel de Kaggle: prueba de punta a punta del clasificador (Fase 2, paso 6b).

ARCHIVO GENERADO por src/construir_kernel.py; no se edita a mano. Lleva incrustado el texto
exacto de {modulos}, con su SHA-256, y los SHA-256 esperados de los datos, leídos de
data/manifest.json. Al correr, escribe esos módulos en {DIR_EXTRACCION}, verifica sus hashes y
llama a src.humo_kaggle.main().
"""

import hashlib
import sys
import time
from pathlib import Path

T_ARRANQUE = time.time()

# SHA-256 de las muestras del dataset privado (data/manifest.json, {SECCION}.{ENTRADA}).
ESPERADOS = {_dict_literal(hashes_datos)}

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
    from src.humo_kaggle import main

    main(ESPERADOS, T_ARRANQUE, SHA256_FUENTES)


if __name__ == "__main__":
    arrancar()
'''


def slug(titulo: str) -> str:
    """Slug como el de Kaggle para títulos ASCII: minúsculas y guiones entre palabras."""
    return re.sub(r"[^a-z0-9]+", "-", titulo.lower()).strip("-")


def metadata_kernel(manifest: Mapping) -> bytes:
    """kernel-metadata.json en ASCII: privado, GPU T4 (machine_shape), internet y el dataset como fuente."""
    duenio, _, nombre = ID_KERNEL.partition("/")
    if not duenio or "/" in nombre or slug(TITULO) != nombre:
        raise ErrorKernel("El slug del título debe ser el del id del kernel.")
    meta = {
        "id": ID_KERNEL,
        "title": TITULO,
        "code_file": ARCHIVO_KERNEL,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [manifest[SECCION][ENTRADA]["id_kaggle"]],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image_pinning_type": "original",
    }
    return (json.dumps(meta, ensure_ascii=True, indent=2) + "\n").encode("ascii")


def candado(kernel: str, manifest: Mapping) -> None:
    """El kernel no menciona test.jsonl ni lleva más hashes del manifest que los de las muestras."""
    if "test.jsonl" in kernel:
        raise ErrorKernel("El kernel menciona test.jsonl.")
    permitidos = set(esperados(manifest).values())
    presentes = {h for h in hashes_del_manifest(manifest) if h in kernel}
    if manifest["splits"][ENTRADA_TEST]["sha256"] in kernel or presentes - permitidos:
        raise ErrorKernel("El kernel lleva un SHA-256 del manifest que no es de las muestras.")
    if presentes != permitidos:
        raise ErrorKernel("El kernel no lleva los SHA-256 de las dos muestras.")


def generar(manifest: Mapping, raiz: Path = REPO_RAIZ) -> dict[str, bytes]:
    """Bytes de los dos archivos de la carpeta del kernel."""
    kernel = generar_kernel(leer_fuentes(raiz), esperados(manifest))
    candado(kernel, manifest)
    compile(kernel, ARCHIVO_KERNEL, "exec")
    return {ARCHIVO_KERNEL: kernel.encode("utf-8"), ARCHIVO_METADATA: metadata_kernel(manifest)}


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
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--escribir", action="store_true", help="escribe el kernel y su metadata")
    modo.add_argument("--verificar", action="store_true", help="falla si lo escrito no es lo que se generaría")
    args = parser.parse_args(argv)

    manifest = leer_manifest()
    try:
        generados = generar(manifest)
    except ErrorKernel as e:
        sys.exit(f"Me detengo: {e}")
    textos = leer_fuentes()
    print("Módulos incrustados (SHA-256 del texto con saltos \\n):")
    for f, t in textos.items():
        print(f"  {f}: {sha256_bytes(t.encode('utf-8'))}")
    print("SHA-256 esperados de los datos (data/manifest.json):")
    for a, h in esperados(manifest).items():
        print(f"  {a}: {h}")
    for nombre, datos in generados.items():
        print(f"{CARPETA_RELATIVA}/{nombre}: {len(datos):,} bytes, SHA-256 {sha256_bytes(datos)}")
    print("Candado: sin test.jsonl y sin otros SHA-256 del manifest.")

    if args.escribir:
        CARPETA.mkdir(parents=True, exist_ok=True)
        for nombre, datos in generados.items():
            (CARPETA / nombre).write_bytes(datos)
        print(f"Escritos en {CARPETA}.")
    pendientes = diferencias(generados, CARPETA)
    if pendientes:
        mensaje = f"Distintos de lo generado (o faltan): {pendientes}."
        if args.verificar:
            sys.exit(f"Me detengo: {mensaje}")
        print(mensaje)
    else:
        print(f"{CARPETA_RELATIVA}: los archivos escritos son exactamente lo generado.")


if __name__ == "__main__":
    main()
