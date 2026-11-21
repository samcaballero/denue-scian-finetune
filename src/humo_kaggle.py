"""Kernel de Kaggle de la prueba de punta a punta (Fase 2, paso 6b).

Prueba desechable y no reportable con Qwen3-0.6B: QLoRA sobre `humo_train.jsonl`, inferencia
en `humo_val.jsonl`, exportación a GGUF Q4_K_M y una medición corta de Qwen3-4B. No escribe en
`results/`; las métricas de la regla 5 se calculan en la laptop (paso 6c) a partir de
`predicciones.jsonl`.

Este módulo no se sube solo: `src/construir_kernel.py` lo copia, junto con
`src/prompt_clasificador.py` y `src/normalizacion.py`, dentro de
`notebooks/kaggle-humo/humo_6b.py`, que es lo que se empuja a Kaggle. Arriba solo importa la
biblioteca estándar y esos dos módulos; torch, Unsloth y transformers se importan dentro de
las funciones, después de instalar las versiones fijadas.

Orden de las etapas (lo que puede fallar barato va antes de instalar):
 1. nvidia-smi: toda GPU debe ser T4; se usa una sola (CUDA_VISIBLE_DEVICES=0).
 2. Datos: SHA-256 de cada JSONL contra el manifest (valores incrustados por el generador).
 3. Instalación con versiones fijadas (pip freeze antes y después).
 4. Imports, una sola GPU visible y semillas.
 5. Qwen/Qwen3-0.6B en la revisión fijada (use_exact_model_name=True; NF4 de bitsandbytes).
 6. Plantilla y tokens: comprobaciones (a)-(d), etiquetas solo en la respuesta, filas > 128.
 7. LoRA y entrenamiento (125 pasos con lote 16); adaptador guardado.
 8. Inferencia greedy en las 200 filas de validación -> predicciones.jsonl (id y respuesta).
 9. Base de 16 bits de la fusión y GGUF Q4_K_M, con el prompt renderizado de un nombre inventado.
10. Mediciones de lote desechables en el 0.6B (32 y 64) con un modelo nuevo.
11. Qwen/Qwen3-4B: 50 pasos con lote 16 y unos 20 con lote 32; se omite si el reloj pasa de 60 min.

Ningún nom_estab va al log ni a `humo_mediciones.json`: el log solo lleva conteos, hashes y
tiempos; los errores se imprimen con los nombres tapados, y cada escritura del JSON tapa
cualquier texto que, normalizado, sea la llave de una fila de la muestra.
"""

from __future__ import annotations

import contextlib
import copy
import datetime
import gc
import hashlib
import json
import os
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.normalizacion import normalizar_nombre
from src.prompt_clasificador import (
    FIN_DE_TURNO,
    IGNORAR,
    INICIO_DE_TURNO,
    INSTRUCCION,
    RE_CODIGO,
    ErrorPrompt,
    etiquetas,
    mensajes,
    prompt_generacion,
    texto_entrenamiento,
    tiene_marca_especial,
)

# --- Configuración ---

SEMILLA = 42
ARCHIVO_TRAIN = "humo_train.jsonl"
ARCHIVO_VAL = "humo_val.jsonl"
CAMPOS = ("id", "nom_estab", "codigo_act")
RE_ID = re.compile(r"[0-9]+")

DIR_ENTRADA = Path("/kaggle/input")
DIR_SALIDA = Path("/kaggle/working")  # lo único que se descarga
DIR_TRABAJO = Path("/tmp/denue_trabajo")  # llama.cpp, fusión de 16 bits, caché de Unsloth
ARCHIVO_MEDICIONES = "humo_mediciones.json"
ARCHIVO_PREDICCIONES = "predicciones.jsonl"
CARPETA_GGUF = "gguf"
CARPETA_ADAPTADOR = "adaptador"
NOMBRE_GGUF = "qwen3-0.6b-humo-Q4_K_M.gguf"

GPU_ESPERADA = "T4"
MACHINE_SHAPE = "NvidiaTeslaT4"  # "GPU T4 x2" en la CLI de Kaggle 2.2.5
CONSULTA_GPU = ("index", "name", "memory.total", "driver_version")

# Versiones fijadas (plan del paso 6b). torch no se fija: se usa el de la imagen si cumple
# torch<2.15.0, que exige unsloth 2026.11.9; si no, pip instala el 2.14 más reciente.
PAQUETES = (
    "unsloth==2026.11.9",
    "unsloth_zoo==2026.11.8",
    "transformers==5.17.0",
    "trl==1.15.0",
    "datasets==4.8.5",
    "peft==0.21.3",
    "bitsandbytes==0.50.2",
    "accelerate==1.15.0",
)
VERSIONES_A_REGISTRAR = (
    "unsloth", "unsloth_zoo", "transformers", "trl", "datasets", "peft", "bitsandbytes",
    "accelerate", "torch", "torchvision", "xformers", "triton", "huggingface_hub",
    "tokenizers", "safetensors", "numpy",
)

# Revisiones: rama main de cada repo según la API de Hugging Face, consultada el 2026-11-21.
MODELO_06B = {"repo": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
MODELO_4B = {"repo": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c"}
# Base de 16 bits a la que la fusión de Unsloth 2026.11.9 resuelve Qwen/Qwen3-0.6B
# (get_model_name(..., load_in_4bit=False) usa MAP_TO_UNSLOTH_16bit, sin use_exact_model_name).
BASE_16_ESPEJO = {"repo": "unsloth/Qwen3-0.6B", "revision": "db4e90b8359f816f09bf5c944797ce116f26da3c"}
TENSOR_ENTRADA = "model.embed_tokens.weight"
TENSOR_SALIDA = "lm_head.weight"

MAX_SEQ = 128
LORA = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "use_gradient_checkpointing": "unsloth",
    "random_state": SEMILLA,
}
LOTE = 16
ENTRENAMIENTO = {
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 5,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "seed": SEMILLA,
    "data_seed": SEMILLA,
}
PASOS_CALENTAMIENTO = 5  # primeros pasos de cada corrida, fuera de las estadísticas de tiempo
PASOS_4B = 50
PASOS_LOTE = PASOS_CALENTAMIENTO + 20  # mediciones de lote: 5 de calentamiento + 20 medidos
LOTES_06B = (32, 64)
LOTES_4B_EXTRA = (32,)
PRESUPUESTO_4B_S = 3600  # el 4B se omite si al llegar a esa etapa pasaron más de 60 min

MAX_NEW_TOKENS = 16
TOKEN_FIN_DE_TEXTO = "<|endoftext|>"

# Ejemplo inventado (no sale del DENUE): tabla de etiquetas y prompt guardado junto al GGUF.
NOMBRE_INVENTADO = "Tlapalería La Ejemplar de Prueba"
CODIGO_INVENTADO = "467111"  # Comercio al por menor en ferreterías y tlapalerías (SCIAN 2023)

TAPADO = "[tapado]"


class AbortoHumo(RuntimeError):
    """Condición que detiene el kernel; el mensaje nunca incluye un nom_estab."""


def informar(mensaje: str) -> None:
    print(f"[humo {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


# --- Funciones puras (se prueban en la laptop) ---


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: str | Path, bloque: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while b := f.read(bloque):
            h.update(b)
    return h.hexdigest()


def buscar_unico(raiz: str | Path, nombre: str) -> Path:
    """La única ruta bajo `raiz` con ese nombre de archivo; aborta si hay cero o varias."""
    encontrados = sorted(p for p in Path(raiz).rglob(nombre) if p.is_file())
    if len(encontrados) != 1:
        raise AbortoHumo(f"Se esperaba exactamente un {nombre} bajo {raiz} y hay {len(encontrados)}.")
    return encontrados[0]


def leer_jsonl_verificado(ruta: str | Path, sha256_esperado: str) -> list[dict]:
    """Registros de una muestra, solo si su SHA-256 es el del manifest (se verifica antes de parsear)."""
    datos = Path(ruta).read_bytes()
    sha = sha256_bytes(datos)
    if sha != sha256_esperado:
        raise AbortoHumo(
            f"El SHA-256 de {Path(ruta).name} no coincide con el manifest: calculado {sha}, esperado {sha256_esperado}."
        )
    if datos.startswith(b"\xef\xbb\xbf"):
        raise AbortoHumo(f"{Path(ruta).name} tiene BOM.")
    texto = datos.decode("utf-8")
    if not texto.endswith("\n"):
        raise AbortoHumo(f"{Path(ruta).name} está vacío o no termina en salto de línea.")
    registros = []
    for i, linea in enumerate(texto[:-1].split("\n"), start=1):
        r = json.loads(linea)
        if not isinstance(r, dict) or tuple(r) != CAMPOS:
            raise AbortoHumo(f"{Path(ruta).name}, línea {i}: las claves no son id, nom_estab y codigo_act en orden.")
        if not all(isinstance(r[c], str) for c in CAMPOS):
            raise AbortoHumo(f"{Path(ruta).name}, línea {i}: un campo no es texto.")
        if not RE_ID.fullmatch(r["id"]):
            raise AbortoHumo(f"{Path(ruta).name}, línea {i}: el id no es solo dígitos ASCII.")
        if not RE_CODIGO.fullmatch(r["codigo_act"]):
            raise AbortoHumo(f"{Path(ruta).name}, línea {i}: codigo_act no es de seis dígitos ASCII.")
        registros.append(r)
    if len({r["id"] for r in registros}) != len(registros):
        raise AbortoHumo(f"{Path(ruta).name} tiene ids repetidos.")
    return registros


def parsear_nvidia_smi(salida: str) -> list[dict]:
    """Filas de `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits`."""
    gpus = []
    for linea in salida.strip().splitlines():
        if not linea.strip():
            continue
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) != len(CONSULTA_GPU):
            raise AbortoHumo("La salida de nvidia-smi no tiene el formato esperado.")
        try:
            gpus.append(
                {"indice": int(partes[0]), "nombre": partes[1], "memoria_mib": int(float(partes[2])), "driver": partes[3]}
            )
        except ValueError as e:
            raise AbortoHumo("La salida de nvidia-smi no tiene el formato esperado.") from e
    return gpus


def verificar_t4(gpus: Sequence[Mapping]) -> None:
    """Aborta si no hay GPU o si alguna no es T4."""
    if not gpus:
        raise AbortoHumo("nvidia-smi no reporta ninguna GPU.")
    otras = sorted({g["nombre"] for g in gpus if GPU_ESPERADA not in g["nombre"]})
    if otras:
        raise AbortoHumo(f"La sesión no es T4: {otras}.")


def diferencias_freeze(antes: str, despues: str) -> dict:
    """Paquetes agregados, quitados y cambiados entre dos salidas de `pip freeze`."""

    def leer(texto: str) -> dict[str, str]:
        res = {}
        for linea in texto.splitlines():
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            nombre, sep, resto = linea.partition("==")
            if not sep:
                nombre, sep, resto = linea.partition(" @ ")
            res[nombre.strip().lower().replace("_", "-")] = resto.strip() if sep else linea
        return res

    a, d = leer(antes), leer(despues)
    return {
        "agregados": {k: d[k] for k in sorted(set(d) - set(a))},
        "quitados": {k: a[k] for k in sorted(set(a) - set(d))},
        "cambiados": {k: [a[k], d[k]] for k in sorted(set(a) & set(d)) if a[k] != d[k]},
    }


def percentil(valores: Sequence[float], p: float) -> float | None:
    """Percentil con interpolación lineal (la regla por omisión de numpy)."""
    v = sorted(valores)
    if not v:
        return None
    pos = (len(v) - 1) * p / 100
    i = int(pos)
    if i + 1 >= len(v):
        return v[-1]
    return v[i] + (v[i + 1] - v[i]) * (pos - i)


def estadisticas_pasos(
    tiempos: Sequence[float],
    ejemplos: Sequence[int],
    fichas: Sequence[int],
    calentamiento: int = PASOS_CALENTAMIENTO,
) -> dict:
    """Tiempo por paso, ejemplos/s y fichas/s sin los pasos de calentamiento."""
    if not len(tiempos) == len(ejemplos) == len(fichas):
        raise ValueError("tiempos, ejemplos y fichas deben tener el mismo largo.")
    t, e, f = list(tiempos[calentamiento:]), list(ejemplos[calentamiento:]), list(fichas[calentamiento:])
    res = {
        "pasos_total": len(tiempos),
        "pasos_calentamiento": min(calentamiento, len(tiempos)),
        "pasos_medidos": len(t),
        "segundos_con_calentamiento": round(sum(tiempos), 4),
    }
    total = sum(t)
    if not t or total <= 0:
        return res
    res.update(
        {
            "segundos_medidos": round(total, 4),
            "s_por_paso_media": round(total / len(t), 5),
            "s_por_paso_mediana": round(percentil(t, 50), 5),
            "s_por_paso_p90": round(percentil(t, 90), 5),
            "s_por_paso_min": round(min(t), 5),
            "s_por_paso_max": round(max(t), 5),
            "ejemplos_por_s": round(sum(e) / total, 3),
            "fichas_por_s": round(sum(f) / total, 1),
        }
    )
    return res


def colar(ejemplos: Sequence[Mapping], pad_id: int) -> tuple[dict[str, list[list[int]]], dict[str, int]]:
    """Lote con relleno a la derecha: pad_id en input_ids, 0 en la máscara e IGNORAR en las etiquetas."""
    if not ejemplos:
        raise ValueError("Lote vacío.")
    largo = max(len(e["input_ids"]) for e in ejemplos)
    ids, mascara, etiq = [], [], []
    reales = etiquetadas = 0
    for e in ejemplos:
        x, y = list(e["input_ids"]), list(e["labels"])
        if len(x) != len(y):
            raise ValueError("input_ids y labels tienen largos distintos.")
        relleno = largo - len(x)
        ids.append(x + [pad_id] * relleno)
        mascara.append([1] * len(x) + [0] * relleno)
        etiq.append(y + [IGNORAR] * relleno)
        reales += len(x)
        etiquetadas += sum(1 for v in y if v != IGNORAR)
    medidas = {
        "ejemplos": len(ejemplos),
        "fichas_reales": reales,
        "fichas_con_relleno": largo * len(ejemplos),
        "fichas_etiquetadas": etiquetadas,
    }
    return {"input_ids": ids, "attention_mask": mascara, "labels": etiq}, medidas


def cortar_respuesta(ids: Sequence[int], ids_fin: Sequence[int]) -> tuple[list[int], str]:
    """Tokens generados hasta antes del primer token de fin, y por qué terminó."""
    fin = set(ids_fin)
    for i, t in enumerate(ids):
        if t in fin:
            return list(ids[:i]), "token_de_fin"
    return list(ids), "limite_de_tokens"


def serializar_predicciones(pares: Sequence[tuple[str, str]]) -> bytes:
    """predicciones.jsonl: solo id y respuesta cruda; UTF-8 sin BOM, "\\n" al final de cada línea."""
    lineas = []
    for id_, respuesta in pares:
        if not isinstance(id_, str) or not RE_ID.fullmatch(id_) or not isinstance(respuesta, str):
            raise ValueError("Cada predicción necesita un id de dígitos y una respuesta de texto.")
        lineas.append(json.dumps({"id": id_, "respuesta": respuesta}, ensure_ascii=False) + "\n")
    return "".join(lineas).encode("utf-8")


def tapar_nombres(texto: str, nombres: Sequence[str], minimo: int = 3) -> str:
    """Texto con cada nom_estab (de `minimo` caracteres o más, sin espacios en las orillas) tapado."""
    for n in sorted({n.strip() for n in nombres if len(n.strip()) >= minimo}, key=len, reverse=True):
        texto = texto.replace(n, "[nombre]")
    return texto


def tapar_llaves(obj, llaves: frozenset[str] | set[str], ruta: str = "$") -> tuple[object, list[str]]:
    """Copia del objeto JSON con cada texto que, normalizado, sea una llave de la muestra cambiado por TAPADO.

    Los textos que normalizados solo tienen dígitos (códigos, ids, fechas) se aceptan. Devuelve
    también las rutas tapadas (nunca el texto).
    """
    rutas: list[str] = []
    if isinstance(obj, str):
        k = normalizar_nombre(obj)
        if k and not k.replace(" ", "").isdigit() and k in llaves:
            return TAPADO, [ruta]
        return obj, rutas
    if isinstance(obj, Mapping):
        nuevo = {}
        for k, v in obj.items():
            nuevo[k], r = tapar_llaves(v, llaves, f"{ruta}.{k}")
            rutas.extend(r)
        return nuevo, rutas
    if isinstance(obj, (list, tuple)):
        nuevo_l = []
        for i, v in enumerate(obj):
            x, r = tapar_llaves(v, llaves, f"{ruta}[{i}]")
            nuevo_l.append(x)
            rutas.extend(r)
        return nuevo_l, rutas
    return obj, rutas


def tabla_etiquetas(piezas: Sequence[str], ids: Sequence[int], etiq: Sequence[int]) -> list[str]:
    """Tabla token por token: posición, id, etiqueta (-100 = ignorada) y texto del token."""
    lineas = [f"{'i':>3}  {'id':>7}  {'etiqueta':>8}  token"]
    for i, (p, t, e) in enumerate(zip(piezas, ids, etiq)):
        lineas.append(f"{i:>3}  {t:>7}  {e:>8}  {p!r}")
    return lineas


def hashes_safetensors(rutas: Sequence[str | Path]) -> dict[str, dict]:
    """SHA-256 de los bytes de cada tensor (con su dtype y forma), leyendo safetensors sin dependencias."""
    res: dict[str, dict] = {}
    for ruta in rutas:
        with open(ruta, "rb") as f:
            cabeza = f.read(8)
            if len(cabeza) != 8:
                raise AbortoHumo(f"{Path(ruta).name} no es un safetensors válido.")
            n = struct.unpack("<Q", cabeza)[0]
            if n > 100 * 2**20:
                raise AbortoHumo(f"La cabecera de {Path(ruta).name} es demasiado grande.")
            cabecera = json.loads(f.read(n))
            cabecera.pop("__metadata__", None)
            base = 8 + n
            for nombre, info in sorted(cabecera.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                if nombre in res:
                    raise AbortoHumo(f"Tensor repetido entre archivos: {nombre}.")
                inicio, fin = info["data_offsets"]
                f.seek(base + inicio)
                h = hashlib.sha256()
                restante = fin - inicio
                while restante > 0:
                    b = f.read(min(1 << 24, restante))
                    if not b:
                        raise AbortoHumo(f"{Path(ruta).name} está truncado.")
                    h.update(b)
                    restante -= len(b)
                res[nombre] = {"dtype": info["dtype"], "shape": list(info["shape"]), "sha256": h.hexdigest()}
    return res


def comparar_tensores(
    oficial: Mapping[str, Mapping], espejo: Mapping[str, Mapping], atado_oficial: bool, atado_espejo: bool
) -> dict:
    """¿Los tensores del espejo son los del repo oficial?

    Cada tensor común debe tener el mismo dtype, forma y SHA-256. Un tensor de un solo lado solo
    se acepta si es `lm_head.weight`, los dos config dicen `tie_word_embeddings` y, en ese lado,
    `lm_head.weight` es idéntico a `model.embed_tokens.weight` (la salida atada a la entrada).
    """
    comunes = sorted(set(oficial) & set(espejo))
    distintos = [n for n in comunes if dict(oficial[n]) != dict(espejo[n])]
    solo_oficial = sorted(set(oficial) - set(espejo))
    solo_espejo = sorted(set(espejo) - set(oficial))

    def extra_aceptable(extras: list[str], lado: Mapping[str, Mapping]) -> bool:
        if not extras:
            return True
        return (
            extras == [TENSOR_SALIDA]
            and atado_oficial
            and atado_espejo
            and TENSOR_ENTRADA in lado
            and dict(lado[TENSOR_SALIDA]) == dict(lado[TENSOR_ENTRADA])
        )

    ok_oficial = extra_aceptable(solo_oficial, oficial)
    ok_espejo = extra_aceptable(solo_espejo, espejo)
    return {
        "identicos": bool(comunes) and not distintos and ok_oficial and ok_espejo,
        "tensores_oficial": len(oficial),
        "tensores_espejo": len(espejo),
        "comunes": len(comunes),
        "distintos": distintos,
        "solo_oficial": solo_oficial,
        "solo_espejo": solo_espejo,
        "solo_un_lado_aceptado_por_atadura": (bool(solo_oficial) and ok_oficial) or (bool(solo_espejo) and ok_espejo),
        "tie_word_embeddings": {"oficial": atado_oficial, "espejo": atado_espejo},
    }


def configuracion() -> dict:
    """Configuración de la corrida, para el JSON de mediciones."""
    return {
        "semilla": SEMILLA,
        "paquetes_fijados": list(PAQUETES),
        "modelo_06b": MODELO_06B,
        "modelo_4b": MODELO_4B,
        "base_16_espejo": BASE_16_ESPEJO,
        "use_exact_model_name": True,
        "load_in_4bit": True,
        "max_seq_length": MAX_SEQ,
        "lora": LORA,
        "lote": LOTE,
        "entrenamiento": ENTRENAMIENTO,
        "pasos_calentamiento_excluidos": PASOS_CALENTAMIENTO,
        "pasos_4b": PASOS_4B,
        "pasos_mediciones_de_lote": PASOS_LOTE,
        "lotes_06b": list(LOTES_06B),
        "lotes_4b_extra": list(LOTES_4B_EXTRA),
        "presupuesto_4b_s": PRESUPUESTO_4B_S,
        "max_new_tokens": MAX_NEW_TOKENS,
        "instruccion": INSTRUCCION,
        "machine_shape": MACHINE_SHAPE,
    }


# --- Registro de mediciones ---


class Registro:
    """humo_mediciones.json: se reescribe al terminar cada etapa, así un fallo deja lo anterior."""

    def __init__(self, ruta: str | Path, t_arranque: float):
        self.ruta = Path(ruta)
        self.t0 = t_arranque
        self.nombres: tuple[str, ...] = ()
        self.llaves: frozenset[str] = frozenset()
        self.datos: dict = {
            "descripcion": "Prueba de punta a punta desechable (Fase 2, paso 6b); sus números no se reportan.",
            "estado": "en_curso",
            "inicio_utc": datetime.datetime.fromtimestamp(t_arranque, datetime.timezone.utc).isoformat(timespec="seconds"),
            "duraciones_s": {},
            "fallas": [],
        }

    def transcurrido(self) -> float:
        return time.time() - self.t0

    def escribir(self) -> None:
        limpio, rutas = tapar_llaves(self.datos, self.llaves)
        if rutas:
            limpio["tapados"] = rutas
        texto = json.dumps(limpio, ensure_ascii=False, indent=2, default=str) + "\n"
        temporal = self.ruta.with_name(self.ruta.name + ".tmp")
        temporal.write_bytes(texto.encode("utf-8"))
        temporal.replace(self.ruta)

    def falla(self, etapa: str, error: BaseException | str) -> None:
        mensaje = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self.datos["fallas"].append({"etapa": etapa, "error": tapar_nombres(str(mensaje), self.nombres)[:2000]})

    @contextlib.contextmanager
    def etapa(self, nombre: str):
        informar(f"inicio: {nombre}")
        t = time.perf_counter()
        try:
            yield
        finally:
            duracion = time.perf_counter() - t
            self.datos["duraciones_s"][nombre] = round(duracion, 3)
            self.datos["segundos_desde_arranque"] = round(self.transcurrido(), 1)
            self.escribir()
            informar(f"fin: {nombre} ({duracion:.1f} s; {self.transcurrido() / 60:.1f} min desde el arranque)")


# --- Etapas en Kaggle (no se prueban en la laptop: necesitan GPU, torch y Unsloth) ---


def consultar_gpus() -> list[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(CONSULTA_GPU)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError as e:
        raise AbortoHumo("No hay nvidia-smi: la sesión no tiene GPU.") from e
    if r.returncode != 0:
        raise AbortoHumo(f"nvidia-smi falló con código {r.returncode}.")
    return parsear_nvidia_smi(r.stdout)


def _pip(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True)


def instalar(reg: Registro) -> None:
    antes = _pip("freeze").stdout
    (DIR_SALIDA / "pip_freeze_antes.txt").write_text(antes, encoding="utf-8")
    t = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", *PAQUETES],
        text=True,
    )
    segundos = time.perf_counter() - t
    if r.returncode != 0:
        raise AbortoHumo(f"pip install falló con código {r.returncode}.")
    despues = _pip("freeze").stdout
    (DIR_SALIDA / "pip_freeze_despues.txt").write_text(despues, encoding="utf-8")
    reg.datos["instalacion"] = {
        "paquetes": list(PAQUETES),
        "segundos_pip_install": round(segundos, 1),
        "cambios_freeze": diferencias_freeze(antes, despues),
    }


def importar(reg: Registro) -> types.SimpleNamespace:
    """Importa Unsloth antes que transformers, verifica una sola T4 visible y fija las semillas."""
    import unsloth  # noqa: F401  (debe ir antes de transformers, trl y peft)
    from unsloth import FastLanguageModel

    import datasets
    import huggingface_hub
    import numpy
    import torch
    import transformers
    from transformers import Trainer, TrainerCallback, TrainingArguments

    if not torch.cuda.is_available():
        raise AbortoHumo("torch no ve ninguna GPU.")
    if torch.cuda.device_count() != 1:
        raise AbortoHumo(f"torch ve {torch.cuda.device_count()} GPU; se esperaba una sola.")
    nombre = torch.cuda.get_device_name(0)
    if GPU_ESPERADA not in nombre:
        raise AbortoHumo(f"torch ve una GPU distinta de T4: {nombre}.")

    random.seed(SEMILLA)
    numpy.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)
    torch.cuda.manual_seed_all(SEMILLA)
    transformers.set_seed(SEMILLA)

    from importlib import metadata

    versiones = {}
    for p in VERSIONES_A_REGISTRAR:
        try:
            versiones[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versiones[p] = None
    reg.datos["gpu"].update(
        {
            "torch_device_count": torch.cuda.device_count(),
            "torch_device_name": nombre,
            "memoria_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
            "capacidad_de_computo": list(torch.cuda.get_device_capability(0)),
            "bf16_nativo": bf16_nativo(torch),
        }
    )
    reg.datos["entorno"] = {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "versiones": versiones,
    }
    return types.SimpleNamespace(
        torch=torch, transformers=transformers, datasets=datasets, hf=huggingface_hub,
        FastLanguageModel=FastLanguageModel, Trainer=Trainer, TrainingArguments=TrainingArguments,
        TrainerCallback=TrainerCallback,
    )


def bf16_nativo(torch) -> bool:
    """bf16 nativo solo desde la capacidad de cómputo 8.0 (la T4 es 7.5).

    No se usa `torch.cuda.is_bf16_supported()`: por omisión cuenta la emulación y en una T4
    puede responder True.
    """
    return torch.cuda.get_device_capability(0)[0] >= 8


def _lfs_sha256(sibling) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    return lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)


def descargar_verificado(mods, repo: str, revision: str, patrones: Sequence[str] | None = None) -> tuple[Path, dict]:
    """Descarga la revisión fijada y verifica la carpeta de la revisión y el SHA-256 de los archivos LFS."""
    ruta = Path(mods.hf.snapshot_download(repo_id=repo, revision=revision, allow_patterns=patrones))
    if ruta.name != revision:
        raise AbortoHumo(f"{repo}: se descargó la revisión {ruta.name}, no la fijada {revision}.")
    info = mods.hf.HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise AbortoHumo(f"{repo}: Hugging Face reporta la revisión {info.sha}, no la fijada {revision}.")
    verificados = {}
    for s in info.siblings:
        esperado = _lfs_sha256(s)
        local = ruta / s.rfilename
        if esperado is None or not local.exists():
            continue
        calculado = sha256_archivo(local)
        if calculado != esperado:
            raise AbortoHumo(f"{repo}/{s.rfilename}: el SHA-256 no es el de Hugging Face.")
        verificados[s.rfilename] = {"sha256": calculado, "bytes": local.stat().st_size}
    if not verificados:
        raise AbortoHumo(f"{repo}: no se verificó ningún archivo LFS.")
    return ruta, verificados


def cargar(mods, modelo: Mapping[str, str]) -> tuple[object, object, dict]:
    """Modelo en 4 bits (NF4 de bitsandbytes) desde el repo exacto y la revisión fijada."""
    torch = mods.torch
    torch.cuda.reset_peak_memory_stats()
    model, tok = mods.FastLanguageModel.from_pretrained(
        model_name=modelo["repo"],
        revision=modelo["revision"],
        use_exact_model_name=True,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ,
        dtype=None,
        random_state=SEMILLA,
    )
    qc = getattr(model.config, "quantization_config", None)
    qd = qc if isinstance(qc, Mapping) else (qc.to_dict() if qc is not None else {})
    if not qd.get("load_in_4bit") or qd.get("bnb_4bit_quant_type") != "nf4":
        raise AbortoHumo(f"{modelo['repo']}: no se cargó en NF4 de 4 bits ({qd.get('bnb_4bit_quant_type')}).")
    if tok.pad_token_id is None:
        raise AbortoHumo(f"{modelo['repo']}: el tokenizador no tiene token de relleno.")
    capas_4bit = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
    info = {
        "repo_pedido": modelo["repo"],
        "revision_fijada": modelo["revision"],
        "name_or_path": model.config._name_or_path,
        "dtype": str(getattr(model, "dtype", None)),
        "cuantizacion": {k: qd.get(k) for k in sorted(qd) if k.startswith(("load_in", "bnb_", "quant_method"))},
        "capas_linear4bit": capas_4bit,
        "vram_tras_carga_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "tokenizador": {
            "eos": tok.eos_token,
            "pad": tok.pad_token,
            "pad_id": tok.pad_token_id,
            "padding_side": getattr(tok, "padding_side", None),
            "plantilla_sha256": sha256_bytes(str(tok.chat_template).encode("utf-8")),
        },
    }
    return model, tok, info


def aplicar_lora(mods, model) -> tuple[object, dict]:
    mods.transformers.set_seed(SEMILLA)
    model = mods.FastLanguageModel.get_peft_model(model, **LORA)
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, {"parametros_entrenables": entrenables, "parametros_totales": total}


def preparar_tokens(reg: Registro, tok, train: Sequence[Mapping], val: Sequence[Mapping]) -> dict:
    """Comprobaciones (a)-(c), etiquetas y filas de más de MAX_SEQ fichas; aborta si algo falla.

    (d) (nombres con "<|") se cuenta en la etapa de datos. (b) solo se registra.
    """
    comp = reg.datos["comprobaciones"]
    id_inicio = tok.convert_tokens_to_ids(INICIO_DE_TURNO)
    id_fin = tok.convert_tokens_to_ids(FIN_DE_TURNO)
    id_fin_texto = tok.convert_tokens_to_ids(TOKEN_FIN_DE_TEXTO)
    c_ok = (
        tok(INICIO_DE_TURNO, add_special_tokens=False)["input_ids"] == [id_inicio]
        and tok(FIN_DE_TURNO, add_special_tokens=False)["input_ids"] == [id_fin]
        and tok(TOKEN_FIN_DE_TEXTO, add_special_tokens=False)["input_ids"] == [id_fin_texto]
        and len({id_inicio, id_fin, id_fin_texto}) == 3
    )
    comp["c_tokens_especiales_un_solo_token"] = {
        "ok": c_ok, "im_start": id_inicio, "im_end": id_fin, "endoftext": id_fin_texto,
    }
    if not c_ok:
        raise AbortoHumo("(c) <|im_start|>, <|im_end|> o <|endoftext|> no son un solo token.")

    filas = [("train", r) for r in train] + [("val", r) for r in val]
    # (a) prompt de generación de la plantilla == prompt_generacion, byte a byte.
    nombres = [r["nom_estab"] for _, r in filas] + [NOMBRE_INVENTADO]
    distintos_a = sum(
        tok.apply_chat_template(mensajes(n), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        != prompt_generacion(n)
        for n in nombres
    )
    comp["a_prompt_generacion_igual_a_plantilla"] = {"textos": len(nombres), "distintos": distintos_a}
    if distintos_a:
        raise AbortoHumo(f"(a) La plantilla no da prompt_generacion en {distintos_a} de {len(nombres)} textos.")

    # (b) plantilla con la respuesta frente al texto de entrenamiento construido: solo se registra.
    cuenta_b = {"igual": 0, "igual_mas_salto_final": 0, "otro": 0}
    for _, r in filas:
        plantilla = tok.apply_chat_template(
            mensajes(r["nom_estab"]) + [{"role": "assistant", "content": r["codigo_act"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        texto = texto_entrenamiento(r["nom_estab"], r["codigo_act"])
        clave = "igual" if plantilla == texto else ("igual_mas_salto_final" if plantilla == texto + "\n" else "otro")
        cuenta_b[clave] += 1
    plantilla_inv = tok.apply_chat_template(
        mensajes(NOMBRE_INVENTADO) + [{"role": "assistant", "content": CODIGO_INVENTADO}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    texto_inv = texto_entrenamiento(NOMBRE_INVENTADO, CODIGO_INVENTADO)
    if plantilla_inv.startswith(texto_inv):
        diferencia = {"plantilla_agrega_al_final": plantilla_inv[len(texto_inv):]}
    else:
        n = next((i for i, (x, y) in enumerate(zip(plantilla_inv, texto_inv)) if x != y), min(len(plantilla_inv), len(texto_inv)))
        diferencia = {"primera_diferencia_en": n, "plantilla": plantilla_inv[n:n + 40], "construido": texto_inv[n:n + 40]}
    comp["b_texto_entrenamiento_frente_a_plantilla"] = {
        "solo_registro": True, "conteo": cuenta_b, "ejemplo_inventado": diferencia,
    }

    # Etiquetas y largos.
    ejemplos: dict[str, list[dict]] = {"train": [], "val": []}
    no_prefijo = etiquetas_mal = 0
    largos, etiquetados = [], {}
    for conjunto, r in filas:
        ids_p = tok(prompt_generacion(r["nom_estab"]), add_special_tokens=False)["input_ids"]
        ids_c = tok(texto_entrenamiento(r["nom_estab"], r["codigo_act"]), add_special_tokens=False)["input_ids"]
        largos.append(len(ids_c))
        try:
            etq = etiquetas(ids_p, ids_c)
        except ErrorPrompt:
            no_prefijo += 1
            continue
        con_etiqueta = [t for t, e in zip(ids_c, etq) if e != IGNORAR]
        etiquetados[len(con_etiqueta)] = etiquetados.get(len(con_etiqueta), 0) + 1
        if tok.decode(con_etiqueta, skip_special_tokens=False) != r["codigo_act"] + FIN_DE_TURNO or con_etiqueta[-1] != id_fin:
            etiquetas_mal += 1
        ejemplos[conjunto].append({"input_ids": ids_c, "labels": etq})
    mayores = sum(1 for n in largos if n > MAX_SEQ)
    reg.datos["tokens"] = {
        "filas": len(filas),
        "largo_max": max(largos),
        "largo_medio": round(sum(largos) / len(largos), 2),
        "mayores_a_max_seq": mayores,
        "max_seq_length": MAX_SEQ,
        "prompt_no_es_prefijo": no_prefijo,
        "etiquetas_distintas_de_codigo_y_fin": etiquetas_mal,
        "fichas_etiquetadas_por_fila": {str(k): v for k, v in sorted(etiquetados.items())},
    }
    if no_prefijo or etiquetas_mal:
        raise AbortoHumo(f"Etiquetas: {no_prefijo} filas sin prefijo y {etiquetas_mal} con etiquetas incorrectas.")
    if mayores:
        raise AbortoHumo(f"{mayores} filas pasan de {MAX_SEQ} fichas; no se trunca.")

    # Ejemplo inventado: tabla de etiquetas en el log y en el JSON.
    ids_p = tok(prompt_generacion(NOMBRE_INVENTADO), add_special_tokens=False)["input_ids"]
    ids_c = tok(texto_inv, add_special_tokens=False)["input_ids"]
    etq = etiquetas(ids_p, ids_c)
    piezas = [tok.decode([t], skip_special_tokens=False) for t in ids_c]
    tabla = tabla_etiquetas(piezas, ids_c, etq)
    print("\n".join(["Tokens del ejemplo inventado (etiqueta -100 = no entra a la pérdida):", *tabla]), flush=True)
    con_etiqueta = [p for p, e in zip(piezas, etq) if e != IGNORAR]
    print(f"Tokens con etiqueta: {con_etiqueta}", flush=True)
    reg.datos["ejemplo_etiquetas"] = {
        "codigo": CODIGO_INVENTADO,
        "fichas": len(ids_c),
        "fichas_con_etiqueta": con_etiqueta,
        "tabla": [{"i": i, "id": t, "etiqueta": e, "token": p} for i, (p, t, e) in enumerate(zip(piezas, ids_c, etq))],
    }
    return {"ejemplos_train": ejemplos["train"], "ids_fin": [id_fin, id_fin_texto]}


def _cronometro(mods):
    """Callback que mide el tiempo entre fines de paso consecutivos (con sincronización de CUDA)."""
    torch = mods.torch

    class Cronometro(mods.TrainerCallback):
        def __init__(self):
            self.tiempos: list[float] = []
            self._t = None

        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self._t = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            t = time.perf_counter()
            self.tiempos.append(t - self._t)
            self._t = t

    return Cronometro()


class ColadorMedido:
    """Colador que registra ejemplos y fichas de cada lote, en orden (dataloader sin workers)."""

    def __init__(self, torch, pad_id: int):
        self.torch = torch
        self.pad_id = pad_id
        self.lotes: list[dict] = []

    def __call__(self, ejemplos):
        lote, medidas = colar(ejemplos, self.pad_id)
        self.lotes.append(medidas)
        return {k: self.torch.tensor(v, dtype=self.torch.long) for k, v in lote.items()}


def entrenar(mods, model, tok, ejemplos: Sequence[Mapping], *, lote: int, etiqueta: str, max_pasos: int | None = None) -> dict:
    """Entrena con Trainer y el colador propio; devuelve tiempos, fichas, pérdida y VRAM pico."""
    torch = mods.torch
    colador = ColadorMedido(torch, tok.pad_token_id)
    cronometro = _cronometro(mods)
    bf16 = bf16_nativo(torch)
    args = mods.TrainingArguments(
        output_dir=str(DIR_TRABAJO / f"trainer_{etiqueta}"),
        per_device_train_batch_size=lote,
        max_steps=max_pasos if max_pasos else -1,
        fp16=not bf16,
        bf16=bf16,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        disable_tqdm=True,
        **ENTRENAMIENTO,
    )
    ds = mods.datasets.Dataset.from_list([{"input_ids": e["input_ids"], "labels": e["labels"]} for e in ejemplos])
    mods.transformers.set_seed(SEMILLA)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    trainer = mods.Trainer(model=model, args=args, train_dataset=ds, data_collator=colador, callbacks=[cronometro])
    trainer.train()
    torch.cuda.synchronize()
    segundos = time.perf_counter() - t
    pasos = len(cronometro.tiempos)
    lotes = colador.lotes[:pasos]
    res = {
        "lote": lote,
        "max_pasos": max_pasos,
        "fp16": not bf16,
        "segundos_train_total": round(segundos, 2),
        "lotes_registrados_por_el_colador": len(colador.lotes),
        "vram_pico_mib": {
            "asignada": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "reservada": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        },
    }
    if len(lotes) == pasos:
        res["estadisticas"] = estadisticas_pasos(
            cronometro.tiempos, [x["ejemplos"] for x in lotes], [x["fichas_reales"] for x in lotes]
        )
        res["pasos"] = [
            {"paso": i + 1, "s": round(s, 5), **x} for i, (s, x) in enumerate(zip(cronometro.tiempos, lotes))
        ]
    else:
        res["estadisticas"] = {"error": "El colador registró menos lotes que pasos; no se emparejan."}
        res["pasos_s"] = [round(s, 5) for s in cronometro.tiempos]
    res["perdida"] = [
        {"paso": h.get("step"), "loss": h.get("loss"), "grad_norm": h.get("grad_norm"), "learning_rate": h.get("learning_rate")}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    del trainer
    return res


def _es_sin_memoria(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def medir_lote(mods, model, tok, ejemplos, *, lote: int, etiqueta: str) -> dict:
    """Medición desechable de un tamaño de lote; si se queda sin memoria, lo registra y sigue."""
    try:
        return entrenar(mods, model, tok, ejemplos, lote=lote, etiqueta=etiqueta, max_pasos=PASOS_LOTE)
    except RuntimeError as e:
        if not _es_sin_memoria(e):
            raise
        return {"lote": lote, "sin_memoria": True, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        gc.collect()
        mods.torch.cuda.empty_cache()


def inferir(reg: Registro, mods, model, tok, val: Sequence[Mapping], ids_fin: Sequence[int]) -> None:
    """Greedy, lote de 1, misma plantilla; escribe predicciones.jsonl (id y respuesta cruda)."""
    torch = mods.torch
    mods.FastLanguageModel.for_inference(model)
    gen = copy.deepcopy(model.generation_config)
    gen.do_sample = False
    gen.num_beams = 1
    for atributo in ("temperature", "top_p", "top_k", "min_p"):
        if hasattr(gen, atributo):
            setattr(gen, atributo, None)
    gen.max_new_tokens = MAX_NEW_TOKENS
    gen.eos_token_id = list(ids_fin)
    gen.pad_token_id = tok.pad_token_id
    torch.cuda.reset_peak_memory_stats()
    pares, motivos, generadas = [], {}, []
    respaldo = False
    t = time.perf_counter()
    for i, r in enumerate(val, start=1):
        enc = tok(prompt_generacion(r["nom_estab"]), add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            try:
                salida = model.generate(**enc, generation_config=gen)
            except TypeError:
                # Respaldo, solo una vez: si la generación no acepta None en los parámetros de
                # muestreo, se ponen valores neutros. Con do_sample=False siguen sin usarse (greedy).
                if respaldo or i != 1:
                    raise
                respaldo = True
                gen.temperature, gen.top_p, gen.top_k = 1.0, 1.0, 50
                salida = model.generate(**enc, generation_config=gen)
        nuevos = salida[0, enc["input_ids"].shape[1]:].tolist()
        corte, motivo = cortar_respuesta(nuevos, ids_fin)
        pares.append((r["id"], tok.decode(corte, skip_special_tokens=False)))
        motivos[motivo] = motivos.get(motivo, 0) + 1
        generadas.append(len(nuevos))
        if i % 50 == 0:
            informar(f"inferencia: {i} de {len(val)} filas")
    torch.cuda.synchronize()
    segundos = time.perf_counter() - t
    datos = serializar_predicciones(pares)
    ruta = DIR_SALIDA / ARCHIVO_PREDICCIONES
    ruta.write_bytes(datos)
    reg.datos["inferencia"] = {
        "filas": len(pares),
        "lote": 1,
        "segundos": round(segundos, 2),
        "s_por_fila": round(segundos / max(len(pares), 1), 4),
        "terminacion": motivos,
        "fichas_generadas_max": max(generadas) if generadas else 0,
        "respaldo_parametros_de_muestreo": respaldo,
        "generation_config": {
            k: getattr(gen, k, None)
            for k in ("do_sample", "num_beams", "temperature", "top_p", "top_k", "max_new_tokens", "eos_token_id", "pad_token_id")
        },
        "vram_pico_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "archivo": {"nombre": ARCHIVO_PREDICCIONES, "bytes": len(datos), "sha256": sha256_bytes(datos)},
    }


def _atado(ruta_config: Path) -> bool:
    return bool(json.loads(Path(ruta_config).read_text(encoding="utf-8")).get("tie_word_embeddings"))


def verificar_base_16(reg: Registro, mods, model, ruta_oficial: Path) -> bool:
    """¿La fusión para el GGUF usará pesos idénticos a los oficiales en la revisión fijada?

    Unsloth fusiona sobre la base de 16 bits que resuelve `get_model_name(_name_or_path,
    load_in_4bit=False)`, en la revisión main del repo resuelto. Camino "exacto": resuelve al
    repo oficial (o a nuestra carpeta local) y main es la revisión fijada. Camino "espejo":
    resuelve a unsloth/Qwen3-0.6B; main debe ser la revisión fijada del espejo y sus tensores,
    idénticos a los oficiales (SHA-256 por tensor, con lm_head atado a los embeddings).
    """
    info: dict = {"name_or_path": model.config._name_or_path}
    reg.datos["base_16_bits"] = info
    try:
        from unsloth.models.loader_utils import get_model_name

        resuelto = get_model_name(model.config._name_or_path, load_in_4bit=False) or model.config._name_or_path
    except Exception as e:  # la fusión hace lo mismo: si falla, usa _name_or_path
        info["get_model_name_error"] = f"{type(e).__name__}"
        resuelto = model.config._name_or_path
    info["base_resuelta_por_unsloth"] = str(resuelto)
    api = mods.hf.HfApi()
    if os.path.isdir(str(resuelto)):
        info["camino"] = "exacto_local"
        ok = Path(resuelto).resolve() == Path(ruta_oficial).resolve()
        info["carpeta_es_la_revision_fijada"] = ok
        return ok
    if str(resuelto).lower() == MODELO_06B["repo"].lower():
        info["camino"] = "exacto"
        main = api.model_info(MODELO_06B["repo"]).sha
        info["main_actual"] = main
        return main == MODELO_06B["revision"]
    if str(resuelto).lower() != BASE_16_ESPEJO["repo"].lower():
        info["camino"] = "desconocido"
        return False

    info["camino"] = "espejo"
    main = api.model_info(BASE_16_ESPEJO["repo"]).sha
    info["main_actual_del_espejo"] = main
    info["revision_fijada_del_espejo"] = BASE_16_ESPEJO["revision"]
    if main != BASE_16_ESPEJO["revision"]:
        info["motivo"] = "La rama main del espejo no es la revisión fijada: la fusión usaría pesos no verificados."
        return False
    ruta_espejo, verif = descargar_verificado(
        mods, BASE_16_ESPEJO["repo"], BASE_16_ESPEJO["revision"], ["config.json", "*.safetensors", "*.json"]
    )
    info["espejo_lfs_verificado"] = verif
    t = time.perf_counter()
    oficial = hashes_safetensors(sorted(ruta_oficial.glob("*.safetensors")))
    espejo = hashes_safetensors(sorted(ruta_espejo.glob("*.safetensors")))
    comparacion = comparar_tensores(
        oficial, espejo, _atado(ruta_oficial / "config.json"), _atado(ruta_espejo / "config.json")
    )
    comparacion["segundos"] = round(time.perf_counter() - t, 1)
    info["comparacion_tensores"] = comparacion
    return bool(comparacion["identicos"])


def _plantilla_del_gguf(ruta: Path) -> str | None:
    """Plantilla de chat guardada dentro del GGUF (mejor esfuerzo, con el gguf-py de llama.cpp)."""
    for c in (DIR_TRABAJO / "llama.cpp" / "gguf-py", Path.home() / ".unsloth" / "llama.cpp" / "gguf-py"):
        if c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
    import gguf

    lector = gguf.GGUFReader(str(ruta))
    campo = lector.fields.get("tokenizer.chat_template")
    if campo is None:
        return None
    if hasattr(campo, "contents"):
        valor = campo.contents()
        return valor if isinstance(valor, str) else bytes(valor).decode("utf-8")
    return bytes(campo.parts[campo.data[0]]).decode("utf-8")


def _info_llama_cpp() -> dict:
    """Archivos de nivel superior de la carpeta de llama.cpp que dejó Unsloth (mejor esfuerzo)."""
    res = {}
    for c in (DIR_TRABAJO / "llama.cpp", Path.home() / ".unsloth" / "llama.cpp"):
        if c.is_dir():
            res[str(c)] = sorted(os.listdir(c))[:80]
    return res


def exportar_gguf(reg: Registro, mods, model, tok) -> None:
    """GGUF Q4_K_M en una carpeta temporal; a la salida solo se copia el .gguf."""
    res = model.save_pretrained_gguf(str(DIR_TRABAJO / "gguf_06b"), tok, quantization_method="q4_k_m")
    archivos = list(res.get("gguf_files", [])) if isinstance(res, Mapping) else []
    if not archivos:
        archivos = [str(p) for p in (DIR_TRABAJO).rglob("*.gguf")]
    q4 = [p for p in archivos if "q4_k_m" in Path(p).name.lower()]
    if len(q4) != 1:
        raise RuntimeError(f"Se esperaba un .gguf Q4_K_M y hay {len(q4)} entre {len(archivos)} archivos .gguf.")
    destino = DIR_SALIDA / CARPETA_GGUF / NOMBRE_GGUF
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(q4[0], destino)
    gguf_info = {
        "exportado": True,
        "nombre_original": Path(q4[0]).name,
        "archivo": f"{CARPETA_GGUF}/{NOMBRE_GGUF}",
        "bytes": destino.stat().st_size,
        "sha256": sha256_archivo(destino),
        "otros_gguf_generados": sorted(Path(p).name for p in archivos if p != q4[0]),
        "llama_cpp": _info_llama_cpp(),
    }
    try:
        plantilla = _plantilla_del_gguf(destino)
        gguf_info["plantilla_gguf_sha256"] = sha256_bytes(plantilla.encode("utf-8")) if plantilla else None
        gguf_info["plantilla_gguf_igual_a_tokenizador"] = plantilla == tok.chat_template if plantilla else None
    except Exception as e:  # mejor esfuerzo
        gguf_info["plantilla_gguf_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    reg.datos["gguf"].update(gguf_info)


def escribir_prompt_inventado(reg: Registro, tok, modelo: Mapping[str, str]) -> None:
    """prompt_inventado.txt (bytes exactos) y .json, junto al GGUF, para compararlos en la laptop (6d)."""
    carpeta = DIR_SALIDA / CARPETA_GGUF
    carpeta.mkdir(parents=True, exist_ok=True)
    prompt = prompt_generacion(NOMBRE_INVENTADO)
    datos = prompt.encode("utf-8")
    (carpeta / "prompt_inventado.txt").write_bytes(datos)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    meta = {
        "nombre_inventado": NOMBRE_INVENTADO,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(datos),
        "prompt_bytes": len(datos),
        "ids": ids,
        "plantilla_tokenizador_sha256": sha256_bytes(str(tok.chat_template).encode("utf-8")),
        "repo": modelo["repo"],
        "revision": modelo["revision"],
        "enable_thinking": False,
        "add_generation_prompt": True,
    }
    (carpeta / "prompt_inventado.json").write_bytes((json.dumps(meta, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8"))
    reg.datos["gguf"]["prompt_inventado"] = {"sha256": meta["prompt_sha256"], "fichas": len(ids)}


def liberar(mods) -> None:
    gc.collect()
    mods.torch.cuda.empty_cache()


def descargas_hf(mods) -> list[dict]:
    """Repos y revisiones en la caché de Hugging Face al final de la corrida."""
    try:
        cache = mods.hf.scan_cache_dir()
    except Exception as e:
        return [{"error": f"{type(e).__name__}"}]
    return sorted(
        (
            {"repo": r.repo_id, "tipo": r.repo_type, "revision": rev.commit_hash, "bytes": rev.size_on_disk}
            for r in cache.repos
            for rev in r.revisions
        ),
        key=lambda x: (x["repo"], x["revision"]),
    )


def archivos_salida() -> list[dict]:
    res = []
    for p in sorted(DIR_SALIDA.rglob("*")):
        if p.is_file() and p.name not in (ARCHIVO_MEDICIONES, ARCHIVO_MEDICIONES + ".tmp"):
            res.append({"ruta": p.relative_to(DIR_SALIDA).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_archivo(p)})
    return res


def _correr(reg: Registro, esperados: Mapping[str, str]) -> None:
    with reg.etapa("gpu"):
        gpus = consultar_gpus()
        verificar_t4(gpus)
        reg.datos["gpu"] = {
            "nvidia_smi": gpus,
            "machine_shape_pedido": MACHINE_SHAPE,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus_usadas": 1,
        }
        informar(f"GPU: {[g['nombre'] for g in gpus]}; se usa solo la 0 (CUDA_VISIBLE_DEVICES=0)")

    with reg.etapa("datos"):
        muestras = {}
        for archivo in (ARCHIVO_TRAIN, ARCHIVO_VAL):
            muestras[archivo] = leer_jsonl_verificado(buscar_unico(DIR_ENTRADA, archivo), esperados[archivo])
        train, val = muestras[ARCHIVO_TRAIN], muestras[ARCHIVO_VAL]
        if {r["id"] for r in train} & {r["id"] for r in val}:
            raise AbortoHumo("Train y validación comparten id.")
        reg.nombres = tuple(r["nom_estab"] for r in train + val)
        reg.llaves = frozenset(k for k in map(normalizar_nombre, reg.nombres) if k)
        reg.datos["datos"] = {
            a: {"sha256": esperados[a], "sha256_verificado": True, "filas": len(rs), "clases": len({r["codigo_act"] for r in rs})}
            for a, rs in muestras.items()
        }
        marcas = sum(tiene_marca_especial(n) for n in reg.nombres)
        reg.datos["comprobaciones"] = {"d_nombres_con_marca_especial": marcas}
        if marcas:
            raise AbortoHumo(f"(d) {marcas} nombres contienen '<|'.")
        informar(f"datos verificados: {len(train)} filas de train y {len(val)} de validación")

    with reg.etapa("instalacion"):
        instalar(reg)

    with reg.etapa("imports"):
        mods = importar(reg)

    with reg.etapa("descarga_06b"):
        ruta_06b, verif = descargar_verificado(mods, MODELO_06B["repo"], MODELO_06B["revision"])
        reg.datos["modelo_06b"] = {"revision_descargada": ruta_06b.name, "lfs_verificado": verif}

    with reg.etapa("carga_06b"):
        model, tok, info = cargar(mods, MODELO_06B)
        reg.datos["modelo_06b"].update(info)

    with reg.etapa("tokens_y_plantilla"):
        prep = preparar_tokens(reg, tok, train, val)

    with reg.etapa("lora_06b"):
        model, info_lora = aplicar_lora(mods, model)
        reg.datos["modelo_06b"]["lora"] = info_lora

    with reg.etapa("entrenamiento_06b"):
        reg.datos["entrenamiento_06b"] = entrenar(mods, model, tok, prep["ejemplos_train"], lote=LOTE, etiqueta="06b")

    with reg.etapa("adaptador"):
        model.save_pretrained(str(DIR_SALIDA / CARPETA_ADAPTADOR))

    with reg.etapa("inferencia"):
        inferir(reg, mods, model, tok, val, prep["ids_fin"])

    reg.datos["gguf"] = {"exportado": False}
    with reg.etapa("prompt_inventado"):
        escribir_prompt_inventado(reg, tok, MODELO_06B)

    with reg.etapa("base_16_bits"):
        try:
            exportable = verificar_base_16(reg, mods, model, ruta_06b)
        except Exception as e:
            reg.falla("base_16_bits", e)
            exportable = False
        reg.datos["base_16_bits"]["exportar"] = exportable
        informar(f"base de 16 bits: camino {reg.datos['base_16_bits'].get('camino')}, exportar = {exportable}")

    if exportable:
        with reg.etapa("gguf"):
            try:
                exportar_gguf(reg, mods, model, tok)
            except Exception as e:
                reg.falla("gguf", e)
    else:
        reg.datos["gguf"]["motivo"] = "No se exportó: la base de 16 bits de la fusión no quedó verificada (ver base_16_bits)."
        reg.falla("gguf", "omitido: base de 16 bits no verificada")

    del model
    liberar(mods)

    with reg.etapa("lotes_06b"):
        reg.datos["lotes_06b"] = {}
        try:
            model, tok2, _ = cargar(mods, MODELO_06B)
            model, _ = aplicar_lora(mods, model)
            for lote in LOTES_06B:
                reg.datos["lotes_06b"][str(lote)] = medir_lote(
                    mods, model, tok2, prep["ejemplos_train"], lote=lote, etiqueta=f"06b_lote{lote}"
                )
                reg.escribir()
            del model
        except Exception as e:
            reg.falla("lotes_06b", e)
        liberar(mods)

    if reg.transcurrido() > PRESUPUESTO_4B_S:
        reg.datos["modelo_4b"] = {"omitido": f"El reloj pasó de {PRESUPUESTO_4B_S / 60:.0f} min antes del 4B."}
        informar("4B omitido por el presupuesto de tiempo")
    else:
        with reg.etapa("modelo_4b"):
            r4: dict = {"duraciones_s": {}}
            reg.datos["modelo_4b"] = r4
            try:
                t = time.perf_counter()
                ruta_4b, verif4 = descargar_verificado(mods, MODELO_4B["repo"], MODELO_4B["revision"])
                r4.update({"revision_descargada": ruta_4b.name, "lfs_verificado": verif4})
                r4["duraciones_s"]["descarga"] = round(time.perf_counter() - t, 1)
                t = time.perf_counter()
                model, tok4, info4 = cargar(mods, MODELO_4B)
                r4.update(info4)
                r4["duraciones_s"]["carga"] = round(time.perf_counter() - t, 1)
                r4["tokenizador_igual_al_06b"] = all(
                    tok4(prompt_generacion(n), add_special_tokens=False)["input_ids"]
                    == tok(prompt_generacion(n), add_special_tokens=False)["input_ids"]
                    for n in reg.nombres[:200] + (NOMBRE_INVENTADO,)
                )
                if not r4["tokenizador_igual_al_06b"]:
                    raise RuntimeError("El tokenizador del 4B no da los mismos ids que el del 0.6B.")
                model, r4["lora"] = aplicar_lora(mods, model)
                reg.escribir()
                t = time.perf_counter()
                r4["lote_16"] = entrenar(mods, model, tok4, prep["ejemplos_train"], lote=LOTE, etiqueta="4b", max_pasos=PASOS_4B)
                r4["duraciones_s"]["entrenamiento_lote_16"] = round(time.perf_counter() - t, 1)
                reg.escribir()
                for lote in LOTES_4B_EXTRA:
                    t = time.perf_counter()
                    r4[f"lote_{lote}"] = medir_lote(mods, model, tok4, prep["ejemplos_train"], lote=lote, etiqueta=f"4b_lote{lote}")
                    r4["duraciones_s"][f"lote_{lote}"] = round(time.perf_counter() - t, 1)
                del model
            except Exception as e:
                reg.falla("modelo_4b", e)
            liberar(mods)

    with reg.etapa("cierre"):
        reg.datos["descargas_hf"] = descargas_hf(mods)
        reg.datos["archivos_salida"] = archivos_salida()
        reg.datos["estado"] = "completo" if not reg.datos["fallas"] else "completo_con_fallas"


def main(esperados: Mapping[str, str], t_arranque: float | None = None, fuentes: Mapping[str, str] | None = None) -> None:
    """Punto de entrada en Kaggle. `esperados`: SHA-256 de cada JSONL, incrustados por el generador."""
    t_arranque = time.time() if t_arranque is None else t_arranque
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # antes de importar torch: una sola T4
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    if set(esperados) != {ARCHIVO_TRAIN, ARCHIVO_VAL}:
        raise AbortoHumo("Los SHA-256 esperados deben ser exactamente los de humo_train.jsonl y humo_val.jsonl.")
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    DIR_TRABAJO.mkdir(parents=True, exist_ok=True)
    os.chdir(DIR_TRABAJO)  # Unsloth deja su caché compilada y llama.cpp en la carpeta actual
    reg = Registro(DIR_SALIDA / ARCHIVO_MEDICIONES, t_arranque)
    reg.datos["configuracion"] = configuracion()
    reg.datos["fuentes_incrustadas_sha256"] = dict(fuentes or {})
    reg.datos["datos_sha256_esperados"] = dict(esperados)
    reg.escribir()
    try:
        _correr(reg, esperados)
    except AbortoHumo as e:
        mensaje = tapar_nombres(str(e), reg.nombres)
        reg.datos["estado"] = "abortado"
        reg.datos["aborto"] = mensaje
        reg.escribir()
        informar(f"ABORTO: {mensaje}")
        sys.exit(1)
    except Exception:
        tb = tapar_nombres(traceback.format_exc(), reg.nombres)
        reg.datos["estado"] = "error"
        reg.datos["error"] = tb[-4000:]
        reg.escribir()
        print(tb, flush=True)
        sys.exit(1)
    reg.escribir()
    informar(f"estado final: {reg.datos['estado']}; fallas: {len(reg.datos['fallas'])}")
