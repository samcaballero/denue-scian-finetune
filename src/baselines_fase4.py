"""Baselines del clasificador en test y hard set (Fase 4, paso 16).

Manda evals/fase4_preregistro.json (modelos.diccionario, modelos.tfidf-lr-C10 y uso_unico.baselines).
src/baselines.py se usa sin cambios: este módulo compone sus funciones públicas en el mismo orden que
correr_diccionario y correr_tfidf, porque esas dos no devuelven el modelo ajustado.

Orden de la corrida:

0. Árbol limpio y este script versionado; commit y versiones. Memoria libre impresa; en --escribir,
   se detiene si está por debajo de MINIMO_LIBRE_GB.
a. SHA-256 del config (constante SHA_CONFIG), configuración de los baselines igual a la de
   src/baselines.py, los dos registros de validación, train final, train candidato sin hard,
   validación y las dos copias sin codigo_act (con el candado de src/paquete_evaluacion.py).
b. Diccionario: se ajusta como en validación; sus aciertos de clase en validación (orden de int(id))
   deben ser idénticos fila por fila a los del registro. Si no, se detiene sin predecir test ni hard.
c. TF-IDF + LR con el C del config: se ajusta con train final, se guarda (joblib, con su SHA-256 en un
   JSON al lado) antes de comparar, se recarga desde el archivo y con él se predice validación, que
   debe reproducir el registro fila por fila; si no, se detiene sin predecir test ni hard. Un modelo ya
   guardado se reusa sin reentrenar si su SHA-256 coincide con su JSON y coinciden la huella de la
   configuración, el SHA-256 de train final y las versiones de numpy, scipy, scikit-learn y joblib
   (ajuste de Samuel al aprobar el plan del paso 16; el commit de entrenamiento solo se registra).
d. Predicciones {"id", "respuesta"} de validación, test y hard de cada baseline en data/fase4/baselines/
   (ignorado por git) y un registro de la corrida en results/, sin ids, nombres ni métricas de test o hard.
e. Antes de escribir en results/ se comprueba que HEAD y el árbol no cambiaron durante la corrida.

Uso:
  .venv/Scripts/python -m src.baselines_fase4                  # punto de control: no escribe
  .venv/Scripts/python -m src.baselines_fase4 --humo CARPETA   # datos inventados, fuera del repo
  .venv/Scripts/python -m src.baselines_fase4 --escribir       # la corrida real (~1 h de CPU)

Nunca abre data/splits/test.jsonl ni data/splits/hard.jsonl: test y hard se leen solo desde las copias
sin etiquetas, y sobre ellos no se calcula ninguna métrica. En la terminal y en el log solo van avance,
tiempos, conteos, hashes y memoria; nunca nombres, llaves ni ids.

El chat de Claude del proyecto propuso este paso; Samuel lo aprobó; Claude Code lo redactó.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import datetime
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as version_de_paquete
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import src.baselines as bl
import src.paquete_evaluacion as pe
from src.baselines import ErrorBaseline
from src.denue import REPO_RAIZ
from src.evaluador_clasificador import SECTORES_2023, ErrorEvaluador, cargar_codigos
from src.exploracion_clasificador import ErrorPrivacidad, textos, verificar_privacidad
from src.muestra_humo import ErrorMuestra, codigo_versionado, escribir_carpeta, leer_verificado
from src.normalizacion import normalizar_nombre
from src.preregistro_fase4 import ErrorPreregistro, parsear_jsonl
from src.scian import leer_manifest
from src.splits import esta_ignorado, sha256_bytes
from src.splits import serializar as serializar_split

SCRIPT = "src/baselines_fase4.py"
EXPERIMENTO = "fase4_paso16_baselines"
SEMILLA = 42
RUTA_CONFIG = "evals/fase4_preregistro.json"
SHA_CONFIG = "7de9d3fc1335179540e35f91cba0ac98920f9f00a4ddc28a680b148898b6156e"
ID_DICC = "diccionario"
ID_TFIDF = "tfidf-lr-C10"
CONJUNTOS = ("validacion", "test", "hard")
RESERVADOS = ("test", "hard")
ENTRADA_COPIAS = pe.ENTRADA
ARCHIVO_COPIA = {"test": pe.ARCHIVO_TEST, "hard": pe.ARCHIVO_HARD}
CAMPOS_COPIA = pe.CAMPOS
CAMPOS_PREDICCION = ("id", "respuesta")  # las mismas claves que las predicciones del kernel de la Fase 4
CARPETA_SALIDA = "data/fase4/baselines"
PATRON_RESULTADO = "*_fase4_baselines-predicciones.json"
PAQUETES_MODELO = ("numpy", "scipy", "scikit-learn", "joblib")
RE_CODIGO = re.compile(r"[0-9]{6}")

# Memoria (paso 16). Pico estimado del proceso ~5.3 GB: datos cargados (~1 GB) + matriz TF-IDF de train y
# vectorizador (~0.3 GB) + ajuste saga (4,171 MB medidos con tracemalloc en el registro de validación de
# C = 10). Cota superior: el pico de 5,297 MB de la corrida de validación, que además tenía cargado el
# split aleatorio. Mínimo para lanzar = pico + ~1.7 GB de margen para el sistema.
PICO_ESTIMADO_GB = 5.3
MINIMO_LIBRE_GB = 7.0
MARGEN_GB = 1.5  # el punto de control recomienda: pico medido + margen, redondeado hacia arriba a 0.5 GB

CONTENIDO_MODELO = (
    "joblib de un dict con 'vectorizador' (TfidfVectorizer ajustado con los nombres de train final) y "
    "'modelo' (LogisticRegression ajustada). Lleva n-gramas de nombres del DENUE: solo en data/ (ignorado)."
)
FORMATO_PREDICCIONES = (
    "JSONL: un objeto por línea con las claves id y respuesta (el código SCIAN de 6 dígitos que da el "
    "baseline), en orden de int(id); UTF-8 sin BOM; saltos \\n."
)


class ErrorFase4(ValueError):
    """Condición no cumplida; el mensaje nunca incluye nombres, llaves ni ids."""


ERRORES_PROPIOS = (ErrorFase4, ErrorBaseline, ErrorMuestra, ErrorPreregistro, ErrorEvaluador, ErrorPrivacidad)


# --- Entorno: git, versiones, memoria y suspensión ---


@dataclass(frozen=True)
class EstadoGit:
    commit: str
    arbol_limpio: bool
    versionado: bool


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True).stdout.strip()


def estado_git() -> EstadoGit:
    return EstadoGit(
        commit=_git("rev-parse", "HEAD"),
        arbol_limpio=_git("status", "--porcelain", "--untracked-files=no") == "",
        versionado=codigo_versionado(SCRIPT),
    )


def versiones() -> dict[str, str]:
    return {p: version_de_paquete(p) for p in PAQUETES_MODELO}


def memoria_libre_gb() -> float | None:
    """Memoria física disponible (GiB) según el sistema; None si no se puede medir."""
    try:
        if sys.platform == "win32":
            class Estado(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            e = Estado()
            e.dwLength = ctypes.sizeof(e)
            if not ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(e)):
                return None
            return e.ullAvailPhys / 2**30
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, AttributeError, ValueError):
        return None


def _texto_gb(gb: float | None) -> str:
    return "no se pudo medir" if gb is None else f"{gb:.1f} GB"


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _estado_de_ejecucion(banderas: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        f = ctypes.WinDLL("kernel32").SetThreadExecutionState
        f.restype = ctypes.c_uint32
        f.argtypes = [ctypes.c_uint32]
        return f(banderas) != 0
    except (OSError, AttributeError):
        return False


def evitar_suspension() -> bool:
    """Pide a Windows no suspender el equipo por inactividad mientras corre el proceso."""
    return _estado_de_ejecucion(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def permitir_suspension() -> bool:
    return _estado_de_ejecucion(ES_CONTINUOUS)


# --- Bitácora de la corrida ---


class Bitacora:
    """Cada línea va a la terminal y, si hay ruta, al log (con hora). Solo avance, tiempos y conteos."""

    def __init__(self, ruta: Path | None = None, eco: Callable[[str], None] = print) -> None:
        self.ruta = Path(ruta) if ruta else None
        self.eco = eco
        self.lineas: list[str] = []

    def __call__(self, texto: str) -> None:
        linea = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {texto}"
        self.eco(linea)
        self.lineas.append(linea)
        if self.ruta is not None:
            with open(self.ruta, "a", encoding="utf-8", newline="\n") as f:
                f.write(linea + "\n")


def registrar_error(e: BaseException, avance: Callable[[str], None]) -> None:
    """Un error propio va con su mensaje; uno inesperado, solo con su tipo y sus marcos (el mensaje
    podría traer un id o un nombre)."""
    if isinstance(e, ERRORES_PROPIOS):
        avance(f"ME DETENGO: {e}")
        return
    if isinstance(e, KeyboardInterrupt):
        avance("ME DETENGO: corrida interrumpida (KeyboardInterrupt).")
    else:
        avance(f"ME DETENGO por un error inesperado {type(e).__name__} (sin su mensaje, que podría traer datos). Marcos:")
    for m in traceback.extract_tb(e.__traceback__):
        avance(f"  {Path(m.filename).name}:{m.lineno} en {m.name}")


# --- Utilidades de archivos ---


def sha256_archivo(ruta: Path, bloque: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while b := f.read(bloque):
            h.update(b)
    return h.hexdigest()


def _rel(ruta: Path, raiz: Path) -> str:
    try:
        return Path(ruta).resolve().relative_to(Path(raiz).resolve()).as_posix()
    except ValueError:
        return Path(ruta).name


def escribir_o_comparar(ruta: Path, datos: bytes) -> str:
    """Escribe de forma atómica; si el archivo ya existe, solo se acepta idéntico byte a byte."""
    ruta = Path(ruta)
    if ruta.exists():
        if ruta.read_bytes() == datos:
            return "ya existía idéntico"
        raise ErrorFase4(f"{ruta.name} ya existe con otro contenido; no se sobrescribe. Decide Samuel.")
    bl.escribir_atomico(datos, ruta)
    return "escrito"


# --- a. Config y entradas ---


def modelo_config(config: Mapping, id_modelo: str) -> dict:
    encontrados = [m for m in config.get("modelos", []) if m.get("id") == id_modelo]
    if len(encontrados) != 1:
        raise ErrorFase4(f"El config no tiene exactamente un modelo {id_modelo}.")
    return encontrados[0]


def _como_json(obj):
    return json.loads(json.dumps(obj))  # tuplas → listas, para comparar con lo leído del config


@dataclass(frozen=True)
class Parametros:
    config: dict
    sha_config: str
    C: float
    huella_dicc: str
    huella_tfidf: str


def verificar_config(config_bytes: bytes, sha_esperado: str) -> Parametros:
    """SHA-256 del config y configuración de los baselines igual a la de src/baselines.py (sin cambios)."""
    sha = sha256_bytes(config_bytes)
    if sha != sha_esperado:
        raise ErrorFase4(f"El SHA-256 de {RUTA_CONFIG} es {sha}, no el esperado {sha_esperado}; me detengo.")
    config = json.loads(config_bytes.decode("utf-8"))
    dicc, tfidf = modelo_config(config, ID_DICC), modelo_config(config, ID_TFIDF)
    for m in (dicc, tfidf):
        if m.get("tipo") != "baseline" or not m.get("entrena_con") or m["entrena_con"][0] != bl.ENTRADA_TRAIN_FINAL:
            raise ErrorFase4(f"{m['id']}: el config no dice que entrena con {bl.ENTRADA_TRAIN_FINAL}.")
    if len(dicc["entrena_con"]) != 2 or not dicc["entrena_con"][1].startswith(bl.ENTRADA_ORIGEN):
        raise ErrorFase4(f"{ID_DICC}: el config no dice que desempata con {bl.ENTRADA_ORIGEN}.")
    cfg = tfidf["configuracion"]
    lr = dict(cfg["regresion_logistica"])
    C = lr.pop("C")
    if not isinstance(C, (int, float)) or isinstance(C, bool):
        raise ErrorFase4(f"{ID_TFIDF}: C no es un número.")
    C = float(C)
    if cfg["tfidf"] != _como_json(bl.CONFIG_TFIDF):
        raise ErrorFase4(f"{ID_TFIDF}: la configuración del TF-IDF del config no es la de src/baselines.py.")
    if lr != _como_json(bl.CONFIG_LR):
        raise ErrorFase4(f"{ID_TFIDF}: la configuración de la regresión logística del config no es la de src/baselines.py.")
    huella_tfidf = bl.huella(bl.config_tfidf(C))
    if cfg.get("huella_config") != huella_tfidf:
        raise ErrorFase4(f"{ID_TFIDF}: la huella de la configuración no es la del config.")
    conj = config["conjuntos"]
    esperadas = {"train_final": bl.ENTRADA_TRAIN_FINAL, "validacion": bl.ENTRADA_VAL,
                 "test": "clasificador_test", "hard": "clasificador_hard"}
    for nombre, entrada in esperadas.items():
        if conj[nombre]["entrada_manifest"] != entrada:
            raise ErrorFase4(f"conjuntos.{nombre} del config no apunta a splits.{entrada}.")
    return Parametros(config=config, sha_config=sha, C=C, huella_dicc=bl.huella(bl.config_diccionario()),
                      huella_tfidf=huella_tfidf)


def leer_registro(raiz: Path, modelo: Mapping, huella_esperada: str, filas_val: int) -> tuple[dict, str]:
    """Registro de validación del baseline con el SHA-256, el commit y la huella del config."""
    rv = modelo["registro_validacion"]
    datos = leer_verificado(Path(raiz) / rv["ruta"], rv["sha256"])
    reg = json.loads(datos.decode("utf-8"))
    if reg.get("commit") != rv["commit"]:
        raise ErrorFase4(f"{modelo['id']}: el commit del registro no es el del config.")
    if reg.get("huella_config") != huella_esperada:
        raise ErrorFase4(f"{modelo['id']}: la huella del registro no es la de la configuración de src/baselines.py.")
    clase = reg.get("aciertos", {}).get("clase")
    if not isinstance(clase, list) or len(clase) != filas_val or not all(isinstance(x, bool) for x in clase):
        raise ErrorFase4(f"{modelo['id']}: aciertos.clase del registro no son {filas_val:,} valores bool.")
    return reg, rv["sha256"]


def leer_copias(manifest: Mapping, raiz: Path, config: Mapping) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Test y hard desde las copias sin codigo_act, con el candado de src/paquete_evaluacion.py."""
    e = manifest.get("kaggle_datasets", {}).get(ENTRADA_COPIAS)
    if not e:
        raise ErrorFase4(f"El manifest no tiene kaggle_datasets.{ENTRADA_COPIAS}.")
    if list(e["campos"]) != list(CAMPOS_COPIA):
        raise ErrorFase4("Las copias no son solo id y nom_estab.")
    carpeta = (Path(raiz) / e["carpeta"]).resolve()
    _, registrada = pe.verificar(manifest, carpeta, raiz)
    if not registrada:
        raise ErrorFase4(f"kaggle_datasets.{ENTRADA_COPIAS} no está registrada en el manifest.")
    copias, info = {}, {}
    for conj, archivo in ARCHIVO_COPIA.items():
        a = e["archivos"][archivo]
        if a["origen_sha256"] != config["conjuntos"][conj]["sha256"]:
            raise ErrorFase4(f"La copia de {conj} no sale del conjunto congelado del config.")
        regs = parsear_jsonl(leer_verificado(Path(raiz) / a["ruta"], a["sha256"]), CAMPOS_COPIA)
        if not (len(regs) == a["filas"] == config["conjuntos"][conj]["filas"]):
            raise ErrorFase4(f"La copia de {conj} no tiene las filas del manifest y del config.")
        copias[conj] = regs
        info[conj] = {"ruta": a["ruta"], "sha256": a["sha256"], "origen_sha256": a["origen_sha256"], "filas": len(regs)}
    return copias, info


@dataclass
class Entradas:
    params: Parametros
    registros: dict[str, dict]
    sha_registros: dict[str, str]
    split: bl.Split
    copias: dict[str, list[dict]]
    info_copias: dict[str, dict]
    codigos: frozenset
    sha_splits: dict[str, str]


def verificar_entradas(manifest: Mapping, raiz: Path, config_bytes: bytes, sha_config: str,
                       avance: Callable[[str], None]) -> Entradas:
    """Paso a: todo lo verificable antes de entrenar. Config y registros van primero: si fallan, no se lee
    ningún split."""
    params = verificar_config(config_bytes, sha_config)
    avance(f"Config {RUTA_CONFIG}: SHA-256 {params.sha_config} (el esperado); C = {params.C:g}; configuración "
           "de los baselines igual a la de src/baselines.py.")
    conj = params.config["conjuntos"]
    registros, sha_registros = {}, {}
    for id_modelo, huella_esperada in ((ID_DICC, params.huella_dicc), (ID_TFIDF, params.huella_tfidf)):
        reg, sha = leer_registro(raiz, modelo_config(params.config, id_modelo), huella_esperada,
                                 conj["validacion"]["filas"])
        registros[id_modelo], sha_registros[id_modelo] = reg, sha
        avance(f"Registro de validación de {id_modelo}: SHA-256 {sha} (el del config), commit y huella "
               f"correctos, {sum(reg['aciertos']['clase']):,} aciertos de clase en {len(reg['aciertos']['clase']):,} filas.")
    sp = manifest["splits"]
    for nombre, entrada in (("train_final", bl.ENTRADA_TRAIN_FINAL), ("validacion", bl.ENTRADA_VAL)):
        if sp[entrada]["sha256"] != conj[nombre]["sha256"]:
            raise ErrorFase4(f"El SHA-256 de splits.{entrada} del manifest no es el de conjuntos.{nombre} del config.")
    t0 = time.perf_counter()
    split = bl.cargar_split_nombre(manifest, raiz)
    if len(split.train) != conj["train_final"]["filas"] or len(split.val) != conj["validacion"]["filas"]:
        raise ErrorFase4("Las filas de train final o de validación no son las del config.")
    sha_splits = {"train_final": sp[bl.ENTRADA_TRAIN_FINAL]["sha256"],
                  "train_candidatos_sin_hard": sp[bl.ENTRADA_ORIGEN]["sha256"],
                  "validacion": sp[bl.ENTRADA_VAL]["sha256"]}
    avance(f"Train final ({len(split.train):,} filas, SHA-256 {sha_splits['train_final']}), train candidato sin hard "
           f"({len(split.pre_tope):,}, {sha_splits['train_candidatos_sin_hard']}) y validación ({len(split.val):,}, "
           f"{sha_splits['validacion']}): SHA-256 del manifest verificados; ninguna llave compartida entre train y "
           f"validación ({time.perf_counter() - t0:.0f} s).")
    copias, info_copias = leer_copias(manifest, raiz, params.config)
    ids_reservados = {r["id"] for c in copias.values() for r in c}
    if len(ids_reservados) != sum(len(c) for c in copias.values()):
        raise ErrorFase4("Test y hard comparten id.")
    if ids_reservados & {r["id"] for regs in (split.train, split.val, split.pre_tope) for r in regs}:
        raise ErrorFase4("Las copias de test o hard comparten id con train o validación.")
    for c in RESERVADOS:
        avance(f"Copia de {c} sin codigo_act: {info_copias[c]['filas']:,} filas, SHA-256 {info_copias[c]['sha256']} "
               f"(candado de src/paquete_evaluacion.py y manifest correctos; origen {info_copias[c]['origen_sha256']}).")
    codigos = cargar_codigos(manifest, raiz)
    avance(f"Catálogo SCIAN {manifest['scian_version']}: {len(codigos):,} clases (SHA-256 verificado).")
    return Entradas(params=params, registros=registros, sha_registros=sha_registros, split=split, copias=copias,
                    info_copias=info_copias, codigos=codigos, sha_splits=sha_splits)


# --- Predicciones y comparación con validación ---


def filas_prediccion(regs: Sequence[Mapping], respuestas: Sequence) -> list[dict]:
    """[{id, respuesta}] en orden de int(id); cada respuesta, un código de 6 dígitos ASCII."""
    filas = [{"id": r["id"], "respuesta": str(c)} for r, c in zip(regs, respuestas, strict=True)]
    malas = sum(1 for f in filas if not RE_CODIGO.fullmatch(f["respuesta"]))
    if malas:
        raise ErrorFase4(f"{malas} respuestas no son un código de 6 dígitos ASCII.")
    ordenadas = sorted(filas, key=lambda f: int(f["id"]))
    if [f["id"] for f in ordenadas] != sorted((r["id"] for r in regs), key=int):
        raise ErrorFase4("Los ids de las predicciones no son los del conjunto.")
    return ordenadas


def serializar_predicciones(filas: Sequence[Mapping]) -> bytes:
    for f in filas:
        if tuple(f) != CAMPOS_PREDICCION:
            raise ErrorFase4("Una predicción no tiene exactamente id y respuesta.")
    return "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas).encode("utf-8")


def comparar_aciertos(nuevos: Sequence[bool], registro: Sequence[bool]) -> dict:
    identicas = sum(1 for a, b in zip(nuevos, registro) if a == b) if len(nuevos) == len(registro) else 0
    return {
        "filas": len(registro),
        "identicas": identicas,
        "aciertos_registro": int(sum(registro)),
        "aciertos_reproducidos": int(sum(nuevos)),
        "criterio": "aciertos.clase idénticos fila por fila, en orden de int(id) (reproduccion_validacion del config)",
        "cumple": len(nuevos) == len(registro) and identicas == len(registro),
    }


def exigir_reproduccion(id_modelo: str, comp: Mapping, avance: Callable[[str], None]) -> None:
    avance(f"{id_modelo}: {comp['identicas']:,} de {comp['filas']:,} filas de validación idénticas al registro "
           f"({comp['aciertos_reproducidos']:,} aciertos de clase; el registro tiene {comp['aciertos_registro']:,}).")
    if not comp["cumple"]:
        raise ErrorFase4(f"{id_modelo} no reproduce su validación fila por fila ({comp['identicas']:,} de "
                         f"{comp['filas']:,} idénticas): me detengo sin predecir test ni hard. Decide Samuel.")


@dataclass
class Salidas:
    carpeta: Path
    raiz: Path
    archivos: dict[str, dict[str, dict]]

    def guardar(self, id_modelo: str, conjunto: str, filas: Sequence[Mapping], avance: Callable[[str], None]) -> None:
        datos = serializar_predicciones(filas)
        ruta = self.carpeta / f"{id_modelo}_{conjunto}.jsonl"
        estado = escribir_o_comparar(ruta, datos)
        sha = sha256_bytes(datos)
        self.archivos.setdefault(id_modelo, {})[conjunto] = {
            "ruta": _rel(ruta, self.raiz), "filas": len(filas), "bytes": len(datos), "sha256": sha}
        avance(f"  {ruta.name}: {len(filas):,} filas, SHA-256 {sha} ({estado}).")


# --- b. Diccionario ---


def paso_diccionario(e: Entradas, salidas: Salidas | None, avance: Callable[[str], None]) -> tuple[dict, dict]:
    """Ajuste y reproducción de validación; si reproduce y hay `salidas`, predice y guarda test y hard."""
    llave = bl.Normalizador()
    t0 = time.perf_counter()
    bl.verificar_llaves(e.split.train, llave)
    bl.verificar_llaves(e.split.val, llave)
    d = bl.ajustar_diccionario(e.split.train, e.split.pre_tope, llave)
    resp_val = bl.predecir_diccionario(d, [r["nom_estab"] for r in e.split.val], llave)
    _, aciertos = bl.evaluar_respuestas(resp_val, e.split.val, e.codigos)
    comp = comparar_aciertos(aciertos["clase"], e.registros[ID_DICC]["aciertos"]["clase"])
    segundos = round(time.perf_counter() - t0, 2)
    avance(f"Diccionario ajustado en {segundos:.1f} s: {d.empates['llaves_en_train']:,} llaves en train.")
    exigir_reproduccion(ID_DICC, comp, avance)
    if salidas is None:
        return comp, {"tiempo_s": segundos}
    pred = {"validacion": filas_prediccion(e.split.val, resp_val)}
    for conj in RESERVADOS:
        nombres = [r["nom_estab"] for r in e.copias[conj]]
        vistas = sum(1 for n in nombres if llave(n) in d.por_llave)
        pred[conj] = filas_prediccion(e.copias[conj], bl.predecir_diccionario(d, nombres, llave))
        avance(f"Diccionario, {conj}: {len(nombres):,} filas predichas; {vistas:,} con la llave vista en train "
               "(el config espera 0: responde la clase global).")
    avance("Predicciones del diccionario:")
    for conj in CONJUNTOS:
        salidas.guardar(ID_DICC, conj, pred[conj], avance)
    return comp, {"tiempo_s": round(time.perf_counter() - t0, 2)}


# --- c. TF-IDF + regresión logística ---


def rutas_modelo(carpeta: Path) -> tuple[Path, Path]:
    return Path(carpeta) / f"{ID_TFIDF}_modelo.joblib", Path(carpeta) / f"{ID_TFIDF}_modelo.json"


def esperado_modelo(e: Entradas) -> dict:
    return {"huella_config": e.params.huella_tfidf, "C": e.params.C,
            "sha256_train_final": e.sha_splits["train_final"], "versiones": versiones()}


def revisar_modelo_guardado(carpeta: Path | None, esperado: Mapping) -> dict | None:
    """None si no hay modelo guardado; sus metadatos si se puede reusar; se detiene en cualquier otro caso.

    Se reusa si el SHA-256 del joblib es el de su JSON y coinciden la huella de la configuración, C, el
    SHA-256 de train final y las versiones de numpy, scipy, scikit-learn y joblib. No depende del commit.
    """
    if carpeta is None:
        return None
    ruta_modelo, ruta_meta = rutas_modelo(carpeta)
    if not ruta_modelo.exists() and not ruta_meta.exists():
        return None
    if not (ruta_modelo.exists() and ruta_meta.exists()):
        raise ErrorFase4(f"Solo existe uno de {ruta_modelo.name} y {ruta_meta.name}; no se reusa ni se sobrescribe. "
                         "Decide Samuel.")
    meta = json.loads(ruta_meta.read_bytes())
    distintos = [k for k in ("huella_config", "C", "sha256_train_final") if meta.get(k) != esperado[k]]
    distintos += [f"versiones.{p}" for p in PAQUETES_MODELO
                  if meta.get("versiones", {}).get(p) != esperado["versiones"][p]]
    if distintos:
        raise ErrorFase4(f"El modelo guardado no coincide en {distintos}; no se reusa ni se sobrescribe. Decide Samuel.")
    sha = sha256_archivo(ruta_modelo)
    if sha != meta.get("sha256"):
        raise ErrorFase4(f"El SHA-256 de {ruta_modelo.name} ({sha}) no es el de su JSON; no se reusa. Decide Samuel.")
    return meta


def guardar_modelo(vectorizador, modelo, carpeta: Path, meta: Mapping) -> dict:
    """joblib atómico (temporal y renombre) y, al lado, su JSON con SHA-256, bytes y lo necesario para reusarlo."""
    ruta_modelo, ruta_meta = rutas_modelo(carpeta)
    if ruta_modelo.exists() or ruta_meta.exists():
        raise ErrorFase4(f"{ruta_modelo.name} ya existe; no se sobrescribe.")
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    temporal = ruta_modelo.with_name(ruta_modelo.name + ".tmp")
    joblib.dump({"vectorizador": vectorizador, "modelo": modelo}, temporal, compress=0)
    sha = sha256_archivo(temporal)
    tamano = temporal.stat().st_size
    os.replace(temporal, ruta_modelo)
    completo = {**meta, "archivo": ruta_modelo.name, "sha256": sha, "bytes": tamano, "contenido": CONTENIDO_MODELO}
    bl.escribir_atomico(bl.a_bytes(completo), ruta_meta)
    return completo


def cargar_modelo(carpeta: Path, meta: Mapping):
    """(vectorizador, modelo) desde el joblib, solo si su SHA-256 es el de su JSON."""
    ruta_modelo, _ = rutas_modelo(carpeta)
    if sha256_archivo(ruta_modelo) != meta["sha256"]:
        raise ErrorFase4(f"El SHA-256 de {ruta_modelo.name} no es el de su JSON; no se carga.")
    obj = joblib.load(ruta_modelo)
    if not (isinstance(obj, dict) and isinstance(obj.get("vectorizador"), TfidfVectorizer)
            and isinstance(obj.get("modelo"), LogisticRegression)):
        raise ErrorFase4(f"{ruta_modelo.name} no tiene un vectorizador TF-IDF y una regresión logística.")
    return obj["vectorizador"], obj["modelo"]


def predecir_tfidf(vectorizador, modelo, nombres: Sequence[str]) -> list[str]:
    return [str(c) for c in modelo.predict(vectorizador.transform(list(nombres)))]


def paso_tfidf(e: Entradas, salidas: Salidas, meta_previa: Mapping | None, commit: str,
               avance: Callable[[str], None]) -> tuple[dict, dict, dict]:
    """Ajuste (o reúso), guardado antes de comparar, recarga, reproducción de validación y test y hard."""
    tiempos: dict[str, float] = {}
    reg_val = e.registros[ID_TFIDF]
    if meta_previa is not None:
        meta = dict(meta_previa)
        avance(f"TF-IDF + LR, C = {e.params.C:g}: reuso el modelo guardado (SHA-256 {meta['sha256']}, entrenado con el "
               f"commit {meta['commit_entrenamiento'][:7]}); no se reentrena.")
        reusado = True
    else:
        reusado = False
        v, X, seg_vec = bl.ajustar_vectorizador([r["nom_estab"] for r in e.split.train], bl.CONFIG_TFIDF)
        tiempos["vectorizador_s"] = seg_vec
        actual, _ = bl._memoria_proceso()
        avance(f"TF-IDF: vectorizador ajustado con train final en {seg_vec:.0f} s ({X.shape[0]:,} filas, "
               f"{X.shape[1]:,} rasgos, {X.nnz:,} no ceros). Memoria del proceso {actual} MB; libre "
               f"{_texto_gb(memoria_libre_gb())}.")
        avance(f"TF-IDF + LR, C = {e.params.C:g}: ajustando la regresión logística ({time.strftime('%H:%M')}; en "
               f"validación tardó {reg_val['cpu']['tiempo_ajuste_s'] / 60:.0f} min)...")
        a = bl.ajustar_lr(X, [r["codigo_act"] for r in e.split.train], e.params.C, bl.CONFIG_LR)
        tiempos["ajuste_s"] = a.segundos
        avance(f"TF-IDF + LR ajustada en {a.segundos / 60:.1f} min: {a.n_iter} iteraciones, convergió: {a.convergio}, "
               f"avisos: {list(a.avisos)}, pico del ajuste {a.memoria_pico_mb:,.0f} MB (en validación: "
               f"{reg_val['extra']['n_iter']} iteraciones, {reg_val['cpu']['memoria_pico_ajuste_mb']:,.0f} MB).")
        ajuste = {"tiempo_vectorizador_s": seg_vec, "tiempo_ajuste_s": a.segundos, "n_iter": a.n_iter,
                  "convergio": a.convergio, "avisos": list(a.avisos), "memoria_pico_ajuste_mb": a.memoria_pico_mb,
                  "rasgos": int(X.shape[1]), "no_ceros": int(X.nnz), "clases_en_train": int(len(a.modelo.classes_))}
        t0 = time.perf_counter()
        meta = guardar_modelo(v, a.modelo, salidas.carpeta, {
            **esperado_modelo(e), "commit_entrenamiento": commit, "python": platform.python_version(),
            "fecha": datetime.date.today().isoformat(), "ajuste": ajuste})
        tiempos["guardar_modelo_s"] = round(time.perf_counter() - t0, 2)
        avance(f"Modelo guardado antes de comparar: {rutas_modelo(salidas.carpeta)[0].name}, {meta['bytes']:,} bytes, "
               f"SHA-256 {meta['sha256']} ({tiempos['guardar_modelo_s']:.0f} s).")
        del v, X, a
        gc.collect()
    t0 = time.perf_counter()
    v, modelo = cargar_modelo(salidas.carpeta, meta)
    tiempos["cargar_modelo_s"] = round(time.perf_counter() - t0, 2)
    avance(f"Modelo recargado desde el archivo (SHA-256 verificado) en {tiempos['cargar_modelo_s']:.0f} s; "
           "las predicciones salen de él.")
    t0 = time.perf_counter()
    resp_val = predecir_tfidf(v, modelo, [r["nom_estab"] for r in e.split.val])
    _, aciertos = bl.evaluar_respuestas(resp_val, e.split.val, e.codigos)
    comp = comparar_aciertos(aciertos["clase"], reg_val["aciertos"]["clase"])
    exigir_reproduccion(ID_TFIDF, comp, avance)
    pred = {"validacion": filas_prediccion(e.split.val, resp_val)}
    for conj in RESERVADOS:
        pred[conj] = filas_prediccion(e.copias[conj], predecir_tfidf(v, modelo, [r["nom_estab"] for r in e.copias[conj]]))
        avance(f"TF-IDF + LR, {conj}: {len(pred[conj]):,} filas predichas.")
    tiempos["prediccion_s"] = round(time.perf_counter() - t0, 2)
    avance("Predicciones de TF-IDF + LR:")
    for conj in CONJUNTOS:
        salidas.guardar(ID_TFIDF, conj, pred[conj], avance)
    info = {**meta["ajuste"], "modelo_reusado": reusado, "commit_entrenamiento": meta["commit_entrenamiento"]}
    return comp, tiempos, {"meta": meta, "info": info}


# --- d. Registro de la corrida en results/ ---


def verificar_sin_ids(obj, ids: set[str]) -> None:
    """Falla si algún texto del objeto es un id; el error solo da las rutas de los campos."""
    rutas = [ruta for ruta, t in textos(obj) if t in ids]
    if rutas:
        raise ErrorFase4(f"Textos iguales a un id en: {rutas}")


def comprobar_head(git: Callable[[], EstadoGit], inicial: EstadoGit, avance: Callable[[str], None]) -> None:
    """Paso e: HEAD y árbol limpio iguales a los del inicio."""
    final = git()
    if final.commit != inicial.commit or not final.arbol_limpio:
        raise ErrorFase4(f"El repo cambió durante la corrida (HEAD {inicial.commit[:7]} → {final.commit[:7]}, árbol "
                         f"limpio: {final.arbol_limpio}): no se escribe en results/. Las predicciones ya guardadas "
                         "quedan en data/fase4/baselines/. Decide Samuel.")
    avance(f"HEAD sin cambios durante la corrida ({final.commit[:7]}) y árbol limpio.")


def escribir_resultado(registro: Mapping, ruta: Path, temporal_en: Path) -> None:
    bl.escribir_atomico(bl.a_bytes(registro), ruta, temporal_en=temporal_en)


def armar_registro(*, fecha: str, inicial: EstadoGit, e: Entradas, comps: Mapping[str, Mapping], tfidf: Mapping,
                   salidas: Salidas, tiempos: Mapping, memoria: Mapping, log: str | None) -> dict:
    meta = tfidf["meta"]
    ruta_modelo, ruta_meta = rutas_modelo(salidas.carpeta)
    modelos_cfg = {i: modelo_config(e.params.config, i) for i in (ID_DICC, ID_TFIDF)}
    return {
        "fecha": fecha,
        "experimento": EXPERIMENTO,
        "commit": inicial.commit,
        "arbol_limpio": inicial.arbol_limpio,
        "head_sin_cambios": True,
        "modelo": [bl.MODELO_DICCIONARIO, bl.MODELO_TFIDF],
        "ids_preregistro": [ID_DICC, ID_TFIDF],
        "configuracion": {
            "script": SCRIPT,
            "baselines": "src/baselines.py sin cambios (funciones públicas en el orden de correr_diccionario y correr_tfidf)",
            "preregistro": {"ruta": RUTA_CONFIG, "sha256": e.params.sha_config},
            "python": platform.python_version(),
            "versiones": versiones(),
            "diccionario": bl.config_diccionario(),
            "tfidf": bl.config_tfidf(e.params.C),
            "huellas": {ID_DICC: e.params.huella_dicc, ID_TFIDF: e.params.huella_tfidf},
            "entrada_test_hard": (f"Copias sin codigo_act de kaggle_datasets.{ENTRADA_COPIAS} (solo id y nom_estab), "
                                  "las mismas que suben a Kaggle."),
            "formato_predicciones": FORMATO_PREDICCIONES,
        },
        "semilla": SEMILLA,
        "split": "nombre: train final y validación congelada; test y hard desde las copias sin codigo_act",
        "hash_split": {
            **e.sha_splits,
            **{f"{c}_copia": e.info_copias[c]["sha256"] for c in RESERVADOS},
            **{f"{c}_origen": e.info_copias[c]["origen_sha256"] for c in RESERVADOS},
        },
        "hash_split_origen": (f"data/manifest.json (splits y kaggle_datasets.{ENTRADA_COPIAS}) y {RUTA_CONFIG} "
                              "(conjuntos)"),
        "registros_validacion": {
            i: {"ruta": modelos_cfg[i]["registro_validacion"]["ruta"], "sha256": e.sha_registros[i],
                "commit": modelos_cfg[i]["registro_validacion"]["commit"]} for i in (ID_DICC, ID_TFIDF)},
        "tamano_muestra": {"train_final": len(e.split.train), "train_antes_del_tope": len(e.split.pre_tope),
                           "validacion": len(e.split.val), **{c: len(e.copias[c]) for c in RESERVADOS}},
        "horas_gpu": 0,
        "reproduccion_validacion": dict(comps),
        "tfidf_ajuste": dict(tfidf["info"]),
        "modelo_guardado": {"ruta": _rel(ruta_modelo, salidas.raiz), "bytes": meta["bytes"], "sha256": meta["sha256"],
                            "metadatos": {"ruta": _rel(ruta_meta, salidas.raiz), "sha256": sha256_archivo(ruta_meta)}},
        "predicciones": salidas.archivos,
        "cpu": {"tiempos_s": dict(tiempos), "memoria": dict(memoria)},
        "log": log,
        "advertencias": [
            "Sin métricas de test ni de hard: se calculan una sola vez, en la evaluación final, con las predicciones "
            "de los 8 modelos (uso_unico del config).",
            "Diccionario: ninguna llave de test ni de hard está en train final (nota del config), así que responde la "
            "clase global en todas sus filas, como en validación.",
        ],
    }


# --- Corrida ---


def correr(*, modo: str, raiz: Path, manifest: Mapping, config_bytes: bytes, sha_config: str,
           carpeta_salida: Path | None, carpeta_resultados: Path, minimo_libre_gb: float | None, fecha: str,
           avance: Callable[[str], None] = print, git: Callable[[], EstadoGit] = estado_git,
           memoria: Callable[[], float | None] = memoria_libre_gb, log: str | None = None) -> dict:
    """modo: "control" (no escribe nada ni predice test ni hard), "humo" o "escribir"."""
    if modo not in ("control", "humo", "escribir"):
        raise ErrorFase4(f"Modo desconocido: {modo}.")
    escribe = modo != "control"
    t_inicio = time.perf_counter()
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)  # regla 1; los sorteos de scikit-learn usan random_state = 42

    # 0. Arranque.
    inicial = git()
    avance(f"commit {inicial.commit[:7]} | árbol limpio: {inicial.arbol_limpio} | {SCRIPT} versionado sin cambios: "
           f"{inicial.versionado} | Python {platform.python_version()}, "
           + ", ".join(f"{k} {v}" for k, v in versiones().items()))
    if escribe and not (inicial.arbol_limpio and inicial.versionado):
        raise ErrorFase4(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
    libre = memoria()
    if minimo_libre_gb is not None:
        avance(f"Memoria libre: {_texto_gb(libre)} (mínimo para lanzar: {minimo_libre_gb:.1f} GB; pico estimado "
               f"~{PICO_ESTIMADO_GB:.1f} GB).")
        if libre is None or libre < minimo_libre_gb:
            raise ErrorFase4(f"Memoria libre {_texto_gb(libre)}, por debajo del mínimo de {minimo_libre_gb:.1f} GB: me "
                             "detengo antes de leer datos. Cierra programas (navegador, editores) y vuelve a lanzar el "
                             "mismo comando.")
    else:
        avance(f"Memoria libre: {_texto_gb(libre)} (en el modo {modo} no se exige mínimo; --escribir exige "
               f"{MINIMO_LIBRE_GB:.1f} GB).")
    previos = sorted(Path(carpeta_resultados).glob(PATRON_RESULTADO)) if Path(carpeta_resultados).exists() else []
    if previos:
        if escribe:
            raise ErrorFase4(f"Ya existe {previos[0].name}: esta corrida ya se hizo; no se repite ni se sobrescribe.")
        avance(f"Aviso: ya existe {previos[0].name}; --escribir se detendría.")

    # a. Verificación de entradas.
    t0 = time.perf_counter()
    e = verificar_entradas(manifest, raiz, config_bytes, sha_config, avance)
    meta_previa = revisar_modelo_guardado(carpeta_salida, esperado_modelo(e))
    avance("Modelo TF-IDF guardado: " + ("no hay; se entrenará." if meta_previa is None else
           f"hay uno reusable (SHA-256 {meta_previa['sha256']}, commit {meta_previa['commit_entrenamiento'][:7]})."))
    tiempos = {"carga_y_verificacion_s": round(time.perf_counter() - t0, 2)}
    actual, _ = bl._memoria_proceso()
    avance(f"Entradas verificadas en {tiempos['carga_y_verificacion_s']:.0f} s. Memoria del proceso {actual} MB; "
           f"libre {_texto_gb(memoria())}.")

    salidas = Salidas(carpeta=Path(carpeta_salida), raiz=Path(raiz), archivos={}) if escribe else None

    # b. Diccionario.
    comp_dicc, t_dicc = paso_diccionario(e, salidas, avance)
    tiempos["diccionario_s"] = t_dicc["tiempo_s"]

    if not escribe:
        return _control_estimar(e, tiempos, avance, meta_previa)

    # c. TF-IDF + LR.
    comp_tfidf, t_tfidf, tfidf = paso_tfidf(e, salidas, meta_previa, inicial.commit, avance)
    tiempos.update(t_tfidf)
    tiempos["total_s"] = round(time.perf_counter() - t_inicio, 2)
    _, pico = bl._memoria_proceso()
    mem = {"libre_al_inicio_gb": None if libre is None else round(libre, 2), "minimo_libre_gb": minimo_libre_gb,
           "pico_ajuste_mb": tfidf["info"].get("memoria_pico_ajuste_mb"), "pico_proceso_mb": pico}

    # d y e. HEAD sin cambios, luego el registro en results/.
    comprobar_head(git, inicial, avance)
    registro = armar_registro(fecha=fecha, inicial=inicial, e=e, comps={ID_DICC: comp_dicc, ID_TFIDF: comp_tfidf},
                              tfidf=tfidf, salidas=salidas, tiempos=tiempos, memoria=mem, log=log)
    llave = bl.Normalizador()
    llaves = {r["llave"] for regs in (e.split.train, e.split.val, e.split.pre_tope) for r in regs}
    llaves |= {k for c in e.copias.values() for r in c if (k := llave(r["nom_estab"]))}
    ids = {r["id"] for regs in (e.split.train, e.split.val, e.split.pre_tope, *e.copias.values()) for r in regs}
    verificar_privacidad(registro, llaves, set())
    verificar_sin_ids(registro, ids)
    ruta = Path(carpeta_resultados) / f"{fecha}_fase4_baselines-predicciones.json"
    escribir_resultado(registro, ruta, salidas.carpeta)
    avance(f"Escrito {_rel(ruta, raiz)} (revisión de privacidad: ningún texto es una llave ni un id).")
    avance(f"RESUMEN: diccionario {comp_dicc['identicas']:,}/{comp_dicc['filas']:,} y TF-IDF + LR "
           f"{comp_tfidf['identicas']:,}/{comp_tfidf['filas']:,} filas de validación idénticas; 6 archivos de "
           f"predicciones en {_rel(salidas.carpeta, raiz)}; total {tiempos['total_s'] / 60:.1f} min; pico del "
           f"proceso {pico} MB.")
    return registro


def _control_estimar(e: Entradas, tiempos: Mapping, avance: Callable[[str], None], meta_previa) -> dict:
    """Punto de control: vectoriza train final para medir la memoria antes del ajuste y estima pico y tiempo."""
    reg = e.registros[ID_TFIDF]
    seg_vec, nnz = 0.0, None
    if meta_previa is None:
        v, X, seg_vec = bl.ajustar_vectorizador([r["nom_estab"] for r in e.split.train], bl.CONFIG_TFIDF)
        nnz = int(X.nnz)
    actual, pico_hasta_ahora = bl._memoria_proceso()
    if meta_previa is None:
        del v, X
        gc.collect()
    ajuste_mb = reg["cpu"]["memoria_pico_ajuste_mb"]
    pico_mb = (actual or 0) + ajuste_mb
    recomendado = math.ceil((pico_mb / 1024 + MARGEN_GB) * 2) / 2
    minutos = (sum(tiempos.values()) + seg_vec + reg["cpu"]["tiempo_ajuste_s"] + 60) / 60
    avance(f"Vectorizador de control: {seg_vec:.0f} s, {nnz if nnz is None else f'{nnz:,}'} no ceros. Memoria del "
           f"proceso tras cargar y vectorizar: {actual} MB.")
    avance(f"Pico estimado de --escribir: {actual} MB + {ajuste_mb:,.0f} MB del ajuste (registro de validación) = "
           f"{pico_mb / 1024:.1f} GB; mínimo recomendado (pico + {MARGEN_GB} GB, a 0.5 GB): {recomendado:.1f} GB; "
           f"mínimo del script: {MINIMO_LIBRE_GB:.1f} GB" + (" (alcanza)." if MINIMO_LIBRE_GB >= recomendado else
                                                              " (NO alcanza: hay que subirlo)."))
    avance(f"Tiempo estimado de --escribir: ~{minutos:.0f} min (ajuste de validación: "
           f"{reg['cpu']['tiempo_ajuste_s'] / 60:.0f} min).")
    _, pico_control = bl._memoria_proceso()
    avance(f"Pico de memoria de este punto de control: {pico_control} MB. PUNTO DE CONTROL: no se escribió nada ni "
           "se predijo test ni hard.")
    return {"memoria_proceso_mb": actual, "pico_estimado_gb": round(pico_mb / 1024, 2), "minimo_recomendado_gb": recomendado,
            "pico_control_mb": pico_control, "minutos_estimados": round(minutos, 1), "no_ceros": nnz}


# --- Juguete para --humo y las pruebas (solo datos inventados) ---

A, B, C_, D = "461110", "311110", "721110", "111110"
PALABRA = {A: "Abarrotes", B: "Panadería", C_: "Hotel", D: "Rancho"}
COMMIT_JUGUETE = "1" * 40


def _reg(i: int, nombre: str, codigo: str) -> dict:
    return {"id": str(i), "nom_estab": nombre, "codigo_act": codigo, "llave": normalizar_nombre(nombre),
            "sin_nombre": False, "generico": False, "igual_titulo": False}


@dataclass
class Juguete:
    carpeta: Path
    raiz: Path
    manifest: dict
    config: dict
    config_bytes: bytes
    salida: Path
    resultados: Path

    def nombres(self) -> dict[str, list[dict]]:
        """Filas de los cinco conjuntos (con nombre, llave e id), solo para las pruebas de privacidad."""
        res = {}
        for nombre, entrada in (("train", bl.ENTRADA_TRAIN_FINAL), ("pre_tope", bl.ENTRADA_ORIGEN),
                                ("validacion", bl.ENTRADA_VAL), ("test", "clasificador_test"),
                                ("hard", "clasificador_hard")):
            texto = (self.raiz / self.manifest["splits"][entrada]["ruta"]).read_text(encoding="utf-8")
            res[nombre] = [json.loads(linea) for linea in texto.splitlines()]
        return res


def serializar_config(config: Mapping) -> bytes:
    return (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def guardar_config_juguete(j: Juguete) -> None:
    j.config_bytes = serializar_config(j.config)
    (j.raiz / RUTA_CONFIG).write_bytes(j.config_bytes)


def armar_juguete(carpeta: Path) -> Juguete:
    """Repo inventado en carpeta/repo y copias sin etiquetas en carpeta/kaggle-datasets/denue-scian-evaluacion,
    armadas con src/paquete_evaluacion.py; registros de validación generados con src/baselines.py; config = el
    real con los hashes, rutas y commit del juguete."""
    carpeta = Path(carpeta)
    raiz = carpeta / "repo"
    (raiz / "data" / "splits").mkdir(parents=True)
    (raiz / "results").mkdir()
    (raiz / "evals").mkdir()
    train = sorted((_reg(31_000_000 + 1000 * i + k, f"{PALABRA[c]} inventado {k % 7}", c)
                    for i, c in enumerate((A, B, C_, D)) for k in range(25)), key=lambda r: int(r["id"]))
    extra = [_reg(32_000_000 + k, f"{PALABRA[c]} inventado {k % 7}", c) for k, c in enumerate((A, B, C_, D) * 3)]
    pre_tope = sorted(train + extra, key=lambda r: int(r["id"]))
    val = [_reg(41_000_000 + k, f"{PALABRA[c]} único {k}", c) for k, c in enumerate((A, B, C_, D) * 3)]
    test = [_reg(51_000_000 + k, f"{PALABRA[c]} prueba {k}", c) for k, c in enumerate((B, A, D, C_, A, B))]
    hard = [_reg(61_000_000 + k, f"Raro inventado {k}", c) for k, c in enumerate((C_, D, A))]
    manifest: dict = {"scian_version": "2023", "splits": {}, "catalogos": {}, "kaggle_datasets": {}}
    for entrada, archivo, regs in ((bl.ENTRADA_TRAIN_FINAL, "train_final.jsonl", train),
                                   (bl.ENTRADA_ORIGEN, "train_candidatos_sin_hard.jsonl", pre_tope),
                                   (bl.ENTRADA_VAL, "val.jsonl", val),
                                   ("clasificador_test", "test.jsonl", test),
                                   ("clasificador_hard", "hard.jsonl", hard)):
        datos = serializar_split(regs)
        (raiz / "data" / "splits" / archivo).write_bytes(datos)
        manifest["splits"][entrada] = {"ruta": f"data/splits/{archivo}", "sha256": sha256_bytes(datos),
                                       "filas": len(regs), "campos": list(regs[0])}
    codigos = sorted({A, B, C_, D, *(f"{x[:2]}1110" for x in SECTORES_2023)})
    cat = ("codigo,titulo\n" + "".join(f"{c},Título inventado {c}\n" for c in codigos)).encode("utf-8")
    (raiz / "data" / "cat.csv").write_bytes(cat)
    manifest["catalogos"]["scian_2023_clases"] = {"ruta": "data/cat.csv", "sha256": sha256_bytes(cat)}

    datos, filas, _ = pe.preparar(manifest, raiz)
    escribir_carpeta(datos, (raiz / pe.CARPETA_RELATIVA).resolve(), raiz=raiz, permitidos=pe.PERMITIDOS)
    manifest["kaggle_datasets"][pe.ENTRADA] = pe.entrada_kaggle(manifest=manifest, datos=datos, filas=filas,
                                                                commit=COMMIT_JUGUETE, fecha="2026-11-25")

    split = bl.cargar_split_nombre(manifest, raiz)
    cods = cargar_codigos(manifest, raiz)
    config = copy.deepcopy(json.loads((REPO_RAIZ / RUTA_CONFIG).read_text(encoding="utf-8")))
    carpeta_reg = raiz / "data" / "baselines" / COMMIT_JUGUETE[:12]
    carpeta_reg.mkdir(parents=True)
    calculos = {
        ID_DICC: ("diccionario_nombre", bl.huella(bl.config_diccionario()), lambda: bl.correr_diccionario(split, cods)),
        ID_TFIDF: ("tfidf_nombre_C10", bl.huella(bl.config_tfidf(10.0)), lambda: bl.correr_tfidf(split, 10.0, cods)[0]),
    }
    for id_modelo, (clave, h, calcular) in calculos.items():
        reg = {"clave": clave, "commit": COMMIT_JUGUETE, "huella_config": h, **calcular()}
        datos_reg = bl.a_bytes(reg)
        (carpeta_reg / f"{clave}.json").write_bytes(datos_reg)
        modelo_config(config, id_modelo)["registro_validacion"] = {
            "ruta": f"data/baselines/{COMMIT_JUGUETE[:12]}/{clave}.json", "sha256": sha256_bytes(datos_reg),
            "commit": COMMIT_JUGUETE}
    for nombre, entrada in (("train_final", bl.ENTRADA_TRAIN_FINAL), ("validacion", bl.ENTRADA_VAL),
                            ("test", "clasificador_test"), ("hard", "clasificador_hard")):
        config["conjuntos"][nombre].update(sha256=manifest["splits"][entrada]["sha256"],
                                           filas=manifest["splits"][entrada]["filas"])
    j = Juguete(carpeta=carpeta, raiz=raiz, manifest=manifest, config=config, config_bytes=b"",
                salida=raiz / CARPETA_SALIDA, resultados=raiz / "results")
    guardar_config_juguete(j)
    return j


def verificar_carpeta_humo(carpeta: Path) -> Path:
    """La carpeta del humo queda fuera del repo (y así fuera de results/ y de data/fase4/) y vacía."""
    c, r = Path(carpeta).resolve(), REPO_RAIZ.resolve()
    if c == r or r in c.parents:
        raise ErrorFase4(f"La carpeta del humo {c} está dentro del repo; debe quedar fuera (nunca results/ ni data/fase4/).")
    if c.exists() and any(c.iterdir()):
        raise ErrorFase4(f"La carpeta del humo {c} ya tiene archivos.")
    return c


# --- Programa principal ---


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--escribir", action="store_true", help="la corrida real: predicciones y results/")
    grupo.add_argument("--humo", metavar="CARPETA", help="corrida de humo con datos inventados en CARPETA (fuera del repo)")
    args = parser.parse_args(argv)

    fecha = datetime.date.today().isoformat()
    marca = time.strftime("%Y-%m-%d_%H%M%S")
    bitacora = Bitacora(None)
    bloqueada = False
    codigo = 0
    try:
        if args.humo:
            carpeta = verificar_carpeta_humo(Path(args.humo))
            j = armar_juguete(carpeta)
            j.salida.mkdir(parents=True, exist_ok=True)
            bitacora = Bitacora(j.salida / f"baselines_{marca}.log")
            bitacora(f"HUMO con datos inventados en {carpeta}: no toca results/ ni data/fase4/ del repo.")
            kwargs = dict(modo="humo", raiz=j.raiz, manifest=j.manifest, config_bytes=j.config_bytes,
                          sha_config=sha256_bytes(j.config_bytes), carpeta_salida=j.salida,
                          carpeta_resultados=j.resultados, minimo_libre_gb=None, log=bitacora.ruta.name)
        elif args.escribir:
            salida = REPO_RAIZ / CARPETA_SALIDA
            if not esta_ignorado(salida / "prueba.log"):
                sys.exit(f"{CARPETA_SALIDA}/ no está ignorado por git; me detengo.")
            salida.mkdir(parents=True, exist_ok=True)
            bitacora = Bitacora(salida / f"baselines_{marca}.log")
            bloqueada = evitar_suspension()
            bitacora(f"Inicio de --escribir. Log: {CARPETA_SALIDA}/{bitacora.ruta.name}. Suspensión del equipo por "
                     f"inactividad bloqueada mientras corre: {bloqueada}.")
            kwargs = dict(modo="escribir", raiz=REPO_RAIZ, manifest=leer_manifest(),
                          config_bytes=(REPO_RAIZ / RUTA_CONFIG).read_bytes(), sha_config=SHA_CONFIG,
                          carpeta_salida=salida, carpeta_resultados=REPO_RAIZ / "results",
                          minimo_libre_gb=MINIMO_LIBRE_GB, log=f"{CARPETA_SALIDA}/{bitacora.ruta.name}")
        else:
            bitacora("PUNTO DE CONTROL con los datos reales: no escribe nada ni predice test ni hard.")
            kwargs = dict(modo="control", raiz=REPO_RAIZ, manifest=leer_manifest(),
                          config_bytes=(REPO_RAIZ / RUTA_CONFIG).read_bytes(), sha_config=SHA_CONFIG,
                          carpeta_salida=REPO_RAIZ / CARPETA_SALIDA, carpeta_resultados=REPO_RAIZ / "results",
                          minimo_libre_gb=None)
        correr(fecha=fecha, avance=bitacora, **kwargs)
        bitacora("Fin." + (" Corrida de humo terminada." if args.humo else ""))
    except BaseException as e:  # noqa: BLE001 (todo paro queda en el log, sin datos)
        if isinstance(e, SystemExit):
            raise
        registrar_error(e, bitacora)
        codigo = 1
    finally:
        if bloqueada:
            permitir_suspension()
    sys.exit(codigo)


if __name__ == "__main__":
    main()
