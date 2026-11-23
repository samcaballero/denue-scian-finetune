"""Baselines del clasificador y experimento de fuga (Fase 1, paso 10).

Los dos baselines obligatorios (CLAUDE.md, sección a), evaluados con src/evaluador_clasificador.py:

- Diccionario: la entrada es nom_estab, que se pasa por normalizar_nombre(). Responde la clase
  más frecuente de la llave en train y, si la llave no se vio, la clase más frecuente global.
  Empates: el mismo conteo medido en el train antes del tope (dentro de una llave, el de
  (llave, clase); en el global, el de la clase) y, si persiste, el código menor.
- TF-IDF de n-gramas de caracteres + regresión logística multinomial sobre el nom_estab
  original. El vectorizador se ajusta solo con los nombres de train de cada split, nunca con
  validación. Barrido chico de C, elegido por accuracy en validación (en empate, el C menor).

Splits: por nombre (train_final.jsonl y val.jsonl) y aleatorio (aleatorio_train_final.jsonl y
aleatorio_val.jsonl, de src/split_aleatorio.py). Cada baseline usa en los dos splits la misma
configuración: la elegida en el split por nombre. La diferencia de accuracy entre los dos splits
es la medida de la fuga.

Uso:
  .venv/Scripts/python -m src.baselines --punto-de-control   # PARADA 1: tiempo y memoria en submuestras
  .venv/Scripts/python -m src.baselines --escribir           # ajustes completos y results/

--escribir guarda cada ajuste terminado en data/baselines/<commit>/ (ignorado por git; escritura
atómica) y escribe cada JSON de results/ solo cuando terminan todos sus ajustes. Si la corrida se
corta, al lanzarla de nuevo con el mismo commit reusa los ajustes terminados. Nunca abre test.jsonl
ni hard.jsonl. En la terminal y en results/ solo van conteos, métricas y códigos SCIAN: nunca
nombres, llaves, ids ni el vocabulario del vectorizador. Los modelos no se guardan.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import tracemalloc
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version as version_de_paquete
from pathlib import Path

import numpy as np

from src.denue import REPO_RAIZ
from src.evaluador_clasificador import (
    aciertos_por_nivel,
    cargar_codigos,
    diferencia_bootstrap,
    evaluar_validacion,
    validar_referencias,
)
from src.muestra_humo import leer_verificado, parsear
from src.normalizacion import normalizar_nombre

SCRIPT = "src/baselines.py"
SEMILLA = 42
EXPERIMENTO = "fase1_paso10_baselines"

ENTRADA_TRAIN_FINAL = "clasificador_train_final"
ENTRADA_ORIGEN = "clasificador_train_candidatos_sin_hard"
ENTRADA_VAL = "clasificador_validacion"
ENTRADA_A_TRAIN = "experimento_aleatorio_train_final"
ENTRADA_A_VAL = "experimento_aleatorio_validacion"
SPLITS = ("nombre", "aleatorio")

MODELO_DICCIONARIO = "baseline_diccionario"
MODELO_TFIDF = "baseline_tfidf_lr"
ARCHIVO_MODELO = {MODELO_DICCIONARIO: "baseline-diccionario", MODELO_TFIDF: "baseline-tfidf-lr"}

CONFIG_TFIDF = {
    "analyzer": "char_wb",
    "ngram_range": (2, 5),
    "min_df": 2,
    "max_features": 100_000,
    "sublinear_tf": True,
    "norm": "l2",
    "lowercase": True,
    "strip_accents": None,
}
CONFIG_LR = {"solver": "saga", "tol": 1e-3, "max_iter": 100, "random_state": SEMILLA}
VALORES_C = (1.0, 10.0)
DECISION_VALORES_C = (
    "Barrido en el train completo del split por nombre con C ∈ {1, 10}. El plan aprobado tenía C ∈ {1, 10, 100}. "
    "C = 100 se descartó porque en el punto de control de costo (PARADA 1) no convergió en 100 iteraciones ni con "
    "10,000 filas, y en el train completo costaría 3.9 h o más. También se descartó elegir C en una submuestra de "
    "30,000 filas (alternativa que propuso Claude Code en la PARADA 1), porque un C elegido con pocas filas puede no "
    "ser el mejor con el train completo. El chat de Claude del proyecto propuso esta rejilla y Samuel la aprobó el "
    "2026-11-22."
)
PREPROCESAMIENTO_TFIDF = (
    "Entrada: nom_estab original. Preprocesamiento propio del vectorizador: todo pasa a minúsculas "
    "(lowercase=True); los acentos y la Ñ se conservan (strip_accents=None); analyzer='char_wb' saca los n-gramas "
    "de 2 a 5 caracteres dentro de cada palabra (separadas por espacios), rellenada con un espacio a cada lado. Solo "
    "se conservan los n-gramas de al menos 2 nombres de train (min_df=2), y de ellos los 100,000 más frecuentes; "
    "TF sublineal (1 + log tf) y norma L2 por fila. El vectorizador se ajusta solo con los nombres de train."
)
DESCRIPCION_LR = (
    "LogisticRegression de scikit-learn con solver saga (descenso de gradiente estocástico con promedio; memoria "
    "lineal) y penalización L2 (l1_ratio=0, el valor por omisión). Con 3 o más clases, scikit-learn ajusta el modelo "
    "multinomial. C es el inverso de la fuerza de la regularización."
)
REGLA_C = "Gana el C con más aciertos de clase en la validación del split por nombre; en empate, el C menor."
SUBMUESTRAS = (10_000, 30_000)
LIMITE_HORAS = 2
LIMITE_GB = 10


class ErrorBaseline(ValueError):
    """Entrada o estado inválido; el mensaje nunca incluye nombres, llaves ni ids."""


# --- Memoria y tiempo ---


def _memoria_proceso() -> tuple[float | None, float | None]:
    """(MB del conjunto de trabajo actual, MB del pico) del proceso; (None, None) si no se puede medir."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Contadores(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            c = Contadores()
            c.cb = ctypes.sizeof(c)
            kernel32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Contadores), wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return None, None
            return round(c.WorkingSetSize / 2**20, 1), round(c.PeakWorkingSetSize / 2**20, 1)
        import resource

        return None, round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # Linux: KB
    except (OSError, AttributeError, ImportError):
        return None, None


def memoria_formula_mb(filas: int, clases: int, rasgos: int, nnz: int) -> float:
    """Memoria de saga multinomial: coeficientes y suma de gradientes (2 × clases × (rasgos + 1)), memoria de
    gradientes (filas × clases), en float64, más la matriz dispersa (nnz × 12 B + índices de filas)."""
    return round((8 * (2 * clases * (rasgos + 1) + filas * clases) + 12 * nnz + 4 * (filas + 1)) / 2**20, 1)


# --- Lectura ---


@dataclass(frozen=True)
class Split:
    nombre: str
    train: list[dict]
    val: list[dict]
    pre_tope: list[dict]
    hash_split: dict[str, str]
    hash_split_origen: str
    hash_pre_tope: dict[str, str]
    descripcion: str


def _leer(manifest: Mapping, entrada: str, raiz: Path) -> list[dict]:
    if entrada in ("clasificador_test", "clasificador_hard"):
        raise ErrorBaseline("Test y hard no se abren antes de la Fase 4 (regla 3).")
    e = manifest["splits"][entrada]
    regs = parsear(leer_verificado(Path(raiz) / e["ruta"], e["sha256"]))
    if len(regs) != e["filas"]:
        raise ErrorBaseline(f"Las filas de {entrada} no son las del manifest.")
    return regs


def _ids(regs: Sequence[Mapping]) -> set[str]:
    return {r["id"] for r in regs}


def cargar_split_nombre(manifest: Mapping, raiz: Path = REPO_RAIZ, origen: list[dict] | None = None,
                        val: list[dict] | None = None) -> Split:
    """train_final y validación, con el train candidato sin hard para los desempates."""
    sp = manifest["splits"]
    train = _leer(manifest, ENTRADA_TRAIN_FINAL, raiz)
    origen = _leer(manifest, ENTRADA_ORIGEN, raiz) if origen is None else origen
    val = _leer(manifest, ENTRADA_VAL, raiz) if val is None else val
    if not _ids(train) <= _ids(origen):
        raise ErrorBaseline("train_final tiene filas fuera del train candidato sin hard.")
    if {r["llave"] for r in train} & {r["llave"] for r in val}:
        raise ErrorBaseline("train_final comparte llaves con validación (regla 2).")
    return Split(
        nombre="nombre", train=train, val=val, pre_tope=origen,
        hash_split={"train": sp[ENTRADA_TRAIN_FINAL]["sha256"], "validacion": sp[ENTRADA_VAL]["sha256"]},
        hash_split_origen=f"data/manifest.json, splits.{ENTRADA_TRAIN_FINAL} y splits.{ENTRADA_VAL}",
        hash_pre_tope={"train_candidatos_sin_hard": sp[ENTRADA_ORIGEN]["sha256"]},
        descripcion="Split por nombre: train final (tope 1,000) y validación congelada; ninguna llave en los dos (regla 2).",
    )


def cargar_splits(manifest: Mapping, raiz: Path = REPO_RAIZ) -> dict[str, Split]:
    """Los dos splits, todos los archivos con su SHA-256 verificado; nunca abre test ni hard."""
    sp = manifest["splits"]
    origen = _leer(manifest, ENTRADA_ORIGEN, raiz)
    val = _leer(manifest, ENTRADA_VAL, raiz)
    nombre = cargar_split_nombre(manifest, raiz, origen, val)
    a_train = _leer(manifest, ENTRADA_A_TRAIN, raiz)
    a_val = _leer(manifest, ENTRADA_A_VAL, raiz)
    p = sorted([*origen, *val], key=lambda r: int(r["id"]))
    if len(_ids(p)) != len(p):
        raise ErrorBaseline("Train candidato sin hard y validación comparten id.")
    ids_a_val = _ids(a_val)
    if not ids_a_val <= _ids(p):
        raise ErrorBaseline("La validación aleatoria tiene filas fuera de P.")
    resto = [r for r in p if r["id"] not in ids_a_val]
    if not _ids(a_train) <= _ids(resto):
        raise ErrorBaseline("El train aleatorio tiene filas fuera del resto de P.")
    aleatorio = Split(
        nombre="aleatorio", train=a_train, val=a_val, pre_tope=resto,
        hash_split={"train": sp[ENTRADA_A_TRAIN]["sha256"], "validacion": sp[ENTRADA_A_VAL]["sha256"]},
        hash_split_origen=f"data/manifest.json, splits.{ENTRADA_A_TRAIN} y splits.{ENTRADA_A_VAL}",
        hash_pre_tope={"train_candidatos_sin_hard": sp[ENTRADA_ORIGEN]["sha256"], "validacion": sp[ENTRADA_VAL]["sha256"]},
        descripcion=(
            "Split aleatorio del experimento de fuga (src/split_aleatorio.py): validación de 1,000 filas de P = train "
            "candidato sin hard ∪ validación sorteadas sin mirar llaves; train = el resto de P con tope 1,000. Las "
            "llaves de validación sí pueden estar en train."
        ),
    )
    return {"nombre": nombre, "aleatorio": aleatorio}


# --- Diccionario ---


@dataclass(frozen=True)
class Diccionario:
    por_llave: dict[str, str]
    clase_global: str
    empates: dict = field(default_factory=dict)


class Normalizador:
    """normalizar_nombre() con caché por nom_estab."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def __call__(self, nombre: str) -> str:
        k = self._cache.get(nombre)
        if k is None:
            k = self._cache[nombre] = normalizar_nombre(nombre)
        return k


def verificar_llaves(regs: Sequence[Mapping], llave: Callable[[str], str]) -> None:
    """La llave calculada desde nom_estab es la del campo `llave` (src/splits.py)."""
    malas = sum(1 for r in regs if llave(r["nom_estab"]) != r["llave"])
    if malas:
        raise ErrorBaseline(f"{malas} filas tienen una llave distinta de normalizar_nombre(nom_estab).")


def ajustar_diccionario(train: Sequence[Mapping], pre_tope: Sequence[Mapping],
                        llave: Callable[[str], str] | None = None) -> Diccionario:
    """Clase más frecuente por llave y global, con los desempates del módulo."""
    llave = llave or Normalizador()
    if not train:
        raise ErrorBaseline("El train está vacío.")
    if not _ids(train) <= _ids(pre_tope):
        raise ErrorBaseline("El train tiene filas fuera del train antes del tope.")
    por_llave_cnt: dict[str, Counter] = defaultdict(Counter)
    for r in train:
        por_llave_cnt[llave(r["nom_estab"])][r["codigo_act"]] += 1
    pre_llave = Counter((llave(r["nom_estab"]), r["codigo_act"]) for r in pre_tope)
    pre_clase = Counter(r["codigo_act"] for r in pre_tope)

    por_llave: dict[str, str] = {}
    empatadas = por_pre = por_codigo = 0
    for k, cnt in por_llave_cnt.items():
        maximo = max(cnt.values())
        candidatas = [c for c, n in cnt.items() if n == maximo]
        if len(candidatas) > 1:
            empatadas += 1
            mejor_pre = max(pre_llave[(k, c)] for c in candidatas)
            if sum(pre_llave[(k, c)] == mejor_pre for c in candidatas) > 1:
                por_codigo += 1
            else:
                por_pre += 1
        por_llave[k] = min(candidatas, key=lambda c: (-pre_llave[(k, c)], c))

    glob = Counter(r["codigo_act"] for r in train)
    maximo = max(glob.values())
    candidatas = [c for c, n in glob.items() if n == maximo]
    clase_global = min(candidatas, key=lambda c: (-pre_clase[c], c))
    mejor_pre = pre_clase[clase_global]
    return Diccionario(
        por_llave=por_llave,
        clase_global=clase_global,
        empates={
            "llaves_en_train": len(por_llave),
            "llaves_con_empate": empatadas,
            "resueltas_por_el_conteo_antes_del_tope": por_pre,
            "resueltas_por_el_codigo_menor": por_codigo,
            "global": {
                "clase": clase_global,
                "filas_en_train": maximo,
                "clases_empatadas_en_train": len(candidatas),
                "filas_antes_del_tope": mejor_pre,
                "resuelto_por": (
                    "sin empate" if len(candidatas) == 1
                    else "código menor" if sum(pre_clase[c] == mejor_pre for c in candidatas) > 1
                    else "conteo antes del tope"
                ),
            },
        },
    )


def predecir_diccionario(d: Diccionario, nombres: Sequence[str], llave: Callable[[str], str] | None = None) -> list[str]:
    llave = llave or Normalizador()
    return [d.por_llave.get(llave(n), d.clase_global) for n in nombres]


# --- TF-IDF + regresión logística ---


def vectorizador(config: Mapping = CONFIG_TFIDF):
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(**config)


def ajustar_vectorizador(nombres_train: Sequence[str], config: Mapping = CONFIG_TFIDF):
    """(vectorizador, X de train, segundos). Se ajusta solo con los nombres de train, nunca con validación."""
    t0 = time.perf_counter()
    v = vectorizador(config)
    X = v.fit_transform(list(nombres_train))
    return v, X, round(time.perf_counter() - t0, 2)


@dataclass(frozen=True)
class AjusteLR:
    modelo: object
    C: float
    segundos: float
    n_iter: int
    convergio: bool
    memoria_pico_mb: float
    avisos: tuple[str, ...]


def ajustar_lr(X, y: Sequence[str], C: float, config: Mapping = CONFIG_LR) -> AjusteLR:
    """Ajusta la regresión logística; mide tiempo de pared y memoria pico (tracemalloc) del ajuste."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    modelo = LogisticRegression(C=C, **config)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        tracemalloc.start()
        try:
            t0 = time.perf_counter()
            modelo.fit(X, list(y))
            segundos = time.perf_counter() - t0
            _, pico = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    categorias = tuple(sorted({a.category.__name__ for a in avisos}))
    return AjusteLR(
        modelo=modelo, C=float(C), segundos=round(segundos, 2), n_iter=int(np.max(modelo.n_iter_)),
        convergio=not any(issubclass(a.category, ConvergenceWarning) for a in avisos),
        memoria_pico_mb=round(pico / 2**20, 1), avisos=categorias,
    )


def elegir_c(barrido: Sequence[Mapping]) -> float:
    """El C con más aciertos; en empate, el menor."""
    if not barrido:
        raise ErrorBaseline("El barrido está vacío.")
    return min(barrido, key=lambda b: (-b["aciertos"], b["C"]))["C"]


# --- Evaluación de un ajuste ---


def evaluar_respuestas(respuestas: Sequence[str], val: Sequence[Mapping], codigos) -> tuple[dict, dict[str, list[bool]]]:
    """(métricas del evaluador, aciertos de clase y de sector en orden de int(id))."""
    pred = {r["id"]: str(c) for r, c in zip(val, respuestas, strict=True)}
    metricas = evaluar_validacion(pred, val, codigos)
    niveles = aciertos_por_nivel(pred, validar_referencias(val), codigos)
    return metricas, {"clase": niveles["clase"], "sector": niveles["sector"]}


def correr_diccionario(split: Split, codigos, llave: Callable[[str], str] | None = None) -> dict:
    llave = llave or Normalizador()
    verificar_llaves(split.train, llave)
    verificar_llaves(split.val, llave)
    t0 = time.perf_counter()
    d = ajustar_diccionario(split.train, split.pre_tope, llave)
    segundos = round(time.perf_counter() - t0, 2)
    respuestas = predecir_diccionario(d, [r["nom_estab"] for r in split.val], llave)
    metricas, aciertos = evaluar_respuestas(respuestas, split.val, codigos)
    vistas = sum(1 for r in split.val if llave(r["nom_estab"]) in d.por_llave)
    return {
        "metricas": metricas,
        "aciertos": aciertos,
        "cpu": {"tiempo_ajuste_s": segundos},
        "extra": {
            "empates": d.empates,
            "validacion_con_llave_vista_en_train": {"filas": vistas, "pct_filas": round(100 * vistas / len(split.val), 2)},
        },
    }


def correr_tfidf(split: Split, C: float, codigos, vec=None, config_tfidf: Mapping = CONFIG_TFIDF,
                 config_lr: Mapping = CONFIG_LR) -> tuple[dict, tuple]:
    """Un ajuste de TF-IDF + LR; `vec` = (vectorizador, X de train, segundos) para reusarlo en el barrido."""
    if vec is None:
        vec = ajustar_vectorizador([r["nom_estab"] for r in split.train], config_tfidf)
    v, X, seg_vec = vec
    a = ajustar_lr(X, [r["codigo_act"] for r in split.train], C, config_lr)
    respuestas = a.modelo.predict(v.transform([r["nom_estab"] for r in split.val]))
    metricas, aciertos = evaluar_respuestas(list(respuestas), split.val, codigos)
    _, pico_proceso = _memoria_proceso()
    registro = {
        "metricas": metricas,
        "aciertos": aciertos,
        "cpu": {
            "tiempo_vectorizador_s": seg_vec,
            "tiempo_ajuste_s": a.segundos,
            "memoria_pico_ajuste_mb": a.memoria_pico_mb,
            "memoria_pico_proceso_mb": pico_proceso,
        },
        "extra": {
            "C": a.C,
            "n_iter": a.n_iter,
            "convergio": a.convergio,
            "avisos": list(a.avisos),
            "rasgos": int(X.shape[1]),
            "clases_en_train": int(len(a.modelo.classes_)),
        },
    }
    return registro, vec


# --- Punto de control ---


def medir_submuestras(train: Sequence[Mapping], submuestras: Sequence[int] = SUBMUESTRAS,
                      valores_c: Sequence[float] = VALORES_C, config_tfidf: Mapping = CONFIG_TFIDF,
                      config_lr: Mapping = CONFIG_LR, avance: Callable[[str], None] = lambda _: None) -> list[dict]:
    """Tiempo, iteraciones y memoria de cada C en submuestras anidadas de train (prefijos de una permutación)."""
    perm = np.random.default_rng(SEMILLA).permutation(len(train))
    filas = []
    for k in submuestras:
        if k > len(train):
            raise ErrorBaseline(f"La submuestra de {k} filas pasa del train ({len(train)}).")
        sub = [train[int(i)] for i in np.sort(perm[:k])]
        y = [r["codigo_act"] for r in sub]
        _, X, seg_vec = ajustar_vectorizador([r["nom_estab"] for r in sub], config_tfidf)
        clases = len(set(y))
        for C in valores_c:
            avance(f"submuestra {k:,}, C = {C:g}: ajustando...")
            a = ajustar_lr(X, y, C, config_lr)
            filas.append({
                "filas": k, "clases": clases, "rasgos": int(X.shape[1]), "nnz": int(X.nnz),
                "tiempo_vectorizador_s": seg_vec, "C": float(C), "tiempo_ajuste_s": a.segundos, "n_iter": a.n_iter,
                "convergio": a.convergio, "memoria_pico_ajuste_mb": a.memoria_pico_mb,
                "memoria_formula_mb": memoria_formula_mb(k, clases, int(X.shape[1]), int(X.nnz)),
            })
            avance(f"  {a.segundos:,.1f} s, {a.n_iter} iteraciones, convergió: {a.convergio}, "
                   f"memoria pico {a.memoria_pico_mb:,.0f} MB")
    return filas


def estimar(filas: Sequence[Mapping], *, filas_full: int, clases_full: int, memoria_datos_mb: float | None,
            config_tfidf: Mapping = CONFIG_TFIDF, config_lr: Mapping = CONFIG_LR) -> dict:
    """Extrapola de la submuestra más grande al train completo: los ajustes del barrido + 1 del split aleatorio."""
    k = max(f["filas"] for f in filas)
    grandes = [f for f in filas if f["filas"] == k]
    escala = (filas_full * clases_full) / (k * grandes[0]["clases"])  # costo por iteración ∝ filas × clases
    por_c = []
    for f in grandes:
        s_iter = f["tiempo_ajuste_s"] / max(f["n_iter"], 1)
        por_c.append({
            "C": f["C"],
            "lineal_s": round(s_iter * escala * f["n_iter"], 0),
            "pesimista_s": round(s_iter * escala * config_lr["max_iter"], 0),
        })
    vec_s = round(grandes[0]["tiempo_vectorizador_s"] * filas_full / k, 0)
    total = {
        "lineal_s": sum(c["lineal_s"] for c in por_c) + max(c["lineal_s"] for c in por_c) + 2 * vec_s,
        "pesimista_s": sum(c["pesimista_s"] for c in por_c) + max(c["pesimista_s"] for c in por_c) + 2 * vec_s,
    }
    nnz_full = int(grandes[0]["nnz"] * filas_full / k)
    rasgos_full = int(config_tfidf["max_features"] or max(f["rasgos"] for f in filas))
    formula_full = memoria_formula_mb(filas_full, clases_full, rasgos_full, nnz_full)
    razon = max(f["memoria_pico_ajuste_mb"] / f["memoria_formula_mb"] for f in grandes)
    ajuste_mb = round(formula_full * max(1.0, razon), 0)
    memoria_total = round(ajuste_mb + (memoria_datos_mb or 0), 0)
    return {
        "submuestra_de_referencia": k,
        "escala_filas_por_clases": round(escala, 2),
        "por_c": por_c,
        "vectorizador_s": vec_s,
        "total_horas": {n: round(v / 3600, 2) for n, v in total.items()},
        "memoria": {
            "formula_full_mb": formula_full,
            "razon_medida_contra_formula": round(razon, 2),
            "ajuste_estimado_mb": ajuste_mb,
            "datos_en_memoria_mb": memoria_datos_mb,
            "total_estimado_gb": round(memoria_total / 1024, 2),
        },
        "pasa_limite_horas": total["pesimista_s"] / 3600 > LIMITE_HORAS,
        "pasa_limite_lineal_horas": total["lineal_s"] / 3600 > LIMITE_HORAS,
        "pasa_limite_memoria": memoria_total / 1024 > LIMITE_GB,
    }


# --- Registros de ajustes terminados y results/ ---


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=list)


def huella(config: Mapping) -> str:
    return hashlib.sha256(_json(config).encode("utf-8")).hexdigest()


def escribir_atomico(datos: bytes, ruta: Path, temporal_en: Path | None = None) -> None:
    """Escribe a un temporal y lo renombra: un corte nunca deja el archivo a medias."""
    ruta = Path(ruta)
    carpeta = Path(temporal_en) if temporal_en else ruta.parent
    carpeta.mkdir(parents=True, exist_ok=True)
    temporal = carpeta / (ruta.name + ".tmp")
    temporal.write_bytes(datos)
    os.replace(temporal, ruta)


def a_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


class Registros:
    """Ajustes terminados en data/baselines/<commit>/; se reusan si el commit y la configuración coinciden."""

    def __init__(self, carpeta: Path, commit: str) -> None:
        self.carpeta = Path(carpeta) / commit[:12]
        self.commit = commit

    def obtener(self, clave: str, config: Mapping, calcular: Callable[[], dict], avance=print) -> dict:
        ruta = self.carpeta / f"{clave}.json"
        h = huella(config)
        if ruta.exists():
            reg = json.loads(ruta.read_bytes())
            if reg.get("commit") != self.commit or reg.get("huella_config") != h:
                raise ErrorBaseline(f"El registro {ruta.name} es de otro commit o de otra configuración; me detengo.")
            avance(f"{clave}: reuso el ajuste ya terminado.")
            return reg
        reg = {"clave": clave, "commit": self.commit, "huella_config": h, **calcular()}
        escribir_atomico(a_bytes(reg), ruta)
        return reg


def _versiones() -> dict:
    return {p: version_de_paquete(p) for p in ("numpy", "scipy", "scikit-learn")}


def config_diccionario() -> dict:
    return {
        "entrada": "nom_estab, normalizado con src/normalizacion.py:normalizar_nombre",
        "regla": "Clase más frecuente de la llave en train; si la llave no se vio en train, la clase más frecuente global.",
        "desempate": (
            "Dentro de una llave: el conteo de (llave, clase) en el train antes del tope; en el global: el conteo de "
            "la clase en el train antes del tope. Si persiste, el código menor."
        ),
        "train_antes_del_tope": {
            "nombre": f"splits.{ENTRADA_ORIGEN}",
            "aleatorio": f"P menos la validación aleatoria (splits.{ENTRADA_ORIGEN} ∪ splits.{ENTRADA_VAL} menos splits.{ENTRADA_A_VAL})",
        },
    }


def config_tfidf(C: float | None = None, tfidf: Mapping = CONFIG_TFIDF, lr: Mapping = CONFIG_LR) -> dict:
    res = {
        "vectorizador": {"clase": "sklearn.feature_extraction.text.TfidfVectorizer", **dict(tfidf)},
        "preprocesamiento": PREPROCESAMIENTO_TFIDF,
        "regresion_logistica": {"clase": "sklearn.linear_model.LogisticRegression", **dict(lr), "l1_ratio": 0.0},
        "descripcion": DESCRIPCION_LR,
    }
    if C is not None:
        res["regresion_logistica"]["C"] = C
    return res


def armar_resultado(*, modelo: str, split: Split, registros: Sequence[Mapping], ctx: Mapping, configuracion: Mapping,
                    elegido: Mapping | None = None, barrido: Sequence[Mapping] | None = None,
                    advertencias: Sequence[str] = ()) -> dict:
    """JSON de results/ con los campos de la regla 4 (sin nombres, llaves ni ids)."""
    reg = elegido or registros[0]
    cpu = {
        "tiempo_ajuste_s_total": round(sum(r["cpu"]["tiempo_ajuste_s"] for r in registros), 2),
        **{k: v for k, v in reg["cpu"].items()},
    }
    res = {
        "fecha": ctx["fecha"],
        "experimento": EXPERIMENTO,
        "commit": ctx["commit"],
        "arbol_limpio": ctx["arbol_limpio"],
        "modelo": modelo,
        "configuracion": {
            **configuracion,
            "script": SCRIPT,
            "evaluador": "src/evaluador_clasificador.py",
            "python": platform.python_version(),
            "versiones": _versiones(),
        },
        "semilla": SEMILLA,
        "split": split.nombre,
        "descripcion_split": split.descripcion,
        "hash_split": dict(split.hash_split),
        "hash_split_origen": split.hash_split_origen,
        "hash_train_antes_del_tope": dict(split.hash_pre_tope),
        "tamano_muestra": {"train": len(split.train), "validacion": len(split.val)},
        "horas_gpu": 0,
        "cpu": cpu,
        "metricas": reg["metricas"],
        "detalle": reg["extra"],
        "advertencias": list(advertencias),
    }
    if barrido is not None:
        res["barrido"] = list(barrido)
    return res


def fila_barrido(reg: Mapping) -> dict:
    m = reg["metricas"]
    return {
        "C": reg["extra"]["C"],
        "aciertos": m["regla5"]["aciertos"],
        "accuracy": m["regla5"]["accuracy"],
        "ic95": m["ic95_accuracy"],
        "accuracy_sector": m["por_nivel"]["sector"]["accuracy"],
        "n_iter": reg["extra"]["n_iter"],
        "convergio": reg["extra"]["convergio"],
        "tiempo_ajuste_s": reg["cpu"]["tiempo_ajuste_s"],
        "memoria_pico_ajuste_mb": reg["cpu"]["memoria_pico_ajuste_mb"],
    }


def armar_fuga(*, regs: Mapping[str, Mapping[str, Mapping]], ctx: Mapping, splits: Mapping[str, Split],
               archivos: Mapping[str, Mapping[str, str]]) -> dict:
    """Diferencia de accuracy (aleatorio − nombre) de cada baseline, con IC 95 % por bootstrap independiente."""
    por_modelo = {}
    for modelo, por_split in regs.items():
        n, a = por_split["nombre"], por_split["aleatorio"]
        por_modelo[modelo] = {
            "accuracy_split_nombre": n["metricas"]["regla5"]["accuracy"],
            "accuracy_split_aleatorio": a["metricas"]["regla5"]["accuracy"],
            "fuga_clase": diferencia_bootstrap(n["aciertos"]["clase"], a["aciertos"]["clase"]),
            "accuracy_sector_split_nombre": n["metricas"]["por_nivel"]["sector"]["accuracy"],
            "accuracy_sector_split_aleatorio": a["metricas"]["por_nivel"]["sector"]["accuracy"],
            "fuga_sector": diferencia_bootstrap(n["aciertos"]["sector"], a["aciertos"]["sector"]),
            "resultados": dict(archivos[modelo]),
        }
    return {
        "fecha": ctx["fecha"],
        "experimento": "fase1_paso10_fuga_split_aleatorio",
        "commit": ctx["commit"],
        "arbol_limpio": ctx["arbol_limpio"],
        "modelo": [MODELO_DICCIONARIO, MODELO_TFIDF],
        "configuracion": {
            "definicion": (
                "Fuga = accuracy en la validación del split aleatorio menos accuracy en la validación del split por "
                "nombre, con la misma configuración de cada baseline. IC 95 %: bootstrap de percentiles independiente "
                "de las dos validaciones, 1,000 remuestreos de cada una con un solo numpy.random.default_rng(42) "
                "(primero los del split por nombre)."
            ),
            "script": SCRIPT,
            "evaluador": "src/evaluador_clasificador.py",
            "python": platform.python_version(),
            "versiones": _versiones(),
        },
        "semilla": SEMILLA,
        "split": "nombre y aleatorio",
        "hash_split": {s: dict(splits[s].hash_split) for s in SPLITS},
        "hash_split_origen": "data/manifest.json, splits",
        "tamano_muestra": {s: {"train": len(splits[s].train), "validacion": len(splits[s].val)} for s in SPLITS},
        "horas_gpu": 0,
        "por_modelo": por_modelo,
        "advertencias": [
            "P (de donde sale el split aleatorio) no tiene las filas que la regla 7 excluyó por llaves de la validación "
            "por nombre ni las del hard set; si acaso, la fuga medida queda un poco por debajo de la real "
            "(ver results/*_fase1_split-aleatorio.json).",
        ],
    }


def ruta_resultado(carpeta: Path, fecha: str, modelo: str, split: str) -> Path:
    return Path(carpeta) / f"{fecha}_fase1_{ARCHIVO_MODELO[modelo]}_split-{split}.json"


def resultado_existente(carpeta: Path, patron: str, commit: str) -> Path | None:
    """Un resultado previo de este paso: None si no hay; la ruta si es del mismo commit; si no, se detiene."""
    previos = sorted(Path(carpeta).glob(patron))
    if not previos:
        return None
    if len(previos) > 1:
        raise ErrorBaseline(f"Hay {len(previos)} resultados que coinciden con {patron}; me detengo.")
    if json.loads(previos[0].read_bytes()).get("commit") != commit:
        raise ErrorBaseline(f"{previos[0].name} es de otro commit; no se sobrescribe.")
    return previos[0]


ADVERTENCIA_IC = "IC 95 % por bootstrap de percentiles sobre las filas de validación (1,000 remuestreos, semilla 42)."
ADVERTENCIA_DICC_NOMBRE = (
    "En el split por nombre ninguna llave de validación está en train (regla 2): el diccionario responde la clase "
    "global en todas las filas y equivale al baseline de clase mayoritaria."
)
ADVERTENCIA_C = (
    "La accuracy del C elegido es optimista: C se eligió en esta misma validación. La medida sin ese sesgo será la del "
    "test (Fase 4)."
)
ADVERTENCIA_ALEATORIO = (
    "Split aleatorio solo para medir la fuga; nunca para elegir modelos. P no tiene las filas que la regla 7 excluyó "
    "por llaves de la validación por nombre ni las del hard set (ver results/*_fase1_split-aleatorio.json)."
)


def correr(manifest: Mapping, *, raiz: Path, carpeta_registros: Path, carpeta_resultados: Path, ctx: Mapping,
           valores_c: Sequence[float] = VALORES_C, tfidf: Mapping = CONFIG_TFIDF, lr: Mapping = CONFIG_LR,
           avance: Callable[[str], None] = print, privacidad: Callable[[Mapping, set[str]], None] | None = None) -> list[Path]:
    """Todos los ajustes del paso 10 en orden; cada JSON de results/ se escribe al terminar sus ajustes."""
    if privacidad is None:
        from src.exploracion_clasificador import verificar_privacidad

        def privacidad(obj, llaves):
            verificar_privacidad(obj, llaves, set())

    random.seed(SEMILLA)
    np.random.seed(SEMILLA)  # regla 1; los sorteos usan generadores propios con semilla 42
    splits = cargar_splits(manifest, raiz)
    codigos = cargar_codigos(manifest, raiz)
    llaves = {r["llave"] for s in splits.values() for regs in (s.train, s.val, s.pre_tope) for r in regs}
    normalizador = Normalizador()
    registros = Registros(carpeta_registros, ctx["commit"])
    escritos: list[Path] = []
    regs_fuga: dict[str, dict[str, Mapping]] = {MODELO_DICCIONARIO: {}, MODELO_TFIDF: {}}
    archivos: dict[str, dict[str, str]] = {MODELO_DICCIONARIO: {}, MODELO_TFIDF: {}}

    def publicar(resultado: Mapping, modelo: str, split: str) -> None:
        patron = f"*_fase1_{ARCHIVO_MODELO[modelo]}_split-{split}.json"
        previo = resultado_existente(carpeta_resultados, patron, ctx["commit"])
        if previo is not None:
            avance(f"{previo.name} ya existe con este commit; no se reescribe.")
            archivos[modelo][split] = previo.name
            return
        privacidad(resultado, llaves)
        ruta = ruta_resultado(carpeta_resultados, ctx["fecha"], modelo, split)
        escribir_atomico(a_bytes(resultado), ruta, temporal_en=registros.carpeta)
        archivos[modelo][split] = ruta.name
        escritos.append(ruta)
        avance(f"Escrito {ruta.name}")

    def linea(reg: Mapping, etiqueta: str) -> None:
        m = reg["metricas"]
        ic = m["ic95_accuracy"]
        avance(f"{etiqueta}: accuracy {m['regla5']['accuracy']:.4f} (IC 95 % {ic['inferior']:.4f}-{ic['superior']:.4f}), "
               f"sector {m['por_nivel']['sector']['accuracy']:.4f}, % formato válido {m['regla5']['pct_formato_valido']}")

    # 1. Diccionario en los dos splits.
    for s in SPLITS:
        split = splits[s]
        reg = registros.obtener(f"diccionario_{s}", config_diccionario(),
                                lambda split=split: correr_diccionario(split, codigos, normalizador), avance)
        linea(reg, f"Diccionario, split {s}")
        regs_fuga[MODELO_DICCIONARIO][s] = reg
        adv = [ADVERTENCIA_IC] + ([ADVERTENCIA_DICC_NOMBRE] if s == "nombre" else [ADVERTENCIA_ALEATORIO])
        publicar(armar_resultado(modelo=MODELO_DICCIONARIO, split=split, registros=[reg], ctx=ctx,
                                 configuracion=config_diccionario(), advertencias=adv), MODELO_DICCIONARIO, s)

    # 2. Barrido de TF-IDF + LR en el split por nombre (un solo vectorizador, ajustado con su train).
    vec = None
    barrido_regs = []
    for C in valores_c:
        def calcular(C=C):
            nonlocal vec
            avance(f"TF-IDF + LR, split nombre, C = {C:g}: ajustando ({time.strftime('%H:%M')})...")
            reg, vec = correr_tfidf(splits["nombre"], C, codigos, vec, tfidf, lr)
            return reg
        reg = registros.obtener(f"tfidf_nombre_C{C:g}", config_tfidf(C, tfidf, lr), calcular, avance)
        linea(reg, f"TF-IDF + LR, split nombre, C = {C:g} ({reg['cpu']['tiempo_ajuste_s']:,.0f} s)")
        barrido_regs.append(reg)
    vec = None  # libera la memoria del vectorizador del split por nombre
    barrido = [fila_barrido(r) for r in barrido_regs]
    c_elegido = elegir_c(barrido)
    elegido = next(r for r in barrido_regs if r["extra"]["C"] == c_elegido)
    avance(f"C elegido: {c_elegido:g}")
    publicar(armar_resultado(
        modelo=MODELO_TFIDF, split=splits["nombre"], registros=barrido_regs, ctx=ctx,
        configuracion={**config_tfidf(c_elegido, tfidf, lr), "valores_c": list(valores_c),
                       "decision_valores_c": DECISION_VALORES_C, "regla_eleccion_c": REGLA_C,
                       "c_elegido": c_elegido},
        elegido=elegido, barrido=barrido, advertencias=[ADVERTENCIA_IC, ADVERTENCIA_C],
    ), MODELO_TFIDF, "nombre")
    regs_fuga[MODELO_TFIDF]["nombre"] = elegido

    # 3. TF-IDF + LR en el split aleatorio, con el C elegido.
    def calcular_aleatorio():
        avance(f"TF-IDF + LR, split aleatorio, C = {c_elegido:g}: ajustando ({time.strftime('%H:%M')})...")
        return correr_tfidf(splits["aleatorio"], c_elegido, codigos, None, tfidf, lr)[0]

    reg = registros.obtener(f"tfidf_aleatorio_C{c_elegido:g}", config_tfidf(c_elegido, tfidf, lr), calcular_aleatorio, avance)
    linea(reg, f"TF-IDF + LR, split aleatorio, C = {c_elegido:g}")
    regs_fuga[MODELO_TFIDF]["aleatorio"] = reg
    publicar(armar_resultado(
        modelo=MODELO_TFIDF, split=splits["aleatorio"], registros=[reg], ctx=ctx,
        configuracion={**config_tfidf(c_elegido, tfidf, lr), "c_elegido_en": "split por nombre (mismo paso)",
                       "decision_valores_c": DECISION_VALORES_C},
        advertencias=[ADVERTENCIA_IC, ADVERTENCIA_ALEATORIO],
    ), MODELO_TFIDF, "aleatorio")

    # 4. Fuga.
    fuga = armar_fuga(regs=regs_fuga, ctx=ctx, splits=splits, archivos=archivos)
    previo = resultado_existente(carpeta_resultados, "*_fase1_fuga-split-aleatorio.json", ctx["commit"])
    if previo is None:
        privacidad(fuga, llaves)
        ruta = Path(carpeta_resultados) / f"{ctx['fecha']}_fase1_fuga-split-aleatorio.json"
        escribir_atomico(a_bytes(fuga), ruta, temporal_en=registros.carpeta)
        escritos.append(ruta)
        avance(f"Escrito {ruta.name}")
    for modelo, d in fuga["por_modelo"].items():
        f = d["fuga_clase"]
        avance(f"Fuga {modelo}: {f['diferencia']:+.4f} (IC 95 % {f['ic95']['inferior']:+.4f} a {f['ic95']['superior']:+.4f})")
    return escritos


# --- Programa principal ---


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True).stdout.strip()


def _imprimir_punto_de_control(filas: Sequence[Mapping], est: Mapping, filas_full: int, clases_full: int) -> None:
    print("Submuestras de train_final (prefijos de default_rng(42).permutation), una fila por C:")
    print(f"{'filas':>8}{'clases':>8}{'rasgos':>9}{'C':>6}{'vect. s':>9}{'ajuste s':>10}{'iter':>6}{'conv.':>7}"
          f"{'pico MB':>9}{'fórmula MB':>12}")
    for f in filas:
        print(f"{f['filas']:>8,}{f['clases']:>8}{f['rasgos']:>9,}{f['C']:>6g}{f['tiempo_vectorizador_s']:>9.1f}"
              f"{f['tiempo_ajuste_s']:>10.1f}{f['n_iter']:>6}{str(f['convergio']):>7}{f['memoria_pico_ajuste_mb']:>9,.0f}"
              f"{f['memoria_formula_mb']:>12,.0f}")
    print(f"Extrapolación a {filas_full:,} filas y {clases_full} clases (escala filas × clases = "
          f"{est['escala_filas_por_clases']}, desde la submuestra de {est['submuestra_de_referencia']:,}):")
    for c in est["por_c"]:
        print(f"  C = {c['C']:g}: lineal {c['lineal_s'] / 60:,.0f} min (iteraciones observadas), "
              f"pesimista {c['pesimista_s'] / 60:,.0f} min (max_iter = {CONFIG_LR['max_iter']})")
    print(f"  vectorizador: {est['vectorizador_s'] / 60:,.1f} min por split")
    h = est["total_horas"]
    print(f"  TOTAL ({len(est['por_c'])} ajustes del barrido + 1 del split aleatorio, con el C más lento, + 2 vectorizadores): "
          f"lineal {h['lineal_s']:.2f} h, pesimista {h['pesimista_s']:.2f} h")
    m = est["memoria"]
    print(f"  Memoria: fórmula a escala completa {m['formula_full_mb']:,.0f} MB × razón medida/fórmula "
          f"{m['razon_medida_contra_formula']} = {m['ajuste_estimado_mb']:,.0f} MB del ajuste, más "
          f"{m['datos_en_memoria_mb'] or 0:,.0f} MB de datos cargados en --escribir: {m['total_estimado_gb']:.2f} GB")
    print(f"  ¿Pasa de {LIMITE_HORAS} h? lineal: {est['pasa_limite_lineal_horas']}, pesimista: {est['pasa_limite_horas']}"
          f" | ¿Pasa de {LIMITE_GB} GB? {est['pasa_limite_memoria']}")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    from src.muestra_humo import ErrorMuestra, codigo_versionado
    from src.scian import leer_manifest
    from src.splits import esta_ignorado

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--punto-de-control", action="store_true", help="mide tiempo y memoria en submuestras; no escribe")
    modo.add_argument("--escribir", action="store_true", help="ajustes completos y results/")
    args = parser.parse_args(argv)

    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    versionado = codigo_versionado(SCRIPT)
    manifest = leer_manifest()
    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | {SCRIPT} versionado sin cambios: {versionado} | "
          + ", ".join(f"{k} {v}" for k, v in _versiones().items()))

    if args.punto_de_control:
        try:
            t0 = time.perf_counter()
            nombre = cargar_split_nombre(manifest, REPO_RAIZ)
            seg_carga = time.perf_counter() - t0
        except (ErrorMuestra, ErrorBaseline) as e:
            sys.exit(f"Me detengo: {e}")
        datos_mb, _ = _memoria_proceso()
        filas_full = len(nombre.train)
        clases_full = len({r["codigo_act"] for r in nombre.train})
        print(f"Cargados train_final, train candidato sin hard y validación (SHA-256 verificados) en {seg_carga:.0f} s; "
              f"memoria del proceso {datos_mb} MB. Test y hard no se abren.")
        filas = medir_submuestras(nombre.train, avance=print)
        # En --escribir se cargan además los dos archivos del split aleatorio (~ +45 % de filas).
        est = estimar(filas, filas_full=filas_full, clases_full=clases_full,
                      memoria_datos_mb=round(datos_mb * 1.45, 0) if datos_mb else None)
        _imprimir_punto_de_control(filas, est, filas_full, clases_full)
        _, pico = _memoria_proceso()
        print(f"Pico de memoria del proceso en este punto de control: {pico} MB")
        print("PUNTO DE CONTROL (PARADA 1): no se escribió nada.")
        return

    if not (arbol_limpio and versionado):
        sys.exit(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
    faltan = [e for e in (ENTRADA_A_TRAIN, ENTRADA_A_VAL) if e not in manifest["splits"]]
    if faltan:
        sys.exit(f"Falta el split aleatorio en el manifest ({faltan}); primero src/split_aleatorio.py --escribir.")
    carpeta_registros = REPO_RAIZ / "data" / "baselines"
    if not esta_ignorado(carpeta_registros / "prueba.json"):
        sys.exit("data/baselines/ no está ignorado por git.")
    ctx = {"fecha": datetime.date.today().isoformat(), "commit": commit, "arbol_limpio": arbol_limpio}
    print(f"Inicio: {time.strftime('%Y-%m-%d %H:%M')}")
    try:
        escritos = correr(manifest, raiz=REPO_RAIZ, carpeta_registros=carpeta_registros,
                          carpeta_resultados=REPO_RAIZ / "results", ctx=ctx)
    except (ErrorMuestra, ErrorBaseline, ValueError) as e:
        sys.exit(f"Me detengo: {e}")
    print(f"Fin: {time.strftime('%Y-%m-%d %H:%M')}. Escritos en results/: {[p.name for p in escritos]}")


if __name__ == "__main__":
    main()
