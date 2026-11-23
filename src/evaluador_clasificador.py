"""Evaluador único de validación del clasificador (Fase 1, paso 10).

Lo usan los dos baselines y los modelos Qwen. Se apoya en `src/metricas_clasificador.py`
(regla 5) y agrega:

- accuracy por nivel del SCIAN: sector oficial (31-33 y 48-49 agrupados; 43 y 46 separados),
  subsector (3 dígitos), rama (4), subrama (5) y clase (6). Una respuesta con formato inválido
  o con código inexistente cuenta como error en todos los niveles;
- macro-F1 de sector: promedio simple del F1 de cada sector presente en la referencia;
- accuracy por subgrupo de la regla 8 (con y sin "SIN NOMBRE"; genéricas y no genéricas). Las
  marcas se leen de las filas de referencia por id y nunca son entrada de un modelo;
- intervalo de confianza al 95 % por bootstrap (1,000 remuestreos, semilla 42) de cada accuracy;
- diferencia de accuracy entre dos modelos: independiente (dos conjuntos distintos, como en la
  fuga) o pareada (los dos modelos sobre las mismas filas; Fase 2, paso 11).

Solo devuelve conteos, porcentajes y códigos SCIAN: nunca nombres, llaves, ids ni respuestas
crudas.

Uso (solo imprime; no escribe nada):
  .venv/Scripts/python -m src.evaluador_clasificador --predicciones <ruta de predicciones.jsonl>
Evalúa contra la validación congelada (data/splits/val.jsonl, SHA-256 verificado). Nunca abre
test.jsonl ni hard.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import numpy as np

from src.denue import REPO_RAIZ
from src.exploracion_clasificador import codigo_en_nivel, mapa_sectores
from src.metricas_clasificador import (
    CODIGO_EXISTENTE,
    ErrorMetricas,
    categoria,
    evaluar,
    leer_predicciones,
    limpiar,
)
from src.metricas_clasificador import DEFINICION as DEFINICION_REGLA5
from src.prompt_clasificador import RE_CODIGO

SEMILLA = 42
REMUESTREOS = 1000
PERCENTILES = (2.5, 97.5)
DECIMALES = 6

# Sectores de la estructura oficial del SCIAN 2023 (los 20 de por_nivel.sector en
# results/2026-11-20_fase1_exploracion-clasificador.json, extraídos del PDF del INEGI).
SECTORES_2023 = (
    "11", "21", "22", "23", "31-33", "43", "46", "48-49", "51", "52",
    "53", "54", "55", "56", "61", "62", "71", "72", "81", "93",
)
SECTOR_DE_PREFIJO = mapa_sectores(SECTORES_2023)
NIVELES = (("sector", 2), ("subsector", 3), ("rama", 4), ("subrama", 5), ("clase", 6))
MARCAS_REPORTADAS = ("sin_nombre", "generico")  # regla 8: se reportan aparte
ENTRADA_VAL = "clasificador_validacion"
RE_ID = re.compile(r"[0-9]+")

DEFINICIONES = {
    "regla5": DEFINICION_REGLA5,
    "por_nivel": (
        "Acierto en un nivel: la respuesta es código existente (regla 5) y cae en la misma categoría de ese nivel "
        "que la referencia. Sector oficial del SCIAN 2023 (31-33 y 48-49 agrupados; 43 y 46 separados), subsector "
        "(3 dígitos), rama (4), subrama (5) y clase (6). Formato inválido y código inexistente cuentan como error en "
        "todos los niveles. Accuracy sobre el total de respuestas."
    ),
    "macro_f1_sector": (
        "Promedio simple, sobre los sectores presentes en la referencia, de F1 = 2·VP / (2·VP + FP + FN); 0 si el "
        "sector no tiene ningún acierto. Una respuesta inválida o inexistente es falso negativo de su sector "
        "verdadero y no es falso positivo de ningún sector; un sector predicho que no está en la referencia no "
        "entra al promedio."
    ),
    "subgrupos": (
        "Accuracy de las filas con y sin cada marca de src/subgrupos.py (sin_nombre y generico), leídas de las filas "
        "de referencia por id; las marcas nunca son entrada de un modelo."
    ),
    "ic95": (
        "Bootstrap de percentiles: 1,000 remuestreos con reemplazo de las filas (en orden de int(id)), del mismo "
        "tamaño, con numpy.random.default_rng(42) nuevo en cada intervalo; límites = percentiles 2.5 y 97.5 de las "
        "accuracies remuestreadas (numpy.percentile, interpolación lineal)."
    ),
    "concentracion": "Clases distintas entre las respuestas de código existente y el código válido más predicho.",
}
# Aparte de DEFINICIONES para que la salida de evaluar_validacion siga igual a la de los baselines.
DEFINICION_PAREADA = (
    "Accuracy de b menos la de a sobre las mismas filas (en orden de int(id)). IC 95 %: bootstrap de percentiles "
    "pareado, 1,000 remuestreos con reemplazo de las posiciones de las filas, las mismas para a y b, con un solo "
    "numpy.random.default_rng(42); límites = percentiles 2.5 y 97.5 de las diferencias remuestreadas."
)


class ErrorEvaluador(ValueError):
    """Entrada inválida para el evaluador; el mensaje nunca incluye nombres, ids ni respuestas."""


# --- Niveles del SCIAN ---


def verificar_sectores(
    codigos: Collection[str], sectores: Sequence[str] = SECTORES_2023, exigir_todos: bool = False
) -> None:
    """Cada código cae en exactamente un sector; con `exigir_todos`, además cada sector tiene alguna clase."""
    mapa = mapa_sectores(sectores)
    if len(mapa) != sum(len(range(int(s[:2]), int(s[-2:]) + 1)) for s in sectores):
        raise ErrorEvaluador("Dos sectores comparten un prefijo.")
    sin_sector = sum(1 for c in codigos if c[:2] not in mapa)
    if sin_sector:
        raise ErrorEvaluador(f"{sin_sector} clases del catálogo no caen en ningún sector.")
    if not exigir_todos:
        return
    usados = {mapa[c[:2]] for c in codigos}
    vacios = [s for s in sectores if s not in usados]
    if vacios:
        raise ErrorEvaluador(f"Sectores sin clases en el catálogo: {vacios}.")


def en_nivel(codigo: str, n: int) -> str:
    """Categoría de nivel n (2 = sector oficial; 3 a 6 = prefijo) que contiene a una clase."""
    return codigo_en_nivel(codigo, n, SECTOR_DE_PREFIJO)


# --- Medidas ---


def _ratio(n: int, total: int) -> float | None:
    return round(n / total, DECIMALES) if total else None


def intervalo_bootstrap(aciertos: Sequence[bool], remuestreos: int = REMUESTREOS, semilla: int = SEMILLA) -> dict | None:
    """IC 95 % de percentiles de la accuracy; None si no hay filas."""
    x = np.asarray(aciertos, dtype=np.float64)
    if x.size == 0:
        return None
    rng = np.random.default_rng(semilla)
    medias = x[rng.integers(0, x.size, size=(remuestreos, x.size))].mean(axis=1)
    inf, sup = np.percentile(medias, PERCENTILES)
    return {"inferior": round(float(inf), DECIMALES), "superior": round(float(sup), DECIMALES)}


def diferencia_bootstrap(
    aciertos_a: Sequence[bool], aciertos_b: Sequence[bool], remuestreos: int = REMUESTREOS, semilla: int = SEMILLA
) -> dict:
    """Accuracy de b menos la de a, con IC 95 % por bootstrap independiente de los dos conjuntos.

    Un solo generador default_rng(semilla): primero los remuestreos de a y luego los de b.
    """
    a = np.asarray(aciertos_a, dtype=np.float64)
    b = np.asarray(aciertos_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ErrorEvaluador("No se puede comparar un conjunto vacío.")
    rng = np.random.default_rng(semilla)
    ma = a[rng.integers(0, a.size, size=(remuestreos, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(remuestreos, b.size))].mean(axis=1)
    inf, sup = np.percentile(mb - ma, PERCENTILES)
    return {
        "diferencia": round(float(b.mean() - a.mean()), DECIMALES),
        "ic95": {"inferior": round(float(inf), DECIMALES), "superior": round(float(sup), DECIMALES)},
        "n_a": int(a.size),
        "n_b": int(b.size),
    }


def diferencia_pareada_bootstrap(
    aciertos_a: Sequence[bool], aciertos_b: Sequence[bool], remuestreos: int = REMUESTREOS, semilla: int = SEMILLA
) -> dict:
    """Accuracy de b menos la de a sobre las mismas filas, con IC 95 % por bootstrap pareado.

    Las dos listas deben venir en el mismo orden de filas (el de int(id)). Cada remuestreo toma
    las mismas posiciones para a y para b, con un solo default_rng(semilla).
    """
    a = np.asarray(aciertos_a, dtype=np.float64)
    b = np.asarray(aciertos_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ErrorEvaluador("Los aciertos pareados deben tener el mismo largo.")
    if a.size == 0:
        raise ErrorEvaluador("No se puede comparar un conjunto vacío.")
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, a.size, size=(remuestreos, a.size))
    inf, sup = np.percentile(b[idx].mean(axis=1) - a[idx].mean(axis=1), PERCENTILES)
    return {
        "diferencia": round(float(b.mean() - a.mean()), DECIMALES),
        "ic95": {"inferior": round(float(inf), DECIMALES), "superior": round(float(sup), DECIMALES)},
        "n": int(a.size),
        "filas_solo_a_acierta": int(((a == 1) & (b == 0)).sum()),
        "filas_solo_b_acierta": int(((a == 0) & (b == 1)).sum()),
    }


def macro_f1(verdaderos: Sequence[str], predichos: Sequence[str | None]) -> dict:
    """Macro-F1 sobre las etiquetas de la referencia; `None` en predichos = ninguna etiqueta."""
    if len(verdaderos) != len(predichos):
        raise ErrorEvaluador("Verdaderos y predichos no tienen el mismo largo.")
    if not verdaderos:
        raise ErrorEvaluador("No hay filas.")
    vp, fp, fn = Counter(), Counter(), Counter()
    for v, p in zip(verdaderos, predichos):
        if p == v:
            vp[v] += 1
        else:
            fn[v] += 1
            if p is not None:
                fp[p] += 1
    etiquetas = sorted(set(verdaderos))
    f1 = {e: 2 * vp[e] / (2 * vp[e] + fp[e] + fn[e]) for e in etiquetas}  # el denominador es > 0
    return {
        "valor": round(sum(f1.values()) / len(f1), DECIMALES),
        "etiquetas": len(etiquetas),
        "por_etiqueta": {e: round(f, DECIMALES) for e, f in f1.items()},
    }


def _bloque(aciertos: Sequence[bool]) -> dict:
    n = len(aciertos)
    k = int(sum(aciertos))
    return {"filas": n, "aciertos": k, "accuracy": _ratio(k, n), "ic95": intervalo_bootstrap(aciertos)}


def _mas_frecuente(valores) -> tuple[str | None, int]:
    """El valor más frecuente y su conteo; en empate, el menor."""
    conteo = Counter(valores)
    if not conteo:
        return None, 0
    return min(conteo.items(), key=lambda kv: (-kv[1], kv[0]))


# --- Validación de entradas ---


def validar_referencias(filas: Sequence[Mapping]) -> list[Mapping]:
    """Las filas de referencia en orden de int(id); cada una con id, codigo_act y las marcas en bool."""
    if not filas:
        raise ErrorEvaluador("No hay filas de referencia.")
    for f in filas:
        if not isinstance(f.get("id"), str) or not RE_ID.fullmatch(f["id"]):
            raise ErrorEvaluador("Un id de referencia no es solo dígitos ASCII.")
        if not isinstance(f.get("codigo_act"), str) or not RE_CODIGO.fullmatch(f["codigo_act"]):
            raise ErrorEvaluador("Un codigo_act de referencia no es de seis dígitos ASCII.")
        if any(not isinstance(f.get(m), bool) for m in MARCAS_REPORTADAS):
            raise ErrorEvaluador("Una fila de referencia no trae las marcas sin_nombre y generico como bool.")
    ordenadas = sorted(filas, key=lambda f: int(f["id"]))
    ids = [f["id"] for f in ordenadas]
    if len(set(ids)) != len(ids):
        raise ErrorEvaluador("Hay id repetidos en la referencia.")
    return ordenadas


# --- Evaluación completa ---


def aciertos_por_nivel(
    predicciones: Mapping[str, str], filas: Sequence[Mapping], codigos: Collection[str]
) -> dict[str, list[bool]]:
    """Acierto de cada fila (en el orden de `filas`) en cada nivel."""
    res: dict[str, list[bool]] = {nombre: [] for nombre, _ in NIVELES}
    for f in filas:
        respuesta = predicciones[f["id"]]
        existe = categoria(respuesta, codigos) == CODIGO_EXISTENTE
        limpia = limpiar(respuesta)
        for nombre, n in NIVELES:
            res[nombre].append(existe and en_nivel(limpia, n) == en_nivel(f["codigo_act"], n))
    return res


def evaluar_validacion(
    predicciones: Mapping[str, str], filas: Sequence[Mapping], codigos: Collection[str]
) -> dict:
    """Todas las métricas del evaluador para {id: respuesta cruda} contra las filas de referencia."""
    filas = validar_referencias(filas)
    codigos = frozenset(codigos)
    verificar_sectores(codigos)
    try:
        regla5 = evaluar(predicciones, {f["id"]: f["codigo_act"] for f in filas}, codigos)
    except ErrorMetricas as e:
        raise ErrorEvaluador(str(e)) from e

    por_nivel = aciertos_por_nivel(predicciones, filas, codigos)
    if sum(por_nivel["clase"]) != regla5["aciertos"]:  # no debería pasar: misma definición
        raise ErrorEvaluador("La accuracy de clase no coincide con la de la regla 5.")
    for (a, _), (b, _) in zip(NIVELES, NIVELES[1:]):
        if any(fino and not grueso for grueso, fino in zip(por_nivel[a], por_nivel[b])):
            raise ErrorEvaluador(f"Un acierto en {b} no es acierto en {a}.")

    sector_pred = []
    for f in filas:
        respuesta = predicciones[f["id"]]
        existe = categoria(respuesta, codigos) == CODIGO_EXISTENTE
        sector_pred.append(en_nivel(limpiar(respuesta), 2) if existe else None)
    f1 = macro_f1([en_nivel(f["codigo_act"], 2) for f in filas], sector_pred)

    clase = por_nivel["clase"]
    subgrupos = {}
    for m in MARCAS_REPORTADAS:
        subgrupos[m] = {
            "con": _bloque([a for a, f in zip(clase, filas) if f[m]]),
            "sin": _bloque([a for a, f in zip(clase, filas) if not f[m]]),
        }

    limpias = [limpiar(predicciones[f["id"]]) for f in filas]
    codigo, veces = _mas_frecuente(c for c in limpias if RE_CODIGO.fullmatch(c))
    return {
        "regla5": regla5,
        "ic95_accuracy": intervalo_bootstrap(clase),
        "por_nivel": {nombre: _bloque(por_nivel[nombre]) for nombre, _ in NIVELES},
        "macro_f1_sector": {
            "valor": f1["valor"],
            "sectores_en_la_referencia": f1["etiquetas"],
            "por_sector": f1["por_etiqueta"],
        },
        "subgrupos": subgrupos,
        "concentracion": {
            "clases_existentes_distintas_predichas": len({c for c in limpias if c in codigos}),
            "codigo_valido_mas_predicho": codigo,
            "filas_con_ese_codigo": veces,
            "pct_filas_con_ese_codigo": round(100 * veces / len(filas), 2),
        },
        "clases_en_la_referencia": len({f["codigo_act"] for f in filas}),
        "definiciones": DEFINICIONES,
        "implementacion": "src/evaluador_clasificador.py (regla 5: src/metricas_clasificador.py)",
        "semilla_bootstrap": SEMILLA,
        "remuestreos_bootstrap": REMUESTREOS,
    }


# --- Entradas verificadas ---


def cargar_codigos(manifest: Mapping, raiz: Path = REPO_RAIZ) -> frozenset[str]:
    """Códigos del catálogo de la versión fijada, con su SHA-256 verificado y el candado de sectores."""
    from src.muestra_humo import leer_verificado
    from src.scian import CATALOGO, leer_catalogo

    entrada = manifest["catalogos"][CATALOGO[manifest["scian_version"]]]
    ruta = Path(raiz) / entrada["ruta"]
    leer_verificado(ruta, entrada["sha256"])
    codigos = frozenset(leer_catalogo(ruta))
    verificar_sectores(codigos, exigir_todos=True)
    return codigos


def cargar_referencias(manifest: Mapping, entrada: str = ENTRADA_VAL, raiz: Path = REPO_RAIZ) -> list[dict]:
    """Filas de un split del manifest (por omisión, la validación), con su SHA-256 verificado.

    Se niega con test y hard: esos conjuntos solo se evalúan en la Fase 4 (regla 3).
    """
    from src.muestra_humo import leer_verificado, parsear

    if entrada in ("clasificador_test", "clasificador_hard"):
        raise ErrorEvaluador("Test y hard no se evalúan antes de la Fase 4 (regla 3).")
    e = manifest["splits"][entrada]
    regs = parsear(leer_verificado(Path(raiz) / e["ruta"], e["sha256"]))
    if len(regs) != e["filas"]:
        raise ErrorEvaluador(f"Las filas de {entrada} no son las del manifest.")
    return regs


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.muestra_humo import ErrorMuestra
    from src.scian import leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predicciones", required=True, help="JSONL con id y respuesta (src/metricas_clasificador.py)")
    args = parser.parse_args(argv)
    manifest = leer_manifest()
    try:
        filas = cargar_referencias(manifest)
        codigos = cargar_codigos(manifest)
        predicciones = leer_predicciones(Path(args.predicciones).read_bytes())
        res = evaluar_validacion(predicciones, filas, codigos)
    except (ErrorEvaluador, ErrorMetricas, ErrorMuestra) as e:
        sys.exit(f"Me detengo: {e}")
    res = {k: v for k, v in res.items() if k != "definiciones"}
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
