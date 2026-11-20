"""Análisis exploratorio para el clasificador (Fase 1, paso 4).

Solo mide y propone: no filtra, no congela y no cambia CLAUDE.md.

1. Clases: códigos presentes contra el catálogo SCIAN 2023, filas por clase y por
   sector, subsector, rama y subrama.
2. Nombres problemáticos: vacíos, la frase "SIN NOMBRE" y llaves iguales a un
   título del catálogo o a `nombre_act`.
3. Grupos por llave normalizada y techo de ambigüedad en los 5 niveles del SCIAN.
4. Estimación de la regla 7 (filas de train excluidas por llave compartida).
5. Topes por clase en train.
6. Efectos medidos de las opciones de decisión.

Escribe un JSON en results/ (solo conteos, códigos SCIAN y títulos del catálogo),
cuatro figuras en docs/figuras/ y un archivo local de revisión en data/ (ignorado
por git). En la terminal imprime solo conteos, porcentajes y códigos.

Uso:
  .venv/Scripts/python -m src.exploracion_clasificador
  .venv/Scripts/python -m src.exploracion_clasificador --figuras results/<archivo>.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import re
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import version as version_de_paquete
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from src.denue import REPO_RAIZ, cargar_denue, entrada_manifest
from src.grupos_nombres import grupos_por_tamano
from src.normalizacion import normalizar_nombre
from src.scian import (
    CANDADO,
    CATALOGO,
    FUENTE_ESTRUCTURA,
    archivo_verificado,
    extraer_estructura,
    leer_catalogo,
    leer_manifest,
    nivel,
    texto_de_paginas,
)

COLUMNAS = ["nom_estab", "codigo_act", "nombre_act"]
VERSION = "2023"
SEMILLA = 42
NOTA_SEMILLA = (
    "La semilla 42 no se usa en este paso: las simulaciones usan las semillas 1000 a 1099, "
    "como pidió Samuel, para no adelantar el split real."
)
SEMILLAS_SIMULACION = tuple(range(1000, 1100))
MUESTRA_PRINCIPAL = 3000  # validación y test, en total
MUESTRAS = (3000, 6000)  # 6,000 = 3,000 en cada conjunto (extra)
UMBRALES_CLASES = (10, 50, 100)
TOP_CLASES = (10, 50, 100)
TOPES = (100, 200, 500, 1000)
NIVELES = {6: "clase", 5: "subrama", 4: "rama", 3: "subsector", 2: "sector"}
GRUPO_GRANDE = 100
TOP_REVISION = 200
TRAMOS_PUREZA = (("menos_de_0.5", 0.0, 0.5), ("0.5_a_0.7", 0.5, 0.7), ("0.7_a_0.9", 0.7, 0.9), ("0.9_a_menos_de_1", 0.9, 1.0))
TRAMOS_CLASES_POR_LLAVE = ("1", "2", "3", "4", "5", "6-9", "10+")
TRAMOS_TAMANO = (("2-9", 2, 9), ("10-99", 10, 99), ("100+", 100, None))
CARPETA_FIGURAS = REPO_RAIZ / "docs" / "figuras"
RESULTADO_PASO3 = REPO_RAIZ / "results" / "2026-11-20_fase1_normalizacion-nombres.json"

# Frase con la que el DENUE marca a los negocios sin nombre comercial. No es ninguna
# llave (ninguna llave es solo la frase), así que puede ir en el código y en results/.
MARCADOR = "SIN NOMBRE"
RE_MARCADOR = re.compile(r"\bSIN NOMBRE\b")
RE_PALABRA_TRAS_SIN = re.compile(r"\bSIN (?=(\w+))")  # con lookahead para no saltarse casos

DEFINICIONES = {
    "marcador": 'La frase "SIN NOMBRE" como palabras completas en cualquier parte de la llave.',
    "errata_marcador": (
        'Llave sin la frase "SIN NOMBRE" con "SIN" seguida de una palabra a distancia de edición 1 '
        'de "NOMBRE"; se cuenta aparte y no entra en el marcador.'
    ),
    "generico": (
        'Llave no vacía cuyas palabras, sin la frase "SIN NOMBRE", aparecen todas en el vocabulario '
        "de los títulos normalizados del SCIAN 2023 de todos los niveles (se acepta la palabra con o "
        "sin S o ES final), con al menos una palabra de 4 letras o más."
    ),
    "techo": (
        "Suma, sobre las llaves no vacías con 2 o más filas, de las filas de la clase mayoritaria de "
        "cada llave, entre el total de filas de esas llaves."
    ),
    "regla7": (
        "Validación y test se sacan al azar, fila por fila, entre las filas con nombre; se excluyen de "
        "train las demás filas de toda llave tocada. Excluidas = filas de las llaves tocadas - m; "
        "porcentaje sobre las N - m filas candidatas a train."
    ),
    "nivel_2": "Sector oficial del SCIAN: 31-33 y 48-49 cuentan como un sector cada uno.",
    "cuartiles": "numpy.percentile con interpolación lineal, sobre las clases con al menos una fila.",
}

COLUMNAS_REVISION = (
    "rango",
    "llave",
    "filas",
    "clases_distintas",
    "clase_mayoritaria",
    "titulo_clase_mayoritaria",
    "proporcion_mayoritaria",
    "marcador",
    "generico_vocabulario",
    "filas_igual_titulo_su_clase",
    "filas_igual_titulo_otra_clase",
    "filas_igual_nombre_act",
)


class ErrorPrivacidad(ValueError):
    """Un texto del resultado es igual a una llave normalizada del DENUE."""


def _git(*args: str) -> str:
    resultado = subprocess.run(
        ["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True
    )
    return resultado.stdout.strip()


def _pct(parte: float, total: float) -> float | None:
    return round(100 * parte / total, 2) if total else None


# --- Niveles del SCIAN ---


def mapa_sectores(sectores: Iterable[str]) -> dict[str, str]:
    """Prefijo de 2 dígitos -> código del sector oficial ("31" -> "31-33")."""
    mapa = {}
    for s in sectores:
        inicio, _, fin = s.partition("-")
        for p in range(int(inicio), int(fin or inicio) + 1):
            mapa[f"{p:02d}"] = s
    return mapa


def codigo_en_nivel(codigo: str, n: int, sector_de_prefijo: Mapping[str, str]) -> str:
    """Código de la categoría de nivel n (2 a 6 dígitos) que contiene a una clase."""
    return sector_de_prefijo[codigo[:2]] if n == 2 else codigo[:n]


# --- Clases ---


def resumen_clases(filas_por_clase: Mapping[str, int], catalogo: Iterable[str]) -> dict:
    """Clases presentes contra el catálogo, cuartiles, umbrales y cobertura."""
    cat = set(catalogo)
    presentes = {c: f for c, f in filas_por_clase.items() if f > 0}
    valores = np.array(sorted(presentes.values()))
    total = int(valores.sum())
    orden = sorted(presentes.values(), reverse=True)
    q1, q2, q3 = (round(float(q), 2) for q in np.percentile(valores, [25, 50, 75]))
    return {
        "clases_catalogo": len(cat),
        "clases_presentes": len(presentes),
        "clases_del_catalogo_sin_filas": len(cat - presentes.keys()),
        "clases_presentes_fuera_del_catalogo": sorted(presentes.keys() - cat),
        "filas": total,
        "filas_por_clase_presente": {
            "minimo": int(valores.min()),
            "q1": q1,
            "mediana": q2,
            "q3": q3,
            "maximo": int(valores.max()),
            "media": round(float(valores.mean()), 2),
        },
        "clases_con_menos_de": {str(u): int((valores < u).sum()) for u in UMBRALES_CLASES},
        "cobertura_de_las_clases_mas_grandes": {
            str(k): {"filas": int(sum(orden[:k])), "pct_filas": _pct(sum(orden[:k]), total)}
            for k in TOP_CLASES
        },
    }


def por_nivel(
    filas_por_clase: Mapping[str, int],
    categorias: Mapping[str, str],
    catalogo: Iterable[str],
    n: int,
    sector_de_prefijo: Mapping[str, str],
    con_titulo: bool,
) -> dict:
    """Clases del catálogo, clases presentes y filas de cada categoría de nivel n."""
    detalle: dict[str, dict] = {}
    for c in sorted(categorias):
        detalle[c] = {"titulo": categorias[c]} if con_titulo else {}
        detalle[c].update(clases_catalogo=0, clases_presentes=0, filas=0)
    for clase in catalogo:
        detalle[codigo_en_nivel(clase, n, sector_de_prefijo)]["clases_catalogo"] += 1
    for clase, f in filas_por_clase.items():
        d = detalle[codigo_en_nivel(clase, n, sector_de_prefijo)]
        d["clases_presentes"] += 1
        d["filas"] += int(f)
    total = sum(d["filas"] for d in detalle.values())
    for d in detalle.values():
        d["pct_filas"] = _pct(d["filas"], total)
    return {
        "categorias_catalogo": len(detalle),
        "categorias_con_filas": sum(d["filas"] > 0 for d in detalle.values()),
        "filas": total,
        "detalle": detalle,
    }


# --- Marcador, erratas, genéricos y títulos ---


def tiene_marcador(llave: str) -> bool:
    return bool(RE_MARCADOR.search(llave))


def quitar_marcador(llave: str) -> str:
    """La llave sin la frase "SIN NOMBRE" (el resto), con espacios colapsados."""
    return " ".join(RE_MARCADOR.sub(" ", llave).split())


def posicion_marcador(llave: str) -> str | None:
    """Dónde está la frase: sola, al final, al inicio o en medio (None si no está)."""
    if not tiene_marcador(llave):
        return None
    if llave == MARCADOR:
        return "sola"
    if llave.endswith(" " + MARCADOR):
        return "final"
    if llave.startswith(MARCADOR + " "):
        return "inicio"
    return "en_medio"


def a_distancia_1(a: str, b: str) -> bool:
    """True si a y b están a distancia de edición exactamente 1."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    corta, larga = (a, b) if len(a) < len(b) else (b, a)
    return any(larga[:i] + larga[i + 1 :] == corta for i in range(len(larga)))


def tiene_errata_marcador(llave: str) -> bool:
    if tiene_marcador(llave):
        return False
    return any(a_distancia_1(p, "NOMBRE") for p in RE_PALABRA_TRAS_SIN.findall(llave))


def vocabulario(titulos: Iterable[str]) -> frozenset[str]:
    """Palabras de los títulos normalizados con `normalizar_nombre()`."""
    return frozenset(p for t in titulos for p in normalizar_nombre(t).split())


def en_vocabulario(palabra: str, vocab: frozenset[str]) -> bool:
    """La palabra, o la palabra con o sin S o ES final, está en el vocabulario."""
    candidatas = {palabra, palabra + "S", palabra + "ES"}
    if palabra.endswith("S"):
        candidatas.add(palabra[:-1])
    if palabra.endswith("ES"):
        candidatas.add(palabra[:-2])
    return bool(candidatas & vocab)


def es_generico(llave: str, vocab: frozenset[str]) -> bool:
    palabras = quitar_marcador(llave).split()
    return (
        bool(palabras)
        and all(en_vocabulario(p, vocab) for p in palabras)
        and any(len(p) >= 4 for p in palabras)
    )


def clasificar_titulo(llave: str, codigo: str, codigo_por_titulo: Mapping[str, str]) -> str | None:
    """"su_clase", "otra_clase" o None, según el título del catálogo igual a la llave."""
    c = codigo_por_titulo.get(llave) if llave else None
    if c is None:
        return None
    return "su_clase" if c == codigo else "otra_clase"


# --- Grupos y techo ---


def conteos_por_llave(llaves: Iterable[str], clases: Iterable[str]) -> dict[str, Counter]:
    """llave -> Counter(clase -> filas)."""
    conteos: dict[str, Counter] = defaultdict(Counter)
    for k, c in zip(llaves, clases):
        conteos[k][c] += 1
    return dict(conteos)


def techo(conteos: Mapping[str, Mapping[str, int]], minimo_filas: int = 2) -> dict:
    """Techo de ambigüedad sobre las llaves no vacías con al menos `minimo_filas` filas."""
    aciertos = filas = llaves = 0
    for k, cnt in conteos.items():
        if k == "":
            continue
        n = sum(cnt.values())
        if n < minimo_filas:
            continue
        aciertos += max(cnt.values())
        filas += n
        llaves += 1
    return {
        "llaves": llaves,
        "filas": filas,
        "filas_clase_mayoritaria": aciertos,
        "pct": _pct(aciertos, filas),
    }


def clases_por_llave(conteos: Mapping[str, Mapping[str, int]]) -> dict:
    """Llaves y filas según el número de clases distintas (llaves no vacías de 2 o más filas)."""
    res = {t: {"llaves": 0, "filas": 0} for t in TRAMOS_CLASES_POR_LLAVE}
    for k, cnt in conteos.items():
        n = sum(cnt.values())
        if k == "" or n < 2:
            continue
        d = len(cnt)
        tramo = str(d) if d <= 5 else ("6-9" if d <= 9 else "10+")
        res[tramo]["llaves"] += 1
        res[tramo]["filas"] += n
    return res


def tramo_pureza(p: float) -> str:
    if p >= 1.0:
        return "1"
    for nombre, desde, hasta in TRAMOS_PUREZA:
        if desde <= p < hasta:
            return nombre
    raise ValueError(f"Pureza fuera de rango: {p}")


def pureza_grupos(conteos: Mapping[str, Mapping[str, int]], minimo: int = GRUPO_GRANDE) -> dict:
    """Pureza (proporción de la clase mayoritaria) de las llaves no vacías con `minimo` filas o más."""
    tramos = {t: {"grupos": 0, "filas": 0} for t in [*(n for n, _, _ in TRAMOS_PUREZA), "1"]}
    purezas = []
    for k, cnt in conteos.items():
        n = sum(cnt.values())
        if k == "" or n < minimo:
            continue
        p = max(cnt.values()) / n
        purezas.append(p)
        tramos[tramo_pureza(p)]["grupos"] += 1
        tramos[tramo_pureza(p)]["filas"] += n
    return {
        "grupos": len(purezas),
        "tramos": tramos,
        "minimo": round(min(purezas), 4) if purezas else None,
        "mediana": round(float(np.median(purezas)), 4) if purezas else None,
        "maximo": round(max(purezas), 4) if purezas else None,
    }


# --- Regla 7 ---


def prob_sin_tocar(tamanos: Sequence[int] | np.ndarray, total: int, muestra: int) -> np.ndarray:
    """P(ninguna fila del grupo en la muestra) = C(total - n, m) / C(total, m) (hipergeométrica)."""
    n = np.asarray(tamanos, dtype=float)
    libre = total - n
    p = np.zeros_like(n)
    ok = libre >= muestra
    p[ok] = np.exp(
        gammaln(libre[ok] + 1)
        - gammaln(libre[ok] - muestra + 1)
        - gammaln(total + 1)
        + gammaln(total - muestra + 1)
    )
    return p


def exclusion_esperada(tamanos: Sequence[int] | np.ndarray, muestra: int) -> np.ndarray:
    """Filas excluidas de train esperadas por grupo: n·(1 - P0) - m·n/N."""
    n = np.asarray(tamanos, dtype=float)
    total = int(n.sum())
    return n * (1 - prob_sin_tocar(n, total, muestra)) - muestra * n / total


def simular_exclusion(
    id_por_fila: np.ndarray, tamanos: np.ndarray, muestra: int, semilla: int
) -> np.ndarray:
    """Filas excluidas por grupo en una corrida (muestra sin reemplazo con la semilla dada)."""
    rng = np.random.default_rng(semilla)
    elegidas = rng.choice(len(id_por_fila), size=muestra, replace=False)
    tocadas, en_muestra = np.unique(id_por_fila[elegidas], return_counts=True)
    excluidas = np.zeros(len(tamanos), dtype=np.int64)
    excluidas[tocadas] = tamanos[tocadas] - en_muestra
    return excluidas


def _resumen_corridas(valores: Sequence[float]) -> dict:
    v = np.asarray(valores, dtype=float)
    return {
        "media": round(float(v.mean()), 2),
        "p5": round(float(np.percentile(v, 5)), 2),
        "p95": round(float(np.percentile(v, 95)), 2),
    }


def estimar_regla7(
    id_por_fila: np.ndarray,
    tamanos: np.ndarray,
    muestra: int,
    semillas: Sequence[int],
    secciones: Mapping[str, Mapping[str, np.ndarray]],
) -> dict:
    """Estimación analítica y simulada; `secciones` agrupa máscaras de grupos para el aporte."""
    total = len(id_por_fila)
    candidatas = total - muestra
    esperado = exclusion_esperada(tamanos, muestra)
    total_esperado = float(esperado.sum())
    corridas: dict[str, list[int]] = defaultdict(list)
    for s in semillas:
        exc = simular_exclusion(id_por_fila, tamanos, muestra, s)
        corridas["total"].append(int(exc.sum()))
        for seccion, mascaras in secciones.items():
            for nombre, m in mascaras.items():
                corridas[f"{seccion}/{nombre}"].append(int(exc[m].sum()))
    aporte = {
        seccion: {
            nombre: {
                "grupos": int(m.sum()),
                "filas": int(tamanos[m].sum()),
                "analitico_filas_excluidas": round(float(esperado[m].sum()), 1),
                "analitico_pct_de_lo_excluido": _pct(float(esperado[m].sum()), total_esperado),
                "simulacion_media_filas_excluidas": round(float(np.mean(corridas[f"{seccion}/{nombre}"])), 1),
            }
            for nombre, m in mascaras.items()
        }
        for seccion, mascaras in secciones.items()
    }
    pct_corridas = [100 * v / candidatas for v in corridas["total"]]
    analitico_pct = _pct(total_esperado, candidatas)
    return {
        "filas_universo": total,
        "muestra_validacion_y_test": muestra,
        "filas_candidatas_train": candidatas,
        "analitico": {"filas_excluidas": round(total_esperado, 1), "pct_filas_candidatas": analitico_pct},
        "simulacion": {
            "corridas": len(semillas),
            "semillas": f"{semillas[0]} a {semillas[-1]}",
            "filas_excluidas": _resumen_corridas(corridas["total"]),
            "pct_filas_candidatas": _resumen_corridas(pct_corridas),
        },
        "pasa_del_10_pct": bool(analitico_pct is not None and analitico_pct > 10),
        "aporte": aporte,
    }


def regla7_analitica_pct(llaves: Sequence[str], muestra: int) -> float | None:
    """Porcentaje esperado de filas de train excluidas, solo analítico."""
    tamanos = np.array(list(Counter(llaves).values()))
    if tamanos.sum() <= muestra:
        return None
    return _pct(float(exclusion_esperada(tamanos, muestra).sum()), int(tamanos.sum()) - muestra)


# --- Topes y efectos ---


def topes_por_clase(filas_por_clase: Mapping[str, int], topes: Iterable[int] = TOPES) -> dict:
    total = sum(filas_por_clase.values())
    res = {}
    for t in topes:
        quedan = sum(min(f, t) for f in filas_por_clase.values())
        res[str(t)] = {
            "filas_que_quedan": quedan,
            "pct_que_queda": _pct(quedan, total),
            "filas_recortadas": total - quedan,
            "clases_recortadas": sum(f > t for f in filas_por_clase.values()),
        }
    return res


def efecto_de_excluir(
    llaves: Sequence[str], clases: Sequence[str], mascara: Sequence[bool], muestra: int = MUESTRA_PRINCIPAL
) -> dict:
    """Tamaño de una categoría y lo que cambiaría si se excluyera del universo dado."""
    llaves, clases, mascara = list(llaves), list(clases), [bool(m) for m in mascara]
    universo = len(llaves)
    filas = sum(mascara)
    en_cat = [i for i, m in enumerate(mascara) if m]
    fuera = [i for i, m in enumerate(mascara) if not m]
    clases_universo = Counter(clases)
    clases_cat = Counter(clases[i] for i in en_cat)
    llaves_sin = [llaves[i] for i in fuera]
    clases_sin = [clases[i] for i in fuera]
    return {
        "filas_universo": universo,
        "filas": filas,
        "pct_filas": _pct(filas, universo),
        "llaves": len({llaves[i] for i in en_cat}),
        "clases_afectadas": len(clases_cat),
        "clases_que_se_quedan_sin_filas": sum(clases_universo[c] == n for c, n in clases_cat.items()),
        f"filas_esperadas_en_muestra_de_{muestra}": round(muestra * filas / universo, 1) if universo else None,
        "techo_clase_llaves_repetidas_pct": {
            "con_la_categoria": techo(conteos_por_llave(llaves, clases))["pct"],
            "sin_la_categoria": techo(conteos_por_llave(llaves_sin, clases_sin))["pct"],
        },
        f"regla7_analitico_pct_m{muestra}": {
            "con_la_categoria": regla7_analitica_pct(llaves, muestra),
            "sin_la_categoria": regla7_analitica_pct(llaves_sin, muestra),
        },
    }


# --- Archivo de revisión ---


def filas_de_revision(
    conteos: Mapping[str, Mapping[str, int]],
    catalogo: Mapping[str, str],
    marcas_por_llave: Mapping[str, Mapping[str, int]],
    vocab: frozenset[str],
    top: int = TOP_REVISION,
) -> list[dict]:
    """Las `top` llaves no vacías con más filas (empates por orden alfabético)."""
    llaves = sorted((k for k in conteos if k), key=lambda k: (-sum(conteos[k].values()), k))[:top]
    filas = []
    for i, k in enumerate(llaves, 1):
        cnt = conteos[k]
        n = sum(cnt.values())
        clase, f = max(sorted(cnt.items()), key=lambda kv: kv[1])  # empate: el código menor
        m = marcas_por_llave.get(k, {})
        filas.append(
            {
                "rango": i,
                "llave": k,
                "filas": n,
                "clases_distintas": len(cnt),
                "clase_mayoritaria": clase,
                "titulo_clase_mayoritaria": catalogo[clase],
                "proporcion_mayoritaria": f"{f / n:.4f}",
                "marcador": "SI" if tiene_marcador(k) else "NO",
                "generico_vocabulario": "SI" if es_generico(k, vocab) else "NO",
                "filas_igual_titulo_su_clase": int(m.get("su_clase", 0)),
                "filas_igual_titulo_otra_clase": int(m.get("otra_clase", 0)),
                "filas_igual_nombre_act": int(m.get("nombre_act", 0)),
            }
        )
    return filas


def escribir_revision(filas: Sequence[Mapping], ruta: str | Path) -> None:
    """CSV en UTF-8 con BOM, para que Excel muestre bien los acentos."""
    with open(ruta, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_REVISION, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


# --- Candado de privacidad ---


def textos(obj, ruta: str = "$"):
    """(ruta, texto) de cada valor de texto de un objeto JSON (no revisa las llaves de los dict)."""
    if isinstance(obj, str):
        yield ruta, obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from textos(v, f"{ruta}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from textos(v, f"{ruta}[{i}]")


def verificar_privacidad(resultado, llaves: set[str], permitidos: set[str]) -> None:
    """Falla si un texto, normalizado, es una llave del DENUE que no es título del catálogo.

    Los textos que normalizados solo tienen dígitos (códigos, fechas, versiones) se aceptan.
    El error solo da las rutas de los campos, nunca el texto.
    """
    problemas = []
    for ruta, t in textos(resultado):
        k = normalizar_nombre(t)
        if k and not k.replace(" ", "").isdigit() and k in llaves and k not in permitidos:
            problemas.append(ruta)
    if problemas:
        raise ErrorPrivacidad(f"Textos iguales a una llave del DENUE en: {problemas}")


# --- Figuras ---

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
SERIE_1 = "#2a78d6"
SERIE_2 = "#eb6834"
FUENTE_DATOS = "Fuente: INEGI, DENUE (CDMX, 05/2026) y SCIAN 2023."


def _miles(x: float) -> str:
    return f"{x:,.0f}"


def _fmt_pct(v: float) -> str:
    """Un decimal; dos si el valor es menor que 1 %, para no redondear 0.05 a 0.1."""
    return f"{v:.2f} %" if v < 1 else f"{v:.1f} %"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 24,
            "axes.titlesize": 34,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "axes.edgecolor": EJE,
            "axes.linewidth": 1.5,
            "axes.labelcolor": TINTA_2,
            "xtick.color": TINTA_2,
            "ytick.color": TINTA_2,
            "text.color": TINTA,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def _figura(plt):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    return fig


def _guardar(fig, ruta: Path) -> None:
    fig.savefig(ruta, dpi=100, metadata={"Software": None})


def _titulos(fig, titulo: str, subtitulo: str) -> None:
    fig.text(0.04, 0.95, titulo, fontsize=36, fontweight="bold", color=TINTA, va="top")
    fig.text(0.04, 0.885, subtitulo, fontsize=24, color=TINTA_2, va="top")
    fig.text(0.04, 0.025, FUENTE_DATOS, fontsize=18, color=TENUE, va="bottom")


def figura_cola_larga(resultado: Mapping, ruta: Path) -> None:
    plt = _pyplot()
    clases = resultado["clases"]
    valores = sorted(clases["filas_por_clase"].values(), reverse=True)
    x = np.arange(1, len(valores) + 1)
    fig = _figura(plt)
    _titulos(
        fig,
        "Filas por clase SCIAN: la cola larga",
        f"Clases ordenadas de mayor a menor: {clases['clases_presentes']:,} de las "
        f"{clases['clases_catalogo']:,} clases del catálogo 2023 tienen filas",
    )
    ax = fig.add_axes([0.09, 0.14, 0.86, 0.68])
    ax.plot(x, valores, color=SERIE_1, linewidth=3.5, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_xlim(0, len(valores) + 5)
    ax.set_ylim(0.8, max(valores) * 2.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _miles(v)))
    ax.grid(axis="y", color=REJILLA, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Clases ordenadas de mayor a menor número de filas")
    ax.set_ylabel("Filas (escala logarítmica)")
    for u in UMBRALES_CLASES:
        n = clases["clases_con_menos_de"][str(u)]
        ax.axhline(u, color=TENUE, linewidth=1.5)
        ax.text(len(valores), u * 1.12, f"{n:,} clases con menos de {u} filas", ha="right",
                va="bottom", fontsize=20, color=TINTA_2)
    cobertura = clases["cobertura_de_las_clases_mas_grandes"]
    for k in TOP_CLASES:
        y = valores[k - 1]
        ax.plot([k], [y], marker="o", markersize=12, color=SERIE_1, markeredgecolor=SUPERFICIE,
                markeredgewidth=2.5, zorder=3)
        ax.annotate(
            f"Las {k} mayores: {_fmt_pct(cobertura[str(k)]['pct_filas'])} de las filas",
            xy=(k, y), xytext=(k + 40, y * (2.2 if k == 10 else 1.9)), fontsize=22, color=TINTA,
            arrowprops={"arrowstyle": "-", "color": TENUE, "linewidth": 1.5},
        )
    _guardar(fig, ruta)
    plt.close(fig)


def figura_sectores(resultado: Mapping, ruta: Path) -> None:
    plt = _pyplot()
    datos = resultado["figuras_datos"]["sectores"]
    principales = datos["principales"]
    otros = datos["otros"]
    etiquetas = [textwrap.fill(f"{s['codigo']} · {s['titulo']}", 46) for s in principales]
    valores = [s["filas"] for s in principales]
    pcts = [s["pct_filas"] for s in principales]
    colores = [SERIE_1] * len(principales)
    if otros["sectores"]:
        etiquetas.append(f"Otros {otros['sectores']} sectores")
        valores.append(otros["filas"])
        pcts.append(otros["pct_filas"])
        colores.append(EJE)
    fig = _figura(plt)
    _titulos(
        fig,
        "Filas por sector SCIAN",
        f"Los {len(principales)} sectores con más filas, de {datos['sectores_con_filas']} sectores con filas "
        "(títulos del catálogo 2023)",
    )
    ax = fig.add_axes([0.40, 0.08, 0.50, 0.76])
    y = np.arange(len(valores))[::-1]
    ax.barh(y, valores, color=colores, height=0.72, edgecolor=SUPERFICIE, linewidth=2)
    ax.set_yticks(y, etiquetas, fontsize=19, color=TINTA)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(EJE)
    tope = max(valores)
    ax.set_xlim(0, tope * 1.32)
    for yi, v, p in zip(y, valores, pcts):
        ax.text(v + tope * 0.012, yi, f"{v:,} ({_fmt_pct(p)})", va="center", fontsize=20, color=TINTA_2)
    _guardar(fig, ruta)
    plt.close(fig)


def figura_techo(resultado: Mapping, ruta: Path) -> None:
    plt = _pyplot()
    techos = resultado["grupos"]["techo_por_nivel"]
    nombres = {"clase": "Clase\n(6 dígitos)", "subrama": "Subrama\n(5)", "rama": "Rama\n(4)",
               "subsector": "Subsector\n(3)", "sector": "Sector\n(2)"}
    orden = [NIVELES[n] for n in (6, 5, 4, 3, 2)]
    valores = [techos[n]["pct"] for n in orden]
    fig = _figura(plt)
    _titulos(
        fig,
        "¿Cuánto se puede acertar solo con el nombre?",
        f"Techo de ambigüedad: en los nombres que se repiten ({techos['clase']['filas']:,} filas), "
        "% de filas de la clase mayoritaria de cada nombre",
    )
    ax = fig.add_axes([0.08, 0.16, 0.88, 0.64])
    x = np.arange(len(orden))
    ax.bar(x, valores, color=SERIE_1, width=0.6, edgecolor=SUPERFICIE, linewidth=2)
    ax.set_xticks(x, [nombres[n] for n in orden], fontsize=24, color=TINTA)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(axis="y", color=REJILLA, linewidth=1)
    ax.set_axisbelow(True)
    for xi, v in zip(x, valores):
        ax.text(xi, v + 1.5, _fmt_pct(v), ha="center", va="bottom", fontsize=28, fontweight="bold",
                color=TINTA)
    _guardar(fig, ruta)
    plt.close(fig)


def figura_tamanos(resultado: Mapping, ruta: Path) -> None:
    plt = _pyplot()
    datos = resultado["grupos"]["tamanos_de_grupo"]
    tramos = list(datos["tramos"])
    llaves = [datos["tramos"][t]["pct_llaves"] for t in tramos]
    filas = [datos["tramos"][t]["pct_filas"] for t in tramos]
    nombres = {"1": "1 fila", "2-9": "2 a 9", "10-99": "10 a 99", "100+": "100 o más"}
    fig = _figura(plt)
    _titulos(
        fig,
        "Tamaño de los grupos de nombre normalizado",
        f"Grupos por nombre normalizado: {datos['llaves']:,} nombres y {datos['filas']:,} filas "
        "(sin los nombres vacíos)",
    )
    ax = fig.add_axes([0.08, 0.16, 0.88, 0.62])
    x = np.arange(len(tramos))
    ancho = 0.38
    ax.bar(x - ancho / 2, llaves, width=ancho, color=SERIE_1, edgecolor=SUPERFICIE, linewidth=2,
           label="% de los nombres")
    ax.bar(x + ancho / 2, filas, width=ancho, color=SERIE_2, edgecolor=SUPERFICIE, linewidth=2,
           label="% de las filas")
    ax.set_xticks(x, [nombres.get(t, t) for t in tramos], fontsize=24, color=TINTA)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Filas por nombre normalizado")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.grid(axis="y", color=REJILLA, linewidth=1)
    ax.set_axisbelow(True)
    for xi, a, b in zip(x, llaves, filas):
        ax.text(xi - ancho / 2, a + 1.5, _fmt_pct(a), ha="center", va="bottom", fontsize=22, color=TINTA)
        ax.text(xi + ancho / 2, b + 1.5, _fmt_pct(b), ha="center", va="bottom", fontsize=22, color=TINTA)
    ax.legend(loc="upper right", fontsize=24)
    _guardar(fig, ruta)
    plt.close(fig)


FIGURAS = {
    "cola-larga-clases": figura_cola_larga,
    "filas-por-sector": figura_sectores,
    "techo-ambiguedad": figura_techo,
    "tamanos-grupo": figura_tamanos,
}


def dibujar_figuras(resultado: Mapping, carpeta: Path = CARPETA_FIGURAS) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    for nombre, funcion in FIGURAS.items():
        ruta = carpeta / f"{resultado['fecha']}_fase1_{nombre}.png"
        funcion(resultado, ruta)
        rutas.append(ruta)
    return rutas


# --- Programa principal ---


def _top_codigos(serie_conteos, n: int = 10) -> list[dict]:
    pares = sorted(((str(c), int(f)) for c, f in serie_conteos.items()), key=lambda kv: (-kv[1], kv[0]))
    return [{"codigo": c, "filas": f} for c, f in pares[:n]]


def datos_figura_sectores(sectores: Mapping, n: int = 10) -> dict:
    con_filas = [(c, d) for c, d in sectores["detalle"].items() if d["filas"] > 0]
    orden = sorted(con_filas, key=lambda cd: (-cd[1]["filas"], cd[0]))
    principales = [
        {"codigo": c, "titulo": d["titulo"], "filas": d["filas"], "pct_filas": d["pct_filas"]}
        for c, d in orden[:n]
    ]
    resto = orden[n:]
    filas_resto = sum(d["filas"] for _, d in resto)
    return {
        "sectores_con_filas": len(con_filas),
        "principales": principales,
        "otros": {
            "sectores": len(resto),
            "filas": filas_resto,
            "pct_filas": _pct(filas_resto, sectores["filas"]),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figuras", type=Path, help="redibuja las figuras desde un JSON de results/")
    args = parser.parse_args(argv)
    if args.figuras:
        with open(args.figuras, encoding="utf-8") as f:
            resultado = json.load(f)
        for r in dibujar_figuras(resultado):
            print(f"Figura: {r.relative_to(REPO_RAIZ).as_posix()}")
        return

    import pandas as pd

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    manifest = leer_manifest()
    entrada = entrada_manifest()

    # Catálogo y estructura oficial del SCIAN 2023.
    entrada_cat = manifest["catalogos"][CATALOGO[VERSION]]
    catalogo = leer_catalogo(archivo_verificado(entrada_cat))
    entrada_pdf = manifest["fuentes"][FUENTE_ESTRUCTURA[VERSION]]
    estructura = extraer_estructura(texto_de_paginas(archivo_verificado(entrada_pdf)), CANDADO[VERSION])
    if estructura.clases != catalogo:
        sys.exit("Las clases de la estructura oficial no son idénticas a las del catálogo CSV.")
    categorias = {n: {c: t for c, t in estructura.categorias.items() if nivel(c) == n} for n in NIVELES}
    sector_de_prefijo = mapa_sectores(categorias[2])
    titulo_norm = {c: normalizar_nombre(t) for c, t in catalogo.items()}
    codigo_por_titulo = {t: c for c, t in titulo_norm.items()}
    if len(codigo_por_titulo) != len(catalogo):
        sys.exit("Hay títulos de clase repetidos tras normalizar.")
    vocab = vocabulario(estructura.categorias.values())
    permitidos = {normalizar_nombre(t) for t in estructura.categorias.values()}

    # DENUE.
    df = cargar_denue(COLUMNAS)
    llave_de = {n: normalizar_nombre(n) for n in df["nom_estab"].unique()}
    nombre_act_de = {n: normalizar_nombre(n) for n in df["nombre_act"].unique()}
    df = pd.DataFrame(
        {
            "llave": df["nom_estab"].map(llave_de),
            "codigo_act": df["codigo_act"],
            "nombre_act_norm": df["nombre_act"].map(nombre_act_de),
        }
    )
    llaves_unicas = df["llave"].unique()
    marcador_de = {k: tiene_marcador(k) for k in llaves_unicas}
    generico_de = {k: es_generico(k, vocab) for k in llaves_unicas}
    errata_de = {k: tiene_errata_marcador(k) for k in llaves_unicas}
    df["vacia"] = df["llave"] == ""
    df["marcador"] = df["llave"].map(marcador_de).astype(bool)
    df["generico"] = df["llave"].map(generico_de).astype(bool)
    df["errata"] = df["llave"].map(errata_de).astype(bool)
    codigo_titulo = df["llave"].map(codigo_por_titulo)
    df["titulo_su_clase"] = codigo_titulo.notna() & (codigo_titulo == df["codigo_act"])
    df["titulo_otra_clase"] = codigo_titulo.notna() & ~df["titulo_su_clase"]
    df["igual_nombre_act"] = ~df["vacia"] & (df["llave"] == df["nombre_act_norm"])
    con_nombre = df[~df["vacia"]]
    filas_total, filas_con_nombre = len(df), len(con_nombre)

    # 2. Clases.
    filas_por_clase = {str(c): int(f) for c, f in sorted(df["codigo_act"].value_counts().items())}
    clases = resumen_clases(filas_por_clase, catalogo)
    clases["filas_por_clase"] = filas_por_clase
    niveles = {
        NIVELES[n]: por_nivel(filas_por_clase, categorias[n], catalogo, n, sector_de_prefijo, n in (2, 3))
        for n in (2, 3, 4, 5)
    }
    for nombre, res in niveles.items():
        if res["filas"] != filas_total:
            sys.exit(f"Las filas por {nombre} no suman {filas_total}.")

    # 3. Nombres problemáticos.
    vacias = df[df["vacia"]]
    con_marcador = df[df["marcador"]]
    resto_de = {k: quitar_marcador(k) for k in con_marcador["llave"].unique()}
    resto = con_marcador["llave"].map(resto_de)
    resto_su = resto == con_marcador["codigo_act"].map(titulo_norm)
    resto_otra = resto.isin(list(codigo_por_titulo)) & ~resto_su
    posiciones = Counter(posicion_marcador(k) for k in con_marcador["llave"])
    casos = df[["titulo_su_clase", "titulo_otra_clase", "igual_nombre_act"]].sum(axis=1)
    otra = df[df["titulo_otra_clase"]]
    pares_otra = Counter(zip(otra["llave"].map(codigo_por_titulo), otra["codigo_act"]))
    nombres_problematicos = {
        "vacios": {
            "filas": len(vacias),
            "clases_distintas": int(vacias["codigo_act"].nunique()),
            "top10_clases": _top_codigos(vacias["codigo_act"].value_counts()),
        },
        "marcador": {
            "frase": MARCADOR,
            "definicion": DEFINICIONES["marcador"],
            "filas": len(con_marcador),
            "pct_filas_con_nombre": _pct(len(con_marcador), filas_con_nombre),
            "llaves": int(con_marcador["llave"].nunique()),
            "clases_distintas": int(con_marcador["codigo_act"].nunique()),
            "filas_por_posicion": {p: posiciones.get(p, 0) for p in ("sola", "final", "inicio", "en_medio")},
            "resto_sin_la_frase": {
                "restos_distintos": len(set(resto_de.values())),
                "filas_igual_titulo_su_clase": int(resto_su.sum()),
                "filas_igual_titulo_otra_clase": int(resto_otra.sum()),
                "filas_igual_nombre_act": int((resto == con_marcador["nombre_act_norm"]).sum()),
            },
            "top10_clases": _top_codigos(con_marcador["codigo_act"].value_counts()),
        },
        "errata_marcador": {
            "definicion": DEFINICIONES["errata_marcador"],
            "filas": int(df["errata"].sum()),
            "llaves": int(df.loc[df["errata"], "llave"].nunique()),
        },
        "iguales_a_un_titulo_o_a_nombre_act": {
            "titulo_su_clase": {
                "filas": int(df["titulo_su_clase"].sum()),
                "llaves": int(df.loc[df["titulo_su_clase"], "llave"].nunique()),
            },
            "titulo_otra_clase": {
                "filas": int(df["titulo_otra_clase"].sum()),
                "llaves": int(df.loc[df["titulo_otra_clase"], "llave"].nunique()),
            },
            "nombre_act": {
                "filas": int(df["igual_nombre_act"].sum()),
                "llaves": int(df.loc[df["igual_nombre_act"], "llave"].nunique()),
            },
            "en_algun_caso": {"filas": int((casos >= 1).sum()), "llaves": int(df.loc[casos >= 1, "llave"].nunique())},
            "filas_en_mas_de_un_caso": int((casos >= 2).sum()),
            "top10_pares_otra_clase": [
                {"codigo_del_titulo": a, "codigo_act": b, "filas": f}
                for (a, b), f in sorted(pares_otra.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            ],
            "nota": "Comparación con normalizar_nombre() en los dos lados; catálogo SCIAN 2023.",
        },
        "genericos": {
            "definicion": DEFINICIONES["generico"],
            "palabras_del_vocabulario": len(vocab),
            "filas": int(con_nombre["generico"].sum()),
            "llaves": int(con_nombre.loc[con_nombre["generico"], "llave"].nunique()),
            "filas_genericas_con_marcador": int((con_nombre["generico"] & con_nombre["marcador"]).sum()),
        },
    }

    # 4. Grupos y techo (sin la llave vacía).
    conteos = {
        NIVELES[n]: conteos_por_llave(
            con_nombre["llave"],
            con_nombre["codigo_act"].map(lambda c, n=n: codigo_en_nivel(c, n, sector_de_prefijo)),
        )
        for n in NIVELES
    }
    filas_por_llave = df["llave"].value_counts()
    tamanos_grupo = grupos_por_tamano({str(k): int(v) for k, v in filas_por_llave.items()})
    with open(RESULTADO_PASO3, encoding="utf-8") as f:
        paso3 = json.load(f)["variante_elegida"]["grupos_por_tamano_en_filas"]
    if tamanos_grupo["grupos"] != paso3:
        sys.exit("Los tamaños de grupo no coinciden con los del paso 3.")
    llaves_con_nombre = sum(c["grupos"] for c in tamanos_grupo["grupos"].values())
    techo_clase = techo(conteos["clase"])
    grandes = {k: v for k, v in conteos["clase"].items() if k and sum(v.values()) >= GRUPO_GRANDE}

    def por_tipo(llaves_sel: Mapping[str, Mapping[str, int]]) -> dict:
        tipos = {
            "con_marcador": lambda k: marcador_de[k],
            "genericos": lambda k: generico_de[k],
            "con_marcador_y_genericos": lambda k: marcador_de[k] and generico_de[k],
            "ni_marcador_ni_generico": lambda k: not marcador_de[k] and not generico_de[k],
        }
        return {
            t: {
                "grupos": sum(1 for k in llaves_sel if f(k)),
                "filas": sum(sum(v.values()) for k, v in llaves_sel.items() if f(k)),
            }
            for t, f in tipos.items()
        }

    grupos = {
        "grupos_por_tamano": tamanos_grupo,
        "tamanos_de_grupo": {
            "llaves": llaves_con_nombre,
            "filas": filas_con_nombre,
            "tramos": {
                t: {
                    **c,
                    "pct_llaves": _pct(c["grupos"], llaves_con_nombre),
                    "pct_filas": _pct(c["filas"], filas_con_nombre),
                }
                for t, c in tamanos_grupo["grupos"].items()
            },
        },
        "filas_en_llaves_repetidas": {
            "filas": techo_clase["filas"],
            "llaves": techo_clase["llaves"],
            "pct_filas_con_nombre": _pct(techo_clase["filas"], filas_con_nombre),
            "pct_todas_las_filas": _pct(techo_clase["filas"], filas_total),
        },
        "techo_por_nivel": {NIVELES[n]: techo(conteos[NIVELES[n]]) for n in NIVELES},
        "techo_por_nivel_definicion": DEFINICIONES["techo"],
        "extra_techo_todas_las_filas_con_nombre": {
            "por_nivel": {NIVELES[n]: techo(conteos[NIVELES[n]], minimo_filas=1) for n in NIVELES},
            "nota": "Extra: las llaves de 1 fila cuentan como aciertos; es una cota dentro de la muestra.",
        },
        "clases_distintas_por_llave_repetida": clases_por_llave(conteos["clase"]),
        "grupos_100_o_mas": {
            "grupos": len(grandes),
            "filas": sum(sum(v.values()) for v in grandes.values()),
            "pureza": pureza_grupos(conteos["clase"]),
            "por_tipo": por_tipo(grandes),
        },
        "llaves_repetidas_por_tipo": por_tipo(
            {k: v for k, v in conteos["clase"].items() if k and sum(v.values()) >= 2}
        ),
    }

    # 5. Regla 7.
    ids, uniques = pd.factorize(con_nombre["llave"])
    tamanos = np.bincount(ids)
    es_marcador = np.array([marcador_de[k] for k in uniques])
    es_gen = np.array([generico_de[k] for k in uniques])
    grande = tamanos >= GRUPO_GRANDE
    secciones = {
        "por_tamano": {
            nombre: (tamanos >= desde) & (tamanos <= (hasta if hasta is not None else tamanos.max()))
            for nombre, desde, hasta in TRAMOS_TAMANO
        },
        "grupos_100_o_mas_por_tipo": {
            "con_marcador": grande & es_marcador,
            "genericos": grande & es_gen,
            "con_marcador_y_genericos": grande & es_marcador & es_gen,
            "ni_marcador_ni_generico": grande & ~es_marcador & ~es_gen,
        },
        "todos_los_grupos_por_tipo": {
            "con_marcador": es_marcador,
            "genericos": es_gen,
            "con_marcador_y_genericos": es_marcador & es_gen,
            "ni_marcador_ni_generico": ~es_marcador & ~es_gen,
        },
    }
    regla7 = {
        str(m): estimar_regla7(ids, tamanos, m, SEMILLAS_SIMULACION, secciones) for m in MUESTRAS
    }

    # 6. Topes por clase (filas con nombre).
    filas_por_clase_con_nombre = {str(c): int(f) for c, f in con_nombre["codigo_act"].value_counts().items()}
    topes = topes_por_clase(filas_por_clase_con_nombre)

    # 9. Efectos de las opciones.
    efectos = {
        "vacios": {
            "universo": "todas las filas (la llave vacía cuenta como un grupo)",
            **efecto_de_excluir(df["llave"], df["codigo_act"], df["vacia"]),
        },
        "marcador": {
            "universo": "filas con nombre",
            **efecto_de_excluir(con_nombre["llave"], con_nombre["codigo_act"], con_nombre["marcador"]),
        },
        "iguales_a_un_titulo_o_a_nombre_act": {
            "universo": "filas con nombre",
            **efecto_de_excluir(
                con_nombre["llave"],
                con_nombre["codigo_act"],
                con_nombre["titulo_su_clase"] | con_nombre["titulo_otra_clase"] | con_nombre["igual_nombre_act"],
            ),
        },
        "genericos": {
            "universo": "filas con nombre",
            **efecto_de_excluir(con_nombre["llave"], con_nombre["codigo_act"], con_nombre["generico"]),
        },
    }

    # 8. Archivo local de revisión.
    marcas = (
        df[~df["vacia"]]
        .groupby("llave")[["titulo_su_clase", "titulo_otra_clase", "igual_nombre_act"]]
        .sum()
        .rename(columns={"titulo_su_clase": "su_clase", "titulo_otra_clase": "otra_clase", "igual_nombre_act": "nombre_act"})
    )
    revision = filas_de_revision(conteos["clase"], catalogo, marcas.to_dict(orient="index"), vocab)
    ruta_revision = REPO_RAIZ / "data" / f"{fecha}_fase1_revision-llaves.csv"

    resultado = {
        "fecha": fecha,
        "experimento": "fase1_paso4_exploracion_clasificador",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": "src/exploracion_clasificador.py",
            "cargador": "src/denue.py:cargar_denue",
            "normalizacion": "src/normalizacion.py:normalizar_nombre",
            "columnas": COLUMNAS,
            "codificacion": entrada["codificacion"],
            "scian_version": manifest["scian_version"],
            "catalogo": {"ruta": entrada_cat["ruta"], "sha256": entrada_cat["sha256"]},
            "estructura_oficial": {"ruta": entrada_pdf["ruta"], "sha256": entrada_pdf["sha256"]},
            "definiciones": DEFINICIONES,
            "muestras_regla7": {"principal": MUESTRA_PRINCIPAL, "extra": 6000},
            "simulacion": "numpy.random.default_rng(semilla).choice(N, m, replace=False)",
            "topes_por_clase": list(TOPES),
            "grupo_grande_filas": GRUPO_GRANDE,
            "python": platform.python_version(),
            "versiones": {p: version_de_paquete(p) for p in ("numpy", "pandas", "scipy", "matplotlib")},
        },
        "semilla": SEMILLA,
        "nota_semilla": NOTA_SEMILLA,
        "semillas_simulacion": f"{SEMILLAS_SIMULACION[0]} a {SEMILLAS_SIMULACION[-1]}",
        "split": "ninguno: CSV completo (todavía no hay splits)",
        "hash_split": entrada["sha256"],
        "hash_split_origen": "data/manifest.json, fuentes.denue_cdmx.sha256",
        "tamano_muestra": filas_total,
        "horas_gpu": None,
        "poblaciones": {"todas_las_filas": filas_total, "filas_con_nombre": filas_con_nombre},
        "clases": clases,
        "por_nivel": niveles,
        "nombres_problematicos": nombres_problematicos,
        "grupos": grupos,
        "regla7_estimacion": regla7,
        "topes_por_clase": {"universo": "filas con nombre, sin separar validación y test", **topes},
        "efectos_de_las_opciones": efectos,
        "figuras_datos": {"sectores": datos_figura_sectores(niveles["sector"])},
        "figuras": [f"docs/figuras/{fecha}_fase1_{n}.png" for n in FIGURAS],
        "archivo_de_revision": {
            "ruta": f"data/{ruta_revision.name}",
            "llaves": len(revision),
            "nota": "Local e ignorado por git; contiene llaves normalizadas, nunca nom_estab original.",
        },
    }

    verificar_privacidad(resultado, set(llaves_unicas), permitidos)
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_exploracion-clasificador.json"
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")
    escribir_revision(revision, ruta_revision)
    rutas_figuras = dibujar_figuras(resultado)

    # Resumen en la terminal: solo conteos, porcentajes y códigos.
    print(f"Filas: {filas_total:,} (con nombre {filas_con_nombre:,})  |  commit {commit[:7]}  |  árbol limpio: {arbol_limpio}")
    print(f"Clases: {json.dumps({k: v for k, v in clases.items() if k != 'filas_por_clase'}, ensure_ascii=False)}")
    for nombre, res in niveles.items():
        print(f"  {nombre}: {res['categorias_con_filas']} de {res['categorias_catalogo']} con filas")
    print(f"Nombres problemáticos: {json.dumps(nombres_problematicos, ensure_ascii=False)}")
    print(f"Grupos: {json.dumps({k: v for k, v in grupos.items() if k != 'techo_por_nivel_definicion'}, ensure_ascii=False)}")
    for m, r in regla7.items():
        print(f"Regla 7 (m={m}): analítico {r['analitico']}; simulación {r['simulacion']}; >10 %: {r['pasa_del_10_pct']}")
        print(f"  aporte: {json.dumps(r['aporte'], ensure_ascii=False)}")
    print(f"Topes: {json.dumps(topes, ensure_ascii=False)}")
    print(f"Efectos: {json.dumps(efectos, ensure_ascii=False)}")
    print(f"Guardado en {salida.relative_to(REPO_RAIZ).as_posix()}")
    print(f"Revisión local: {ruta_revision.relative_to(REPO_RAIZ).as_posix()} ({len(revision)} llaves)")
    for r in rutas_figuras:
        print(f"Figura: {r.relative_to(REPO_RAIZ).as_posix()}")


if __name__ == "__main__":
    main()
