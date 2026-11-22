"""Hard set del clasificador (Fase 1, paso 7).

150 filas de `data/splits/train_candidatos.jsonl`, una por llave, curadas a mano por Samuel
(CLAUDE.md, sección c, reglas 2, 3, 7 y 8). Dos tipos: ambiguo (el nombre admite varias
actividades) e indirecto (sin palabras del vocabulario SCIAN, pero una persona deduciría la
actividad). Se trata como el test: se congela con su SHA-256 y no se vuelve a abrir hasta la
Fase 4. Todas las filas de sus llaves salen de train.

Candidatos (semilla 42):

1. Fuera las llaves con alguna fila marcada `sin_nombre` o `igual_titulo` (marcas de
   `src/subgrupos.py`, ya guardadas en cada fila de los splits).
2. Fuente (i), clases mezcladas: llaves de 2 a 20 filas cuya clase más común no pasa del
   60 % de sus filas.
3. Fuente (ii), sin vocabulario: llaves de 1 fila que, sin las palabras gramaticales de
   PALABRAS_GRAMATICALES, tienen al menos una palabra y ninguna pasa `en_vocabulario` de
   `src/subgrupos.py`.
4. rng = numpy.random.default_rng(42), en este orden: permutación de las llaves de (i)
   (ordenadas), una fila por llave de (i) con rng.integers(n) en el orden de esa permutación
   (filas en orden de int(id)) y permutación de las llaves de (ii) (ordenadas).
5. Cada ronda recorre las permutaciones desde donde se quedó la anterior y acepta un
   candidato si su sector (2 primeros dígitos del codigo_act de la fila sorteada) tiene cupo:
   tope de 30 por sector, por fuente y por ronda; cuotas de cada ronda en RONDAS.

Uso:
  .venv/Scripts/python -m src.hard_set candidatos              # PARADA 1: solo conteos, no escribe
  .venv/Scripts/python -m src.hard_set candidatos --escribir   # CSV de curación de la última ronda
  .venv/Scripts/python -m src.hard_set leer CSV [CSV ...]      # lectura de vuelta y PARADA 2; no escribe
  .venv/Scripts/python -m src.hard_set congelar CSV [CSV ...]  # hard.jsonl, train sin hard, manifest y results

Nunca abre el archivo de test ni construye su ruta: del test solo usa el SHA-256 y las filas
que registra el manifest. En la terminal solo van conteos y códigos SCIAN, nunca nombres,
llaves ni ids.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import io
import json
import platform
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.denue import REPO_RAIZ, sha256_archivo
from src.muestra_humo import codigo_versionado, hashes_del_manifest, leer_verificado, parsear
from src.splits import CAMPOS, ESPERADOS, LIMITE_REGLA7_PCT, esta_ignorado, pasa_del_limite, sha256_bytes
from src.splits import FORMATO as FORMATO_SPLITS
from src.splits import serializar as serializar_split
from src.subgrupos import en_vocabulario, quitar_marcador

SCRIPT = "src/hard_set.py"
SEMILLA = 42
N_HARD = 150
MIN_FILAS_TIPO = 50  # aviso en la PARADA 2 si un tipo queda por debajo
MIN_FILAS_LLAVE_I = 2
MAX_FILAS_LLAVE = 20  # tope de tamaño de la llave: 150 × 20 filas no pasan el 10 % de la regla 7
MAX_PCT_CLASE = 60
TOPE_SECTOR = 30
# Cuotas de cada ronda. Una ronda nueva se agrega aquí solo con la aprobación de Samuel.
RONDAS: tuple[dict[str, int], ...] = ({"i": 200, "ii": 200},)
FUENTES = {"i": "i_clases_mezcladas", "ii": "ii_sin_vocabulario"}
TIPOS = ("ambiguo", "indirecto")
CAMPOS_HARD = (*CAMPOS, "tipo")

# Palabras de clase cerrada que no cuentan en la fuente (ii). La lista original la propuso el
# chat de Claude del proyecto y la aprobó Samuel; las agregadas las detectó Claude Code (son
# las demás palabras de clase cerrada que pasaban `en_vocabulario`) y las aprobó el chat.
PALABRAS_GRAMATICALES_ORIGINALES = frozenset(
    "A AL CON DE DEL E EL EN LA LAS LO LOS O PARA POR SIN U UN UNA UNOS UNAS Y".split()
)
PALABRAS_GRAMATICALES_AGREGADAS = frozenset({"MEDIANTE", "QUE", "SOBRE", "SU", "SUS"})
PALABRAS_GRAMATICALES = PALABRAS_GRAMATICALES_ORIGINALES | PALABRAS_GRAMATICALES_AGREGADAS

ENTRADA_TRAIN = "clasificador_train_candidatos"
ENTRADA_VAL = "clasificador_validacion"
ENTRADA_TEST = "clasificador_test"  # de esta entrada solo se leen `sha256` y `filas`
ENTRADA_HARD = "clasificador_hard"
ENTRADA_TRAIN_SIN_HARD = "clasificador_train_candidatos_sin_hard"
CARPETA_SPLITS = REPO_RAIZ / "data" / "splits"
ARCHIVO_HARD = "hard.jsonl"
ARCHIVO_TRAIN_SIN_HARD = "train_candidatos_sin_hard.jsonl"

COLUMNAS_CSV = (
    "id", "nom_estab", "codigo_act", "descripcion_scian", "fuente", "filas_llave",
    "clase_1", "clase_2", "clase_3", "incluir", "tipo", "nota",
)
COLUMNAS_LEIDAS = ("id", "incluir", "tipo")
PREFIJO_ID = "id-"
INICIO_FORMULA = ("=", "+", "-", "@")
RE_ID_CSV = re.compile(r"(?:ID-)?([0-9]+)")
SI = frozenset({"S", "SI"})
NO = frozenset({"", "N", "NO"})
TRAMOS_FILAS = (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-10", 6, 10), ("11-20", 11, 20))

PROCEDIMIENTO = (
    "Origen: splits.clasificador_train_candidatos (SHA-256 verificado antes de parsear). "
    "Fuera las llaves con alguna fila sin_nombre o igual_titulo. "
    f"Fuente (i): llaves de {MIN_FILAS_LLAVE_I} a {MAX_FILAS_LLAVE} filas cuya clase más común no pasa del "
    f"{MAX_PCT_CLASE} % de sus filas. "
    "Fuente (ii): llaves de 1 fila que, sin las palabras gramaticales, tienen al menos una palabra y ninguna "
    "pasa en_vocabulario (src/subgrupos.py). "
    f"rng = numpy.random.default_rng({SEMILLA}): permutación de las llaves de (i) ordenadas; una fila por llave "
    "de (i) con rng.integers(n) en el orden de la permutación (filas por int(id)); permutación de las llaves "
    "de (ii) ordenadas. Cada ronda sigue las permutaciones desde donde quedó la anterior y acepta un candidato "
    f"si su sector (2 primeros dígitos de codigo_act) tiene cupo: tope {TOPE_SECTOR} por sector, fuente y "
    "ronda. Samuel marcó a mano incluir (S/N) y tipo (ambiguo/indirecto) en el CSV de curación; si la "
    "etiqueta del DENUE le pareció equivocada, la fila se descartó (no se corrigió). Hard set: las "
    f"{N_HARD} filas con incluir=S. Train candidato sin hard: las filas de train candidato cuya llave no está "
    "en el hard set."
)
NOTA_HARD = (
    "Se trata como el test (regla 3): no se usa para elegir modelos ni hiperparámetros, no sube a Kaggle "
    "hasta la Fase 4 y se evalúa una sola vez, junto con el test."
)
NOTA_TRAIN_SIN_HARD = "Sin congelar: el tope por clase parte de este archivo y nunca abre hard.jsonl."


class ErrorHardSet(ValueError):
    """El hard set o sus candidatos no cumplen una condición; no se escribe nada."""


class ErrorCuracion(ErrorHardSet):
    """El CSV curado tiene un valor inválido o no cumple una validación."""


class ErrorConteo(ErrorCuracion):
    """El CSV curado no tiene exactamente N_HARD filas con incluir=S."""

    def __init__(self, encontradas: int, esperadas: int, por_fuente: Mapping[str, int]):
        self.encontradas = encontradas
        self.esperadas = esperadas
        self.por_fuente = dict(por_fuente)
        super().__init__(f"Hay {encontradas} filas con incluir=S; se piden exactamente {esperadas}.")


# --- Entradas ---


@dataclass
class Entradas:
    """Train candidato y validación verificados, con los conteos de la regla 7 del manifest."""

    regs: list[dict]
    grupos: dict[str, list[dict]]
    llaves_val: set[str]
    candidatas: int  # filas candidatas a train de la regla 7 (universo − test − validación)
    previas: int  # filas ya excluidas por compartir llave con test o validación
    sha_test: str
    hashes: dict[str, str]


def agrupar(regs: Sequence[Mapping]) -> dict[str, list[dict]]:
    """Llave → sus filas, en el orden del archivo (int(id))."""
    grupos: dict[str, list[dict]] = {}
    for r in regs:
        grupos.setdefault(r["llave"], []).append(r)
    return grupos


def cargar_entradas(manifest: Mapping, raiz: Path = REPO_RAIZ, universo: int = ESPERADOS["universo"]) -> Entradas:
    """Lee train candidato y validación con su SHA-256 verificado; del test solo usa el manifest."""
    s = manifest["splits"]
    e_train, e_val, e_test = s[ENTRADA_TRAIN], s[ENTRADA_VAL], s[ENTRADA_TEST]
    regs = parsear(leer_verificado(Path(raiz) / e_train["ruta"], e_train["sha256"]))
    val = parsear(leer_verificado(Path(raiz) / e_val["ruta"], e_val["sha256"]))
    if len(regs) != e_train["filas"] or len(val) != e_val["filas"]:
        raise ErrorHardSet("Las filas de train candidato o de validación no son las del manifest.")
    grupos = agrupar(regs)
    llaves_val = {r["llave"] for r in val}
    if set(grupos) & llaves_val:
        raise ErrorHardSet("Train candidato comparte llaves con validación.")
    candidatas = universo - e_test["filas"] - e_val["filas"]
    previas = candidatas - e_train["filas"]
    if previas < 0:
        raise ErrorHardSet("Los conteos del manifest no cuadran con el universo.")
    hashes = {"train_candidatos": e_train["sha256"], "validacion": e_val["sha256"]}
    return Entradas(regs, grupos, llaves_val, candidatas, previas, e_test["sha256"], hashes)


# --- Fuentes de candidatos ---


def llave_excluida(filas: Sequence[Mapping]) -> bool:
    """Alguna fila de la llave tiene la marca de "SIN NOMBRE" o la de igual a título."""
    return any(r["sin_nombre"] or r["igual_titulo"] for r in filas)


def clases_mezcladas(filas: Sequence[Mapping], max_filas: int = MAX_FILAS_LLAVE) -> bool:
    """Fuente (i): de 2 a max_filas filas y la clase más común no pasa del 60 % (con enteros)."""
    n = len(filas)
    if not MIN_FILAS_LLAVE_I <= n <= max_filas:
        return False
    mayor = max(Counter(r["codigo_act"] for r in filas).values())
    return mayor * 100 <= MAX_PCT_CLASE * n


def palabras_con_contenido(llave: str, gramaticales: frozenset[str] = PALABRAS_GRAMATICALES) -> list[str]:
    """Palabras de la llave sin la frase "SIN NOMBRE" y sin las palabras gramaticales."""
    return [p for p in quitar_marcador(llave).split() if p not in gramaticales]


def sin_vocabulario(llave: str, vocab: frozenset[str], gramaticales: frozenset[str] = PALABRAS_GRAMATICALES) -> bool:
    """Fuente (ii): al menos una palabra con contenido y ninguna en el vocabulario del SCIAN."""
    palabras = palabras_con_contenido(llave, gramaticales)
    return bool(palabras) and not any(en_vocabulario(p, vocab) for p in palabras)


@dataclass
class Elegibles:
    i: list[str]
    ii: list[str]
    conteos: dict[str, int]


def elegibles(grupos: Mapping[str, Sequence[Mapping]], vocab: frozenset[str]) -> Elegibles:
    """Llaves elegibles de cada fuente (ordenadas) y los conteos de la PARADA 1."""
    i, ii = [], []
    c = Counter()
    for k, filas in grupos.items():
        if any(r["sin_nombre"] for r in filas):
            c["excluidas_sin_nombre"] += 1
            continue
        if any(r["igual_titulo"] for r in filas):
            c["excluidas_igual_titulo"] += 1
            continue
        if len(filas) >= MIN_FILAS_LLAVE_I:
            if clases_mezcladas(filas):
                i.append(k)
            elif clases_mezcladas(filas, max_filas=len(filas)):
                c["i_fuera_por_mas_de_20_filas"] += 1
            continue
        if sin_vocabulario(k, vocab):
            ii.append(k)
        elif not palabras_con_contenido(k):
            c["ii_solo_palabras_gramaticales"] += 1
        if sin_vocabulario(k, vocab, PALABRAS_GRAMATICALES_ORIGINALES):
            c["ii_con_la_lista_original"] += 1
        if sin_vocabulario(k, vocab, frozenset()):
            c["ii_sin_ignorar_palabras"] += 1
    if any(grupos[k][0]["generico"] for k in ii):
        raise ErrorHardSet("Una llave de la fuente (ii) tiene la marca de genérico; las definiciones no cuadran.")
    conteos = {
        "llaves": len(grupos),
        "excluidas_sin_nombre": c["excluidas_sin_nombre"],
        "excluidas_igual_titulo": c["excluidas_igual_titulo"],
        "i": len(i),
        "i_genericas": sum(bool(grupos[k][0]["generico"]) for k in i),
        "i_fuera_por_mas_de_20_filas": c["i_fuera_por_mas_de_20_filas"],
        "ii": len(ii),
        "ii_con_la_lista_original": c["ii_con_la_lista_original"],
        "ii_sin_ignorar_palabras": c["ii_sin_ignorar_palabras"],
        "ii_solo_palabras_gramaticales": c["ii_solo_palabras_gramaticales"],
    }
    return Elegibles(sorted(i), sorted(ii), conteos)


# --- Sorteo y reparto por sector ---


@dataclass(frozen=True, eq=False)
class Candidato:
    ronda: int
    fuente: str  # "i" o "ii"
    posicion: int  # posición en la permutación de su fuente
    registro: Mapping  # la fila sorteada, con las claves de CAMPOS
    filas_llave: int

    @property
    def id(self) -> str:
        return self.registro["id"]

    @property
    def llave(self) -> str:
        return self.registro["llave"]

    @property
    def sector(self) -> str:
        return self.registro["codigo_act"][:2]


def sortear(
    llaves_i: Sequence[str], llaves_ii: Sequence[str], grupos: Mapping[str, Sequence[Mapping]], semilla: int = SEMILLA
) -> dict[str, list[Mapping]]:
    """Las filas de cada fuente en el orden del sorteo (una por llave)."""
    llaves_i, llaves_ii = sorted(llaves_i), sorted(llaves_ii)
    rng = np.random.default_rng(semilla)
    perm_i = rng.permutation(len(llaves_i))
    filas_i = [grupos[llaves_i[j]][int(rng.integers(len(grupos[llaves_i[j]])))] for j in perm_i]
    perm_ii = rng.permutation(len(llaves_ii))
    filas_ii = [grupos[llaves_ii[j]][0] for j in perm_ii]
    return {"i": filas_i, "ii": filas_ii}


def repartir(secuencia: Sequence[Mapping], inicio: int, cuota: int, tope: int = TOPE_SECTOR) -> tuple[list[int], int]:
    """Posiciones aceptadas desde `inicio` hasta llenar la cuota, con tope por sector.

    Devuelve también la posición siguiente a la última examinada (donde sigue la próxima ronda).
    """
    por_sector: Counter = Counter()
    aceptadas: list[int] = []
    pos = inicio
    while len(aceptadas) < cuota and pos < len(secuencia):
        sector = secuencia[pos]["codigo_act"][:2]
        if por_sector[sector] < tope:
            por_sector[sector] += 1
            aceptadas.append(pos)
        pos += 1
    if len(aceptadas) < cuota:
        raise ErrorHardSet(f"Solo hay {len(aceptadas)} candidatos con cupo de sector; se piden {cuota}.")
    return aceptadas, pos


@dataclass
class Generacion:
    candidatos: list[Candidato]
    elegibles: Elegibles
    recorridas: dict[str, int]
    rondas: tuple[dict[str, int], ...] = field(default_factory=tuple)

    def de_ronda(self, ronda: int) -> list[Candidato]:
        return [c for c in self.candidatos if c.ronda == ronda]


def generar_candidatos(
    grupos: Mapping[str, Sequence[Mapping]],
    vocab: frozenset[str],
    rondas: Sequence[Mapping[str, int]] = RONDAS,
    semilla: int = SEMILLA,
    tope: int = TOPE_SECTOR,
) -> Generacion:
    """Candidatos de todas las rondas, desde cero."""
    el = elegibles(grupos, vocab)
    secuencias = sortear(el.i, el.ii, grupos, semilla)
    inicio = {"i": 0, "ii": 0}
    cands: list[Candidato] = []
    for ronda, cuotas in enumerate(rondas, 1):
        for fuente in ("i", "ii"):
            aceptadas, inicio[fuente] = repartir(secuencias[fuente], inicio[fuente], cuotas[fuente], tope)
            for p in aceptadas:
                r = secuencias[fuente][p]
                cands.append(Candidato(ronda, fuente, p, r, len(grupos[r["llave"]])))
    if len({c.llave for c in cands}) != len(cands):
        raise ErrorHardSet("Hay dos candidatos con la misma llave.")
    return Generacion(cands, el, dict(inicio), tuple(dict(r) for r in rondas))


# --- Regla 7 ---


def peor_caso(tamanos: Sequence[int], n: int = N_HARD) -> int:
    """Filas de las n llaves más grandes: lo máximo que el hard set puede sacar de train."""
    return sum(sorted(tamanos, reverse=True)[:n])


def exclusion_regla7(previas: int, candidatas: int, por_hard: int) -> dict:
    total = previas + por_hard
    return {
        "filas_candidatas_train": candidatas,
        "excluidas_por_test_y_validacion": previas,
        "excluidas_por_hard_set": por_hard,
        "filas_excluidas": total,
        "pct_filas_candidatas": round(100 * total / candidatas, 2) if candidatas else None,
        "limite_pct": LIMITE_REGLA7_PCT,
        "pasa_del_limite": pasa_del_limite(total, candidatas),
    }


def tramo_filas(n: int) -> str:
    for nombre, a, b in TRAMOS_FILAS:
        if a <= n <= b:
            return nombre
    raise ErrorHardSet(f"Una llave candidata tiene {n} filas, fuera de los tramos.")


def resumen_candidatos(gen: Generacion, ent: Entradas) -> dict:
    """Conteos de la PARADA 1 (sin nombres, llaves ni ids)."""
    cands = gen.candidatos
    por_sector = {f: Counter(c.sector for c in cands if c.fuente == f) for f in FUENTES}
    sectores = sorted(set(por_sector["i"]) | set(por_sector["ii"]))
    peor = peor_caso([c.filas_llave for c in cands])
    return {
        "palabras_gramaticales": {
            "originales": sorted(PALABRAS_GRAMATICALES_ORIGINALES),
            "agregadas": sorted(PALABRAS_GRAMATICALES_AGREGADAS),
        },
        "elegibles": gen.elegibles.conteos,
        "rondas": [dict(r) for r in gen.rondas],
        "ofrecidos": {f: sum(c.fuente == f for c in cands) for f in FUENTES},
        "posiciones_recorridas": dict(gen.recorridas),
        "por_sector": {s: {f: por_sector[f][s] for f in FUENTES} for s in sectores},
        "filas_por_llave": {t: sum(tramo_filas(c.filas_llave) == t for c in cands) for t, _, _ in TRAMOS_FILAS},
        "genericos": {f: sum(bool(c.registro["generico"]) for c in cands if c.fuente == f) for f in FUENTES},
        "clases": len({c.registro["codigo_act"] for c in cands}),
        "peor_caso": {"filas_llaves_mas_grandes": peor, **exclusion_regla7(ent.previas, ent.candidatas, peor)},
        "cota_garantizada": {
            "filas": N_HARD * MAX_FILAS_LLAVE,
            **exclusion_regla7(ent.previas, ent.candidatas, N_HARD * MAX_FILAS_LLAVE),
        },
    }


# --- CSV de curación ---


def texto_excel(t: str) -> str:
    """Un espacio al inicio si el texto empieza como fórmula de Excel (=, +, -, @)."""
    return " " + t if t.startswith(INICIO_FORMULA) else t


def clases_mas_comunes(filas: Sequence[Mapping], catalogo: Mapping[str, str], k: int = 3) -> list[str]:
    """Las k clases más comunes de la llave: "código (filas) título"; empates por código."""
    cnt = Counter(r["codigo_act"] for r in filas)
    orden = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [f"{c} ({n}) {catalogo[c]}" for c, n in orden] + [""] * (k - len(orden))


def intercalar(cands: Sequence[Candidato]) -> list[Candidato]:
    """Orden del CSV: por ronda, alternando (i) y (ii) en el orden del sorteo."""
    salida: list[Candidato] = []
    for ronda in sorted({c.ronda for c in cands}):
        por_fuente = {f: sorted((c for c in cands if c.ronda == ronda and c.fuente == f), key=lambda c: c.posicion)
                      for f in FUENTES}
        for j in range(max(len(v) for v in por_fuente.values())):
            salida += [por_fuente[f][j] for f in FUENTES if j < len(por_fuente[f])]
    return salida


def filas_curacion(
    cands: Sequence[Candidato], grupos: Mapping[str, Sequence[Mapping]], catalogo: Mapping[str, str]
) -> tuple[list[dict], Counter]:
    """Filas del CSV y cuántas celdas de cada columna llevaron el espacio contra fórmulas."""
    filas, protegidas = [], Counter()
    for c in intercalar(cands):
        r = c.registro
        clases = clases_mas_comunes(grupos[c.llave], catalogo)
        textos = {
            "nom_estab": r["nom_estab"],
            "descripcion_scian": catalogo[r["codigo_act"]],
            "clase_1": clases[0], "clase_2": clases[1], "clase_3": clases[2],
        }
        for col, t in textos.items():
            if texto_excel(t) != t:
                protegidas[col] += 1
        filas.append({
            "id": PREFIJO_ID + r["id"],
            "nom_estab": texto_excel(textos["nom_estab"]),
            "codigo_act": r["codigo_act"],
            "descripcion_scian": texto_excel(textos["descripcion_scian"]),
            "fuente": FUENTES[c.fuente],
            "filas_llave": str(c.filas_llave),
            "clase_1": texto_excel(clases[0]), "clase_2": texto_excel(clases[1]), "clase_3": texto_excel(clases[2]),
            "incluir": "", "tipo": "", "nota": "",
        })
    return filas, protegidas


def serializar_csv(filas: Sequence[Mapping]) -> bytes:
    """CSV en UTF-8 con BOM (para Excel), separado por comas, con saltos "\\n"."""
    buf = io.StringIO()
    escritor = csv.DictWriter(buf, fieldnames=COLUMNAS_CSV, lineterminator="\n")
    escritor.writeheader()
    escritor.writerows(filas)
    return buf.getvalue().encode("utf-8-sig")


def escribir_nuevo(datos: bytes, ruta: Path) -> None:
    """Escribe en binario solo si git ignora la ruta y el archivo no existe."""
    if not esta_ignorado(ruta):
        raise ErrorHardSet(f"{Path(ruta).name} no está ignorado por git; no se escribe.")
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "xb") as f:  # "x": falla si el archivo ya existe
        f.write(datos)


# --- Lectura de vuelta ---


@dataclass(frozen=True)
class Marca:
    archivo: str
    registro: int  # número de registro en el CSV (1 = primera fila después del encabezado)
    id: str
    incluir: bool
    tipo: str | None
    tipo_en_fila_n: bool  # la fila dice N (o vacío) pero trae algo en tipo


def _sin_acentos(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


def _celda(t: str) -> str:
    return _sin_acentos(t.replace("﻿", "")).strip()


def decodificar(datos: bytes) -> str:
    """UTF-8 (con o sin BOM), UTF-16 con BOM o, si nada de eso, cp1252 (Excel en ANSI)."""
    if datos.startswith((b"\xff\xfe", b"\xfe\xff")):
        return datos.decode("utf-16")
    if datos.startswith(b"\xef\xbb\xbf"):
        datos = datos[3:]
    try:
        return datos.decode("utf-8")
    except UnicodeDecodeError:
        return datos.decode("cp1252", errors="replace")  # solo se leen columnas ASCII


def _registros_csv(texto: str) -> tuple[list[list[str]], dict[str, int]]:
    """Registros del CSV y la posición de cada columna leída; detecta `,`, `;` o tabulador."""
    for delim in (",", ";", "\t"):
        filas = list(csv.reader(io.StringIO(texto, newline=""), delimiter=delim))
        if not filas:
            break
        encabezado = [_celda(c).lower() for c in filas[0]]
        if all(c in encabezado for c in COLUMNAS_LEIDAS):
            repetidas = [c for c in COLUMNAS_LEIDAS if encabezado.count(c) > 1]
            if repetidas:
                raise ErrorCuracion(f"Columnas repetidas en el encabezado: {repetidas}.")
            return filas[1:], {c: encabezado.index(c) for c in COLUMNAS_LEIDAS}
    raise ErrorCuracion(f"No encuentro las columnas {list(COLUMNAS_LEIDAS)} en el encabezado.")


def leer_curacion(datos: bytes, archivo: str = "") -> list[Marca]:
    """Solo id, incluir y tipo de cada registro no vacío; tolera lo que Excel produce al guardar."""
    registros, pos = _registros_csv(decodificar(datos))
    marcas, errores = [], []
    for n, fila in enumerate(registros, 1):
        if not any(c.strip() for c in fila):
            continue
        valor = {c: _celda(fila[i]) if i < len(fila) else "" for c, i in pos.items()}
        donde = f"{archivo} registro {n}"
        m = RE_ID_CSV.fullmatch(valor["id"].upper())
        incluir = valor["incluir"].upper()
        tipo = valor["tipo"].lower()
        if not m:
            errores.append(f"{donde}: columna id con formato inválido")
        if incluir not in SI | NO:
            errores.append(f"{donde}: columna incluir debe ser S o N")
        elif incluir in SI and tipo not in TIPOS:
            errores.append(f"{donde}: columna tipo debe ser {' o '.join(TIPOS)} cuando incluir es S")
        if m and incluir in SI | NO:
            si = incluir in SI
            marcas.append(Marca(archivo, n, m.group(1), si, tipo if si else None, not si and tipo != ""))
    if errores:
        raise ErrorCuracion("\n".join(errores))
    return marcas


def validar_curacion(
    marcas: Sequence[Marca], cands: Sequence[Candidato], llaves_val: set[str], n: int = N_HARD
) -> list[tuple[Candidato, str]]:
    """Las filas con incluir=S, ordenadas por int(id), tras todas las validaciones."""
    por_id = {int(c.id): c for c in cands}
    vistos: dict[int, str] = {}
    errores = []
    for m in marcas:
        k, donde = int(m.id), f"{m.archivo} registro {m.registro}"
        if k in vistos:
            errores.append(f"{donde}: id repetido (también en {vistos[k]})")
        vistos.setdefault(k, donde)
        if k not in por_id:
            errores.append(f"{donde}: el id no es de ningún candidato regenerado")
    if errores:
        raise ErrorCuracion("\n".join(errores))
    incluidas = [(por_id[int(m.id)], m.tipo) for m in marcas if m.incluir]
    llaves = [c.llave for c, _ in incluidas]
    if len(set(llaves)) != len(llaves):
        raise ErrorCuracion("Hay dos filas incluidas con la misma llave.")
    if set(llaves) & llaves_val:
        raise ErrorCuracion("Hay filas incluidas cuya llave está en validación.")
    if len(incluidas) != n:
        raise ErrorConteo(len(incluidas), n, Counter(c.fuente for c, _ in incluidas))
    return sorted(incluidas, key=lambda ct: int(ct[0].id))


def leer_archivos(rutas: Sequence[Path]) -> tuple[list[Marca], list[dict]]:
    """Marcas de todos los CSV curados y la ruta y el SHA-256 de cada uno."""
    marcas, archivos = [], []
    for ruta in rutas:
        datos = Path(ruta).read_bytes()
        p = Path(ruta).resolve()
        nombre = p.relative_to(REPO_RAIZ.resolve()).as_posix() if REPO_RAIZ.resolve() in p.parents else p.name
        marcas += leer_curacion(datos, nombre)
        archivos.append({"ruta": nombre, "sha256": sha256_bytes(datos), "tamano_bytes": len(datos)})
    return marcas, archivos


# --- Resumen del hard set (PARADA 2) ---


def resumen_hard(incluidas: Sequence[tuple[Candidato, str]], ent: Entradas) -> dict:
    """Conteos de la PARADA 2 (sin nombres, llaves ni ids)."""
    llaves = {c.llave for c, _ in incluidas}
    por_tipo = Counter(t for _, t in incluidas)
    salen = sum(len(ent.grupos[k]) for k in llaves)
    clases_train = Counter(r["codigo_act"] for r in ent.regs)
    restantes = Counter(r["codigo_act"] for r in ent.regs if r["llave"] not in llaves)
    clases_hard = Counter(c.registro["codigo_act"] for c, _ in incluidas)
    return {
        "filas": len(incluidas),
        "por_tipo": {t: por_tipo[t] for t in TIPOS},
        "por_fuente_y_tipo": {f: {t: sum(c.fuente == f and tt == t for c, tt in incluidas) for t in TIPOS}
                              for f in FUENTES},
        "por_sector": dict(sorted(Counter(c.sector for c, _ in incluidas).items())),
        "clases": len(clases_hard),
        "por_clase": dict(sorted(clases_hard.items())),
        "genericos": {t: sum(bool(c.registro["generico"]) for c, tt in incluidas if tt == t) for t in TIPOS},
        "filas_que_salen_de_train": salen,
        "regla7": exclusion_regla7(ent.previas, ent.candidatas, salen),
        "train_candidatos_sin_hard": {"filas": len(ent.regs) - salen, "clases": len(restantes)},
        "clases_de_train_sin_ejemplos": sorted(c for c in clases_train if c not in restantes),
        "clases_del_hard_sin_ejemplos_en_train": sorted(c for c in clases_hard if c not in restantes),
        "tipos_con_menos_de_50": [t for t in TIPOS if por_tipo[t] < MIN_FILAS_TIPO],
    }


# --- Congelado ---


def registros_hard(incluidas: Sequence[tuple[Candidato, str]]) -> list[dict]:
    regs = [{**{c: ct.registro[c] for c in CAMPOS}, "tipo": t} for ct, t in incluidas]
    return sorted(regs, key=lambda r: int(r["id"]))


def serializar_hard(regs: Sequence[Mapping]) -> bytes:
    """JSONL en UTF-8 sin BOM, con "\\n" al final de cada línea; claves de CAMPOS_HARD en orden."""
    for r in regs:
        if tuple(r) != CAMPOS_HARD or r["tipo"] not in TIPOS:
            raise ErrorHardSet("Un registro del hard set no tiene las claves esperadas en orden o un tipo válido.")
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def contenidos(incluidas: Sequence[tuple[Candidato, str]], ent: Entradas) -> dict[str, bytes]:
    """Bytes de hard.jsonl y de train_candidatos_sin_hard.jsonl."""
    llaves = {c.llave for c, _ in incluidas}
    sin_hard = [r for r in ent.regs if r["llave"] not in llaves]
    return {ARCHIVO_HARD: serializar_hard(registros_hard(incluidas)), ARCHIVO_TRAIN_SIN_HARD: serializar_split(sin_hard)}


def candado_hashes(hashes: Mapping[str, str], manifest: Mapping) -> None:
    """Ningún archivo nuevo puede tener el SHA-256 del test ni el de otra entrada del manifest."""
    del_manifest = hashes_del_manifest(manifest)
    iguales = sorted(n for n, h in hashes.items() if h in del_manifest)
    if iguales:
        raise ErrorHardSet(f"{iguales} tiene el SHA-256 de una entrada del manifest (¿test?); me detengo.")


def escribir_splits(datos: Mapping[str, bytes], carpeta: Path) -> dict[str, Path]:
    """Escribe los dos archivos; se niega si alguno existe o no está ignorado por git."""
    rutas = {n: Path(carpeta) / n for n in datos}
    existentes = sorted(n for n, r in rutas.items() if r.exists())
    if existentes:
        raise ErrorHardSet(f"Ya existen {existentes}; no se sobrescriben.")
    for n, r in rutas.items():
        escribir_nuevo(datos[n], r)
    return rutas


def agregar_entradas(manifest: Mapping, nuevas: Mapping[str, Mapping]) -> dict:
    """Copia del manifest con las entradas nuevas en `splits`; nada más cambia."""
    if set(nuevas) & set(manifest["splits"]):
        raise ErrorHardSet("El manifest ya tiene alguna de las entradas del hard set; no se sobrescriben.")
    nuevo = copy.deepcopy(dict(manifest))
    for k, v in nuevas.items():
        nuevo["splits"][k] = copy.deepcopy(dict(v))
    otras_iguales(manifest, nuevo, set(nuevas))
    return nuevo


def otras_iguales(antes: Mapping, despues: Mapping, nuevas: set[str]) -> None:
    """Falla si cambió algo del manifest fuera de las entradas nuevas de `splits`."""
    claves = (set(antes) | set(despues)) - {"splits"}
    cambiadas = sorted(k for k in claves if antes.get(k) != despues.get(k))
    s_antes = dict(antes.get("splits", {}))
    s_despues = {k: v for k, v in despues.get("splits", {}).items() if k not in nuevas}
    if cambiadas or s_antes != s_despues:
        raise ErrorHardSet(f"Cambiarían otras partes del manifest: {cambiadas or ['splits']}.")


def entradas_manifest(
    *, datos: Mapping[str, bytes], incluidas: Sequence[tuple[Candidato, str]], ent: Entradas, manifest: Mapping,
    archivos_curacion: Sequence[Mapping], commit: str, fecha: str, carpeta: Path = CARPETA_SPLITS,
) -> dict[str, dict]:
    from src.scian import CATALOGO
    from src.subgrupos import VOCABULARIO

    ruta = lambda n: (Path(carpeta) / n).resolve().relative_to(REPO_RAIZ.resolve()).as_posix()  # noqa: E731
    llaves = {c.llave for c, _ in incluidas}
    sin_hard = [r for r in ent.regs if r["llave"] not in llaves]
    fuentes = {
        "train_candidatos_sha256": ent.hashes["train_candidatos"],
        "validacion_sha256": ent.hashes["validacion"],
        "catalogo_sha256": manifest["catalogos"][CATALOGO[manifest["scian_version"]]]["sha256"],
        "vocabulario_sha256": manifest["catalogos"][VOCABULARIO]["sha256"],
    }
    comun = {
        "semilla": SEMILLA, "procedimiento": PROCEDIMIENTO,
        "palabras_gramaticales": {
            "originales": ", ".join(sorted(PALABRAS_GRAMATICALES_ORIGINALES)),
            "agregadas": ", ".join(sorted(PALABRAS_GRAMATICALES_AGREGADAS)),
        },
        "rondas": [dict(r) for r in RONDAS], "tope_sector": TOPE_SECTOR, "max_filas_llave": MAX_FILAS_LLAVE,
        "curacion": [dict(a) for a in archivos_curacion],
        "script": SCRIPT, "commit_codigo": commit, "numpy": np.__version__, "python": platform.python_version(),
        "fuentes": fuentes, "fecha_registro": fecha,
    }
    por_tipo = Counter(t for _, t in incluidas)
    hard = {
        "ruta": ruta(ARCHIVO_HARD), "sha256": sha256_bytes(datos[ARCHIVO_HARD]),
        "tamano_bytes": len(datos[ARCHIVO_HARD]), "filas": len(incluidas), "llaves": len(llaves),
        "clases": len({c.registro["codigo_act"] for c, _ in incluidas}),
        "por_tipo": {t: por_tipo[t] for t in TIPOS}, "congelado": True,
        **comun, "campos": list(CAMPOS_HARD), "formato": FORMATO_SPLITS, "nota": NOTA_HARD,
    }
    train = {
        "ruta": ruta(ARCHIVO_TRAIN_SIN_HARD), "sha256": sha256_bytes(datos[ARCHIVO_TRAIN_SIN_HARD]),
        "tamano_bytes": len(datos[ARCHIVO_TRAIN_SIN_HARD]), "filas": len(sin_hard),
        "llaves": len({r["llave"] for r in sin_hard}), "clases": len({r["codigo_act"] for r in sin_hard}),
        "congelado": False, **comun, "campos": list(CAMPOS), "formato": FORMATO_SPLITS, "nota": NOTA_TRAIN_SIN_HARD,
    }
    return {ENTRADA_HARD: hard, ENTRADA_TRAIN_SIN_HARD: train}


# --- Programa principal ---


def _git(*args: str) -> str:
    resultado = subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True)
    return resultado.stdout.strip()


def _cargar_todo(manifest: Mapping) -> tuple[Entradas, frozenset[str], dict[str, str], Generacion]:
    """Desde cero: entradas verificadas, vocabulario, catálogo y candidatos de todas las rondas."""
    from src.scian import CATALOGO, archivo_verificado, leer_catalogo
    from src.subgrupos import VOCABULARIO, cargar_vocabulario

    ent = cargar_entradas(manifest)
    e_train = manifest["splits"][ENTRADA_TRAIN]
    cat = manifest["catalogos"][CATALOGO[manifest["scian_version"]]]
    if e_train["fuentes"]["vocabulario_sha256"] != manifest["catalogos"][VOCABULARIO]["sha256"]:
        raise ErrorHardSet("El vocabulario del manifest no es el que se usó para los splits.")
    if e_train["fuentes"]["catalogo_sha256"] != cat["sha256"]:
        raise ErrorHardSet("El catálogo del manifest no es el que se usó para los splits.")
    vocab = cargar_vocabulario()
    catalogo = leer_catalogo(archivo_verificado(cat))
    return ent, vocab, catalogo, generar_candidatos(ent.grupos, vocab)


def _miles(n: int) -> str:
    return f"{n:,}"


def imprimir_candidatos(res: Mapping) -> None:
    pg = res["palabras_gramaticales"]
    print(f"Palabras gramaticales que no cuentan en la fuente (ii), {len(pg['originales']) + len(pg['agregadas'])} en total:")
    print(f"  lista original ({len(pg['originales'])}): {', '.join(pg['originales'])}")
    print(f"  agregadas ({len(pg['agregadas'])}): {', '.join(pg['agregadas'])}")
    e = res["elegibles"]
    print(f"Llaves de train candidato: {_miles(e['llaves'])} | fuera por SIN NOMBRE: {_miles(e['excluidas_sin_nombre'])} "
          f"| fuera por igual a título: {_miles(e['excluidas_igual_titulo'])}")
    print(f"  fuente (i) elegibles (2 a {MAX_FILAS_LLAVE} filas, clase más común <= {MAX_PCT_CLASE} %): {_miles(e['i'])} "
          f"(genéricas: {_miles(e['i_genericas'])}); fuera por más de {MAX_FILAS_LLAVE} filas: {_miles(e['i_fuera_por_mas_de_20_filas'])}")
    print(f"  fuente (ii) elegibles con la lista final: {_miles(e['ii'])} | con la lista original: "
          f"{_miles(e['ii_con_la_lista_original'])} | sin ignorar palabras: {_miles(e['ii_sin_ignorar_palabras'])} "
          f"| fuera por tener solo palabras gramaticales: {_miles(e['ii_solo_palabras_gramaticales'])}")
    for n, r in enumerate(res["rondas"], 1):
        print(f"Ronda {n}: cuotas i {r['i']}, ii {r['ii']}; tope {TOPE_SECTOR} por sector y fuente")
    of, rec = res["ofrecidos"], res["posiciones_recorridas"]
    print(f"  candidatos: i {of['i']}, ii {of['ii']} | posiciones recorridas de la permutación: "
          f"i {_miles(rec['i'])} de {_miles(e['i'])}, ii {_miles(rec['ii'])} de {_miles(e['ii'])}")
    print("Candidatos por sector (2 dígitos): sector, i, ii, total")
    for s, c in res["por_sector"].items():
        print(f"  {s:>6} {c['i']:>5} {c['ii']:>5} {c['i'] + c['ii']:>6}")
    print("Filas por llave de los candidatos: " + ", ".join(f"{t}: {n}" for t, n in res["filas_por_llave"].items()))
    print(f"Genéricos: i {res['genericos']['i']}, ii {res['genericos']['ii']} | clases distintas: {res['clases']}")
    p, c = res["peor_caso"], res["cota_garantizada"]
    print(f"Regla 7: ya excluidas {_miles(p['excluidas_por_test_y_validacion'])} de {_miles(p['filas_candidatas_train'])} filas candidatas")
    print(f"  peor caso (las {N_HARD} llaves candidatas más grandes): {_miles(p['excluidas_por_hard_set'])} filas salen de train "
          f"-> {_miles(p['filas_excluidas'])} ({p['pct_filas_candidatas']:.2f} %), pasa de {LIMITE_REGLA7_PCT} %: {p['pasa_del_limite']}")
    print(f"  cota garantizada ({N_HARD} x {MAX_FILAS_LLAVE} = {_miles(c['filas'])} filas): {_miles(c['filas_excluidas'])} "
          f"({c['pct_filas_candidatas']:.2f} %), pasa de {LIMITE_REGLA7_PCT} %: {c['pasa_del_limite']}")


def imprimir_hard(res: Mapping) -> None:
    print(f"Hard set: {res['filas']} filas | por tipo: {res['por_tipo']}")
    print("  por fuente y tipo: " + "; ".join(f"{f}: {v}" for f, v in res["por_fuente_y_tipo"].items()))
    print(f"  genéricos por tipo: {res['genericos']}")
    print("  por sector: " + ", ".join(f"{s}: {n}" for s, n in res["por_sector"].items()))
    print(f"  clases: {res['clases']} | por clase: " + ", ".join(f"{c} {n}" for c, n in res["por_clase"].items()))
    r = res["regla7"]
    print(f"Filas que salen de train (todas las de las {res['filas']} llaves): {_miles(res['filas_que_salen_de_train'])}")
    print(f"Regla 7: {_miles(r['filas_excluidas'])} de {_miles(r['filas_candidatas_train'])} filas candidatas excluidas "
          f"({r['pct_filas_candidatas']:.2f} %): test y validación {_miles(r['excluidas_por_test_y_validacion'])}, "
          f"hard set {_miles(r['excluidas_por_hard_set'])}; pasa de {LIMITE_REGLA7_PCT} %: {r['pasa_del_limite']}")
    t = res["train_candidatos_sin_hard"]
    print(f"Train candidato sin hard: {_miles(t['filas'])} filas, {t['clases']} clases")
    print(f"Clases de train que quedarían sin ejemplos: {len(res['clases_de_train_sin_ejemplos'])} {res['clases_de_train_sin_ejemplos']}")
    print(f"Clases del hard set sin ningún ejemplo en train: {len(res['clases_del_hard_sin_ejemplos_en_train'])} "
          f"{res['clases_del_hard_sin_ejemplos_en_train']}")
    for tipo in res["tipos_con_menos_de_50"]:
        print(f"AVISO: el tipo {tipo} queda con {res['por_tipo'][tipo]} filas, menos de {MIN_FILAS_TIPO}.")


def _leer_y_validar(rutas: Sequence[str], ent: Entradas, gen: Generacion) -> tuple[list, list[dict], list[Marca]]:
    marcas, archivos = leer_archivos([Path(r) for r in rutas])
    try:
        incluidas = validar_curacion(marcas, gen.candidatos, ent.llaves_val)
    except ErrorConteo as e:
        ofrecidos = Counter(c.fuente for c in gen.candidatos)
        print(f"Filas con incluir=S: {e.encontradas} de {e.esperadas} pedidas.")
        for f in FUENTES:
            print(f"  fuente {f}: {e.por_fuente.get(f, 0)} con S de {ofrecidos[f]} candidatos")
        if e.encontradas < e.esperadas:
            sys.exit(f"Faltan {e.esperadas - e.encontradas}. Me detengo: propongo una ronda {len(RONDAS) + 1} "
                     "(ver el reporte); se agrega a RONDAS solo con la aprobación de Samuel.")
        sys.exit("Sobran filas con S. Me detengo: decide Samuel cuáles quedan.")
    return incluidas, archivos, marcas


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.exploracion_clasificador import verificar_privacidad
    from src.scian import escribir_manifest, leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="accion", required=True)
    p_cand = sub.add_parser("candidatos", help="PARADA 1; con --escribir, el CSV de curación")
    p_cand.add_argument("--escribir", action="store_true")
    for nombre in ("leer", "congelar"):
        sub.add_parser(nombre).add_argument("csv", nargs="+")
    args = parser.parse_args(argv)

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    versionado = codigo_versionado(SCRIPT)
    escribe = args.accion == "congelar" or (args.accion == "candidatos" and args.escribir)
    if escribe and not (arbol_limpio and versionado):
        sys.exit(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
    manifest = leer_manifest()

    try:
        ent, vocab, catalogo, gen = _cargar_todo(manifest)
    except (ErrorHardSet, ValueError) as e:
        sys.exit(f"Me detengo: {e}")
    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | {SCRIPT} versionado sin cambios: {versionado} "
          f"| numpy {np.__version__}")
    print(f"Origen (SHA-256 verificado antes de parsear): train candidato {_miles(len(ent.regs))} filas, "
          f"validación {_miles(len(ent.llaves_val))} llaves. El test no se abre.")
    res_cand = resumen_candidatos(gen, ent)

    if args.accion == "candidatos":
        imprimir_candidatos(res_cand)
        if res_cand["cota_garantizada"]["pasa_del_limite"] or res_cand["peor_caso"]["pasa_del_limite"]:
            sys.exit("Regla 7: el peor caso pasa de 10 %. No se escribe nada; decide Samuel.")
        ronda = len(RONDAS)
        filas, protegidas = filas_curacion(gen.de_ronda(ronda), ent.grupos, catalogo)
        print(f"CSV de la ronda {ronda}: {len(filas)} filas | celdas con espacio contra fórmulas de Excel: "
              f"{sum(protegidas.values())} {dict(protegidas)}")
        if not args.escribir:
            print("PUNTO DE CONTROL (PARADA 1): no se escribió nada. Para escribir el CSV, candidatos --escribir "
                  "tras la aprobación de Samuel.")
            return
        datos = serializar_csv(filas)
        _, _, catalogo2, gen2 = _cargar_todo(leer_manifest())
        if serializar_csv(filas_curacion(gen2.de_ronda(ronda), ent.grupos, catalogo2)[0]) != datos:
            sys.exit("Generar de nuevo no da los mismos bytes; no se escribe nada.")
        ruta = REPO_RAIZ / "data" / f"{fecha}_fase1_hard-candidatos-r{ronda}.csv"
        try:
            escribir_nuevo(datos, ruta)
        except (ErrorHardSet, FileExistsError) as e:
            sys.exit(f"Me detengo: {e}")
        print(f"CSV escrito: {ruta.relative_to(REPO_RAIZ).as_posix()} ({_miles(len(datos))} bytes, "
              f"SHA-256 {sha256_archivo(ruta)}; ignorado por git: {esta_ignorado(ruta)})")
        return

    try:
        incluidas, archivos, marcas = _leer_y_validar(args.csv, ent, gen)
    except ErrorHardSet as e:
        sys.exit(f"Me detengo: {e}")
    avisos_tipo = sum(m.tipo_en_fila_n for m in marcas)
    sin_fila = len({int(c.id) for c in gen.candidatos} - {int(m.id) for m in marcas})
    print(f"CSV leídos: {len(archivos)} | registros: {len(marcas)} | candidatos sin registro en los CSV: {sin_fila} "
          f"| filas N con algo en tipo: {avisos_tipo}")
    res_hard = resumen_hard(incluidas, ent)
    imprimir_hard(res_hard)
    if res_hard["regla7"]["pasa_del_limite"]:
        sys.exit("Regla 7: la exclusión pasa de 10 %. No se congela; decide Samuel.")
    if args.accion == "leer":
        print("PUNTO DE CONTROL (PARADA 2): no se escribió nada. Para congelar, congelar <csv> tras la aprobación de Samuel.")
        return

    # --- congelar ---
    nuevas = {ENTRADA_HARD, ENTRADA_TRAIN_SIN_HARD}
    rutas = [CARPETA_SPLITS / ARCHIVO_HARD, CARPETA_SPLITS / ARCHIVO_TRAIN_SIN_HARD]
    if nuevas & set(manifest["splits"]) or any(r.exists() for r in rutas):
        sys.exit("Ya hay un hard set en data/splits/ o en el manifest; no se sobrescribe.")
    if not all(esta_ignorado(r) for r in rutas):
        sys.exit("Algún archivo de data/splits/ no está ignorado por git.")
    datos = contenidos(incluidas, ent)
    if serializar_split(ent.regs) != Path(REPO_RAIZ / manifest["splits"][ENTRADA_TRAIN]["ruta"]).read_bytes():
        sys.exit("Volver a serializar train candidato no da sus mismos bytes; me detengo.")
    hashes = {n: sha256_bytes(b) for n, b in datos.items()}
    try:
        candado_hashes(hashes, manifest)
    except ErrorHardSet as e:
        sys.exit(f"Me detengo: {e}")
    entradas = entradas_manifest(datos=datos, incluidas=incluidas, ent=ent, manifest=manifest,
                                 archivos_curacion=archivos, commit=commit, fecha=fecha)
    resultado = armar_resultado(fecha=fecha, commit=commit, arbol_limpio=arbol_limpio, manifest=manifest,
                                hashes=hashes, archivos=archivos, res_cand=res_cand, res_hard=res_hard, entradas=entradas)
    llaves = set(ent.grupos) | ent.llaves_val
    verificar_privacidad({"resultado": resultado, "manifest": entradas}, llaves, set())
    print("Revisión de privacidad del resultado y de las entradas del manifest: ningún texto es una llave.")

    # Regenerar desde cero (manifest, splits, vocabulario, catálogo y CSV releídos) y comparar bytes.
    ent2, _, _, gen2 = _cargar_todo(leer_manifest())
    incluidas2, _, _ = _leer_y_validar(args.csv, ent2, gen2)
    if contenidos(incluidas2, ent2) != datos:
        sys.exit("Generar de nuevo no da los mismos bytes; no se escribe nada.")
    try:
        escritas = escribir_splits(datos, CARPETA_SPLITS)
    except (ErrorHardSet, FileExistsError) as e:
        sys.exit(f"Me detengo: {e}")
    for n, r in escritas.items():
        if sha256_archivo(r) != hashes[n]:
            sys.exit(f"El SHA-256 releído de {n} no coincide con el generado.")
    nuevo = agregar_entradas(manifest, entradas)
    escribir_manifest(nuevo)
    otras_iguales(manifest, leer_manifest(), nuevas)
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_hard-set.json"
    with open(salida, "x", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")
    for k in (ENTRADA_HARD, ENTRADA_TRAIN_SIN_HARD):
        e = nuevo["splits"][k]
        print(f"{e['ruta']}: {_miles(e['filas'])} filas, {_miles(e['tamano_bytes'])} bytes, SHA-256 {e['sha256']}, "
              f"congelado {e['congelado']}")
    print(f"Manifest actualizado (las demás entradas quedaron idénticas); resultados en {salida.relative_to(REPO_RAIZ).as_posix()}")


def armar_resultado(
    *, fecha: str, commit: str, arbol_limpio: bool, manifest: Mapping, hashes: Mapping[str, str],
    archivos: Sequence[Mapping], res_cand: Mapping, res_hard: Mapping, entradas: Mapping[str, Mapping],
) -> dict:
    """JSON de results/ con los campos de la regla 4; solo conteos y códigos SCIAN."""
    from importlib.metadata import version as version_de_paquete

    from src.scian import CATALOGO
    from src.subgrupos import DEFINICIONES, VOCABULARIO

    cand = dict(res_cand)
    cand["palabras_gramaticales"] = {k: ", ".join(v) for k, v in res_cand["palabras_gramaticales"].items()}
    return {
        "fecha": fecha,
        "experimento": "fase1_paso7_hard_set",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": SCRIPT,
            "subgrupos": "src/subgrupos.py",
            "definiciones_subgrupos": DEFINICIONES,
            "procedimiento": PROCEDIMIENTO,
            "n_hard": N_HARD, "rondas": [dict(r) for r in RONDAS], "tope_sector": TOPE_SECTOR,
            "min_filas_llave_i": MIN_FILAS_LLAVE_I, "max_filas_llave": MAX_FILAS_LLAVE, "max_pct_clase": MAX_PCT_CLASE,
            "scian_version": manifest["scian_version"],
            "catalogo_sha256": manifest["catalogos"][CATALOGO[manifest["scian_version"]]]["sha256"],
            "vocabulario_sha256": manifest["catalogos"][VOCABULARIO]["sha256"],
            "python": platform.python_version(),
            "versiones": {"numpy": version_de_paquete("numpy")},
        },
        "semilla": SEMILLA,
        "split": "clasificador: hard set congelado y train candidato sin hard (data/splits/)",
        "hash_split": {"hard": hashes[ARCHIVO_HARD], "train_candidatos_sin_hard": hashes[ARCHIVO_TRAIN_SIN_HARD]},
        "hash_split_origen": "data/manifest.json, splits",
        "hash_origen": entradas[ENTRADA_HARD]["fuentes"],
        "curacion": [dict(a) for a in archivos],
        "tamano_muestra": {"hard": res_hard["filas"], "train_candidatos_sin_hard": res_hard["train_candidatos_sin_hard"]["filas"]},
        "horas_gpu": None,
        "candidatos": cand,
        "hard_set": dict(res_hard),
    }


if __name__ == "__main__":
    main()
