"""Errores del 4B en test por tipo (Fase 4, paso 18): vista, propuestas de Claude Code y conteo tras la revisión.

Criterios: docs/criterios-errores.md, con su commit antes de clasificar. Los 50 errores son los del paso 17
(data/fase4/errores_4b-lr2e-4-r32_test.jsonl, con el SHA-256 registrado en el resultado de la evaluación final).
CAMBIO DE DISEÑO D7, aprobado por Samuel el 2026-11-25: Claude Code propone un tipo por fila y Samuel confirma o
cambia cada una. Cada fila se identifica por su posición (1 a n, en orden de int(id)): los ids no salen de la laptop.

Modos:
  .venv/Scripts/python -m src.errores_fase4 --vista
      Escribe data/fase4/errores_4b_vista.md: una ficha por posición, sin ids, con el tipo automático o "por
      decidir". Es para que Claude Code la lea con su herramienta de lectura; en la terminal, solo conteos.
  .venv/Scripts/python -m src.errores_fase4 --proponer
      Valida data/fase4/errores_4b_propuestas.json y escribe data/fase4/errores_4b_para_revision.csv (UTF-8 con
      BOM, saltos de línea \\n): posicion en lugar de id, las demás columnas del paso 17, y tipo_propuesto, razon,
      tipo_samuel y nota_samuel (vacías) y candidato_charla.
  .venv/Scripts/python -m src.errores_fase4 --contar RUTA
      Lee el CSV revisado por Samuel (exportado de Google Sheets) y escribe una sola vez
      results/<fecha>_fase4_errores-por-tipo.json, sin nombres, llaves ni ids.

Ningún modo imprime nombres, llaves, ids ni respuestas; los mensajes citan posiciones y columnas, nunca valores.
El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import platform
import re
import sys
import traceback
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import src.baselines as bl
from src.baselines_fase4 import EstadoGit, ErrorFase4, _git, comprobar_head
from src.denue import REPO_RAIZ
from src.evaluacion_final_fase4 import COLUMNAS_ERRORES, ErrorFinal, candados
from src.exploracion_clasificador import ErrorPrivacidad
from src.metricas_clasificador import CODIGO_EXISTENTE
from src.muestra_humo import ErrorMuestra, codigo_versionado, leer_verificado
from src.normalizacion import normalizar_nombre
from src.prompt_clasificador import RE_CODIGO
from src.splits import esta_ignorado, sha256_bytes

SCRIPT = "src/errores_fase4.py"
EXPERIMENTO = "fase4_paso18_errores_por_tipo"
MODELO = "4b-lr2e-4-r32"
CONJUNTO = "test"
SEMILLA = 42
RESULTADO_FINAL = "results/2026-11-25_fase4_evaluacion-final.json"
RUTA_CRITERIOS = "docs/criterios-errores.md"
CARPETA = "data/fase4"
ERRORES = f"{CARPETA}/errores_4b-lr2e-4-r32_test.jsonl"
VISTA = f"{CARPETA}/errores_4b_vista.md"
PROPUESTAS = f"{CARPETA}/errores_4b_propuestas.json"
PARA_REVISION = f"{CARPETA}/errores_4b_para_revision.csv"
PATRON_CONTEO = "*_fase4_errores-por-tipo.json"
ENTRADA_TEST = "clasificador_test"

TIPOS = ("codigo_inexistente", "clases_hermanas", "etiqueta_dudosa", "nombre_ambiguo", "otro")
AUTOMATICOS = TIPOS[:2]
DE_JUICIO = TIPOS[2:]
NIVELES_HERMANAS = ("rama", "subrama")
MAX_PALABRAS_RAZON = 15
MAX_CANDIDATOS = 6
SI = "sí"
CLAVES_PROPUESTA = ("posicion", "tipo", "razon", "candidato_charla")
COLUMNAS_NUEVAS = ("tipo_propuesto", "razon", "tipo_samuel", "nota_samuel", "candidato_charla")
assert COLUMNAS_ERRORES[0] == "id"
COLUMNAS_REVISION = ("posicion", *COLUMNAS_ERRORES[1:], *COLUMNAS_NUEVAS)
COLUMNAS_LEIDAS = ("posicion", "tipo_propuesto", "tipo_samuel", "nota_samuel")
DECIMALES = 2

AUTORIA = ("El chat de Claude del proyecto propuso el paso y los tipos; Samuel lo aprobó. Claude Code redactó los "
           "criterios y el código y propuso un tipo por fila; Samuel confirmó o cambió cada fila.")
CAMBIO_DE_DISENO = ("CAMBIO DE DISEÑO D7, aprobado por Samuel el 2026-11-25: Claude Code propone un tipo por fila con "
                    "criterios escritos y versionados antes de clasificar, y Samuel confirma o cambia cada una. Sesgo "
                    "de anclaje: Samuel vio la propuesta antes de decidir, así que los conteos son de la clasificación "
                    "revisada, no de una independiente; la matriz propuesto_a_final muestra cuánto cambió.")
ADVERTENCIAS = (
    "Muestra de 50 de los 568 errores de clase del 4B en test, sorteada con la semilla 42 (paso 17): los porcentajes "
    "por tipo tienen mucha incertidumbre.",
    "Los tipos de juicio (etiqueta_dudosa, nombre_ambiguo y otro) los propuso un modelo de lenguaje (Claude Code) con "
    "los criterios versionados, y Samuel los revisó viendo la propuesta.",
    "Los tipos codigo_inexistente y clases_hermanas son automáticos (códigos), y Samuel no los cambia.",
)


class ErrorErrores(ValueError):
    """PARADA del paso 18; el mensaje nunca incluye nombres, llaves, ids ni respuestas."""


ERRORES_PROPIOS = (ErrorErrores, ErrorMuestra, ErrorFinal, ErrorPrivacidad, ErrorFase4)


# --- Entradas ---


def sha_criterios(datos: bytes) -> str:
    """SHA-256 de los criterios con saltos de línea \\n (git puede dejar CRLF en la copia de trabajo)."""
    return sha256_bytes(datos.replace(b"\r\n", b"\n"))


def leer_criterios(raiz: Path, versionado: Callable[[str], bool] = codigo_versionado) -> str:
    if not versionado(RUTA_CRITERIOS):
        raise ErrorErrores(f"{RUTA_CRITERIOS} debe estar en git y sin cambios: los criterios se fijan antes de clasificar.")
    return sha_criterios((Path(raiz) / RUTA_CRITERIOS).read_bytes())


def cargar_errores(raiz: Path) -> tuple[list[dict], dict]:
    """Los errores del paso 17, verificados contra el SHA-256 del resultado final, en orden de int(id)."""
    datos_res = (Path(raiz) / RESULTADO_FINAL).read_bytes()
    res = json.loads(datos_res.decode("utf-8"))
    ea = res["errores_para_analisis"]
    archivo = ea["archivos"]["jsonl"]
    if archivo["ruta"] != ERRORES or ea["modelo"] != MODELO or ea["conjunto"] != CONJUNTO:
        raise ErrorErrores(f"{RESULTADO_FINAL} no registra los errores esperados ({ERRORES}, {MODELO}, {CONJUNTO}).")
    datos = leer_verificado(Path(raiz) / ERRORES, archivo["sha256"])
    filas = []
    for linea in datos.decode("utf-8").split("\n"):
        if linea:
            filas.append(json.loads(linea))
    if len(filas) != ea["n"]:
        raise ErrorErrores(f"{ERRORES} tiene {len(filas)} filas, no las {ea['n']} registradas.")
    for i, f in enumerate(filas, 1):
        ok = (tuple(f) == COLUMNAS_ERRORES and isinstance(f["id"], str) and f["id"].isdigit()
              and isinstance(f["acierta_sector"], bool))
        if not ok:
            raise ErrorErrores(f"La fila {i} de {ERRORES} no tiene las columnas o los tipos esperados.")
    ids = [f["id"] for f in filas]
    if len(set(ids)) != len(ids):
        raise ErrorErrores(f"{ERRORES} repite ids.")
    filas.sort(key=lambda f: int(f["id"]))
    info = {"n_errores": ea["n_errores"], "n": ea["n"], "sha256": archivo["sha256"],
            "sha256_resultado_final": sha256_bytes(datos_res), "hash_split_test": res["hash_split"][CONJUNTO]}
    return filas, info


def tipo_automatico(f: Mapping, posicion: int) -> str | None:
    """a (codigo_inexistente) o b (clases_hermanas) de los criterios; None si el tipo es de juicio."""
    if f["categoria_regla5"] != CODIGO_EXISTENTE:
        return "codigo_inexistente"
    resp, etq = f["codigo_respondido"], f["codigo_act"]
    if not RE_CODIGO.fullmatch(resp) or resp == etq:
        raise ErrorErrores(f"Posición {posicion}: un código existente debe ser de 6 dígitos y distinto de codigo_act.")
    hermanas = resp[:4] == etq[:4]
    if hermanas != (f["nivel_mas_fino_acertado"] in NIVELES_HERMANAS):
        raise ErrorErrores(f"Posición {posicion}: la rama en común no coincide con nivel_mas_fino_acertado.")
    return "clases_hermanas" if hermanas else None


def automaticos(filas: Sequence[Mapping]) -> list[str | None]:
    return [tipo_automatico(f, i) for i, f in enumerate(filas, 1)]


# --- Vista (para Claude Code) ---


def _si_no(x: bool) -> str:
    return "sí" if x else "no"


def _linea(texto: str) -> str:
    return " ".join(str(texto).split())


def texto_vista(filas: Sequence[Mapping]) -> str:
    L = ["# Vista de los errores del 4B en test (paso 18)", "",
         "Sin ids: la posición es el orden de int(id). Ignorada por git: lleva nombres.", ""]
    for i, (f, t) in enumerate(zip(filas, automaticos(filas)), 1):
        if f["categoria_regla5"] == CODIGO_EXISTENTE:
            resp = f"{f['codigo_respondido']} {f['titulo_respondido']}"
        else:
            resp = f"{f['categoria_regla5']}: {json.dumps(f['respuesta'], ensure_ascii=False)}"
        L += [f"## {i}",
              f"- nombre: {_linea(f['nom_estab'])}",
              f"- DENUE: {f['codigo_act']} {f['titulo_act']}",
              f"- respuesta: {resp}",
              f"- nivel más fino acertado: {f['nivel_mas_fino_acertado']}; acierta sector: {_si_no(f['acierta_sector'])}",
              f"- marcas: sin_nombre {_si_no(f['sin_nombre'])}, generico {_si_no(f['generico'])}, "
              f"igual_titulo {_si_no(f['igual_titulo'])}",
              f"- tipo automático: {t or 'por decidir (etiqueta_dudosa, nombre_ambiguo u otro)'}", ""]
    return "\n".join(L)


# --- Propuestas y CSV para revisión ---


def validar_propuestas(prop: Mapping, filas: Sequence[Mapping], sha_crit: str) -> list[dict]:
    """Las propuestas de Claude Code, por posición; ErrorErrores ante cualquier incumplimiento de los criterios."""
    if not isinstance(prop, Mapping) or set(prop) != {"criterios_sha256", "filas"}:
        raise ErrorErrores("Las propuestas deben tener exactamente criterios_sha256 y filas.")
    if prop["criterios_sha256"] != sha_crit:
        raise ErrorErrores(f"Las propuestas no se hicieron con los criterios versionados ({RUTA_CRITERIOS}).")
    props = prop["filas"]
    if not isinstance(props, list) or len(props) != len(filas):
        raise ErrorErrores(f"Las propuestas deben ser una lista de {len(filas)} filas.")
    autos = automaticos(filas)
    res = []
    for i, (p, f, auto) in enumerate(zip(props, filas, autos), 1):
        if not isinstance(p, Mapping) or set(p) != set(CLAVES_PROPUESTA):
            raise ErrorErrores(f"Posición {i}: la propuesta debe tener exactamente {list(CLAVES_PROPUESTA)}.")
        pos, tipo, razon, cand = (p[k] for k in CLAVES_PROPUESTA)
        if type(pos) is not int or pos != i:
            raise ErrorErrores(f"Fila {i} de las propuestas: posicion debe ser {i} (orden de int(id)).")
        if tipo not in TIPOS:
            raise ErrorErrores(f"Posición {i}: el tipo no es uno de {list(TIPOS)}.")
        if auto and tipo != auto:
            raise ErrorErrores(f"Posición {i}: el tipo automático es {auto} y la propuesta dice otro.")
        if not auto and tipo in AUTOMATICOS:
            raise ErrorErrores(f"Posición {i}: {tipo} es automático y no aplica a esta fila.")
        if not isinstance(razon, str) or not razon.strip() or "\n" in razon or "\r" in razon:
            raise ErrorErrores(f"Posición {i}: la razón debe ser un texto de una línea, no vacío.")
        if len(razon.split()) > MAX_PALABRAS_RAZON:
            raise ErrorErrores(f"Posición {i}: la razón pasa de {MAX_PALABRAS_RAZON} palabras.")
        llave = normalizar_nombre(f["nom_estab"])
        if llave and f" {llave} " in f" {normalizar_nombre(razon)} ":
            raise ErrorErrores(f"Posición {i}: la razón contiene la llave del nombre de su fila.")
        if not isinstance(cand, bool):
            raise ErrorErrores(f"Posición {i}: candidato_charla debe ser true o false.")
        res.append({"tipo": tipo, "razon": razon.strip(), "candidato_charla": cand})
    tipos_cand = [r["tipo"] for r in res if r["candidato_charla"]]
    if len(tipos_cand) > MAX_CANDIDATOS:
        raise ErrorErrores(f"Hay {len(tipos_cand)} candidatos para la charla; el máximo es {MAX_CANDIDATOS}.")
    if len(set(tipos_cand)) != len(tipos_cand):
        raise ErrorErrores("Los candidatos para la charla deben ser de tipos distintos.")
    return res


def filas_revision(filas: Sequence[Mapping], props: Sequence[Mapping]) -> list[dict]:
    return [{"posicion": i, **{c: f[c] for c in COLUMNAS_ERRORES[1:]}, "tipo_propuesto": p["tipo"],
             "razon": p["razon"], "tipo_samuel": "", "nota_samuel": "",
             "candidato_charla": SI if p["candidato_charla"] else ""}
            for i, (f, p) in enumerate(zip(filas, props), 1)]


def bytes_csv(filas: Sequence[Mapping]) -> bytes:
    """UTF-8 con BOM (para Google Sheets) y saltos de línea \\n, columnas COLUMNAS_REVISION en orden."""
    for f in filas:
        if tuple(f) != COLUMNAS_REVISION:
            raise ErrorErrores("Una fila para revisión no tiene las columnas esperadas en orden.")
    texto = io.StringIO()
    w = csv.DictWriter(texto, fieldnames=COLUMNAS_REVISION, lineterminator="\n")
    w.writeheader()
    w.writerows(filas)
    return texto.getvalue().encode("utf-8-sig")


# --- Lectura del CSV revisado ---


def leer_csv(datos: bytes, columnas: Sequence[str], que: str) -> list[dict]:
    """Solo `columnas` de un CSV con o sin BOM, con \\n o \\r\\n y quizá con columnas extra (exportado de Sheets).
    Omite las filas vacías por completo."""
    try:
        texto = datos.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ErrorErrores(f"El {que} no es UTF-8.") from e
    lector = csv.DictReader(io.StringIO(texto, newline=""))
    encabezado = lector.fieldnames or []
    faltan = [c for c in columnas if c not in encabezado]
    if faltan or len(set(encabezado)) != len(encabezado):
        raise ErrorErrores(f"Al {que} le faltan las columnas {faltan} o repite alguna.")
    res = []
    for r in lector:
        if all(not (v or "").strip() for v in r.values() if isinstance(v, str)):
            continue
        res.append({c: (r.get(c) or "") for c in columnas})
    return res


def normalizar_tipo(t: str) -> str:
    """Sin acentos, en minúsculas y con "_" en lugar de espacios o guiones: "Nombre ambiguo" → nombre_ambiguo."""
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "_", t.strip().lower())


def por_posicion(filas: Sequence[Mapping], n: int, que: str) -> dict[int, Mapping]:
    """Las filas por su posición; exige exactamente 1..n, sin repetir (el orden no importa)."""
    res: dict[int, Mapping] = {}
    malas, repetidas = [], []
    for k, f in enumerate(filas, 1):
        p = f["posicion"].strip()
        if not (p.isascii() and p.isdigit()) or not 1 <= int(p) <= n:
            malas.append(k)
        elif int(p) in res:
            repetidas.append(int(p))
        else:
            res[int(p)] = f
    faltan = sorted(set(range(1, n + 1)) - set(res))
    if malas or repetidas or faltan:
        raise ErrorErrores(f"El {que} debe tener exactamente las posiciones 1 a {n}: filas del CSV con posición "
                           f"inválida {malas}, posiciones repetidas {sorted(set(repetidas))}, faltan {faltan}.")
    return res


def validar_revision(revisado: Sequence[Mapping], propios: Sequence[Mapping], autos: Sequence[str | None]) -> list[dict]:
    """Por posición: tipo_propuesto, tipo_samuel (normalizado o vacío) y si hay nota. Respeta los automáticos."""
    n = len(autos)
    rev = por_posicion(revisado, n, "CSV revisado")
    prop = por_posicion(propios, n, "CSV para revisión")
    editadas, invalidas, automaticas = [], [], []
    res = []
    for i in range(1, n + 1):
        tp = prop[i]["tipo_propuesto"].strip()
        if rev[i]["tipo_propuesto"].strip() != tp:
            editadas.append(i)
        ts = normalizar_tipo(rev[i]["tipo_samuel"])
        if ts and ts not in TIPOS:
            invalidas.append(i)
        auto = autos[i - 1]
        if ts in TIPOS and ((auto and ts != auto) or (not auto and ts in AUTOMATICOS)):
            automaticas.append(i)
        res.append({"tipo_propuesto": tp, "tipo_samuel": ts, "tiene_nota": bool(rev[i]["nota_samuel"].strip())})
    if editadas or invalidas or automaticas:
        raise ErrorErrores(
            f"CSV revisado: tipo_propuesto editado en las posiciones {editadas}; tipo_samuel que no es uno de "
            f"{list(TIPOS)} en {invalidas}; tipo automático asignado o cambiado en {automaticas} (los automáticos no "
            "se cambian: la otra lectura va en nota_samuel).")
    return res


# --- Conteo y resultado ---


def _pct(x: int, n: int) -> float:
    return round(100 * x / n, DECIMALES) if n else 0.0


def contar(filas: Sequence[Mapping], revision: Sequence[Mapping]) -> dict:
    n = len(filas)
    final = [r["tipo_samuel"] or r["tipo_propuesto"] for r in revision]
    sector = [f["acierta_sector"] for f in filas]
    por_tipo = {}
    for t in TIPOS:
        filas_t = [i for i, x in enumerate(final) if x == t]
        s = sum(sector[i] for i in filas_t)
        por_tipo[t] = {"filas": len(filas_t), "pct": _pct(len(filas_t), n), "sector_acertado": s,
                       "pct_sector_acertado_del_tipo": _pct(s, len(filas_t))}
    matriz = {p: {t: 0 for t in TIPOS} for p in TIPOS}
    for r, t in zip(revision, final):
        matriz[r["tipo_propuesto"]][t] += 1
    propuestos = Counter(r["tipo_propuesto"] for r in revision)
    return {
        "por_tipo_final": por_tipo,
        "total": {"filas": n, "sector_acertado": sum(sector), "pct_sector_acertado": _pct(sum(sector), n)},
        "por_tipo_propuesto": {t: propuestos.get(t, 0) for t in TIPOS},
        "revision_de_samuel": {
            "cambiadas": sum(1 for r in revision if r["tipo_samuel"] and r["tipo_samuel"] != r["tipo_propuesto"]),
            "confirmadas_con_el_mismo_tipo": sum(1 for r in revision if r["tipo_samuel"] == r["tipo_propuesto"]),
            "sin_tocar": sum(1 for r in revision if not r["tipo_samuel"]),
            "con_nota": sum(1 for r in revision if r["tiene_nota"]),
        },
        "propuesto_a_final": matriz,
    }


def armar_resultado(*, fecha: str, git: EstadoGit, info: Mapping, criterios: Mapping, fuentes: Mapping,
                    conteo: Mapping) -> dict:
    return {
        "fecha": fecha,
        "experimento": EXPERIMENTO,
        "commit": git.commit,
        "arbol_limpio": git.arbol_limpio,
        "head_sin_cambios": True,
        "modelo": MODELO,
        "configuracion": {
            "script": SCRIPT,
            "criterios": {"ruta": RUTA_CRITERIOS, "sha256": criterios["sha256"], "commit": criterios["commit"],
                          "sha256_de": "los bytes con saltos de línea \\n"},
            "tipos": list(TIPOS),
            "automaticos": list(AUTOMATICOS),
            "precedencia": "En el orden de tipos: la fila recibe el primero que aplica.",
            "tipo_final": "tipo_samuel si no está vacío; si no, tipo_propuesto.",
            "python": platform.python_version(),
        },
        "semilla": SEMILLA,
        "semilla_uso": "La del sorteo de los errores en el paso 17; el conteo no sortea nada.",
        "split": CONJUNTO,
        "hash_split": {CONJUNTO: info["hash_split_test"]},
        "hash_split_origen": f"data/manifest.json (splits.{ENTRADA_TEST})",
        "tamano_muestra": {"errores_del_modelo_en_test": info["n_errores"], "muestra": info["n"]},
        "horas_gpu": 0.0,
        "fuentes": fuentes,
        "conteos": conteo,
        "autoria": AUTORIA,
        "cambio_de_diseno": CAMBIO_DE_DISENO,
        "advertencias": list(ADVERTENCIAS),
    }


# --- Corridas ---


def estado_git() -> EstadoGit:
    return EstadoGit(commit=_git("rev-parse", "HEAD"),
                     arbol_limpio=_git("status", "--porcelain", "--untracked-files=no") == "",
                     versionado=codigo_versionado(SCRIPT))


def commit_de_criterios() -> str:
    """El commit que agregó los criterios."""
    return _git("log", "--diff-filter=A", "--format=%H", "--", RUTA_CRITERIOS).split("\n")[-1]


def referencias_reales(raiz: Path) -> dict[str, list[dict]]:
    """Los 4 conjuntos que vio el kernel (copias de test y hard sin etiquetas, train final y validación)."""
    from src.salida_fase4 import referencias
    from src.scian import leer_manifest

    return referencias(leer_manifest(), raiz)


def manifest_real() -> dict:
    from src.scian import leer_manifest

    return leer_manifest()


def escribir_en_data(datos: bytes, ruta: Path, ignorado: Callable[[Path], bool]) -> None:
    if not ignorado(ruta):
        raise ErrorErrores(f"{ruta.name} no está ignorado por git; no se escribe.")
    bl.escribir_atomico(datos, ruta, temporal_en=ruta.parent)


def correr_vista(raiz: Path = REPO_RAIZ, ignorado: Callable[[Path], bool] = esta_ignorado,
                 avance: Callable[[str], None] = print) -> dict:
    filas, _ = cargar_errores(raiz)
    autos = automaticos(filas)
    datos = texto_vista(filas).encode("utf-8")
    escribir_en_data(datos, Path(raiz) / VISTA, ignorado)
    c = Counter(a or "por_decidir" for a in autos)
    avance(f"Vista escrita en {VISTA}: {len(filas)} fichas, sin ids; automáticas: codigo_inexistente "
           f"{c['codigo_inexistente']}, clases_hermanas {c['clases_hermanas']}; por decidir: {c['por_decidir']}.")
    return dict(c)


def correr_proponer(raiz: Path = REPO_RAIZ, ignorado: Callable[[Path], bool] = esta_ignorado,
                    versionado: Callable[[str], bool] = codigo_versionado,
                    avance: Callable[[str], None] = print) -> dict:
    filas, _ = cargar_errores(raiz)
    sha_crit = leer_criterios(raiz, versionado)
    try:
        prop = json.loads((Path(raiz) / PROPUESTAS).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ErrorErrores(f"No pude leer {PROPUESTAS} como JSON ({type(e).__name__}).") from e
    props = validar_propuestas(prop, filas, sha_crit)
    datos = bytes_csv(filas_revision(filas, props))
    escribir_en_data(datos, Path(raiz) / PARA_REVISION, ignorado)
    c = Counter(p["tipo"] for p in props)
    cand = sum(p["candidato_charla"] for p in props)
    resumen = {"por_tipo_propuesto": {t: c.get(t, 0) for t in TIPOS}, "candidatos": cand, "bytes": len(datos),
               "sha256": sha256_bytes(datos)}
    avance(f"Escrito {PARA_REVISION}: {len(props)} filas, {len(datos):,} bytes, SHA-256 {resumen['sha256']}.")
    avance("  Tipos propuestos: " + ", ".join(f"{t} {k}" for t, k in resumen["por_tipo_propuesto"].items())
           + f"; candidatos para la charla: {cand}.")
    return resumen


def existentes(raiz: Path) -> list[str]:
    return [p.name for p in sorted((Path(raiz) / "results").glob(PATRON_CONTEO))]


def preparar_conteo(raiz: Path, datos_revisado: bytes, *, fecha: str, git: EstadoGit, manifest: Mapping,
                    versionado: Callable[[str], bool], commit_criterios: Callable[[], str]) -> bytes:
    """El resultado en bytes, desde los archivos (sin escribir)."""
    filas, info = cargar_errores(raiz)
    if manifest["splits"][ENTRADA_TEST]["sha256"] != info["hash_split_test"]:
        raise ErrorErrores(f"El SHA-256 de test del manifest no es el de {RESULTADO_FINAL}.")
    autos = automaticos(filas)
    sha_crit = leer_criterios(raiz, versionado)
    prop = json.loads((Path(raiz) / PROPUESTAS).read_text(encoding="utf-8"))
    props = validar_propuestas(prop, filas, sha_crit)
    datos_propios = (Path(raiz) / PARA_REVISION).read_bytes()
    propios = leer_csv(datos_propios, COLUMNAS_LEIDAS, "CSV para revisión")
    if [f["posicion"] for f in propios] != [str(i) for i in range(1, len(filas) + 1)] or \
            [f["tipo_propuesto"] for f in propios] != [p["tipo"] for p in props]:
        raise ErrorErrores(f"{PARA_REVISION} no corresponde a {PROPUESTAS}.")
    revisado = leer_csv(datos_revisado, COLUMNAS_LEIDAS, "CSV revisado")
    revision = validar_revision(revisado, propios, autos)
    fuentes = {
        "evaluacion_final": {"ruta": RESULTADO_FINAL, "sha256": info["sha256_resultado_final"]},
        "errores": {"ruta": ERRORES, "sha256": info["sha256"], "verificado_contra": RESULTADO_FINAL},
        "para_revision": {"ruta": PARA_REVISION, "bytes": len(datos_propios), "sha256": sha256_bytes(datos_propios)},
        "revisado": {"bytes": len(datos_revisado), "sha256": sha256_bytes(datos_revisado),
                     "nota": "Exportado de Google Sheets por Samuel; su ruta no se registra."},
    }
    resultado = armar_resultado(fecha=fecha, git=git, info=info,
                                criterios={"sha256": sha_crit, "commit": commit_criterios()},
                                fuentes=fuentes, conteo=contar(filas, revision))
    return bl.a_bytes(resultado)


def correr_contar(ruta_revisado: Path, *, raiz: Path = REPO_RAIZ, fecha: str,
                  git: Callable[[], EstadoGit] = estado_git, manifest: Mapping | None = None,
                  referencias: Callable[[Path], Mapping[str, Sequence[Mapping]]] = referencias_reales,
                  versionado: Callable[[str], bool] = codigo_versionado,
                  commit_criterios: Callable[[], str] = commit_de_criterios,
                  avance: Callable[[str], None] = print) -> dict:
    """El conteo, una sola vez: escribe results/<fecha>_fase4_errores-por-tipo.json."""
    hay = existentes(raiz)
    if hay:
        raise ErrorErrores(f"Ya existe {hay}: el conteo ya se hizo; no se repite ni se sobrescribe.")
    inicial = git()
    if not (inicial.arbol_limpio and inicial.versionado):
        raise ErrorErrores(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
    manifest = manifest if manifest is not None else manifest_real()
    datos_revisado = Path(ruta_revisado).read_bytes()
    kw = dict(fecha=fecha, git=inicial, manifest=manifest, versionado=versionado, commit_criterios=commit_criterios)
    a, b = preparar_conteo(raiz, datos_revisado, **kw), preparar_conteo(raiz, datos_revisado, **kw)
    if a != b:
        raise ErrorErrores("El doble cálculo no da los mismos bytes; no se escribe nada.")
    resultado = json.loads(a)
    refs = referencias(raiz)
    filas_nombres = [r for c in refs.values() for r in c]
    llaves = {k for k in (normalizar_nombre(r["nom_estab"]) for r in filas_nombres) if k}
    ids = {r["id"] for r in filas_nombres}
    rev = candados(resultado, {"resultado": a.decode("utf-8")}, llaves, ids, filas_nombres)
    avance(f"Candados limpios: verificar_privacidad ({len(llaves):,} llaves), sin ids ({len(ids):,}) y regla acotada "
           f"contra los {len(refs)} conjuntos ({sum(c['llaves_una_palabra_con_letras'] for c in rev['archivos'].values())} "
           "coincidencias de una palabra, solo contadas).")
    comprobar_head(git, inicial, avance)
    ruta = Path(raiz) / "results" / f"{fecha}_fase4_errores-por-tipo.json"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    bl.escribir_atomico(a, ruta, temporal_en=Path(raiz) / CARPETA)
    avance(f"Escrito results/{ruta.name}: {len(a):,} bytes, SHA-256 {sha256_bytes(a)}.")
    return resultado


# --- Terminal ---


def imprimir_conteo(r: Mapping) -> None:
    c = r["conteos"]
    print(f"Errores por tipo final ({c['total']['filas']} filas; aciertan el sector {c['total']['sector_acertado']}):")
    for t, x in c["por_tipo_final"].items():
        print(f"  {t}: {x['filas']} ({x['pct']} %); aciertan el sector {x['sector_acertado']}")
    s = c["revision_de_samuel"]
    print(f"Revisión de Samuel: cambiadas {s['cambiadas']}, confirmadas con el mismo tipo "
          f"{s['confirmadas_con_el_mismo_tipo']}, sin tocar {s['sin_tocar']}, con nota {s['con_nota']}.")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--vista", action="store_true", help="escribe la vista sin ids para Claude Code")
    grupo.add_argument("--proponer", action="store_true", help="valida las propuestas y escribe el CSV para Samuel")
    grupo.add_argument("--contar", metavar="RUTA", help="cuenta el CSV revisado por Samuel (una sola vez)")
    args = parser.parse_args(argv)
    try:
        if args.vista:
            correr_vista()
        elif args.proponer:
            correr_proponer()
        else:
            imprimir_conteo(correr_contar(Path(args.contar), fecha=datetime.date.today().isoformat()))
    except ERRORES_PROPIOS as e:
        sys.exit(f"ME DETENGO: {e}")
    except Exception as e:  # noqa: BLE001 (sin su mensaje, que podría traer un id, un nombre o una respuesta)
        print(f"ME DETENGO por un error inesperado {type(e).__name__} (sin su mensaje). Marcos:")
        for m in traceback.extract_tb(e.__traceback__):
            print(f"  {Path(m.filename).name}:{m.lineno} en {m.name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
