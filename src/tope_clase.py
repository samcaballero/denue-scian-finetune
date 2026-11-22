"""Tope por clase y train final del clasificador (Fase 1, paso 8).

El tope evita que las clases grandes ahoguen a la cola larga del train; validación y test
conservan la distribución natural (CLAUDE.md, sección c, reglas 1, 3, 4 y 8).

Procedimiento (semilla 42):

1. Origen: data/splits/train_candidatos_sin_hard.jsonl, con su SHA-256 verificado antes de
   parsearlo (`leer_verificado` y `parsear` de src/muestra_humo.py, que además rechaza un
   archivo que no esté ordenado por int(id)).
2. Las filas se agrupan por codigo_act; dentro de cada clase van en orden de int(id) y las
   clases se recorren en orden ascendente de código.
3. rng = numpy.random.default_rng(42). En cada clase, aunque no se recorte,
   perm = rng.permutation(n_clase), y se conservan las filas de sort(perm[:min(n_clase, tope)]).
   Todas las clases consumen el generador igual con cualquier tope: el sorteo de una clase no
   depende del tope, y los trains de topes distintos quedan anidados.
4. Train final: las filas conservadas, en orden de int(id), con los campos de src/splits.py.

Punto de control (PARADA 1): para cada tope, filas, llaves, clases, clases con menos de 10
filas, clases recortadas, % de filas de la clase más grande, horas de GPU por época (tasas de
results/2026-11-21_fase2_prueba-humo.json), distancia de variación total contra validación y
contra train sin tope, y % de filas de validación cuya clase queda recortada.

Uso:
  .venv/Scripts/python -m src.tope_clase                   # PARADA 1: solo conteos, no escribe
  .venv/Scripts/python -m src.tope_clase --topes 300 400   # lo mismo, con otros topes
  .venv/Scripts/python -m src.tope_clase --escribir        # train_final.jsonl, manifest y results/

--escribir se niega mientras TOPE no esté fijado en el código (lo deciden Samuel y el chat de
Claude del proyecto). Nunca abre test.jsonl ni hard.jsonl ni construye su ruta: de ellos solo
usa el SHA-256 del manifest. En la terminal solo van conteos y códigos SCIAN, nunca nombres,
llaves ni ids.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np

from src.denue import REPO_RAIZ, sha256_archivo
from src.hard_set import ErrorHardSet, candado_hashes, escribir_nuevo, otras_iguales
from src.muestra_humo import ErrorMuestra, codigo_versionado, leer_verificado, parsear
from src.splits import CAMPOS, esta_ignorado, sha256_bytes
from src.splits import FORMATO as FORMATO_SPLITS
from src.splits import serializar as serializar_split
from src.subgrupos import CAMPOS_MARCAS

SCRIPT = "src/tope_clase.py"
SEMILLA = 42
TOPES_PUNTO_DE_CONTROL: tuple[int | None, ...] = (100, 200, 500, 1000, None)  # None: sin tope
POR_DECIDIR = "por decidir"
SIN_TOPE = "sin tope"
# Tope aprobado: un entero o SIN_TOPE. Se fija en un commit propio después de la PARADA 1;
# mientras valga POR_DECIDIR, --escribir se niega. 1,000: lo recomendó el chat de Claude del
# proyecto y lo aprobó Samuel el 2026-11-22 (el tope más alto con el que una época del 4B cabe
# en una sesión de Kaggle de 12 h, y el que menos se aleja de la distribución natural).
TOPE: int | str = 1000
MIN_FILAS_CLASE = 10
LIMITE_SESION_KAGGLE_H = 12  # sesiones de CPU y GPU: https://www.kaggle.com/docs/notebooks
FUENTE_LIMITE_SESION = "https://www.kaggle.com/docs/notebooks"
RESULTADO_TASAS = "results/2026-11-21_fase2_prueba-humo.json"
MODELOS = ("qwen3_0.6b", "qwen3_4b")
LOTE = 32

ENTRADA_ORIGEN = "clasificador_train_candidatos_sin_hard"
ENTRADA_VAL = "clasificador_validacion"
ENTRADA_FINAL = "clasificador_train_final"
RUTA_FINAL = "data/splits/train_final.jsonl"
RE_CODIGO = re.compile(r"[0-9]{6}")

DEFINICION_TVD = (
    "Distancia de variación total: la mitad de la suma, sobre todas las clases, de la diferencia absoluta "
    "entre la proporción de filas de la clase en una distribución y en la otra. Es la fracción de filas que "
    "habría que cambiar de clase para que las dos distribuciones fueran iguales: 0 si son idénticas, 1 si no "
    "comparten ninguna clase."
)
DEFINICION_HORAS = (
    "Horas de GPU por época = filas / (ejemplos/s) / 3600, con las tasas de lote 32 medidas en la prueba de punta "
    "a punta; con_costo_fijo suma el costo fijo medido por corrida (instalación, imports, descarga y carga)."
)
NOTA_FINAL = (
    "Congelado: no se regenera ni cambia; es el train de los modelos del clasificador. A diferencia de test y "
    "hard, sí se lee para entrenar y sube a Kaggle reducido a id, nom_estab y codigo_act."
)


class ErrorTope(ValueError):
    """El tope o sus entradas no cumplen una condición; no se escribe nada."""


# --- Tope ---


def validar_tope(tope) -> None:
    """Un tope es None (sin tope) o un entero de 1 en adelante."""
    if tope is None:
        return
    if isinstance(tope, bool) or not isinstance(tope, int) or tope < 1:
        raise ErrorTope(f"Tope inválido: {tope!r}; debe ser un entero de 1 en adelante o sin tope.")


def tope_aprobado(valor=TOPE) -> int | None:
    """El tope fijado en el código: un entero, o None si se aprobó sin tope."""
    if valor == SIN_TOPE:
        return None
    if valor == POR_DECIDIR:
        raise ErrorTope("El tope aún no está fijado en el código (PARADA 1): no se escribe nada.")
    validar_tope(valor)
    return valor


def texto_tope(tope: int | None) -> str:
    return SIN_TOPE if tope is None else f"{tope:,}"


def orden_topes(topes: Iterable[int | None]) -> list[int | None]:
    """Topes sin repetir: los enteros de menor a mayor y al final sin tope."""
    topes = set(topes)
    for t in topes:
        validar_tope(t)
    return sorted(t for t in topes if t is not None) + ([None] if None in topes else [])


def por_clase(regs: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    """Clase → sus filas en orden de int(id); las clases en orden ascendente de código."""
    grupos: dict[str, list[Mapping]] = {}
    for r in regs:
        if not isinstance(r["codigo_act"], str) or not RE_CODIGO.fullmatch(r["codigo_act"]):
            raise ErrorTope("Un codigo_act no es de seis dígitos ASCII.")
        grupos.setdefault(r["codigo_act"], []).append(r)
    return {c: sorted(grupos[c], key=lambda r: int(r["id"])) for c in sorted(grupos)}


def aplicar_tope(regs: Sequence[Mapping], tope: int | None, semilla: int = SEMILLA) -> list[dict]:
    """Filas conservadas con el tope, en orden de int(id) (procedimiento del módulo)."""
    validar_tope(tope)
    ids = [int(r["id"]) for r in regs]
    if len(set(ids)) != len(ids):
        raise ErrorTope("Hay id repetidos en el origen.")
    rng = np.random.default_rng(semilla)
    conservadas: list[Mapping] = []
    for filas in por_clase(regs).values():
        perm = rng.permutation(len(filas))  # en todas las clases, aunque no se recorten
        k = len(filas) if tope is None else min(len(filas), tope)
        conservadas.extend(filas[int(i)] for i in np.sort(perm[:k]))
    return sorted((dict(r) for r in conservadas), key=lambda r: int(r["id"]))


# --- Medidas ---


def distancia_variacion_total(a: Mapping[str, int], b: Mapping[str, int]) -> float:
    """½ Σ |a_c/Na − b_c/Nb| sobre todas las clases, calculada con fracciones exactas."""
    na, nb = sum(a.values()), sum(b.values())
    if na <= 0 or nb <= 0:
        raise ErrorTope("No se puede comparar una distribución vacía.")
    total = sum(abs(Fraction(a.get(c, 0), na) - Fraction(b.get(c, 0), nb)) for c in set(a) | set(b))
    return float(total / 2)


def leer_tasas(ruta: Path = REPO_RAIZ / RESULTADO_TASAS) -> dict[str, dict]:
    """Ejemplos/s con lote 32 y costo fijo por corrida de cada modelo, del resultado de la prueba."""
    with open(ruta, encoding="utf-8") as f:
        resultado = json.load(f)
    por_modelo = resultado["estimacion_horas_gpu"]["por_modelo"]
    tasas = {}
    for m in MODELOS:
        d = por_modelo[m]
        if d["lote"] != LOTE:
            raise ErrorTope(f"La tasa de {m} no es la del lote {LOTE}.")
        tasas[m] = {
            "modelo": d["modelo"],
            "lote": d["lote"],
            "ejemplos_por_s": d["ejemplos_por_s"],
            "costo_fijo_s": d["costo_fijo_s"]["total"],
        }
    return tasas


def horas_por_epoca(filas: int, tasa: Mapping) -> dict:
    entrenamiento = filas / tasa["ejemplos_por_s"] / 3600
    con_fijo = entrenamiento + tasa["costo_fijo_s"] / 3600
    return {
        "entrenamiento": round(entrenamiento, 2),
        "con_costo_fijo": round(con_fijo, 2),
        "pasa_del_limite_de_sesion": con_fijo > LIMITE_SESION_KAGGLE_H,
    }


def resumen_tope(
    final: Sequence[Mapping], tope: int | None, origen_por_clase: Mapping[str, int],
    val_por_clase: Mapping[str, int], tasas: Mapping[str, Mapping],
) -> dict:
    """Conteos de un tope (sin nombres, llaves ni ids; solo códigos SCIAN)."""
    cnt = Counter(r["codigo_act"] for r in final)
    filas = len(final)
    recortadas = [c for c, n in origen_por_clase.items() if tope is not None and n > tope]
    mayor_n = max(cnt.values())
    mayor = min(c for c, n in cnt.items() if n == mayor_n)
    val_total = sum(val_por_clase.values())
    val_recortadas = sum(val_por_clase.get(c, 0) for c in recortadas)
    return {
        "tope": tope,
        "filas": filas,
        "llaves": len({r["llave"] for r in final}),
        "clases": len(cnt),
        "clases_con_menos_de_10_filas": sum(n < MIN_FILAS_CLASE for n in cnt.values()),
        "clases_recortadas": len(recortadas),
        "clase_mas_grande": {"codigo": mayor, "filas": mayor_n, "pct_filas": round(100 * mayor_n / filas, 2)},
        "horas_gpu_por_epoca": {m: horas_por_epoca(filas, tasas[m]) for m in MODELOS},
        "tvd_contra_validacion": round(distancia_variacion_total(cnt, val_por_clase), 4),
        "tvd_contra_train_sin_tope": round(distancia_variacion_total(cnt, origen_por_clase), 4),
        "validacion_filas_de_clases_recortadas": val_recortadas,
        "validacion_pct_filas_de_clases_recortadas": round(100 * val_recortadas / val_total, 2),
        "filas_por_marca": {c: sum(bool(r[c]) for r in final) for c in CAMPOS_MARCAS},
    }


def punto_de_control(
    origen: Sequence[Mapping], val: Sequence[Mapping], topes: Iterable[int | None], tasas: Mapping[str, Mapping]
) -> list[dict]:
    """Una fila por tope; comprueba que ningún tope pierda clases y que los trains queden anidados."""
    origen_por_clase = Counter(r["codigo_act"] for r in origen)
    val_por_clase = Counter(r["codigo_act"] for r in val)
    tabla, anterior = [], None
    for tope in orden_topes(topes):
        final = aplicar_tope(origen, tope)
        ids = {r["id"] for r in final}
        if anterior is not None and not anterior <= ids:
            raise ErrorTope("Los trains de topes distintos no quedaron anidados.")
        anterior = ids
        fila = resumen_tope(final, tope, origen_por_clase, val_por_clase, tasas)
        if fila["clases"] != len(origen_por_clase):
            raise ErrorTope(f"Con el tope {texto_tope(tope)} se pierde alguna clase.")
        tabla.append(fila)
    return tabla


def contexto(origen: Sequence[Mapping], val: Sequence[Mapping]) -> dict:
    """Conteos de origen y validación que no dependen del tope."""
    clases_train = {r["codigo_act"] for r in origen}
    val_por_clase = Counter(r["codigo_act"] for r in val)
    ausentes = [c for c in val_por_clase if c not in clases_train]
    return {
        "train_candidatos_sin_hard": {"filas": len(origen), "llaves": len({r["llave"] for r in origen}),
                                      "clases": len(clases_train)},
        "validacion": {"filas": len(val), "clases": len(val_por_clase)},
        "validacion_clases_ausentes_de_train": len(ausentes),
        "validacion_filas_de_clases_ausentes_de_train": sum(val_por_clase[c] for c in ausentes),
    }


# --- Entradas ---


def cargar(manifest: Mapping, raiz: Path = REPO_RAIZ) -> tuple[list[dict], list[dict]]:
    """Train candidato sin hard y validación, con su SHA-256 verificado; nunca abre test ni hard."""
    salida = []
    for entrada in (ENTRADA_ORIGEN, ENTRADA_VAL):
        e = manifest["splits"][entrada]
        regs = parsear(leer_verificado(Path(raiz) / e["ruta"], e["sha256"]))
        if len(regs) != e["filas"]:
            raise ErrorTope(f"Las filas de {entrada} no son las del manifest.")
        salida.append(regs)
    origen, val = salida
    if {r["id"] for r in origen} & {r["id"] for r in val}:
        raise ErrorTope("Train candidato sin hard comparte id con validación.")
    if {r["llave"] for r in origen} & {r["llave"] for r in val}:
        raise ErrorTope("Train candidato sin hard comparte llaves con validación.")
    por_clase(origen)
    por_clase(val)
    return origen, val


def generar(manifest: Mapping, tope: int | None, raiz: Path = REPO_RAIZ) -> tuple[bytes, list[dict], list[dict], list[dict]]:
    """Desde cero: (bytes de train_final.jsonl, train final, origen, validación)."""
    origen, val = cargar(manifest, raiz)
    final = aplicar_tope(origen, tope)
    return serializar_split(final), final, origen, val


# --- Manifest y results/ ---


def procedimiento(tope: int | None) -> str:
    return (
        f"Origen: splits.{ENTRADA_ORIGEN} (SHA-256 verificado antes de parsear). Filas agrupadas por codigo_act; "
        "en cada clase, en orden de int(id); clases en orden ascendente de código. "
        f"rng = numpy.random.default_rng({SEMILLA}); en cada clase, aunque no se recorte, "
        "perm = rng.permutation(n_clase), y se conservan las filas de sort(perm[:min(n_clase, tope)]). "
        f"Tope: {texto_tope(tope)}. Train final: las filas conservadas, en orden de int(id). Todas las clases "
        "consumen el generador igual con cualquier tope, así que los trains de topes distintos quedan anidados."
    )


def entrada_final(
    *, datos: bytes, final: Sequence[Mapping], manifest: Mapping, tope: int | None, commit: str, fecha: str
) -> dict:
    """Entrada splits.clasificador_train_final del manifest."""
    return {
        "ruta": RUTA_FINAL,
        "sha256": sha256_bytes(datos),
        "tamano_bytes": len(datos),
        "filas": len(final),
        "llaves": len({r["llave"] for r in final}),
        "clases": len({r["codigo_act"] for r in final}),
        "tope": tope,
        "congelado": True,
        "semilla": SEMILLA,
        "procedimiento": procedimiento(tope),
        "script": SCRIPT,
        "commit_codigo": commit,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "campos": list(CAMPOS),
        "formato": FORMATO_SPLITS,
        "fuentes": {"train_candidatos_sin_hard_sha256": manifest["splits"][ENTRADA_ORIGEN]["sha256"]},
        "fecha_registro": fecha,
        "nota": NOTA_FINAL,
    }


def agregar_entrada(manifest: Mapping, entrada: Mapping) -> dict:
    """Copia del manifest con la entrada nueva en `splits`; nada más cambia."""
    if ENTRADA_FINAL in manifest["splits"]:
        raise ErrorTope(f"El manifest ya tiene splits.{ENTRADA_FINAL}; no se sobrescribe.")
    nuevo = copy.deepcopy(dict(manifest))
    nuevo["splits"][ENTRADA_FINAL] = copy.deepcopy(dict(entrada))
    otras_iguales(manifest, nuevo, {ENTRADA_FINAL})
    return nuevo


def armar_resultado(
    *, fecha: str, commit: str, arbol_limpio: bool, manifest: Mapping, tope: int | None, tabla: Sequence[Mapping],
    ctx: Mapping, final: Sequence[Mapping], origen: Sequence[Mapping], hash_final: str, tasas: Mapping,
) -> dict:
    """JSON de results/ con los campos de la regla 4; solo conteos y códigos SCIAN."""
    antes = Counter(r["codigo_act"] for r in origen)
    despues = Counter(r["codigo_act"] for r in final)
    elegido = next(f for f in tabla if f["tope"] == tope)
    s = manifest["splits"]
    return {
        "fecha": fecha,
        "experimento": "fase1_paso8_tope_por_clase",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": SCRIPT,
            "procedimiento": procedimiento(tope),
            "tope": tope,
            "topes_comparados": [f["tope"] for f in tabla],
            "min_filas_clase": MIN_FILAS_CLASE,
            "tasas": {"origen": RESULTADO_TASAS, "lote": LOTE, "por_modelo": dict(tasas)},
            "limite_sesion_kaggle_h": LIMITE_SESION_KAGGLE_H,
            "fuente_limite_sesion": FUENTE_LIMITE_SESION,
            "definiciones": {"tvd": DEFINICION_TVD, "horas_gpu_por_epoca": DEFINICION_HORAS},
            "python": platform.python_version(),
            "versiones": {"numpy": np.__version__},
        },
        "semilla": SEMILLA,
        "split": "clasificador: train final con tope por clase (data/splits/train_final.jsonl)",
        "hash_split": {"train_final": hash_final},
        "hash_split_origen": f"data/manifest.json, splits.{ENTRADA_FINAL}",
        "hash_origen": {"train_candidatos_sin_hard": s[ENTRADA_ORIGEN]["sha256"], "validacion": s[ENTRADA_VAL]["sha256"]},
        "tamano_muestra": {"train_final": len(final), "train_candidatos_sin_hard": len(origen),
                           "validacion": ctx["validacion"]["filas"]},
        "horas_gpu": None,
        "contexto": dict(ctx),
        "comparacion_de_topes": [dict(f) for f in tabla],
        "train_final": {
            **elegido,
            "por_clase": {c: {"antes": antes[c], "despues": despues[c]} for c in sorted(antes)},
        },
        "advertencias": [
            "Las horas de GPU son estimaciones con las tasas de una muestra de 2,000 filas; el train completo puede "
            "tener filas más largas.",
            "La TVD contra validación tiene ruido de muestreo (validación tiene 1,000 filas): la fila sin tope da "
            "su piso.",
            f"Una sesión de GPU de Kaggle dura como máximo {LIMITE_SESION_KAGGLE_H} h ({FUENTE_LIMITE_SESION}).",
        ],
        "verificacion": {"regenerado_desde_cero_mismos_bytes": True, "hash_releido_igual_al_generado": True},
    }


# --- Programa principal ---


def _git(*args: str) -> str:
    resultado = subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True)
    return resultado.stdout.strip()


def _h(d: Mapping) -> str:
    return f"{d['entrenamiento']:.2f}{'*' if d['pasa_del_limite_de_sesion'] else ''}"


def imprimir(tabla: Sequence[Mapping], ctx: Mapping, tasas: Mapping) -> None:
    t, v = ctx["train_candidatos_sin_hard"], ctx["validacion"]
    print(f"Origen: train candidato sin hard {t['filas']:,} filas, {t['llaves']:,} llaves, {t['clases']} clases | "
          f"validación {v['filas']:,} filas, {v['clases']} clases | clases de validación ausentes de train: "
          f"{ctx['validacion_clases_ausentes_de_train']} ({ctx['validacion_filas_de_clases_ausentes_de_train']} filas)")
    print("Tasas (lote 32) de " + RESULTADO_TASAS + ": " + "; ".join(
        f"{m}: {d['ejemplos_por_s']} ejemplos/s, costo fijo {d['costo_fijo_s']} s por corrida" for m, d in tasas.items()))
    encabezado = ("tope", "filas", "llaves", "clases", "<10 filas", "recortadas", "clase mayor", "% mayor",
                  "h 0.6B", "h 4B", "TVD val", "TVD sin tope", "% val recort.")
    anchos = (9, 9, 9, 7, 10, 11, 12, 8, 7, 7, 8, 13, 14)
    print("".join(f"{e:>{a}}" for e, a in zip(encabezado, anchos)))
    for f in tabla:
        h = f["horas_gpu_por_epoca"]
        valores = (
            texto_tope(f["tope"]), f"{f['filas']:,}", f"{f['llaves']:,}", f"{f['clases']}",
            f"{f['clases_con_menos_de_10_filas']}", f"{f['clases_recortadas']}", f["clase_mas_grande"]["codigo"],
            f"{f['clase_mas_grande']['pct_filas']:.2f}", _h(h["qwen3_0.6b"]), _h(h["qwen3_4b"]),
            f"{f['tvd_contra_validacion']:.4f}", f"{f['tvd_contra_train_sin_tope']:.4f}",
            f"{f['validacion_pct_filas_de_clases_recortadas']:.2f}",
        )
        print("".join(f"{x:>{a}}" for x, a in zip(valores, anchos)))
    print("h = horas de GPU por época, solo entrenamiento (filas / ejemplos/s / 3600); a cada corrida se suma su "
          f"costo fijo. * = con el costo fijo pasa de {LIMITE_SESION_KAGGLE_H} h, el límite de una sesión de GPU de "
          f"Kaggle ({FUENTE_LIMITE_SESION}).")
    print("TVD = distancia de variación total: la fracción de filas que habría que cambiar de clase para que las dos "
          "distribuciones fueran iguales (0 = idénticas, 1 = sin clases en común). Validación tiene 1,000 filas: la "
          "fila sin tope da el piso de ruido de la TVD contra validación.")
    print("% val recort. = % de las filas de validación cuya clase queda recortada con ese tope.")
    print("Los trains de los topes quedan anidados (comprobado) y ningún tope pierde clases.")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.exploracion_clasificador import ErrorPrivacidad, verificar_privacidad
    from src.scian import escribir_manifest, leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--escribir", action="store_true", help="escribe train_final.jsonl, el manifest y results/")
    modo.add_argument("--topes", type=int, nargs="+", help="topes del punto de control (además de sin tope)")
    args = parser.parse_args(argv)

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    versionado = codigo_versionado(SCRIPT)
    manifest = leer_manifest()
    ruta_final = REPO_RAIZ / RUTA_FINAL
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_train-final.json"
    tope = None
    if args.escribir:
        try:
            tope = tope_aprobado()
        except ErrorTope as e:
            sys.exit(str(e))
        if not (arbol_limpio and versionado):
            sys.exit(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
        if ENTRADA_FINAL in manifest["splits"] or ruta_final.exists() or salida.exists():
            sys.exit("Ya hay un train final en data/splits/, en el manifest o en results/; no se sobrescribe.")
        if not esta_ignorado(ruta_final):
            sys.exit(f"{RUTA_FINAL} no está ignorado por git.")

    topes = list(TOPES_PUNTO_DE_CONTROL) if not args.topes else [*args.topes, None]
    if args.escribir:
        topes.append(tope)
    try:
        origen, val = cargar(manifest)
        tasas = leer_tasas()
        tabla = punto_de_control(origen, val, topes, tasas)
    except (ErrorMuestra, ErrorTope) as e:
        sys.exit(f"Me detengo: {e}")
    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | {SCRIPT} versionado sin cambios: {versionado} "
          f"| numpy {np.__version__}")
    print("SHA-256 de train candidato sin hard y de validación verificados antes de parsear. Test y hard no se abren.")
    ctx = contexto(origen, val)
    imprimir(tabla, ctx, tasas)
    llaves = {r["llave"] for r in origen} | {r["llave"] for r in val}
    try:
        verificar_privacidad({"tabla": tabla, "contexto": ctx}, llaves, set())
    except ErrorPrivacidad as e:
        sys.exit(f"Me detengo: {e}")

    if not args.escribir:
        print("PUNTO DE CONTROL (PARADA 1): no se escribió nada. El tope lo deciden Samuel y el chat.")
        return

    # --- escribir ---
    datos, final, _, _ = generar(manifest, tope)
    hash_final = sha256_bytes(datos)
    try:
        candado_hashes({RUTA_FINAL: hash_final}, manifest)
    except ErrorHardSet as e:
        sys.exit(f"Me detengo: {e}")
    entrada = entrada_final(datos=datos, final=final, manifest=manifest, tope=tope, commit=commit, fecha=fecha)
    resultado = armar_resultado(fecha=fecha, commit=commit, arbol_limpio=arbol_limpio, manifest=manifest, tope=tope,
                                tabla=tabla, ctx=ctx, final=final, origen=origen, hash_final=hash_final, tasas=tasas)
    try:
        verificar_privacidad({"resultado": resultado, "manifest": entrada}, llaves, set())
    except ErrorPrivacidad as e:
        sys.exit(f"Me detengo: {e}")
    print("Revisión de privacidad del resultado y de la entrada del manifest: ningún texto es una llave.")

    # Regenerar desde cero (manifest y origen releídos) y comparar bytes.
    datos2, _, _, _ = generar(leer_manifest(), tope)
    if datos2 != datos:
        sys.exit("Generar de nuevo no da los mismos bytes; no se escribe nada.")
    try:
        escribir_nuevo(datos, ruta_final)
    except (ErrorHardSet, FileExistsError) as e:
        sys.exit(f"Me detengo: {e}")
    if sha256_archivo(ruta_final) != hash_final:
        sys.exit("El SHA-256 releído de train_final.jsonl no coincide con el generado.")
    nuevo = agregar_entrada(manifest, entrada)
    escribir_manifest(nuevo)
    releido = leer_manifest()
    otras_iguales(manifest, releido, {ENTRADA_FINAL})
    if releido["splits"][ENTRADA_FINAL] != entrada:
        sys.exit("La entrada releída del manifest no es la generada.")
    with open(salida, "x", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{RUTA_FINAL}: {len(final):,} filas, {entrada['clases']} clases, {len(datos):,} bytes, SHA-256 {hash_final} "
          f"(tope {texto_tope(tope)})")
    print(f"Manifest actualizado: splits.{ENTRADA_FINAL} (las demás entradas quedaron idénticas); resultados en "
          f"{salida.relative_to(REPO_RAIZ).as_posix()}")


if __name__ == "__main__":
    main()
