"""Split aleatorio del experimento de fuga (Fase 1, paso 10).

Mide cuánto infla la accuracy que las llaves de validación también estén en train, sin tocar
test ni hard. Solo sirve para este experimento: nunca para elegir ni entrenar los modelos finales.

Procedimiento (semilla 42):

1. P = train candidato sin hard ∪ validación (los dos con su SHA-256 verificado antes de
   parsear; sin id ni llaves compartidos), en orden de int(id).
2. perm = numpy.random.default_rng(42).permutation(len(P)).
3. Validación aleatoria: las filas sort(perm[:1000]), sin mirar llaves.
4. Train aleatorio: el resto de P con el tope por clase de src/tope_clase.py (`aplicar_tope`,
   tope 1,000, con su propio default_rng(42)).
5. Piso de ruido de la TVD (solo para los conteos; no se escribe): las filas sort(perm[1000:2000]).

La única diferencia buscada con el split por nombre es que las llaves de la validación sí pueden
estar en train. P, además, no tiene las filas que la regla 7 excluyó por llaves de la validación
por nombre ni las de las llaves del hard set; los conteos lo reportan.

Uso:
  .venv/Scripts/python -m src.split_aleatorio              # PARADA 1: solo conteos, no escribe
  .venv/Scripts/python -m src.split_aleatorio --escribir   # archivos, manifest y results/

Nunca abre test.jsonl ni hard.jsonl ni construye su ruta: de ellos solo usa el SHA-256 del
manifest. En la terminal solo van conteos y códigos SCIAN, nunca nombres, llaves ni ids.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.denue import REPO_RAIZ, sha256_archivo
from src.hard_set import ErrorHardSet, agregar_entradas, candado_hashes, escribir_nuevo, otras_iguales
from src.muestra_humo import ErrorMuestra, codigo_versionado
from src.splits import CAMPOS, esta_ignorado, sha256_bytes
from src.splits import FORMATO as FORMATO_SPLITS
from src.splits import serializar as serializar_split
from src.subgrupos import CAMPOS_MARCAS
from src.tope_clase import ErrorTope, aplicar_tope, distancia_variacion_total, tope_aprobado
from src.tope_clase import cargar as cargar_origen_y_val

SCRIPT = "src/split_aleatorio.py"
SEMILLA = 42
N_VAL = 1000
ENTRADA_ORIGEN = "clasificador_train_candidatos_sin_hard"
ENTRADA_VAL = "clasificador_validacion"
ENTRADA_CANDIDATOS = "clasificador_train_candidatos"  # solo se leen sus conteos (filas del hard set)
ENTRADA_A_VAL = "experimento_aleatorio_validacion"
ENTRADA_A_TRAIN = "experimento_aleatorio_train_final"
RUTA_A_VAL = "data/splits/aleatorio_val.jsonl"
RUTA_A_TRAIN = "data/splits/aleatorio_train_final.jsonl"
RESULTADO_SPLITS = "results/2026-11-21_fase1_splits-clasificador.json"  # filas excluidas por la regla 7
SOLO_EXPERIMENTO = (
    "Solo para el experimento de fuga del paso 10 (split aleatorio contra split por nombre); nunca para elegir "
    "ni entrenar los modelos finales."
)
DEFINICION_TVD = (
    "Distancia de variación total: la mitad de la suma, sobre todas las clases, de la diferencia absoluta entre "
    "la proporción de filas de la clase en una distribución y en la otra (0 = idénticas, 1 = sin clases en común)."
)


class ErrorSplitAleatorio(ValueError):
    """El split aleatorio no cumple una condición; no se escribe nada."""


@dataclass(frozen=True)
class SplitAleatorio:
    p: list[dict]
    val: list[dict]
    resto: list[dict]
    train: list[dict]
    piso: list[dict]


def procedimiento(tope: int) -> str:
    return (
        f"P = splits.{ENTRADA_ORIGEN} ∪ splits.{ENTRADA_VAL} (SHA-256 verificados antes de parsear), en orden de "
        f"int(id). perm = numpy.random.default_rng({SEMILLA}).permutation(len(P)). Validación aleatoria: las filas "
        f"sort(perm[:{N_VAL}]), sin mirar llaves. Train aleatorio: el resto de P con el tope por clase de "
        f"src/tope_clase.py (aplicar_tope, tope {tope:,}, con su propio default_rng({SEMILLA})). Piso de ruido de la "
        f"TVD (no se escribe): las filas sort(perm[{N_VAL}:{2 * N_VAL}])."
    )


# --- Construcción ---


def unir(origen: Sequence[Mapping], val: Sequence[Mapping]) -> list[dict]:
    """P: las dos listas juntas en orden de int(id); sin id repetidos."""
    p = sorted((dict(r) for r in [*origen, *val]), key=lambda r: int(r["id"]))
    ids = [r["id"] for r in p]
    if len(set(ids)) != len(ids):
        raise ErrorSplitAleatorio("Train candidato sin hard y validación comparten id.")
    return p


def partir(p: Sequence[Mapping], tope: int, n_val: int = N_VAL, semilla: int = SEMILLA) -> SplitAleatorio:
    """El procedimiento del módulo sobre P (ya en orden de int(id))."""
    if len(p) < 2 * n_val:
        raise ErrorSplitAleatorio(f"P tiene {len(p)} filas; se piden {2 * n_val} entre validación y piso.")
    ids = [int(r["id"]) for r in p]
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ErrorSplitAleatorio("P no está en orden de int(id) o tiene id repetidos.")
    perm = np.random.default_rng(semilla).permutation(len(p))
    idx_val = np.sort(perm[:n_val])
    idx_piso = np.sort(perm[n_val:2 * n_val])
    en_val = set(int(i) for i in idx_val)
    val = [dict(p[int(i)]) for i in idx_val]
    resto = [dict(r) for i, r in enumerate(p) if i not in en_val]
    train = aplicar_tope(resto, tope, semilla)
    return SplitAleatorio(p=[dict(r) for r in p], val=val, resto=resto, train=train,
                          piso=[dict(p[int(i)]) for i in idx_piso])


def verificar(s: SplitAleatorio) -> None:
    """Tamaños, ids disjuntos, train dentro del resto y todas las clases del resto en train."""
    ids_val = {r["id"] for r in s.val}
    ids_resto = {r["id"] for r in s.resto}
    if len(ids_val) != len(s.val) or ids_val & ids_resto or len(ids_val) + len(ids_resto) != len(s.p):
        raise ErrorSplitAleatorio("Validación aleatoria y resto no parten P.")
    if not {r["id"] for r in s.train} <= ids_resto:
        raise ErrorSplitAleatorio("El train aleatorio tiene filas fuera del resto.")
    if {r["codigo_act"] for r in s.train} != {r["codigo_act"] for r in s.resto}:
        raise ErrorSplitAleatorio("El tope perdió alguna clase.")
    for regs in (s.val, s.train):
        if any(tuple(r) != CAMPOS for r in regs):
            raise ErrorSplitAleatorio("Un registro no tiene los campos de src/splits.py en orden.")


def cargar(manifest: Mapping, raiz: Path = REPO_RAIZ) -> tuple[list[dict], list[dict]]:
    """Train candidato sin hard y validación verificados (src/tope_clase.py); nunca abre test ni hard."""
    try:
        return cargar_origen_y_val(manifest, raiz)
    except ErrorTope as e:
        raise ErrorSplitAleatorio(str(e)) from e


def generar(manifest: Mapping, raiz: Path = REPO_RAIZ) -> tuple[SplitAleatorio, list[dict], dict[str, bytes]]:
    """Desde cero: (split, validación por nombre, bytes de los dos archivos)."""
    origen, val = cargar(manifest, raiz)
    s = partir(unir(origen, val), tope_aprobado())
    verificar(s)
    return s, val, {RUTA_A_VAL: serializar_split(s.val), RUTA_A_TRAIN: serializar_split(s.train)}


# --- Conteos ---


def _marcas(regs: Sequence[Mapping]) -> dict[str, int]:
    return {c: sum(bool(r[c]) for r in regs) for c in CAMPOS_MARCAS}


def _tam(regs: Sequence[Mapping]) -> dict[str, int]:
    return {"filas": len(regs), "llaves": len({r["llave"] for r in regs}), "clases": len({r["codigo_act"] for r in regs})}


def _en(val: Sequence[Mapping], llaves: set[str]) -> dict:
    dentro = [r for r in val if r["llave"] in llaves]
    return {"filas": len(dentro), "pct_filas": round(100 * len(dentro) / len(val), 2),
            "llaves": len({r["llave"] for r in dentro})}


def conteos(s: SplitAleatorio, val_nombre: Sequence[Mapping], manifest: Mapping, excluidas_por_val: int | None) -> dict:
    """Conteos del punto 5 del plan (sin nombres, llaves ni ids)."""
    llaves_train = {r["llave"] for r in s.train}
    llaves_resto = {r["llave"] for r in s.resto}
    ids_nombre = {r["id"] for r in val_nombre}
    llaves_origen = {r["llave"] for r in s.p if r["id"] not in ids_nombre}
    clases_train = {r["codigo_act"] for r in s.train}
    ausentes = [r for r in s.val if r["codigo_act"] not in clases_train]
    cnt_val = Counter(r["codigo_act"] for r in s.val)
    sp = manifest["splits"]
    return {
        "p": _tam(s.p),
        "validacion_aleatoria": _tam(s.val),
        "resto_antes_del_tope": _tam(s.resto),
        "train_aleatorio": _tam(s.train),
        "validacion_por_nombre": _tam(val_nombre),
        "validacion_aleatoria_con_llave_en_train_aleatorio": _en(s.val, llaves_train),
        "validacion_aleatoria_con_llave_en_el_resto_antes_del_tope": _en(s.val, llaves_resto),
        "validacion_por_nombre_con_llave_en_train": _en(val_nombre, llaves_origen)["filas"],  # regla 2: debe ser 0
        "validacion_aleatoria_filas_de_la_validacion_por_nombre": len({r["id"] for r in s.val} & {r["id"] for r in val_nombre}),
        "validacion_aleatoria_con_clase_ausente_de_train_aleatorio": {
            "filas": len(ausentes), "clases": len({r["codigo_act"] for r in ausentes})},
        "filas_por_marca": {"validacion_aleatoria": _marcas(s.val), "validacion_por_nombre": _marcas(val_nombre),
                            "train_aleatorio": _marcas(s.train)},
        "tvd_clases": {
            "validacion_aleatoria_contra_validacion_por_nombre": round(
                distancia_variacion_total(cnt_val, Counter(r["codigo_act"] for r in val_nombre)), 4),
            "piso_de_ruido_validacion_aleatoria_contra_perm_1000_2000": round(
                distancia_variacion_total(cnt_val, Counter(r["codigo_act"] for r in s.piso)), 4),
            "definicion": DEFINICION_TVD,
        },
        "p_frente_a_la_poblacion_de_la_validacion_por_nombre": {
            "filas_excluidas_por_la_regla_7_por_llaves_de_validacion": excluidas_por_val,
            "fuente_de_ese_conteo": RESULTADO_SPLITS,
            "filas_de_las_llaves_del_hard_set": sp[ENTRADA_CANDIDATOS]["filas"] - sp[ENTRADA_ORIGEN]["filas"],
            "nota": (
                "P no tiene esas filas: en P, cada llave de la validación por nombre queda solo con su fila de "
                "validación, y las llaves del hard set no están. Si acaso, la fuga medida queda un poco por debajo de "
                "la real."
            ),
        },
    }


def leer_excluidas_por_val(raiz: Path = REPO_RAIZ) -> int | None:
    """Filas que la regla 7 excluyó por llaves de validación (results del paso 5); None si no está."""
    ruta = Path(raiz) / RESULTADO_SPLITS
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)["regla7"]["por_llaves_de_validacion"]


# --- Manifest y results/ ---


def entradas_manifest(*, datos: Mapping[str, bytes], s: SplitAleatorio, manifest: Mapping, tope: int,
                      commit: str, fecha: str) -> dict[str, dict]:
    sp = manifest["splits"]
    fuentes = {"train_candidatos_sin_hard_sha256": sp[ENTRADA_ORIGEN]["sha256"],
               "validacion_sha256": sp[ENTRADA_VAL]["sha256"]}
    comun = {
        "congelado": True,
        "semilla": SEMILLA,
        "tope": tope,
        "procedimiento": procedimiento(tope),
        "script": SCRIPT,
        "commit_codigo": commit,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "campos": list(CAMPOS),
        "formato": FORMATO_SPLITS,
        "fuentes": fuentes,
        "fecha_registro": fecha,
        "solo_experimento": True,
        "nota": SOLO_EXPERIMENTO,
    }
    res = {}
    for entrada, ruta, regs in ((ENTRADA_A_VAL, RUTA_A_VAL, s.val), (ENTRADA_A_TRAIN, RUTA_A_TRAIN, s.train)):
        res[entrada] = {"ruta": ruta, "sha256": sha256_bytes(datos[ruta]), "tamano_bytes": len(datos[ruta]),
                        **_tam(regs), **comun}
    return res


def armar_resultado(*, fecha: str, commit: str, arbol_limpio: bool, manifest: Mapping, tope: int,
                    cnt: Mapping, hashes: Mapping[str, str]) -> dict:
    sp = manifest["splits"]
    return {
        "fecha": fecha,
        "experimento": "fase1_paso10_split_aleatorio",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": SCRIPT,
            "procedimiento": procedimiento(tope),
            "n_validacion": N_VAL,
            "tope": tope,
            "python": platform.python_version(),
            "versiones": {"numpy": np.__version__},
        },
        "semilla": SEMILLA,
        "split": "clasificador: split aleatorio del experimento de fuga (data/splits/aleatorio_*.jsonl)",
        "hash_split": {"validacion_aleatoria": hashes[RUTA_A_VAL], "train_aleatorio": hashes[RUTA_A_TRAIN]},
        "hash_split_origen": f"data/manifest.json, splits.{ENTRADA_A_VAL} y splits.{ENTRADA_A_TRAIN}",
        "hash_origen": {"train_candidatos_sin_hard": sp[ENTRADA_ORIGEN]["sha256"], "validacion": sp[ENTRADA_VAL]["sha256"]},
        "tamano_muestra": {"p": cnt["p"]["filas"], "validacion_aleatoria": cnt["validacion_aleatoria"]["filas"],
                           "train_aleatorio": cnt["train_aleatorio"]["filas"]},
        "horas_gpu": None,
        "conteos": dict(cnt),
        "nota": SOLO_EXPERIMENTO,
        "verificacion": {"regenerado_desde_cero_mismos_bytes": True, "hash_releido_igual_al_generado": True},
    }


# --- Programa principal ---


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True).stdout.strip()


def imprimir(cnt: Mapping) -> None:
    for nombre in ("p", "validacion_aleatoria", "resto_antes_del_tope", "train_aleatorio", "validacion_por_nombre"):
        t = cnt[nombre]
        print(f"  {nombre}: {t['filas']:,} filas, {t['llaves']:,} llaves, {t['clases']} clases")
    for nombre in ("validacion_aleatoria_con_llave_en_train_aleatorio", "validacion_aleatoria_con_llave_en_el_resto_antes_del_tope"):
        e = cnt[nombre]
        print(f"  {nombre}: {e['filas']:,} filas ({e['pct_filas']} %), {e['llaves']:,} llaves")
    print(f"  validacion_por_nombre_con_llave_en_train: {cnt['validacion_por_nombre_con_llave_en_train']}")
    print(f"  validación aleatoria: filas que son de la validación por nombre: "
          f"{cnt['validacion_aleatoria_filas_de_la_validacion_por_nombre']}")
    a = cnt["validacion_aleatoria_con_clase_ausente_de_train_aleatorio"]
    print(f"  validación aleatoria con clase ausente del train aleatorio: {a['filas']} filas ({a['clases']} clases)")
    for conj, m in cnt["filas_por_marca"].items():
        print(f"  marcas en {conj}: " + ", ".join(f"{k} {v:,}" for k, v in m.items()))
    t = cnt["tvd_clases"]
    print(f"  TVD de clases, validación aleatoria contra validación por nombre: "
          f"{t['validacion_aleatoria_contra_validacion_por_nombre']:.4f} (piso de ruido con perm[1000:2000]: "
          f"{t['piso_de_ruido_validacion_aleatoria_contra_perm_1000_2000']:.4f})")
    q = cnt["p_frente_a_la_poblacion_de_la_validacion_por_nombre"]
    print(f"  P no tiene {q['filas_excluidas_por_la_regla_7_por_llaves_de_validacion']:,} filas excluidas por la regla 7 "
          f"(llaves de validación) ni {q['filas_de_las_llaves_del_hard_set']:,} filas de llaves del hard set.")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.exploracion_clasificador import ErrorPrivacidad, verificar_privacidad
    from src.scian import escribir_manifest, leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escribir", action="store_true", help="escribe los archivos, el manifest y results/")
    args = parser.parse_args(argv)

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    versionado = codigo_versionado(SCRIPT)
    manifest = leer_manifest()
    rutas = {r: REPO_RAIZ / r for r in (RUTA_A_VAL, RUTA_A_TRAIN)}
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_split-aleatorio.json"
    if args.escribir:
        if not (arbol_limpio and versionado):
            sys.exit(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
        if {ENTRADA_A_VAL, ENTRADA_A_TRAIN} & set(manifest["splits"]) or any(r.exists() for r in rutas.values()) \
                or salida.exists():
            sys.exit("Ya hay un split aleatorio en data/splits/, en el manifest o en results/; no se sobrescribe.")
        if not all(esta_ignorado(r) for r in rutas.values()):
            sys.exit("Algún archivo del split aleatorio no está ignorado por git.")

    try:
        s, val_nombre, datos = generar(manifest)
    except (ErrorMuestra, ErrorTope, ErrorSplitAleatorio) as e:
        sys.exit(f"Me detengo: {e}")
    tope = tope_aprobado()
    cnt = conteos(s, val_nombre, manifest, leer_excluidas_por_val())
    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | {SCRIPT} versionado sin cambios: {versionado} "
          f"| numpy {np.__version__}")
    print("SHA-256 de train candidato sin hard y de validación verificados antes de parsear. Test y hard no se abren.")
    print(f"Split aleatorio (semilla {SEMILLA}, validación de {N_VAL:,} filas, tope {tope:,}):")
    imprimir(cnt)
    hashes = {r: sha256_bytes(d) for r, d in datos.items()}
    llaves = {r["llave"] for r in s.p}
    resultado = armar_resultado(fecha=fecha, commit=commit, arbol_limpio=arbol_limpio, manifest=manifest, tope=tope,
                                cnt=cnt, hashes=hashes)
    entradas = entradas_manifest(datos=datos, s=s, manifest=manifest, tope=tope, commit=commit, fecha=fecha)
    try:
        verificar_privacidad({"resultado": resultado, "manifest": entradas}, llaves, set())
        candado_hashes(hashes, manifest)
    except (ErrorPrivacidad, ErrorHardSet) as e:
        sys.exit(f"Me detengo: {e}")
    print("Revisión de privacidad: ningún texto es una llave. Ningún archivo tiene el SHA-256 de una entrada del manifest.")
    for r, h in hashes.items():
        print(f"  {r}: {len(datos[r]):,} bytes, SHA-256 {h}")

    if not args.escribir:
        print("PUNTO DE CONTROL (PARADA 1): no se escribió nada.")
        return

    _, _, datos2 = generar(leer_manifest())
    if datos2 != datos:
        sys.exit("Generar de nuevo no da los mismos bytes; no se escribe nada.")
    try:
        for r, d in datos.items():
            escribir_nuevo(d, rutas[r])
    except (ErrorHardSet, FileExistsError) as e:
        sys.exit(f"Me detengo: {e}")
    for r, h in hashes.items():
        if sha256_archivo(rutas[r]) != h:
            sys.exit(f"El SHA-256 releído de {r} no coincide con el generado.")
    nuevo = agregar_entradas(manifest, entradas)
    escribir_manifest(nuevo)
    releido = leer_manifest()
    otras_iguales(manifest, releido, set(entradas))
    if any(releido["splits"][e] != v for e, v in entradas.items()):
        sys.exit("Las entradas releídas del manifest no son las generadas.")
    with open(salida, "x", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Escritos {RUTA_A_VAL} y {RUTA_A_TRAIN}; manifest actualizado (splits.{ENTRADA_A_VAL} y "
          f"splits.{ENTRADA_A_TRAIN}; lo demás quedó idéntico); resultados en {salida.relative_to(REPO_RAIZ).as_posix()}")


if __name__ == "__main__":
    main()
