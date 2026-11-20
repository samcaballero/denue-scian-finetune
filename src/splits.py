"""Split del clasificador: test y validación congelados (Fase 1, paso 5).

Procedimiento (semilla 42; CLAUDE.md, sección c, reglas 1, 2, 3, 7 y 8):

1. Universo: filas con llave normalizada no vacía, en orden ascendente de int(id).
2. perm = numpy.random.default_rng(42).permutation(N).
3. Test: las filas perm[0:2000].
4. Validación: las primeras 1000 filas de perm[2000:] cuya llave no está en test.
5. Train candidato: las filas cuya llave no está ni en test ni en validación.
6. Excluidas: el resto (regla 7).

Uso:
  .venv/Scripts/python -m src.splits             # punto de control: solo conteos, no escribe splits
  .venv/Scripts/python -m src.splits --escribir  # solo tras la aprobación de Samuel

En la terminal y en results/ solo van conteos; las 20 llaves que más filas excluyen
van solo a un archivo local de data/ (ignorado por git). Nunca carga `clee`; usa
`nombre_act` solo para la marca de igual a título y no lo escribe.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version as version_de_paquete
from pathlib import Path

import numpy as np
import pandas as pd

from src.denue import REPO_RAIZ, cargar_denue, entrada_manifest, sha256_archivo
from src.normalizacion import normalizar_nombre
from src.subgrupos import CAMPOS_MARCAS, VOCABULARIO, cargar_titulos, cargar_vocabulario, marcas
from src.subgrupos import DEFINICIONES as DEFINICIONES_SUBGRUPOS

COLUMNAS = ["id", "nom_estab", "codigo_act", "nombre_act"]
SEMILLA = 42
N_TEST = 2000
N_VAL = 1000
LIMITE_REGLA7_PCT = 10
TOP_LLAVES = 20
CAMPOS = ("id", "nom_estab", "codigo_act", "llave", *CAMPOS_MARCAS)
# Candado: conteos del universo y de las marcas medidos en el paso 4.
ESPERADOS = {"universo": 462520, "sin_nombre": 43979, "generico": 50134, "igual_titulo": 449}
CONJUNTOS = ("test", "validacion", "train_candidatos")
ARCHIVOS = {"test": "test.jsonl", "validacion": "val.jsonl", "train_candidatos": "train_candidatos.jsonl"}
ENTRADAS = {
    "test": "clasificador_test",
    "validacion": "clasificador_validacion",
    "train_candidatos": "clasificador_train_candidatos",
}
CONGELADO = {"test": True, "validacion": True, "train_candidatos": False}
CARPETA_SPLITS = REPO_RAIZ / "data" / "splits"
COLUMNAS_LLAVES = ("rango", "llave", "conjunto", "filas_del_grupo", "filas_en_el_conjunto", "filas_excluidas")
PROCEDIMIENTO = (
    "Universo: filas del DENUE con normalizar_nombre(nom_estab) no vacía, en orden ascendente de int(id). "
    f"perm = numpy.random.default_rng({SEMILLA}).permutation(len(universo)). "
    f"Test: las filas perm[0:{N_TEST}]. "
    f"Validación: las primeras {N_VAL} filas de perm[{N_TEST}:] cuya llave no está en test. "
    "Train candidato: las filas cuya llave no está en test ni en validación. "
    "Excluidas: el resto (regla 7)."
)
FORMATO = (
    "JSONL: un objeto por línea con las claves en el orden de 'campos'; UTF-8 sin BOM; saltos \\n; "
    "ordenado por int(id)."
)
NOTA_TRAIN = "Sin congelar: el hard set y el tope por clase lo cambiarán."
RE_ID = re.compile(r"[0-9]+")


class ErrorSplit(ValueError):
    """El split no cumple una condición; no se escribe nada."""


class ErrorRegla7(ErrorSplit):
    """La exclusión de la regla 7 pasa del límite: no se congela; decide Samuel."""


@dataclass(frozen=True)
class Particion:
    """Índices (posiciones en el universo, ordenadas) de cada conjunto."""

    test: np.ndarray
    validacion: np.ndarray
    train_candidatos: np.ndarray
    excluidas: np.ndarray

    def conjuntos(self) -> dict[str, np.ndarray]:
        return {
            "test": self.test,
            "validacion": self.validacion,
            "train_candidatos": self.train_candidatos,
            "excluidas": self.excluidas,
        }


def _indices(valores) -> np.ndarray:
    return np.sort(np.asarray(list(valores), dtype=np.int64))


# --- Universo y marcas ---


def universo(df: pd.DataFrame) -> pd.DataFrame:
    """Filas con llave no vacía, en orden ascendente de int(id), con la columna `llave`."""
    ids = df["id"].tolist()
    if not all(RE_ID.fullmatch(i) for i in ids):
        raise ErrorSplit("Hay id que no son solo dígitos ASCII.")
    if len(set(ids)) != len(ids):
        raise ErrorSplit("Hay id repetidos.")
    llave_de = {n: normalizar_nombre(n) for n in df["nom_estab"].unique()}
    u = df.assign(llave=df["nom_estab"].map(llave_de).astype(str))
    u = u[u["llave"] != ""]
    orden = np.argsort(np.array([int(i) for i in u["id"]], dtype=np.int64), kind="stable")
    return u.iloc[orden].reset_index(drop=True)


def agregar_marcas(u: pd.DataFrame, vocab: frozenset[str], titulos: frozenset[str]) -> pd.DataFrame:
    """Agrega las tres marcas de `src/subgrupos.py` y quita `nombre_act`."""
    cache: dict[tuple[str, str], dict[str, bool]] = {}
    columnas: dict[str, list[bool]] = {c: [] for c in CAMPOS_MARCAS}
    for k, na in zip(u["llave"], u["nombre_act"]):
        m = cache.get((k, na))
        if m is None:
            m = cache[(k, na)] = marcas(k, na, vocab, titulos)
        for c in CAMPOS_MARCAS:
            columnas[c].append(m[c])
    return u.assign(**columnas).drop(columns=["nombre_act"])


def preparar(df: pd.DataFrame, vocab: frozenset[str], titulos: frozenset[str]) -> pd.DataFrame:
    return agregar_marcas(universo(df), vocab, titulos)


def conteos_universo(u: pd.DataFrame) -> dict[str, int]:
    return {"universo": len(u), **{c: int(u[c].sum()) for c in CAMPOS_MARCAS}}


# --- Partición ---


def partir(llaves: Sequence[str], semilla: int = SEMILLA, n_test: int = N_TEST, n_val: int = N_VAL) -> Particion:
    """Test, validación, train candidato y excluidas según el procedimiento del módulo."""
    llaves = list(llaves)
    if any(k == "" for k in llaves):
        raise ErrorSplit("El universo tiene llaves vacías.")
    n = len(llaves)
    if n < n_test + n_val:
        raise ErrorSplit(f"El universo tiene {n} filas; se piden {n_test} de test y {n_val} de validación.")
    perm = np.random.default_rng(semilla).permutation(n)
    test = [int(i) for i in perm[:n_test]]
    llaves_test = {llaves[i] for i in test}
    val: list[int] = []
    for i in perm[n_test:]:
        if llaves[i] not in llaves_test:
            val.append(int(i))
            if len(val) == n_val:
                break
    if len(val) < n_val:
        raise ErrorSplit(f"Solo hay {len(val)} filas con llave fuera de test para validación; se piden {n_val}.")
    tocadas = llaves_test | {llaves[i] for i in val}
    en_conjunto = set(test) | set(val)
    train = [i for i in range(n) if llaves[i] not in tocadas]
    excluidas = [i for i in range(n) if llaves[i] in tocadas and i not in en_conjunto]
    return Particion(_indices(test), _indices(val), _indices(train), _indices(excluidas))


def verificar_particion(llaves: Sequence[str], p: Particion, n_test: int = N_TEST, n_val: int = N_VAL) -> None:
    """Tamaños exactos, llaves sin cruce y cada fila del universo en exactamente un grupo."""
    llaves = list(llaves)
    if len(p.test) != n_test or len(p.validacion) != n_val:
        raise ErrorSplit("Los tamaños de test o validación no son los pedidos.")
    todos = np.concatenate(list(p.conjuntos().values()))
    if len(todos) != len(llaves) or not np.array_equal(np.sort(todos), np.arange(len(llaves))):
        raise ErrorSplit("Alguna fila del universo no está en exactamente un grupo.")
    por_conjunto = {nombre: {llaves[i] for i in getattr(p, nombre)} for nombre in CONJUNTOS}
    for a, b in (("test", "validacion"), ("test", "train_candidatos"), ("validacion", "train_candidatos")):
        if por_conjunto[a] & por_conjunto[b]:
            raise ErrorSplit(f"Hay llaves en {a} y en {b}.")
    if not {llaves[i] for i in p.excluidas} <= por_conjunto["test"] | por_conjunto["validacion"]:
        raise ErrorSplit("Hay filas excluidas cuya llave no está en test ni en validación.")


# --- Regla 7 ---


def pasa_del_limite(excluidas: int, candidatas: int, limite_pct: int = LIMITE_REGLA7_PCT) -> bool:
    """True si excluidas / candidatas > limite_pct %, comparando con enteros."""
    return excluidas * 100 > limite_pct * candidatas


def candado_regla7(excluidas: int, candidatas: int, limite_pct: int = LIMITE_REGLA7_PCT) -> None:
    if pasa_del_limite(excluidas, candidatas, limite_pct):
        raise ErrorRegla7(
            f"Regla 7: se excluyen {excluidas} de {candidatas} filas candidatas a train "
            f"(más de {limite_pct} %). No se congela: decide Samuel."
        )


def regla7(llaves: Sequence[str], p: Particion, top: int = TOP_LLAVES) -> dict:
    """Filas candidatas a train excluidas por compartir llave con test o validación.

    `top` trae las llaves; solo va al archivo local de data/ (ver `top_sin_llaves`).
    """
    llaves = list(llaves)
    candidatas = len(llaves) - len(p.test) - len(p.validacion)
    llaves_test = {llaves[i] for i in p.test}
    llaves_val = {llaves[i] for i in p.validacion}
    tamano = Counter(llaves)
    en_conjunto = Counter(llaves[i] for i in np.concatenate([p.test, p.validacion]))
    excluidas_por_llave = Counter(llaves[i] for i in p.excluidas)
    por_test = sum(v for k, v in excluidas_por_llave.items() if k in llaves_test)
    orden = sorted(excluidas_por_llave.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return {
        "filas_candidatas_train": candidatas,
        "filas_excluidas": len(p.excluidas),
        "pct_filas_candidatas": round(100 * len(p.excluidas) / candidatas, 2) if candidatas else None,
        "por_llaves_de_test": por_test,
        "por_llaves_de_validacion": len(p.excluidas) - por_test,
        "llaves_tocadas": {"test": len(llaves_test), "validacion": len(llaves_val)},
        "llaves_que_excluyen_filas": len(excluidas_por_llave),
        "limite_pct": LIMITE_REGLA7_PCT,
        "pasa_del_limite": pasa_del_limite(len(p.excluidas), candidatas),
        "top": [
            {
                "rango": r,
                "llave": k,
                "conjunto": "test" if k in llaves_test else "validacion",
                "filas_del_grupo": tamano[k],
                "filas_en_el_conjunto": en_conjunto[k],
                "filas_excluidas": v,
            }
            for r, (k, v) in enumerate(orden, 1)
        ],
    }


def top_sin_llaves(top: Sequence[Mapping]) -> list[dict]:
    """Las filas del top sin la llave: lo único que puede ir a results/ o a la terminal."""
    return [{c: fila[c] for c in COLUMNAS_LLAVES if c != "llave"} for fila in top]


def esta_ignorado(ruta: str | Path) -> bool:
    """True si git ignora la ruta. Usa `git check-ignore -q`: con -v, una excepción (!) también da 0."""
    p = Path(ruta)
    p = p if p.is_absolute() else REPO_RAIZ / p
    relativa = p.resolve().relative_to(REPO_RAIZ.resolve()).as_posix()
    r = subprocess.run(["git", "check-ignore", "-q", relativa], cwd=REPO_RAIZ, capture_output=True)
    if r.returncode not in (0, 1):
        raise ErrorSplit(f"git check-ignore falló con código {r.returncode}.")
    return r.returncode == 0


def escribir_llaves_regla7(top: Sequence[Mapping], ruta: Path) -> None:
    """CSV local en UTF-8 con BOM (para Excel), solo si git ignora la ruta."""
    if not esta_ignorado(ruta):
        raise ErrorSplit(f"{ruta.name} no está ignorado por git; no se escribe.")
    with open(ruta, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_LLAVES, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(top)


# --- Conteos ---


def conteos(u: pd.DataFrame, p: Particion) -> dict:
    """Filas, llaves, clases y filas de cada marca por conjunto (incluidas las excluidas)."""
    res = {}
    for nombre, idx in p.conjuntos().items():
        sub = u.iloc[idx]
        res[nombre] = {
            "filas": len(sub),
            "llaves": int(sub["llave"].nunique()),
            "clases": int(sub["codigo_act"].nunique()),
            **{c: int(sub[c].sum()) for c in CAMPOS_MARCAS},
        }
    return res


def clases_fuera_de_train(u: pd.DataFrame, p: Particion) -> dict:
    """Filas y clases de test y validación cuya clase no aparece en train candidato."""
    en_train = set(u["codigo_act"].iloc[p.train_candidatos])
    res = {}
    for nombre in ("test", "validacion"):
        sub = u.iloc[getattr(p, nombre)]
        fuera = ~sub["codigo_act"].isin(en_train)
        res[nombre] = {"filas": int(fuera.sum()), "clases": int(sub.loc[fuera, "codigo_act"].nunique())}
    return res


# --- Archivos ---


def registros(u: pd.DataFrame, indices: np.ndarray) -> list[dict]:
    """Registros con las claves en el orden de CAMPOS; en el orden del universo (por int(id))."""
    sub = u.iloc[np.sort(indices)]
    columnas = [sub[c].tolist() for c in CAMPOS]
    return [
        {
            "id": str(i),
            "nom_estab": str(n),
            "codigo_act": str(c),
            "llave": str(k),
            "sin_nombre": bool(m1),
            "generico": bool(m2),
            "igual_titulo": bool(m3),
        }
        for i, n, c, k, m1, m2, m3 in zip(*columnas)
    ]


def serializar(regs: Sequence[Mapping]) -> bytes:
    """JSONL en UTF-8 sin BOM, con "\\n" al final de cada línea."""
    for r in regs:
        if tuple(r) != CAMPOS:
            raise ErrorSplit("Un registro no tiene las claves esperadas en orden.")
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def contenidos(u: pd.DataFrame, p: Particion) -> dict[str, bytes]:
    return {nombre: serializar(registros(u, getattr(p, nombre))) for nombre in CONJUNTOS}


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def escribir_archivos(datos: Mapping[str, bytes], carpeta: Path) -> dict[str, Path]:
    """Escribe los archivos en binario; se niega si alguno ya existe."""
    rutas = {nombre: carpeta / ARCHIVOS[nombre] for nombre in datos}
    existentes = sorted(r.name for r in rutas.values() if r.exists())
    if existentes:
        raise ErrorSplit(f"Ya existen {existentes}; un conjunto congelado no se sobrescribe.")
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, ruta in rutas.items():
        with open(ruta, "xb") as f:  # "x": falla si el archivo ya existe
            f.write(datos[nombre])
    return rutas


# --- Programa principal ---


def _git(*args: str) -> str:
    resultado = subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True)
    return resultado.stdout.strip()


def generar() -> tuple[int, pd.DataFrame, Particion]:
    """Desde cero: carga verificada, universo, marcas, candado del paso 4 y partición."""
    vocab, titulos = cargar_vocabulario(), cargar_titulos()
    df = cargar_denue(COLUMNAS)
    u = preparar(df, vocab, titulos)
    observado = conteos_universo(u)
    if observado != ESPERADOS:
        raise ErrorSplit(f"Los conteos del universo no coinciden con el paso 4: {observado} contra {ESPERADOS}.")
    llaves = u["llave"].tolist()
    p = partir(llaves)
    verificar_particion(llaves, p)
    return len(df), u, p


def _imprimir(filas_total: int, u: pd.DataFrame, cnt: dict, fuera: dict, r7: dict, ruta_llaves: Path) -> None:
    print(f"Filas: {filas_total:,} | llave vacía (excluida): {filas_total - len(u):,} | universo: {len(u):,}")
    print("Marcas en el universo (paso 4: 43,979 / 50,134 / 449): " + ", ".join(
        f"{c} {int(u[c].sum()):,}" for c in CAMPOS_MARCAS))
    for nombre, c in cnt.items():
        print(
            f"  {nombre}: {c['filas']:,} filas, {c['llaves']:,} llaves, {c['clases']:,} clases | "
            f"sin_nombre {c['sin_nombre']:,}, generico {c['generico']:,}, igual_titulo {c['igual_titulo']:,}"
        )
    for nombre, f in fuera.items():
        print(f"Clase ausente de train candidato, {nombre}: {f['filas']:,} filas ({f['clases']:,} clases)")
    print(
        f"Regla 7: {r7['filas_excluidas']:,} de {r7['filas_candidatas_train']:,} filas candidatas excluidas "
        f"({r7['pct_filas_candidatas']} %); por llaves de test {r7['por_llaves_de_test']:,}, de validación "
        f"{r7['por_llaves_de_validacion']:,}; llaves tocadas {r7['llaves_tocadas']}; "
        f"pasa de {r7['limite_pct']} %: {r7['pasa_del_limite']}"
    )
    print("Top 20 (sin llaves): rango, conjunto, filas del grupo, filas en el conjunto, filas excluidas")
    for t in top_sin_llaves(r7["top"]):
        print(f"  {t['rango']:>2} {t['conjunto']:<10} {t['filas_del_grupo']:>6,} {t['filas_en_el_conjunto']:>3} {t['filas_excluidas']:>6,}")
    print(f"Llaves de la regla 7 (solo local): {ruta_llaves.relative_to(REPO_RAIZ).as_posix()} "
          f"(ignorado por git: {esta_ignorado(ruta_llaves)})")


def armar_resultado(
    *,
    fecha: str,
    commit: str,
    arbol_limpio: bool,
    manifest: Mapping,
    entrada_csv: Mapping,
    filas_total: int,
    u: pd.DataFrame,
    cnt: Mapping,
    fuera: Mapping,
    r7: Mapping,
    ruta_llaves: Path,
    hashes: Mapping[str, str],
) -> dict:
    """JSON de results/ con los campos de la regla 4; solo conteos, sin llaves."""
    from src.scian import CATALOGO

    entrada_cat = manifest["catalogos"][CATALOGO[manifest["scian_version"]]]
    entrada_voc = manifest["catalogos"][VOCABULARIO]
    return {
        "fecha": fecha,
        "experimento": "fase1_paso5_splits_clasificador",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": "src/splits.py",
            "cargador": "src/denue.py:cargar_denue",
            "normalizacion": "src/normalizacion.py:normalizar_nombre",
            "subgrupos": "src/subgrupos.py",
            "definiciones_subgrupos": DEFINICIONES_SUBGRUPOS,
            "columnas_leidas": COLUMNAS,
            "campos_de_los_archivos": list(CAMPOS),
            "codificacion": entrada_csv["codificacion"],
            "scian_version": manifest["scian_version"],
            "catalogo": {"ruta": entrada_cat["ruta"], "sha256": entrada_cat["sha256"]},
            "vocabulario": {"ruta": entrada_voc["ruta"], "sha256": entrada_voc["sha256"]},
            "procedimiento": PROCEDIMIENTO,
            "n_test": N_TEST,
            "n_validacion": N_VAL,
            "limite_regla7_pct": LIMITE_REGLA7_PCT,
            "python": platform.python_version(),
            "versiones": {paquete: version_de_paquete(paquete) for paquete in ("numpy", "pandas")},
        },
        "semilla": SEMILLA,
        "split": "clasificador: test y validación congelados, train candidato sin congelar (data/splits/)",
        "hash_split": {nombre: hashes[nombre] for nombre in CONJUNTOS},
        "hash_split_origen": "data/manifest.json, splits",
        "hash_fuente": entrada_csv["sha256"],
        "tamano_muestra": {"universo": len(u), **{n: c["filas"] for n, c in cnt.items()}},
        "horas_gpu": None,
        "poblacion": {
            "todas_las_filas": filas_total,
            "llave_vacia_excluida": filas_total - len(u),
            "universo": len(u),
        },
        "subgrupos_en_el_universo": {**conteos_universo(u), "coinciden_con_el_paso_4": True},
        "conjuntos": dict(cnt),
        "clases_fuera_de_train_candidato": dict(fuera),
        "regla7": {
            **{k: v for k, v in r7.items() if k != "top"},
            "top_20_sin_llaves": top_sin_llaves(r7["top"]),
            "archivo_local_de_llaves": {
                "ruta": ruta_llaves.relative_to(REPO_RAIZ).as_posix(),
                "nota": "Local e ignorado por git; contiene llaves normalizadas, nunca nom_estab original.",
            },
        },
        "verificacion": {
            "regenerado_desde_cero_mismos_bytes": True,
            "hash_releido_igual_al_generado": True,
        },
    }


def entrada_split(
    nombre: str, *, ruta: Path, datos: bytes, cnt: Mapping, commit: str, fecha: str, fuentes: Mapping
) -> dict:
    """Entrada de data/manifest.json (splits) de un archivo."""
    entrada = {
        "ruta": ruta.relative_to(REPO_RAIZ).as_posix(),
        "sha256": sha256_bytes(datos),
        "tamano_bytes": len(datos),
        "filas": cnt[nombre]["filas"],
        "llaves": cnt[nombre]["llaves"],
        "clases": cnt[nombre]["clases"],
        "congelado": CONGELADO[nombre],
        "semilla": SEMILLA,
        "procedimiento": PROCEDIMIENTO,
        "script": "src/splits.py",
        "commit_codigo": commit,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "campos": list(CAMPOS),
        "formato": FORMATO,
        "fuentes": dict(fuentes),
        "fecha_registro": fecha,
    }
    if not CONGELADO[nombre]:
        entrada["nota"] = NOTA_TRAIN
    return entrada


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.exploracion_clasificador import verificar_privacidad
    from src.scian import CATALOGO, escribir_manifest, leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escribir", action="store_true", help="escribe y registra los splits (tras la aprobación)")
    args = parser.parse_args(argv)

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    manifest = leer_manifest()
    rutas_splits = {n: CARPETA_SPLITS / ARCHIVOS[n] for n in CONJUNTOS}
    if args.escribir:
        if not arbol_limpio:
            sys.exit("Hay archivos versionados modificados: el commit registrado no sería el código exacto.")
        if any(ENTRADAS[n] in manifest["splits"] for n in CONJUNTOS) or any(r.exists() for r in rutas_splits.values()):
            sys.exit("Ya hay splits en data/splits/ o en el manifest; no se sobrescriben.")
        if not all(esta_ignorado(r) for r in rutas_splits.values()):
            sys.exit("Algún archivo de data/splits/ no está ignorado por git.")

    try:
        filas_total, u, p = generar()
    except ErrorSplit as e:
        sys.exit(f"Me detengo: {e}")
    llaves = u["llave"].tolist()
    r7 = regla7(llaves, p)
    cnt = conteos(u, p)
    fuera = clases_fuera_de_train(u, p)
    ruta_llaves = REPO_RAIZ / "data" / f"{fecha}_fase1_regla7-llaves.csv"
    escribir_llaves_regla7(r7["top"], ruta_llaves)
    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | numpy {np.__version__}")
    _imprimir(filas_total, u, cnt, fuera, r7, ruta_llaves)

    # Bytes y hashes en memoria; el resultado se revisa contra las llaves antes de escribir nada.
    datos = contenidos(u, p)
    hashes = {nombre: sha256_bytes(datos[nombre]) for nombre in CONJUNTOS}
    entrada_csv = entrada_manifest()
    resultado = armar_resultado(
        fecha=fecha, commit=commit, arbol_limpio=arbol_limpio, manifest=manifest, entrada_csv=entrada_csv,
        filas_total=filas_total, u=u, cnt=cnt, fuera=fuera, r7=r7, ruta_llaves=ruta_llaves, hashes=hashes,
    )
    verificar_privacidad(resultado, set(llaves), set())
    print("Revisión de privacidad del resultado: ningún texto es una llave del DENUE.")

    if r7["pasa_del_limite"]:
        sys.exit("Regla 7: la exclusión pasa de 10 %. No se congela; decide Samuel.")
    if not args.escribir:
        print("PUNTO DE CONTROL: no se escribió ningún split. Para escribirlos, --escribir tras la aprobación de Samuel.")
        return

    candado_regla7(r7["filas_excluidas"], r7["filas_candidatas_train"])
    # Regenerar desde cero (CSV, vocabulario, catálogo, universo y partición) y comparar bytes.
    _, u2, p2 = generar()
    if contenidos(u2, p2) != datos:
        sys.exit("Generar de nuevo no da los mismos bytes; no se escribe nada.")
    rutas = escribir_archivos(datos, CARPETA_SPLITS)
    for nombre, ruta in rutas.items():
        if sha256_archivo(ruta) != hashes[nombre]:  # relee solo los bytes
            sys.exit(f"El SHA-256 releído de {ruta.name} no coincide con el generado.")

    fuentes = {
        "denue_cdmx_sha256": entrada_csv["sha256"],
        "catalogo_sha256": manifest["catalogos"][CATALOGO[manifest["scian_version"]]]["sha256"],
        "vocabulario_sha256": manifest["catalogos"][VOCABULARIO]["sha256"],
    }
    for nombre in CONJUNTOS:
        manifest["splits"][ENTRADAS[nombre]] = entrada_split(
            nombre, ruta=rutas[nombre], datos=datos[nombre], cnt=cnt, commit=commit, fecha=fecha, fuentes=fuentes
        )
    escribir_manifest(manifest)

    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_splits-clasificador.json"
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for nombre in CONJUNTOS:
        e = manifest["splits"][ENTRADAS[nombre]]
        print(f"{e['ruta']}: {e['filas']:,} filas, {e['tamano_bytes']:,} bytes, SHA-256 {e['sha256']}, congelado {e['congelado']}")
    print(f"Manifest actualizado; resultados en {salida.relative_to(REPO_RAIZ).as_posix()}")

if __name__ == "__main__":
    main()
