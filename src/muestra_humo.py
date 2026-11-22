"""Muestra de prueba para el dataset privado de Kaggle (Fase 2, paso 6a).

Prueba de punta a punta desechable con Qwen3-0.6B: sus números no se reportan.

Procedimiento (semilla 42; CLAUDE.md, sección c, reglas 1, 3 y 8):

1. Lee los bytes de val.jsonl y train_candidatos.jsonl desde la ruta del manifest y verifica
   su SHA-256 antes de parsearlos. De test solo se usa el SHA-256 registrado en el manifest:
   el archivo de test nunca se abre ni se construye su ruta.
2. Para cada archivo, un generador nuevo:
   idx = sort(numpy.random.default_rng(42).permutation(n)[:k]), sobre las posiciones del
   archivo (ordenado por int(id)). Train candidato: k = 2,000; validación: k = 200.
3. Solo id, nom_estab y codigo_act (sin llave ni marcas; regla 8); JSONL en UTF-8 sin BOM,
   saltos \\n, ordenado por int(id).
4. Candado: en la carpeta solo pueden estar humo_train.jsonl, humo_val.jsonl y
   dataset-metadata.json, y ninguno puede tener el SHA-256 de test.

Uso:
  .venv/Scripts/python -m src.muestra_humo              # punto de control: solo conteos, no escribe
  .venv/Scripts/python -m src.muestra_humo --escribir   # solo tras la aprobación de Samuel
  .venv/Scripts/python -m src.muestra_humo --verificar  # candado sobre la carpeta ya escrita

En la terminal solo van conteos, tamaños y hashes; ningún nom_estab.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.denue import REPO_RAIZ, sha256_archivo
from src.splits import CAMPOS as CAMPOS_ORIGEN
from src.splits import RE_ID, sha256_bytes

SEMILLA = 42
CAMPOS_HUMO = ("id", "nom_estab", "codigo_act")
ARCHIVO_TRAIN = "humo_train.jsonl"
ARCHIVO_VAL = "humo_val.jsonl"
ARCHIVO_METADATA = "dataset-metadata.json"  # nombre que usa el código de la CLI de Kaggle 2.2.5
PERMITIDOS = frozenset({ARCHIVO_TRAIN, ARCHIVO_VAL, ARCHIVO_METADATA})
# Archivo de la muestra: entrada de origen en la sección `splits` del manifest.
ORIGEN = {ARCHIVO_TRAIN: "clasificador_train_candidatos", ARCHIVO_VAL: "clasificador_validacion"}
TAMANOS = {ARCHIVO_TRAIN: 2000, ARCHIVO_VAL: 200}
ENTRADA_TEST = "clasificador_test"  # de esta entrada solo se lee `sha256`
CARPETA_RELATIVA = "../kaggle-datasets/denue-scian-humo"
CARPETA = (REPO_RAIZ / CARPETA_RELATIVA).resolve()
SECCION = "kaggle_datasets"
ENTRADA = "denue_scian_humo"
SCRIPT = "src/muestra_humo.py"

ID_KAGGLE = "samcaballero/denue-scian-humo"
TITULO = "DENUE SCIAN prueba de humo"
LICENCIA = "other"  # "Other (specified in description)": Términos de Libre Uso del INEGI
DESCRIPCION = (
    "Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), CDMX, "
    "edición 05/2026. Términos de Libre Uso de la Información del INEGI. "
    "Muestra desechable para una prueba de punta a punta; privada."
)

PROCEDIMIENTO = (
    "Para cada archivo de origen (ordenado por int(id)), con un generador nuevo: "
    f"idx = sort(numpy.random.default_rng({SEMILLA}).permutation(n)[:k]); "
    f"{ARCHIVO_TRAIN}: k = {TAMANOS[ARCHIVO_TRAIN]} de splits.{ORIGEN[ARCHIVO_TRAIN]}; "
    f"{ARCHIVO_VAL}: k = {TAMANOS[ARCHIVO_VAL]} de splits.{ORIGEN[ARCHIVO_VAL]}. "
    "Solo los campos id, nom_estab y codigo_act."
)
FORMATO = (
    "JSONL: un objeto por línea con las claves en el orden de 'campos'; UTF-8 sin BOM; saltos \\n; "
    "ordenado por int(id)."
)
NOTA = (
    "Prueba de punta a punta desechable con Qwen3-0.6B; sus números no se reportan. Puede incluir "
    "filas que después vayan al hard set: el modelo de prueba se desecha y no se usa para elegir nada."
)
SUBIDA = "La hace Samuel con kaggle datasets create sin -u/--public (privado por omisión en la CLI 2.2.5)."


class ErrorMuestra(ValueError):
    """La muestra o la carpeta no cumplen una condición; no se escribe nada."""


class ErrorCandado(ErrorMuestra):
    """La carpeta tiene un archivo no permitido, le falta uno o alguno tiene el SHA-256 de test."""


# --- Origen ---


def leer_verificado(ruta: Path, sha256_esperado: str) -> bytes:
    """Bytes de un archivo, solo si su SHA-256 es el registrado."""
    datos = Path(ruta).read_bytes()
    sha = sha256_bytes(datos)
    if sha != sha256_esperado:
        raise ErrorMuestra(
            f"El SHA-256 de {Path(ruta).name} no coincide con el manifest: "
            f"calculado {sha}, registrado {sha256_esperado}."
        )
    return datos


def parsear(datos: bytes) -> list[dict]:
    """Registros de un split de `src/splits.py`; verifica claves, ids y orden por int(id).

    Corta solo en "\\n": `splitlines()` también cortaría en U+2028, que JSON deja sin escapar.
    """
    if datos.startswith(b"\xef\xbb\xbf"):
        raise ErrorMuestra("El archivo de origen tiene BOM.")
    texto = datos.decode("utf-8")
    if not texto.endswith("\n"):
        raise ErrorMuestra("El archivo de origen está vacío o no termina en salto de línea.")
    regs = [json.loads(linea) for linea in texto[:-1].split("\n")]
    for r in regs:
        if not isinstance(r, dict) or tuple(r) != CAMPOS_ORIGEN:
            raise ErrorMuestra("Un registro de origen no tiene las claves de src/splits.py en orden.")
        if not isinstance(r["id"], str) or not RE_ID.fullmatch(r["id"]):
            raise ErrorMuestra("Un id de origen no es solo dígitos ASCII.")
    ids = [int(r["id"]) for r in regs]
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ErrorMuestra("El origen no está ordenado por int(id) o tiene id repetidos.")
    return regs


# --- Muestra ---


def muestrear(n: int, k: int, semilla: int = SEMILLA) -> np.ndarray:
    """Posiciones de la muestra: prefijo de una permutación con un generador nuevo, ordenado."""
    if not 0 < k <= n:
        raise ErrorMuestra(f"Se piden {k} filas de {n}.")
    return np.sort(np.random.default_rng(semilla).permutation(n)[:k])


def seleccionar(regs: Sequence[Mapping], indices: Sequence[int]) -> list[dict]:
    """Registros de las posiciones dadas, solo con CAMPOS_HUMO, en orden de int(id)."""
    salida = []
    for i in indices:
        r = regs[int(i)]
        if any(c not in r or not isinstance(r[c], str) for c in CAMPOS_HUMO):
            raise ErrorMuestra("Un registro no tiene id, nom_estab y codigo_act como texto.")
        salida.append({c: r[c] for c in CAMPOS_HUMO})
    salida.sort(key=lambda r: int(r["id"]))
    return salida


def serializar(regs: Sequence[Mapping]) -> bytes:
    """JSONL en UTF-8 sin BOM, con "\\n" al final de cada línea y acentos sin escapar."""
    for r in regs:
        if tuple(r) != CAMPOS_HUMO:
            raise ErrorMuestra("Un registro de la muestra no tiene exactamente id, nom_estab y codigo_act.")
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def metadata_kaggle(
    *, id_kaggle: str = ID_KAGGLE, titulo: str = TITULO, licencia: str = LICENCIA, descripcion: str = DESCRIPCION
) -> bytes:
    """`dataset-metadata.json` en ASCII: la CLI lo lee con `open()` sin codificación (cp1252 en Windows)."""
    duenio, _, slug = id_kaggle.partition("/")
    if not duenio or "/" in slug or not 6 <= len(slug) <= 50:
        raise ErrorMuestra("El id de Kaggle debe ser dueño/slug, con un slug de 6 a 50 caracteres.")
    if not 6 <= len(titulo) <= 50:
        raise ErrorMuestra("El título debe tener de 6 a 50 caracteres.")
    meta = {"title": titulo, "id": id_kaggle, "licenses": [{"name": licencia}], "description": descripcion}
    return (json.dumps(meta, ensure_ascii=True, indent=2) + "\n").encode("ascii")


def preparar(
    manifest: Mapping, raiz: Path = REPO_RAIZ, tamanos: Mapping[str, int] = TAMANOS
) -> tuple[dict[str, bytes], dict[str, list[dict]], dict[str, int]]:
    """Bytes de los tres archivos de la carpeta, en memoria; nunca abre el archivo de test.

    Devuelve (bytes por archivo, registros de cada muestra, filas de cada origen).
    """
    datos: dict[str, bytes] = {}
    regs: dict[str, list[dict]] = {}
    filas_origen: dict[str, int] = {}
    for archivo, entrada in ORIGEN.items():
        e = manifest["splits"][entrada]
        origen = parsear(leer_verificado(Path(raiz) / e["ruta"], e["sha256"]))
        filas_origen[archivo] = len(origen)
        regs[archivo] = seleccionar(origen, muestrear(len(origen), tamanos[archivo]))
        datos[archivo] = serializar(regs[archivo])
    if {r["id"] for r in regs[ARCHIVO_TRAIN]} & {r["id"] for r in regs[ARCHIVO_VAL]}:
        raise ErrorMuestra("Las muestras de train y validación comparten id.")
    datos[ARCHIVO_METADATA] = metadata_kaggle()
    return datos, regs, filas_origen


def conteos(regs: Mapping[str, Sequence[Mapping]]) -> dict:
    """Filas y clases por muestra, y clases de la muestra de validación ausentes de la de train."""
    clases = {a: {r["codigo_act"] for r in rs} for a, rs in regs.items()}
    ausentes = clases[ARCHIVO_VAL] - clases[ARCHIVO_TRAIN]
    return {
        "por_archivo": {a: {"filas": len(rs), "clases": len(clases[a])} for a, rs in regs.items()},
        "val_clases_ausentes_de_train": len(ausentes),
        "val_filas_de_clases_ausentes": sum(r["codigo_act"] in ausentes for r in regs[ARCHIVO_VAL]),
    }


# --- Candado ---


def hashes_del_manifest(obj, ruta: str = "") -> dict[str, list[str]]:
    """SHA-256 → rutas del manifest donde aparece (claves `sha256` o terminadas en `_sha256`)."""
    res: dict[str, list[str]] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            sub = f"{ruta}.{k}" if ruta else str(k)
            if isinstance(v, str) and (k == "sha256" or str(k).endswith("_sha256")):
                res.setdefault(v, []).append(sub)
            else:
                for h, rutas in hashes_del_manifest(v, sub).items():
                    res.setdefault(h, []).extend(rutas)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            for h, rutas in hashes_del_manifest(v, f"{ruta}[{i}]").items():
                res.setdefault(h, []).extend(rutas)
    return res


def verificar_fuera_del_repo(carpeta: Path, raiz: Path = REPO_RAIZ) -> None:
    c, r = Path(carpeta).resolve(), Path(raiz).resolve()
    if c == r or r in c.parents:
        raise ErrorCandado(f"La carpeta {c} está dentro del repo; debe quedar fuera.")


def _comparar_con_test(hashes: Mapping[str, str], sha_test: str, permitidos: frozenset[str] = PERMITIDOS) -> None:
    nombres = set(hashes)
    if nombres != permitidos:
        raise ErrorCandado(
            f"Archivos no permitidos: {sorted(nombres - permitidos)}; faltan: {sorted(permitidos - nombres)}."
        )
    iguales = sorted(n for n, h in hashes.items() if h == sha_test)
    if iguales:
        raise ErrorCandado(f"{iguales} tiene el SHA-256 del test congelado; me detengo (regla 3).")


def candado_bytes(
    datos: Mapping[str, bytes], sha_test: str, permitidos: frozenset[str] = PERMITIDOS
) -> dict[str, str]:
    """Candado sobre los bytes en memoria; devuelve el SHA-256 de cada archivo."""
    hashes = {n: sha256_bytes(b) for n, b in datos.items()}
    _comparar_con_test(hashes, sha_test, permitidos)
    return hashes


def candado_carpeta(
    carpeta: Path, sha_test: str, raiz: Path = REPO_RAIZ, permitidos: frozenset[str] = PERMITIDOS
) -> dict[str, str]:
    """Candado sobre la carpeta escrita: fuera del repo, solo los archivos permitidos y sin el hash de test."""
    carpeta = Path(carpeta)
    verificar_fuera_del_repo(carpeta, raiz)
    if not carpeta.is_dir():
        raise ErrorCandado(f"No existe la carpeta {carpeta}.")
    entradas = sorted(os.listdir(carpeta))
    raras = [e for e in entradas if (carpeta / e).is_symlink() or not (carpeta / e).is_file()]
    if raras:
        raise ErrorCandado(f"Hay subcarpetas o enlaces en la carpeta: {raras}.")
    hashes = {e: sha256_archivo(carpeta / e) for e in entradas}
    _comparar_con_test(hashes, sha_test, permitidos)
    return hashes


# --- Escritura y manifest ---


def escribir_carpeta(
    datos: Mapping[str, bytes], carpeta: Path, raiz: Path = REPO_RAIZ, permitidos: frozenset[str] = PERMITIDOS
) -> None:
    """Escribe los archivos en binario; se niega si la carpeta ya tiene algo o está en el repo."""
    carpeta = Path(carpeta)
    verificar_fuera_del_repo(carpeta, raiz)
    if set(datos) - permitidos:
        raise ErrorCandado(f"Archivos no permitidos: {sorted(set(datos) - permitidos)}.")
    if carpeta.exists() and any(carpeta.iterdir()):
        raise ErrorMuestra(f"La carpeta {carpeta} ya tiene archivos; no se sobrescribe.")
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre in sorted(datos):
        with open(carpeta / nombre, "xb") as f:  # "x": falla si el archivo ya existe
            f.write(datos[nombre])


def entrada_kaggle(
    *, manifest: Mapping, datos: Mapping[str, bytes], regs: Mapping[str, Sequence[Mapping]], commit: str, fecha: str
) -> dict:
    """Entrada de data/manifest.json (sección kaggle_datasets) de la muestra de prueba."""
    cnt = conteos(regs)["por_archivo"]
    archivos = {}
    for archivo, entrada in ORIGEN.items():
        archivos[archivo] = {
            "ruta": f"{CARPETA_RELATIVA}/{archivo}",
            "origen": f"splits.{entrada}",
            "origen_sha256": manifest["splits"][entrada]["sha256"],
            "filas": cnt[archivo]["filas"],
            "clases": cnt[archivo]["clases"],
            "tamano_bytes": len(datos[archivo]),
            "sha256": sha256_bytes(datos[archivo]),
        }
    return {
        "id_kaggle": ID_KAGGLE,
        "titulo": TITULO,
        "licencia": LICENCIA,
        "privado": True,
        "desechable": True,
        "subida": SUBIDA,
        "carpeta": CARPETA_RELATIVA,
        "archivos": archivos,
        "metadata": {
            "ruta": f"{CARPETA_RELATIVA}/{ARCHIVO_METADATA}",
            "codificacion": "ascii",
            "tamano_bytes": len(datos[ARCHIVO_METADATA]),
            "sha256": sha256_bytes(datos[ARCHIVO_METADATA]),
        },
        "semilla": SEMILLA,
        "procedimiento": PROCEDIMIENTO,
        "campos": list(CAMPOS_HUMO),
        "formato": FORMATO,
        "script": SCRIPT,
        "commit_codigo": commit,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "fecha_registro": fecha,
        "nota": NOTA,
    }


def agregar_entrada(manifest: Mapping, entrada: Mapping, nombre: str = ENTRADA) -> dict:
    """Copia del manifest con la entrada nueva; ninguna otra sección cambia."""
    if nombre in manifest.get(SECCION, {}):
        raise ErrorMuestra(f"El manifest ya tiene {SECCION}.{nombre}; no se sobrescribe.")
    nuevo = copy.deepcopy(dict(manifest))
    nuevo.setdefault(SECCION, {})[nombre] = copy.deepcopy(dict(entrada))
    otras_iguales(manifest, nuevo, nombre)
    return nuevo


def otras_iguales(antes: Mapping, despues: Mapping, nombre: str = ENTRADA) -> None:
    """Falla si cambió alguna sección del manifest distinta de la de Kaggle o de la propia entrada."""
    claves = (set(antes) | set(despues)) - {SECCION}
    cambiadas = sorted(k for k in claves if antes.get(k) != despues.get(k))
    otras = {k: v for k, v in antes.get(SECCION, {}).items() if k != nombre}
    if cambiadas or {k: v for k, v in despues.get(SECCION, {}).items() if k != nombre} != otras:
        raise ErrorMuestra(f"Cambiarían otras secciones del manifest: {cambiadas or [SECCION]}.")


# --- Programa principal ---


def _git(*args: str) -> str:
    resultado = subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True)
    return resultado.stdout.strip()


def codigo_versionado(ruta: str = SCRIPT) -> bool:
    """True si el archivo está en git y sin cambios (un archivo nuevo sin commit da False)."""
    r = subprocess.run(["git", "ls-files", "--error-unmatch", ruta], cwd=REPO_RAIZ, capture_output=True)
    return r.returncode == 0 and _git("status", "--porcelain", "--", ruta) == ""


def _coincidencias(hashes: Mapping[str, str], manifest: Mapping) -> tuple[dict, dict]:
    """Coincidencias con hashes del manifest: (con la propia entrada registrada, con cualquier otra)."""
    del_manifest = hashes_del_manifest(manifest)
    propia = f"{SECCION}.{ENTRADA}."
    registradas, otras = {}, {}
    for nombre, h in sorted(hashes.items()):
        for ruta in del_manifest.get(h, []):
            (registradas if ruta.startswith(propia) else otras).setdefault(nombre, []).append(ruta)
    return registradas, otras


def _imprimir_hashes(hashes: Mapping[str, str], datos: Mapping[str, bytes] | None = None) -> None:
    for nombre in sorted(hashes):
        tamano = f"{len(datos[nombre]):,} bytes, " if datos is not None else ""
        print(f"  {nombre}: {tamano}SHA-256 {hashes[nombre]}")


def verificar(manifest: Mapping) -> None:
    """--verificar: candado sobre la carpeta escrita y, si ya está registrada, hashes iguales a los del manifest."""
    sha_test = manifest["splits"][ENTRADA_TEST]["sha256"]
    try:
        hashes = candado_carpeta(CARPETA, sha_test)
    except ErrorMuestra as e:
        sys.exit(f"Me detengo: {e}")
    print(f"Carpeta: {CARPETA}")
    _imprimir_hashes(hashes)
    registrada = manifest.get(SECCION, {}).get(ENTRADA)
    if registrada:
        esperados = {a: d["sha256"] for a, d in registrada["archivos"].items()}
        esperados[ARCHIVO_METADATA] = registrada["metadata"]["sha256"]
        if esperados != hashes:
            sys.exit("Me detengo: los SHA-256 de la carpeta no son los registrados en el manifest.")
        print(f"Los tres SHA-256 coinciden con {SECCION}.{ENTRADA} del manifest.")
    else:
        print(f"Aún no hay entrada {SECCION}.{ENTRADA} en el manifest.")
    _, otras = _coincidencias(hashes, manifest)
    if otras:
        sys.exit(f"Me detengo: archivos con el hash de otra entrada del manifest: {otras}.")
    print("Candado: solo los 3 archivos permitidos, fuera del repo, ninguno con el SHA-256 de test "
          "ni de otra entrada del manifest.")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.exploracion_clasificador import verificar_privacidad
    from src.normalizacion import normalizar_nombre
    from src.scian import escribir_manifest, leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--escribir", action="store_true", help="escribe la carpeta y la entrada del manifest")
    modo.add_argument("--verificar", action="store_true", help="solo el candado sobre la carpeta escrita")
    args = parser.parse_args(argv)

    manifest = leer_manifest()
    if args.verificar:
        verificar(manifest)
        return

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    versionado = codigo_versionado()
    if args.escribir:
        if not (arbol_limpio and versionado):
            sys.exit(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
        if ENTRADA in manifest.get(SECCION, {}):
            sys.exit(f"El manifest ya tiene {SECCION}.{ENTRADA}; no se sobrescribe.")
        if CARPETA.exists() and any(CARPETA.iterdir()):
            sys.exit(f"La carpeta {CARPETA} ya tiene archivos; no se sobrescribe.")

    sha_test = manifest["splits"][ENTRADA_TEST]["sha256"]
    try:
        verificar_fuera_del_repo(CARPETA)
        datos, regs, filas_origen = preparar(manifest)
        hashes = candado_bytes(datos, sha_test)
    except ErrorMuestra as e:
        sys.exit(f"Me detengo: {e}")

    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | {SCRIPT} versionado sin cambios: {versionado} "
          f"| numpy {np.__version__}")
    print("Origen (SHA-256 igual al del manifest, verificado antes de parsear):")
    for archivo, entrada in ORIGEN.items():
        e = manifest["splits"][entrada]
        print(f"  {e['ruta']}: {filas_origen[archivo]:,} filas, SHA-256 {e['sha256']}")
    cnt = conteos(regs)
    print(f"Muestras (semilla {SEMILLA}, un generador nuevo por archivo):")
    for archivo in ORIGEN:
        c = cnt["por_archivo"][archivo]
        print(f"  {archivo}: {c['filas']:,} filas, {c['clases']:,} clases, {len(datos[archivo]):,} bytes, "
              f"SHA-256 {hashes[archivo]}")
    print(f"  {ARCHIVO_METADATA}: {len(datos[ARCHIVO_METADATA]):,} bytes (ASCII), SHA-256 {hashes[ARCHIVO_METADATA]}")
    compartidos = {r["id"] for r in regs[ARCHIVO_TRAIN]} & {r["id"] for r in regs[ARCHIVO_VAL]}
    print(f"Ids compartidos entre las dos muestras: {len(compartidos)}")
    print(f"Clases de {ARCHIVO_VAL} ausentes de {ARCHIVO_TRAIN}: {cnt['val_clases_ausentes_de_train']:,} "
          f"({cnt['val_filas_de_clases_ausentes']:,} filas)")
    _, otras = _coincidencias(hashes, manifest)
    if otras:
        sys.exit(f"Me detengo: archivos con el hash de otra entrada del manifest: {otras}.")
    print("Candado: solo los 3 archivos permitidos; ninguno tiene el SHA-256 de test (leído del manifest; "
          "el archivo no se abrió) ni el de otra entrada del manifest.")
    print(f"Carpeta destino: {CARPETA} (fuera del repo: True; existe: {CARPETA.exists()})")

    entrada = entrada_kaggle(manifest=manifest, datos=datos, regs=regs, commit=commit, fecha=fecha)
    llaves = {normalizar_nombre(r["nom_estab"]) for rs in regs.values() for r in rs}
    verificar_privacidad({"entrada": entrada, "metadata": json.loads(datos[ARCHIVO_METADATA])}, llaves, set())
    print("Revisión de privacidad de la entrada del manifest y del metadata: ningún texto es una llave de las muestras.")

    if not args.escribir:
        print("PUNTO DE CONTROL: no se escribió nada. Para escribir, --escribir tras la aprobación de Samuel.")
        return

    # Regenerar desde cero (manifest y archivos de origen releídos) y comparar bytes.
    datos2, _, _ = preparar(leer_manifest())
    if datos2 != datos:
        sys.exit("Generar de nuevo no da los mismos bytes; no se escribe nada.")
    try:
        escribir_carpeta(datos, CARPETA)
        releidos = candado_carpeta(CARPETA, sha_test)
    except ErrorMuestra as e:
        sys.exit(f"Me detengo: {e}")
    if releidos != hashes:
        sys.exit("Los SHA-256 releídos de la carpeta no coinciden con los generados.")

    nuevo = agregar_entrada(manifest, entrada)
    escribir_manifest(nuevo)
    releido = leer_manifest()
    otras_iguales(manifest, releido)
    if releido[SECCION][ENTRADA] != entrada:
        sys.exit("La entrada releída del manifest no es la generada.")
    print(f"Carpeta escrita y releída: {CARPETA}")
    _imprimir_hashes(releidos, datos)
    print(f"Manifest actualizado: {SECCION}.{ENTRADA} (las demás secciones quedaron idénticas).")


if __name__ == "__main__":
    main()
