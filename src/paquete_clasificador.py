"""Paquete del clasificador para un dataset privado de Kaggle (Fase 1, paso 8).

Carpeta fuera del repo, ../kaggle-datasets/denue-scian-clasificador, con:

- train_final.jsonl: todo data/splits/train_final.jsonl, solo con id, nom_estab y codigo_act;
- val.jsonl: todo data/splits/val.jsonl, con los mismos campos;
- dataset-metadata.json, en ASCII, con el crédito al INEGI.

Reutiliza el patrón y el candado de src/muestra_humo.py. Los archivos de origen se leen con su
SHA-256 verificado antes de parsearlos. El candado corre al escribir y al verificar, y rechaza:

- archivos fuera de la lista (incluidos los ocultos, las subcarpetas y los enlaces) o faltantes,
  y una carpeta dentro del repo;
- cualquier archivo con el SHA-256 del test o del hard set, o de cualquier otra entrada del
  manifest (salvo la propia, ya registrada).

Del test y del hard set solo se lee el SHA-256 del manifest: sus archivos nunca se abren ni se
construye su ruta.

Uso:
  .venv/Scripts/python -m src.paquete_clasificador              # punto de control: no escribe
  .venv/Scripts/python -m src.paquete_clasificador --escribir   # carpeta y entrada del manifest
  .venv/Scripts/python -m src.paquete_clasificador --verificar  # candado sobre la carpeta escrita

La subida la hace Samuel con kaggle datasets create -p <carpeta>, sin -u (privado).
En la terminal solo van conteos, tamaños y hashes; ningún nom_estab.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.denue import REPO_RAIZ
from src.muestra_humo import (
    SECCION,
    ErrorCandado,
    ErrorMuestra,
    agregar_entrada,
    candado_bytes,
    candado_carpeta,
    codigo_versionado,
    escribir_carpeta,
    hashes_del_manifest,
    leer_verificado,
    metadata_kaggle,
    otras_iguales,
    parsear,
    seleccionar,
    serializar,
    verificar_fuera_del_repo,
)
from src.muestra_humo import CAMPOS_HUMO as CAMPOS
from src.muestra_humo import FORMATO
from src.splits import sha256_bytes

SCRIPT = "src/paquete_clasificador.py"
ARCHIVO_TRAIN = "train_final.jsonl"
ARCHIVO_VAL = "val.jsonl"
ARCHIVO_METADATA = "dataset-metadata.json"
PERMITIDOS = frozenset({ARCHIVO_TRAIN, ARCHIVO_VAL, ARCHIVO_METADATA})
# Archivo del paquete: entrada de origen en la sección `splits` del manifest.
ORIGEN = {ARCHIVO_TRAIN: "clasificador_train_final", ARCHIVO_VAL: "clasificador_validacion"}
ENTRADA_TEST = "clasificador_test"  # de esta entrada solo se lee `sha256`
ENTRADA_HARD = "clasificador_hard"  # de esta entrada solo se lee `sha256`
CARPETA_RELATIVA = "../kaggle-datasets/denue-scian-clasificador"
CARPETA = (REPO_RAIZ / CARPETA_RELATIVA).resolve()
ENTRADA = "denue_scian_clasificador"

ID_KAGGLE = "samcaballero/denue-scian-clasificador"
TITULO = "DENUE SCIAN clasificador"
LICENCIA = "other"  # "Other (specified in description)": Términos de Libre Uso del INEGI
DESCRIPCION = (
    "Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), CDMX, "
    "edición 05/2026. Términos de Libre Uso de la Información del INEGI. "
    "Nombre del establecimiento y clase SCIAN 2023: train final (con tope por clase) y validación "
    "del clasificador; privado."
)
PROCEDIMIENTO = (
    f"{ARCHIVO_TRAIN}: todas las filas de splits.{ORIGEN[ARCHIVO_TRAIN]}; {ARCHIVO_VAL}: todas las filas de "
    f"splits.{ORIGEN[ARCHIVO_VAL]}. SHA-256 de cada origen verificado antes de parsear; solo los campos id, "
    "nom_estab y codigo_act, en orden de int(id). Sin sorteo."
)
SUBIDA = "La hace Samuel con kaggle datasets create -p <carpeta> sin -u/--public (privado por omisión en la CLI 2.2.5)."
NOTA = (
    "Datos de entrenamiento y validación del clasificador. Test y hard set no suben a Kaggle hasta la Fase 4 "
    "(regla 3); el candado rechaza sus SHA-256."
)


class ErrorPaquete(ErrorMuestra):
    """El paquete no cumple una condición; no se escribe nada."""


# --- Contenido ---


def preparar(manifest: Mapping, raiz: Path = REPO_RAIZ) -> tuple[dict[str, bytes], dict[str, list[dict]]]:
    """Bytes de los tres archivos del paquete, en memoria, y los registros completos de cada origen."""
    regs: dict[str, list[dict]] = {}
    for archivo, entrada in ORIGEN.items():
        e = manifest["splits"][entrada]
        origen = parsear(leer_verificado(Path(raiz) / e["ruta"], e["sha256"]))
        if len(origen) != e["filas"]:
            raise ErrorPaquete(f"Las filas de {entrada} no son las del manifest.")
        regs[archivo] = origen
    tr, va = regs[ARCHIVO_TRAIN], regs[ARCHIVO_VAL]
    if {r["id"] for r in tr} & {r["id"] for r in va}:
        raise ErrorPaquete("Train final y validación comparten id.")
    if {r["llave"] for r in tr} & {r["llave"] for r in va}:
        raise ErrorPaquete("Train final y validación comparten llaves.")
    datos: dict[str, bytes] = {}
    for archivo, origen in regs.items():
        reducidos = seleccionar(origen, range(len(origen)))
        if [r["id"] for r in reducidos] != [r["id"] for r in origen]:
            raise ErrorPaquete(f"Los ids de {archivo} no son exactamente los de su origen.")
        datos[archivo] = serializar(reducidos)
    datos[ARCHIVO_METADATA] = metadata_kaggle(
        id_kaggle=ID_KAGGLE, titulo=TITULO, licencia=LICENCIA, descripcion=DESCRIPCION
    )
    return datos, regs


def conteos(regs: Mapping[str, Sequence[Mapping]]) -> dict:
    clases = {a: {r["codigo_act"] for r in rs} for a, rs in regs.items()}
    ausentes = clases[ARCHIVO_VAL] - clases[ARCHIVO_TRAIN]
    return {
        "por_archivo": {a: {"filas": len(rs), "clases": len(clases[a])} for a, rs in regs.items()},
        "val_clases_ausentes_de_train": len(ausentes),
        "val_filas_de_clases_ausentes": sum(r["codigo_act"] in ausentes for r in regs[ARCHIVO_VAL]),
    }


# --- Candado ---


def candado_manifest(hashes: Mapping[str, str], manifest: Mapping) -> None:
    """Ningún archivo puede tener el SHA-256 de test, de hard ni de otra entrada del manifest (salvo la propia)."""
    s = manifest["splits"]
    prohibidos = {s[ENTRADA_TEST]["sha256"]: "test", s[ENTRADA_HARD]["sha256"]: "hard set"}
    iguales = sorted(f"{n} ({prohibidos[h]})" for n, h in hashes.items() if h in prohibidos)
    if iguales:
        raise ErrorCandado(f"{iguales} tiene el SHA-256 de un conjunto congelado; me detengo (regla 3).")
    propia = f"{SECCION}.{ENTRADA}."
    del_manifest = hashes_del_manifest(manifest)
    otras = {}
    for nombre, h in sorted(hashes.items()):
        rutas = [r for r in del_manifest.get(h, []) if not r.startswith(propia)]
        if rutas:
            otras[nombre] = rutas
    if otras:
        raise ErrorCandado(f"Archivos con el SHA-256 de otra entrada del manifest: {otras}.")


def candado_en_memoria(datos: Mapping[str, bytes], manifest: Mapping) -> dict[str, str]:
    """Candado sobre los bytes: solo los archivos permitidos y ningún hash del manifest."""
    hashes = candado_bytes(datos, manifest["splits"][ENTRADA_TEST]["sha256"], PERMITIDOS)
    candado_manifest(hashes, manifest)
    return hashes


def candado_en_carpeta(carpeta: Path, manifest: Mapping, raiz: Path = REPO_RAIZ) -> dict[str, str]:
    """Candado sobre la carpeta escrita: fuera del repo, solo los archivos permitidos y ningún hash del manifest."""
    hashes = candado_carpeta(carpeta, manifest["splits"][ENTRADA_TEST]["sha256"], raiz, PERMITIDOS)
    candado_manifest(hashes, manifest)
    return hashes


def verificar(manifest: Mapping, carpeta: Path = CARPETA, raiz: Path = REPO_RAIZ) -> tuple[dict[str, str], bool]:
    """Candado sobre la carpeta y, si la entrada ya está registrada, hashes iguales a los del manifest.

    Devuelve los hashes y si la entrada estaba registrada.
    """
    hashes = candado_en_carpeta(carpeta, manifest, raiz)
    registrada = manifest.get(SECCION, {}).get(ENTRADA)
    if registrada:
        esperados = {a: d["sha256"] for a, d in registrada["archivos"].items()}
        esperados[ARCHIVO_METADATA] = registrada["metadata"]["sha256"]
        if esperados != hashes:
            raise ErrorCandado("Los SHA-256 de la carpeta no son los registrados en el manifest.")
    return hashes, bool(registrada)


# --- Manifest ---


def entrada_kaggle(
    *, manifest: Mapping, datos: Mapping[str, bytes], regs: Mapping[str, Sequence[Mapping]], commit: str, fecha: str
) -> dict:
    """Entrada kaggle_datasets.denue_scian_clasificador del manifest."""
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
        "desechable": False,
        "subida": SUBIDA,
        "carpeta": CARPETA_RELATIVA,
        "archivos": archivos,
        "metadata": {
            "ruta": f"{CARPETA_RELATIVA}/{ARCHIVO_METADATA}",
            "codificacion": "ascii",
            "tamano_bytes": len(datos[ARCHIVO_METADATA]),
            "sha256": sha256_bytes(datos[ARCHIVO_METADATA]),
        },
        "procedimiento": PROCEDIMIENTO,
        "campos": list(CAMPOS),
        "formato": FORMATO,
        "script": SCRIPT,
        "commit_codigo": commit,
        "python": platform.python_version(),
        "fecha_registro": fecha,
        "nota": NOTA,
    }


# --- Programa principal ---


def _git(*args: str) -> str:
    resultado = subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True)
    return resultado.stdout.strip()


def _imprimir_hashes(hashes: Mapping[str, str], datos: Mapping[str, bytes] | None = None) -> None:
    for nombre in sorted(hashes):
        tamano = f"{len(datos[nombre]):,} bytes, " if datos is not None else ""
        print(f"  {nombre}: {tamano}SHA-256 {hashes[nombre]}")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.exploracion_clasificador import ErrorPrivacidad, verificar_privacidad
    from src.scian import escribir_manifest, leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--escribir", action="store_true", help="escribe la carpeta y la entrada del manifest")
    modo.add_argument("--verificar", action="store_true", help="solo el candado sobre la carpeta escrita")
    args = parser.parse_args(argv)

    manifest = leer_manifest()
    if args.verificar:
        try:
            hashes, registrada = verificar(manifest)
        except ErrorMuestra as e:
            sys.exit(f"Me detengo: {e}")
        print(f"Carpeta: {CARPETA}")
        _imprimir_hashes(hashes)
        if registrada:
            print(f"Los tres SHA-256 coinciden con {SECCION}.{ENTRADA} del manifest.")
        else:
            print(f"Aún no hay entrada {SECCION}.{ENTRADA} en el manifest.")
        print("Candado: solo los 3 archivos permitidos, fuera del repo, ninguno con el SHA-256 de test, de hard "
              "ni de otra entrada del manifest.")
        return

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    versionado = codigo_versionado(SCRIPT)
    if ORIGEN[ARCHIVO_TRAIN] not in manifest["splits"]:
        sys.exit(f"El manifest aún no tiene splits.{ORIGEN[ARCHIVO_TRAIN]}: primero va src/tope_clase.py --escribir.")
    if args.escribir:
        if not (arbol_limpio and versionado):
            sys.exit(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
        if ENTRADA in manifest.get(SECCION, {}):
            sys.exit(f"El manifest ya tiene {SECCION}.{ENTRADA}; no se sobrescribe.")
        if CARPETA.exists() and any(CARPETA.iterdir()):
            sys.exit(f"La carpeta {CARPETA} ya tiene archivos; no se sobrescribe.")

    try:
        verificar_fuera_del_repo(CARPETA)
        datos, regs = preparar(manifest)
        hashes = candado_en_memoria(datos, manifest)
    except ErrorMuestra as e:
        sys.exit(f"Me detengo: {e}")

    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | {SCRIPT} versionado sin cambios: {versionado}")
    print("Origen (SHA-256 igual al del manifest, verificado antes de parsear):")
    for archivo, entrada in ORIGEN.items():
        e = manifest["splits"][entrada]
        print(f"  {e['ruta']}: {len(regs[archivo]):,} filas, SHA-256 {e['sha256']}")
    cnt = conteos(regs)
    print("Paquete (todas las filas; solo id, nom_estab y codigo_act):")
    for archivo in ORIGEN:
        c = cnt["por_archivo"][archivo]
        print(f"  {archivo}: {c['filas']:,} filas, {c['clases']:,} clases, {len(datos[archivo]):,} bytes, "
              f"SHA-256 {hashes[archivo]}")
    print(f"  {ARCHIVO_METADATA}: {len(datos[ARCHIVO_METADATA]):,} bytes (ASCII), SHA-256 {hashes[ARCHIVO_METADATA]}")
    print("Ids y llaves compartidos entre train final y validación: 0 y 0 (comprobado).")
    print(f"Clases de {ARCHIVO_VAL} ausentes de {ARCHIVO_TRAIN}: {cnt['val_clases_ausentes_de_train']:,} "
          f"({cnt['val_filas_de_clases_ausentes']:,} filas)")
    print("Candado: solo los 3 archivos permitidos; ninguno tiene el SHA-256 de test ni de hard (leídos del manifest; "
          "los archivos no se abrieron) ni el de otra entrada del manifest.")
    print(f"Carpeta destino: {CARPETA} (fuera del repo: True; existe: {CARPETA.exists()})")

    entrada = entrada_kaggle(manifest=manifest, datos=datos, regs=regs, commit=commit, fecha=fecha)
    llaves = {r["llave"] for rs in regs.values() for r in rs}
    try:
        verificar_privacidad({"entrada": entrada, "metadata": json.loads(datos[ARCHIVO_METADATA])}, llaves, set())
    except ErrorPrivacidad as e:
        sys.exit(f"Me detengo: {e}")
    print("Revisión de privacidad de la entrada del manifest y del metadata: ningún texto es una llave.")

    if not args.escribir:
        print("PUNTO DE CONTROL: no se escribió nada. Para escribir, --escribir.")
        return

    # Regenerar desde cero (manifest y archivos de origen releídos) y comparar bytes.
    datos2, _ = preparar(leer_manifest())
    if datos2 != datos:
        sys.exit("Generar de nuevo no da los mismos bytes; no se escribe nada.")
    try:
        escribir_carpeta(datos, CARPETA, permitidos=PERMITIDOS)
        releidos, _ = verificar(manifest)
    except ErrorMuestra as e:
        sys.exit(f"Me detengo: {e}")
    if releidos != hashes:
        sys.exit("Los SHA-256 releídos de la carpeta no coinciden con los generados.")

    nuevo = agregar_entrada(manifest, entrada, ENTRADA)
    escribir_manifest(nuevo)
    releido = leer_manifest()
    otras_iguales(manifest, releido, ENTRADA)
    if releido[SECCION][ENTRADA] != entrada:
        sys.exit("La entrada releída del manifest no es la generada.")
    try:
        verificar(releido)
    except ErrorMuestra as e:
        sys.exit(f"Me detengo: {e}")
    print(f"Carpeta escrita y releída: {CARPETA}")
    _imprimir_hashes(releidos, datos)
    print(f"Manifest actualizado: {SECCION}.{ENTRADA} (las demás secciones quedaron idénticas); --verificar pasa.")


if __name__ == "__main__":
    main()
