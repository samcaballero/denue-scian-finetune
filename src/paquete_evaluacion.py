"""Paquete de evaluación del clasificador para un dataset privado de Kaggle (Fase 4, paso 14, Tema B).

Carpeta fuera del repo, ../kaggle-datasets/denue-scian-evaluacion, con:

- test.jsonl: las 2,000 filas de data/splits/test.jsonl, solo con id y nom_estab;
- hard.jsonl: las 105 filas de data/splits/hard.jsonl, solo con id y nom_estab;
- dataset-metadata.json, en ASCII, con el crédito al INEGI.

Sin codigo_act, tipo, llave ni marcas: el kernel de la Fase 4 genera las respuestas sin ver las
etiquetas, y los baselines predicen desde estas mismas copias (evals/fase4_preregistro.json). Las
etiquetas solo se leen en la laptop, una vez, al abrir test.

Reutiliza el patrón y el candado de src/paquete_clasificador.py y src/muestra_humo.py. Los archivos
de origen se leen con su SHA-256 verificado antes de parsearlos, y codigo_act, tipo, llave y las
marcas se descartan al parsear. El candado corre al escribir y al verificar, y rechaza:

- archivos fuera de la lista (incluidos los ocultos, las subcarpetas y los enlaces) o faltantes,
  y una carpeta dentro del repo;
- cualquier archivo con el SHA-256 del test o del hard set completos, o de cualquier otra entrada
  del manifest (salvo la propia, ya registrada);
- (nuevo) cualquier línea de test.jsonl o hard.jsonl cuyas claves no sean exactamente id y
  nom_estab, y un número de filas distinto del de su origen.

Uso:
  .venv/Scripts/python -m src.paquete_evaluacion              # punto de control: no escribe
  .venv/Scripts/python -m src.paquete_evaluacion --escribir   # carpeta y entrada del manifest
  .venv/Scripts/python -m src.paquete_evaluacion --verificar  # candado sobre la carpeta escrita

La subida la hace Samuel con kaggle datasets create -p <carpeta>, sin -u (privado).
En la terminal solo van conteos, columnas, tamaños y hashes; ningún nom_estab, id ni código.
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
    verificar_fuera_del_repo,
)
from src.muestra_humo import FORMATO
from src.preregistro_fase4 import ErrorPreregistro, parsear_jsonl
from src.splits import sha256_bytes

SCRIPT = "src/paquete_evaluacion.py"
CAMPOS = ("id", "nom_estab")
ARCHIVO_TEST = "test.jsonl"
ARCHIVO_HARD = "hard.jsonl"
ARCHIVO_METADATA = "dataset-metadata.json"
PERMITIDOS = frozenset({ARCHIVO_TEST, ARCHIVO_HARD, ARCHIVO_METADATA})
# Archivo del paquete: entrada de origen en la sección `splits` del manifest.
ORIGEN = {ARCHIVO_TEST: "clasificador_test", ARCHIVO_HARD: "clasificador_hard"}
CARPETA_RELATIVA = "../kaggle-datasets/denue-scian-evaluacion"
CARPETA = (REPO_RAIZ / CARPETA_RELATIVA).resolve()
ENTRADA = "denue_scian_evaluacion"

ID_KAGGLE = "samcaballero/denue-scian-evaluacion"
TITULO = "DENUE SCIAN evaluacion"
LICENCIA = "other"  # "Other (specified in description)": Términos de Libre Uso del INEGI
DESCRIPCION = (
    "Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), CDMX, "
    "edición 05/2026. Términos de Libre Uso de la Información del INEGI. "
    "Nombre del establecimiento, sin la clase SCIAN, del test y del hard set del clasificador, "
    "para la evaluación final; privado."
)
PROCEDIMIENTO = (
    f"{ARCHIVO_TEST}: todas las filas de splits.{ORIGEN[ARCHIVO_TEST]}; {ARCHIVO_HARD}: todas las filas de "
    f"splits.{ORIGEN[ARCHIVO_HARD]}. SHA-256 de cada origen verificado antes de parsear; solo los campos id y "
    "nom_estab (sin codigo_act, tipo, llave ni marcas), en orden de int(id). Sin sorteo."
)
SUBIDA = "La hace Samuel con kaggle datasets create -p <carpeta> sin -u/--public (privado por omisión en la CLI 2.2.5)."
NOTA = (
    "Fase 4: test y hard set sin etiquetas, para que el kernel genere las respuestas sin ver codigo_act y los "
    "baselines predigan desde las mismas copias. Las etiquetas solo se leen en la laptop, una vez, al abrir test "
    "(evals/fase4_preregistro.json). El candado rechaza cualquier archivo con el SHA-256 de test, de hard o de otra "
    "entrada del manifest, y cualquier línea con claves distintas de id y nom_estab."
)


class ErrorPaquete(ErrorMuestra):
    """El paquete no cumple una condición; no se escribe nada."""


# --- Contenido ---


def serializar(regs: Sequence[Mapping]) -> bytes:
    """JSONL en UTF-8 sin BOM, con "\\n" al final de cada línea y acentos sin escapar."""
    for r in regs:
        if tuple(r) != CAMPOS:
            raise ErrorPaquete("Un registro del paquete no tiene exactamente id y nom_estab.")
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def preparar(manifest: Mapping, raiz: Path = REPO_RAIZ) -> tuple[dict[str, bytes], dict[str, int], set[str]]:
    """Bytes de los tres archivos del paquete, en memoria; filas de cada uno; llaves de origen (solo privacidad).

    De cada fila de origen solo se conservan id y nom_estab; codigo_act, tipo, llave y las marcas
    se descartan aquí mismo y nunca se devuelven.
    """
    datos: dict[str, bytes] = {}
    filas: dict[str, int] = {}
    ids: dict[str, list[str]] = {}
    llaves: set[str] = set()
    for archivo, entrada in ORIGEN.items():
        e = manifest["splits"][entrada]
        try:
            origen = parsear_jsonl(leer_verificado(Path(raiz) / e["ruta"], e["sha256"]), e["campos"])
        except ErrorPreregistro as err:
            raise ErrorPaquete(f"{entrada}: {err}") from err
        if len(origen) != e["filas"]:
            raise ErrorPaquete(f"Las filas de {entrada} no son las del manifest.")
        reducidos = [{"id": r["id"], "nom_estab": r["nom_estab"]} for r in origen]
        llaves |= {r["llave"] for r in origen}
        ids[archivo] = [r["id"] for r in reducidos]
        if ids[archivo] != [r["id"] for r in origen]:  # no debería pasar: misma lista, mismo orden
            raise ErrorPaquete(f"Los ids de {archivo} no son exactamente los de su origen.")
        datos[archivo] = serializar(reducidos)
        filas[archivo] = len(reducidos)
        del origen
    if set(ids[ARCHIVO_TEST]) & set(ids[ARCHIVO_HARD]):
        raise ErrorPaquete("Test y hard comparten id.")
    datos[ARCHIVO_METADATA] = metadata_kaggle(
        id_kaggle=ID_KAGGLE, titulo=TITULO, licencia=LICENCIA, descripcion=DESCRIPCION
    )
    return datos, filas, llaves


# --- Candado ---


def candado_contenido(datos: Mapping[str, bytes], manifest: Mapping) -> dict[str, int]:
    """Cada línea de los dos JSONL tiene exactamente las claves id y nom_estab, y las filas son las del origen."""
    filas = {}
    for archivo, entrada in ORIGEN.items():
        try:
            regs = parsear_jsonl(datos[archivo], CAMPOS)
        except ErrorPreregistro as err:
            raise ErrorCandado(f"{archivo}: {err} Solo se permiten id y nom_estab (regla 3).") from err
        if b"\r" in datos[archivo]:
            raise ErrorCandado(f"{archivo} tiene saltos \\r.")
        if len(regs) != manifest["splits"][entrada]["filas"]:
            raise ErrorCandado(f"{archivo} no tiene las filas de splits.{entrada}.")
        filas[archivo] = len(regs)
    return filas


def candado_manifest(hashes: Mapping[str, str], manifest: Mapping) -> None:
    """Ningún archivo puede tener el SHA-256 de test, de hard ni de otra entrada del manifest (salvo la propia)."""
    s = manifest["splits"]
    prohibidos = {s[ORIGEN[ARCHIVO_TEST]]["sha256"]: "test", s[ORIGEN[ARCHIVO_HARD]]["sha256"]: "hard set"}
    iguales = sorted(f"{n} ({prohibidos[h]})" for n, h in hashes.items() if h in prohibidos)
    if iguales:
        raise ErrorCandado(f"{iguales} tiene el SHA-256 de un conjunto congelado completo; me detengo (regla 3).")
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
    """Candado sobre los bytes: solo los archivos permitidos, ningún hash del manifest y solo id y nom_estab."""
    hashes = candado_bytes(datos, manifest["splits"][ORIGEN[ARCHIVO_TEST]]["sha256"], PERMITIDOS)
    candado_manifest(hashes, manifest)
    candado_contenido(datos, manifest)
    return hashes


def candado_en_carpeta(carpeta: Path, manifest: Mapping, raiz: Path = REPO_RAIZ) -> dict[str, str]:
    """Candado sobre la carpeta escrita: fuera del repo, solo los archivos permitidos, sin hashes del manifest
    y solo id y nom_estab en cada línea."""
    hashes = candado_carpeta(carpeta, manifest["splits"][ORIGEN[ARCHIVO_TEST]]["sha256"], raiz, PERMITIDOS)
    candado_manifest(hashes, manifest)
    candado_contenido({a: (Path(carpeta) / a).read_bytes() for a in ORIGEN}, manifest)
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
    *, manifest: Mapping, datos: Mapping[str, bytes], filas: Mapping[str, int], commit: str, fecha: str
) -> dict:
    """Entrada kaggle_datasets.denue_scian_evaluacion del manifest: filas, bytes y SHA-256 (sin clases)."""
    archivos = {}
    for archivo, entrada in ORIGEN.items():
        archivos[archivo] = {
            "ruta": f"{CARPETA_RELATIVA}/{archivo}",
            "origen": f"splits.{entrada}",
            "origen_sha256": manifest["splits"][entrada]["sha256"],
            "filas": filas[archivo],
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
              "ni de otra entrada del manifest, y cada línea solo con id y nom_estab.")
        return

    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    versionado = codigo_versionado(SCRIPT)
    if args.escribir:
        if not (arbol_limpio and versionado):
            sys.exit(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
        if ENTRADA in manifest.get(SECCION, {}):
            sys.exit(f"El manifest ya tiene {SECCION}.{ENTRADA}; no se sobrescribe.")
        if CARPETA.exists() and any(CARPETA.iterdir()):
            sys.exit(f"La carpeta {CARPETA} ya tiene archivos; no se sobrescribe.")

    try:
        verificar_fuera_del_repo(CARPETA)
        datos, filas, llaves = preparar(manifest)
        hashes = candado_en_memoria(datos, manifest)
    except ErrorMuestra as e:
        sys.exit(f"Me detengo: {e}")

    print(f"commit {commit[:7]} | árbol limpio: {arbol_limpio} | {SCRIPT} versionado sin cambios: {versionado}")
    print("Origen (SHA-256 igual al del manifest, verificado antes de parsear):")
    for archivo, entrada in ORIGEN.items():
        e = manifest["splits"][entrada]
        print(f"  {e['ruta']}: {e['filas']:,} filas, columnas {e['campos']}, SHA-256 {e['sha256']}")
    print("Paquete (todas las filas; codigo_act, tipo, llave y marcas descartados al parsear):")
    for archivo in ORIGEN:
        print(f"  {archivo}: {filas[archivo]:,} filas, columnas {list(CAMPOS)}, {len(datos[archivo]):,} bytes, "
              f"SHA-256 {hashes[archivo]}")
    print(f"  {ARCHIVO_METADATA}: {len(datos[ARCHIVO_METADATA]):,} bytes (ASCII), SHA-256 {hashes[ARCHIVO_METADATA]}")
    print("Ids compartidos entre test y hard: 0 (comprobado); ids y orden iguales a los de cada origen.")
    print("Candado: solo los 3 archivos permitidos; ninguno tiene el SHA-256 de test ni de hard completos ni el de "
          "otra entrada del manifest; cada línea tiene exactamente id y nom_estab.")
    print(f"Carpeta destino: {CARPETA} (fuera del repo: True; existe: {CARPETA.exists()})")

    entrada = entrada_kaggle(manifest=manifest, datos=datos, filas=filas, commit=commit, fecha=fecha)
    try:
        verificar_privacidad({"entrada": entrada, "metadata": json.loads(datos[ARCHIVO_METADATA])}, llaves, set())
    except ErrorPrivacidad as e:
        sys.exit(f"Me detengo: {e}")
    print("Revisión de privacidad de la entrada del manifest y del metadata: ningún texto es una llave.")

    if not args.escribir:
        print("PUNTO DE CONTROL: no se escribió nada. Para escribir, --escribir.")
        return

    # Regenerar desde cero (manifest y archivos de origen releídos) y comparar bytes.
    datos2, _, _ = preparar(leer_manifest())
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
