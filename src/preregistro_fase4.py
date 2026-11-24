"""Pre-registro de la evaluación final del clasificador (Fase 4, paso 14).

Las reglas de la Fase 4 están escritas en `evals/fase4_preregistro.json`, y los scripts de la
Fase 4 las leen de ahí. Este módulo implementa lo que se calcula de ese archivo y lo comprueba:

- `elegir_fewshot()`: los k = 5 ejemplos del few-shot. Se toma
  `perm = numpy.random.default_rng(42).permutation(n)` sobre las filas de train final, en el orden
  del archivo, se recorre y se toma la primera fila de cada sector oficial aún no visto (31-33 y
  48-49 agrupados) hasta tener 5. El orden en el prompt es el orden de elección;
- `sha256_ids()`: huella de los ids elegidos, en orden; con ella el kernel comprueba que eligió
  los mismos ejemplos sin que sus nombres entren al repo;
- `prompt_fewshot()`: el prompt multi-turno. Cada ejemplo es un turno de usuario con el prompt
  único de `src/prompt_clasificador.py` y un turno de asistente con su código, sin bloque
  <think>; al final va `prompt_generacion()` de la fila. Es lo que se espera de
  `apply_chat_template(..., enable_thinking=False)`; la prueba byte a byte va en el kernel
  (paso 15);
- `cota_fichas()` y `limite_largo()`: el límite de largo del few-shot. Cada ficha del BPE por
  bytes de Qwen, especial o no, cubre al menos un byte del texto, así que los bytes UTF-8 del
  prompt son una cota superior exacta de sus fichas. Se toma el máximo entre el texto y su
  forma NFC, la que usa el normalizador del tokenizador. El kernel cuenta las fichas reales.

Arriba solo se importan la biblioteca estándar, numpy y `src/prompt_clasificador.py`, para que el
kernel de la Fase 4 pueda incrustar este módulo. Lo que lee archivos del repo se importa dentro de
las funciones.

Uso (solo imprime; no escribe nada):
  .venv/Scripts/python -m src.preregistro_fase4              # calcula la selección y el límite
  .venv/Scripts/python -m src.preregistro_fase4 --comprobar  # además, los compara con el config

Lee test.jsonl y hard.jsonl solo para las llaves y el largo de los nombres: codigo_act y tipo se
descartan al parsear, y no se calcula ninguna métrica. En la terminal solo van códigos SCIAN y
sectores de los 5 ejemplos de train, hashes y conteos; nunca nombres, ids ni llaves.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.prompt_clasificador import (
    FIN_DE_TURNO,
    INICIO_DE_TURNO,
    RE_CODIGO,
    contenido_usuario,
    prompt_generacion,
    tiene_marca_especial,
    validar_codigo,
)

SEMILLA = 42  # regla 1
K_FEWSHOT = 5
LARGO_GENERAL = 256  # todo, salvo el prompt few-shot (CAMBIO DE DISEÑO aprobado por Samuel el 24 nov)
NUEVAS_BASE = 32  # tope de fichas nuevas de los modelos base
NUEVAS_AFINADOS = 16  # las de la validación de los afinados
MULTIPLO_LARGO = 64
RUTA_CONFIG = "evals/fase4_preregistro.json"
SCRIPT = "src/preregistro_fase4.py"

RE_ID = re.compile(r"[0-9]+")
RE_COMMIT = re.compile(r"[0-9a-f]{40}")

# Conjunto -> entrada de la sección `splits` del manifest.
ENTRADAS = {
    "train_final": "clasificador_train_final",
    "validacion": "clasificador_validacion",
    "test": "clasificador_test",
    "hard": "clasificador_hard",
}
RESERVADOS = ("test", "hard")  # de ellos solo se conservan id, nom_estab y llave
KAGGLE_ENTRADA = "denue_scian_clasificador"
KAGGLE_ARCHIVO = "train_final.jsonl"
CAMPOS_KAGGLE = ("id", "nom_estab", "codigo_act")


class ErrorPreregistro(ValueError):
    """Entrada inválida o comprobación fallida; el mensaje nunca incluye nombres, ids ni llaves."""


# --- Funciones puras (se prueban en la laptop y se incrustan en el kernel) ---


def sector_de_prefijo(sectores: Iterable[str]) -> dict[str, str]:
    """Prefijo de 2 dígitos -> sector oficial ("31" -> "31-33"); falla si dos sectores comparten prefijo."""
    mapa: dict[str, str] = {}
    for s in sectores:
        inicio, _, fin = s.partition("-")
        for p in range(int(inicio), int(fin or inicio) + 1):
            if f"{p:02d}" in mapa:
                raise ErrorPreregistro("Dos sectores comparten un prefijo.")
            mapa[f"{p:02d}"] = s
    return mapa


def elegir_fewshot(
    codigos: Sequence[str], sector_de: Mapping[str, str], k: int = K_FEWSHOT, semilla: int = SEMILLA
) -> list[int]:
    """Posiciones de los k ejemplos, en el orden en que se eligen (el del prompt).

    `codigos`: el codigo_act de cada fila de train final, en el orden del archivo.
    """
    if k < 1:
        raise ErrorPreregistro("Se pide al menos un ejemplo.")
    perm = np.random.default_rng(semilla).permutation(len(codigos))
    vistos: set[str] = set()
    elegidas: list[int] = []
    for i in perm:
        c = codigos[int(i)]
        if not isinstance(c, str) or not RE_CODIGO.fullmatch(c) or c[:2] not in sector_de:
            raise ErrorPreregistro("Un codigo_act no es de seis dígitos ASCII o no cae en ningún sector.")
        s = sector_de[c[:2]]
        if s not in vistos:
            vistos.add(s)
            elegidas.append(int(i))
            if len(elegidas) == k:
                return elegidas
    raise ErrorPreregistro(f"El train solo tiene {len(elegidas)} sectores; se piden {k}.")


def sha256_ids(ids: Sequence[str]) -> str:
    """SHA-256 de los ids en orden, unidos con "\\n" y con "\\n" al final, en ASCII."""
    if not ids or any(not isinstance(i, str) or not RE_ID.fullmatch(i) for i in ids):
        raise ErrorPreregistro("Los ids deben ser texto de dígitos ASCII.")
    return hashlib.sha256(("\n".join(ids) + "\n").encode("ascii")).hexdigest()


def turno_ejemplo(nom_estab: str, codigo: str) -> str:
    """Un ejemplo: turno de usuario con el prompt único y turno de asistente con el código, sin <think>."""
    return (
        f"{INICIO_DE_TURNO}user\n{contenido_usuario(nom_estab)}{FIN_DE_TURNO}\n"
        f"{INICIO_DE_TURNO}assistant\n{validar_codigo(codigo)}{FIN_DE_TURNO}\n"
    )


def prompt_fewshot(ejemplos: Sequence[tuple[str, str]], nom_estab: str) -> str:
    """Los ejemplos (nombre, código) en orden y, al final, el prompt de generación de la fila."""
    if not ejemplos:
        raise ErrorPreregistro("El prompt few-shot necesita al menos un ejemplo.")
    return "".join(turno_ejemplo(n, c) for n, c in ejemplos) + prompt_generacion(nom_estab)


def cota_fichas(texto: str) -> int:
    """Cota superior de las fichas del texto: sus bytes UTF-8 (el máximo entre el texto y su forma NFC)."""
    return max(len(texto.encode("utf-8")), len(unicodedata.normalize("NFC", texto).encode("utf-8")))


def limite_largo(cota_maxima: int, nuevas: int = NUEVAS_BASE, multiplo: int = MULTIPLO_LARGO) -> int:
    """El menor múltiplo de `multiplo` que es mayor o igual que la cota máxima del prompt más las fichas nuevas."""
    if cota_maxima < 1 or nuevas < 1 or multiplo < 1:
        raise ErrorPreregistro("La cota, las fichas nuevas y el múltiplo deben ser positivos.")
    return -(-(cota_maxima + nuevas) // multiplo) * multiplo


def parsear_jsonl(datos: bytes, campos: Sequence[str]) -> list[dict]:
    """Registros de un JSONL del proyecto con las claves `campos` en orden, ordenado por int(id).

    Corta solo en "\\n": `splitlines()` también cortaría en U+2028, que JSON deja sin escapar.
    """
    if datos.startswith(b"\xef\xbb\xbf"):
        raise ErrorPreregistro("El archivo tiene BOM.")
    texto = datos.decode("utf-8")
    if not texto.endswith("\n"):
        raise ErrorPreregistro("El archivo está vacío o no termina en salto de línea.")
    campos = tuple(campos)
    regs = []
    for n, linea in enumerate(texto[:-1].split("\n"), start=1):
        try:
            r = json.loads(linea)
        except json.JSONDecodeError as e:
            raise ErrorPreregistro(f"Línea {n}: no es JSON válido.") from e
        if not isinstance(r, dict) or tuple(r) != campos:
            raise ErrorPreregistro(f"Línea {n}: las claves no son {list(campos)} en ese orden.")
        if not isinstance(r["id"], str) or not RE_ID.fullmatch(r["id"]):
            raise ErrorPreregistro(f"Línea {n}: el id no es solo dígitos ASCII.")
        if not isinstance(r.get("nom_estab", ""), str):
            raise ErrorPreregistro(f"Línea {n}: nom_estab no es texto.")
        regs.append(r)
    ids = [int(r["id"]) for r in regs]
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ErrorPreregistro("El archivo no está ordenado por int(id) o tiene ids repetidos.")
    return regs


# --- Cálculo sobre los archivos del repo ---


@dataclass
class Seleccion:
    """Selección few-shot sobre un archivo de train final (solo en memoria)."""

    posiciones: list[int]
    ids: list[str]
    nombres: list[str]
    codigos: list[str]
    sectores: list[str]

    @property
    def huella(self) -> str:
        return sha256_ids(self.ids)

    def publico(self) -> dict:
        return {"sha256_ids": self.huella, "codigos": list(self.codigos), "sectores": list(self.sectores)}


@dataclass
class Calculo:
    """Lo calculado para el pre-registro. Solo `publico()` se imprime o se compara con el config."""

    filas: dict[str, int]
    sha256: dict[str, str]
    kaggle_sha256: str
    seleccion: dict[str, Seleccion]
    coinciden: bool
    llaves_ejemplos_en: dict[str, int] = field(default_factory=dict)
    ids_ejemplos_en: dict[str, int] = field(default_factory=dict)
    llave_kaggle_igual_a_local: bool = False
    ejemplos_con_marca_especial: int = 0
    cota_bloque_sin_fila: int = 0
    cota_maxima: dict[str, int] = field(default_factory=dict)
    limite_fewshot: int = 0
    cero_shot_confirmadas: dict[str, dict[str, int]] = field(default_factory=dict)
    filas_con_marca_especial: dict[str, int] = field(default_factory=dict)
    llaves_reservadas_en_train: dict[str, int] = field(default_factory=dict)
    llaves: set[str] = field(default_factory=set, repr=False)  # todas las llaves; solo para la privacidad

    def publico(self) -> dict:
        res = {
            "filas": dict(self.filas),
            "sha256": dict(self.sha256),
            "kaggle_train_final_sha256": self.kaggle_sha256,
            "seleccion": {o: s.publico() for o, s in self.seleccion.items()},
            "coinciden": self.coinciden,
        }
        if self.coinciden:
            res.update(
                llaves_ejemplos_en=dict(self.llaves_ejemplos_en),
                ids_ejemplos_en=dict(self.ids_ejemplos_en),
                llave_kaggle_igual_a_local=self.llave_kaggle_igual_a_local,
                ejemplos_con_marca_especial=self.ejemplos_con_marca_especial,
                cota_bloque_sin_fila=self.cota_bloque_sin_fila,
                cota_maxima=dict(self.cota_maxima),
                limite_fewshot=self.limite_fewshot,
                cero_shot_confirmadas={k: dict(v) for k, v in self.cero_shot_confirmadas.items()},
                filas_con_marca_especial=dict(self.filas_con_marca_especial),
                llaves_reservadas_en_train=dict(self.llaves_reservadas_en_train),
            )
        return res


def _leer(ruta: Path, sha256: str) -> bytes:
    datos = Path(ruta).read_bytes()
    calculado = hashlib.sha256(datos).hexdigest()
    if calculado != sha256:
        raise ErrorPreregistro(
            f"El SHA-256 de {Path(ruta).name} no coincide con el manifest: calculado {calculado}, registrado {sha256}."
        )
    return datos


def leer_conjuntos(manifest: Mapping, raiz: Path) -> dict[str, list[dict]]:
    """Los cuatro conjuntos con su SHA-256 verificado antes de parsear.

    De test y hard solo quedan id, nom_estab y llave: codigo_act, tipo y las marcas se descartan aquí.
    """
    regs: dict[str, list[dict]] = {}
    for nombre, entrada in ENTRADAS.items():
        e = manifest["splits"][entrada]
        filas = parsear_jsonl(_leer(Path(raiz) / e["ruta"], e["sha256"]), e["campos"])
        if len(filas) != e["filas"]:
            raise ErrorPreregistro(f"Las filas de {entrada} no son las del manifest.")
        if nombre in RESERVADOS:
            filas = [{"id": r["id"], "nom_estab": r["nom_estab"], "llave": r["llave"]} for r in filas]
        regs[nombre] = filas
    return regs


def leer_train_kaggle(manifest: Mapping, raiz: Path) -> list[dict]:
    """La copia de train final del dataset de Kaggle, con su SHA-256 del manifest verificado."""
    e = manifest["kaggle_datasets"][KAGGLE_ENTRADA]["archivos"][KAGGLE_ARCHIVO]
    return parsear_jsonl(_leer(Path(raiz) / e["ruta"], e["sha256"]), CAMPOS_KAGGLE)


def seleccionar(filas: Sequence[Mapping], sector_de: Mapping[str, str]) -> Seleccion:
    pos = elegir_fewshot([r["codigo_act"] for r in filas], sector_de)
    return Seleccion(
        posiciones=pos,
        ids=[filas[p]["id"] for p in pos],
        nombres=[filas[p]["nom_estab"] for p in pos],
        codigos=[filas[p]["codigo_act"] for p in pos],
        sectores=[sector_de[filas[p]["codigo_act"][:2]] for p in pos],
    )


def calcular(manifest: Mapping, raiz: Path, sectores: Sequence[str]) -> Calculo:
    """Selección few-shot en las dos copias de train final y, si coinciden, el límite de largo.

    Manda la copia de Kaggle, que es la que verá el kernel. Si las dos selecciones no dan el
    mismo hash, los mismos códigos y los mismos sectores, no se calcula nada más (PARADA).
    """
    from src.normalizacion import normalizar_nombre

    regs = leer_conjuntos(manifest, raiz)
    kaggle = leer_train_kaggle(manifest, raiz)
    sector_de = sector_de_prefijo(sectores)
    sel = {"kaggle": seleccionar(kaggle, sector_de), "local": seleccionar(regs["train_final"], sector_de)}
    coinciden = sel["kaggle"].publico() == sel["local"].publico()
    calc = Calculo(
        filas={n: len(r) for n, r in regs.items()} | {"kaggle_train_final": len(kaggle)},
        sha256={n: manifest["splits"][e]["sha256"] for n, e in ENTRADAS.items()},
        kaggle_sha256=manifest["kaggle_datasets"][KAGGLE_ENTRADA]["archivos"][KAGGLE_ARCHIVO]["sha256"],
        seleccion=sel,
        coinciden=coinciden,
    )
    calc.llaves = {r["llave"] for rs in regs.values() for r in rs}
    if not coinciden:
        return calc

    ej = sel["kaggle"]
    llaves_ej = [normalizar_nombre(n) for n in ej.nombres]
    calc.llave_kaggle_igual_a_local = llaves_ej == [regs["train_final"][p]["llave"] for p in sel["local"].posiciones]
    for conj in ("validacion", "test", "hard"):
        llaves = {r["llave"] for r in regs[conj]}
        ids = {r["id"] for r in regs[conj]}
        calc.llaves_ejemplos_en[conj] = sum(k in llaves for k in llaves_ej)
        calc.ids_ejemplos_en[conj] = sum(i in ids for i in ej.ids)
    calc.ejemplos_con_marca_especial = sum(tiene_marca_especial(n) for n in ej.nombres)

    ejemplos = list(zip(ej.nombres, ej.codigos))
    calc.cota_bloque_sin_fila = cota_fichas(prompt_fewshot(ejemplos, ""))
    llaves_train = {r["llave"] for r in regs["train_final"]}
    for conj in RESERVADOS:
        nombres = [r["nom_estab"] for r in regs[conj]]
        calc.cota_maxima[conj] = max(cota_fichas(prompt_fewshot(ejemplos, n)) for n in nombres)
        calc.filas_con_marca_especial[conj] = sum(tiene_marca_especial(n) for n in nombres)
        calc.llaves_reservadas_en_train[conj] = len({r["llave"] for r in regs[conj]} & llaves_train)
    calc.limite_fewshot = limite_largo(max(calc.cota_maxima.values()), NUEVAS_BASE)
    for conj in ("validacion", *RESERVADOS):
        cotas = [cota_fichas(prompt_generacion(r["nom_estab"])) for r in regs[conj]]
        calc.cero_shot_confirmadas[conj] = {
            "filas": len(cotas),
            "base_cota_mas_32_hasta_256": sum(c + NUEVAS_BASE <= LARGO_GENERAL for c in cotas),
            "afinados_cota_mas_16_hasta_256": sum(c + NUEVAS_AFINADOS <= LARGO_GENERAL for c in cotas),
        }
    return calc


# --- Comprobación del config ---


def _git(raiz: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(["git", *args], cwd=raiz, capture_output=True, check=True).stdout.decode("utf-8")


def comprobar_config(config: Mapping, calc: Calculo, manifest: Mapping, raiz: Path, referencias: Mapping) -> list[str]:
    """Problemas del config frente a lo calculado, al manifest y a los archivos locales; [] si todo cuadra.

    `referencias`: {"sectores": SECTORES_2023, "modelos_base": {"06b": {...}, "4b": {...}}}.
    Los mensajes solo traen rutas de campos, conteos y hashes.
    """
    from src.denue import sha256_archivo

    p: list[str] = []
    pub = calc.publico()
    if not calc.coinciden:
        return ["La selección few-shot no coincide entre la copia de Kaggle y data/splits: PARADA."]

    for nombre, entrada in ENTRADAS.items():
        c = config["conjuntos"][nombre]
        e = manifest["splits"][entrada]
        if (c["entrada_manifest"], c["sha256"], c["filas"]) != (entrada, e["sha256"], e["filas"]):
            p.append(f"conjuntos.{nombre} no coincide con splits.{entrada} del manifest.")
    if config["conjuntos"]["hard"].get("por_tipo") != manifest["splits"]["clasificador_hard"]["por_tipo"]:
        p.append("conjuntos.hard.por_tipo no coincide con el manifest.")

    fs = config["few_shot"]
    if fs["k"] != K_FEWSHOT or fs["semilla"] != SEMILLA:
        p.append("few_shot.k o few_shot.semilla no son 5 y 42.")
    if list(fs["sectores_oficiales"]) != list(referencias["sectores"]):
        p.append("few_shot.sectores_oficiales no es SECTORES_2023 del evaluador.")
    if fs["fuente"]["sha256"] != calc.kaggle_sha256:
        p.append("few_shot.fuente.sha256 no es el de la copia de Kaggle de train final.")
    sel = pub["seleccion"]["kaggle"]
    for k in ("sha256_ids", "codigos", "sectores"):
        if fs[k] != sel[k]:
            p.append(f"few_shot.{k} no es el calculado.")
    fuera = all(v == 0 for v in calc.llaves_ejemplos_en.values()) and all(v == 0 for v in calc.ids_ejemplos_en.values())
    if fs["llaves_fuera_de_val_test_hard"] is not True or not fuera or not calc.llave_kaggle_igual_a_local:
        p.append("Las llaves de los ejemplos no quedan fuera de val, test y hard (o el config no lo dice).")
    if calc.ejemplos_con_marca_especial:
        p.append("Un nombre de ejemplo contiene '<|'.")

    lm = config["largo_maximo"]
    esperado = {
        "general": LARGO_GENERAL,
        "few_shot": calc.limite_fewshot,
        "cota_maxima_few_shot": max(calc.cota_maxima.values()),
        "nuevas_base": NUEVAS_BASE,
        "nuevas_afinados": NUEVAS_AFINADOS,
        "multiplo": MULTIPLO_LARGO,
    }
    for k, v in esperado.items():
        if lm.get(k) != v:
            p.append(f"largo_maximo.{k} no es {v}.")

    modelos = {m["id"]: m for m in config["modelos"]}
    if len(modelos) != 8 or len(config["modelos"]) != 8:
        p.append("El config no trae exactamente 8 modelos con id distinto.")
    if set(config["orden_en_el_kernel"]) - set(modelos) or len(config["orden_en_el_kernel"]) != 6:
        p.append("orden_en_el_kernel debe nombrar los 6 LLM del config.")
    for m in config["modelos"]:
        if "base" in m:
            ref = referencias["modelos_base"].get(m["tamano"])
            if m["base"] != ref:
                p.append(f"modelos.{m['id']}.base no es la revisión fijada en src/humo_kaggle.py.")
        for clave in ("adaptador", "adaptador_config", "predicciones_validacion", "registro_validacion"):
            if clave in m:
                ruta = Path(raiz) / m[clave]["ruta"]
                if not ruta.is_file():
                    p.append(f"modelos.{m['id']}.{clave}: no existe el archivo local.")
                elif sha256_archivo(ruta) != m[clave]["sha256"]:
                    p.append(f"modelos.{m['id']}.{clave}: el SHA-256 no es el del archivo local.")
        if "adaptador" in m:
            res = json.loads((Path(raiz) / m["resultado_validacion"]).read_text(encoding="utf-8"))
            if m["adaptador"]["sha256"] != res["modelo"].get("adaptador_sha256"):
                p.append(f"modelos.{m['id']}.adaptador.sha256 no es el de su resultado de validación.")

    fuera_de_test = {x["modelo"] for x in config["no_van_a_test"]}
    if len(config["no_van_a_test"]) != 7 or len(fuera_de_test) != 7 or fuera_de_test & set(modelos):
        p.append("no_van_a_test debe listar 7 modelos distintos, ninguno de los 8 que van a test.")
    comp = config["comparaciones"]
    for grupo, g in comp.items():
        if not isinstance(g, Mapping) or "pares" not in g:
            continue
        if not isinstance(g["pares"], list) or not g["pares"]:
            p.append(f"comparaciones.{grupo}.pares debe ser una lista no vacía.")
            continue
        for par in g["pares"]:
            if isinstance(par, str):  # clave de otro grupo cuyos pares se reutilizan (hard)
                otro = comp.get(par)
                if par == grupo or not isinstance(otro, Mapping) or not isinstance(otro.get("pares"), list) or any(
                    not isinstance(x, Mapping) for x in otro["pares"]
                ):
                    p.append(f"comparaciones.{grupo}: '{par}' no es un grupo de comparaciones con pares explícitos.")
            elif not isinstance(par, Mapping) or par.get("a") not in modelos or par.get("b") not in modelos or par["a"] == par["b"]:
                p.append(f"comparaciones.{grupo}: un par no nombra dos modelos distintos del config.")
    if "secundarias_sector" in comp:
        todas = [*comp["principales"]["pares"], *comp["secundarias"]["pares"]]
        if comp["secundarias_sector"]["pares"] != todas:
            p.append("comparaciones.secundarias_sector.pares no son los pares de principales y secundarias, en ese orden.")

    commit = config["registro"]["commit_codigo"]
    if commit != "pendiente":
        if not RE_COMMIT.fullmatch(commit):
            p.append("registro.commit_codigo no es un commit de 40 hexadecimales.")
        else:
            try:
                en_git = _git(raiz, "show", f"{commit}:{SCRIPT}").replace("\r\n", "\n")
            except Exception:
                en_git = None
            local = (Path(raiz) / SCRIPT).read_bytes().decode("utf-8").replace("\r\n", "\n")
            if en_git != local:
                p.append(f"{SCRIPT} no es el del commit registrado.")
    return p


def privacidad_config(config: Mapping, llaves: Collection[str]) -> None:
    """Falla si algún texto del config, normalizado, es una llave de train final, val, test o hard."""
    from src.exploracion_clasificador import verificar_privacidad

    verificar_privacidad(config, set(llaves), set())


# --- Programa principal ---


def _imprimir(calc: Calculo) -> None:
    pub = calc.publico()
    print("Conjuntos (SHA-256 igual al del manifest, verificado antes de parsear):")
    for n, f in pub["filas"].items():
        sha = pub["sha256"].get(n, pub["kaggle_train_final_sha256"])
        print(f"  {n}: {f:,} filas, SHA-256 {sha}")
    print("De test y hard solo se conservaron id, nom_estab y llave (codigo_act y tipo se descartaron al parsear).")
    for origen, etiqueta in (("kaggle", "copia de Kaggle (manda)"), ("local", "data/splits/train_final.jsonl")):
        s = pub["seleccion"][origen]
        print(f"Few-shot en {etiqueta}:")
        for i, (c, sec) in enumerate(zip(s["codigos"], s["sectores"]), start=1):
            print(f"  {i}. código {c}, sector {sec}")
        print(f"  SHA-256 de los 5 ids en orden: {s['sha256_ids']}")
    if not pub["coinciden"]:
        print("PARADA: la selección no coincide entre las dos copias de train final; no se elige cuál manda.")
        return
    print("Las dos copias dan el mismo hash, los mismos códigos y los mismos sectores.")
    print(f"Llaves de los ejemplos en validación, test y hard: {pub['llaves_ejemplos_en']}; ids: {pub['ids_ejemplos_en']}")
    print(f"Llaves de la copia de Kaggle iguales a las de data/splits: {pub['llave_kaggle_igual_a_local']}")
    print(f"Ejemplos con '<|' en el nombre: {pub['ejemplos_con_marca_especial']}")
    print(f"Cota del bloque few-shot con un nombre vacío: {pub['cota_bloque_sin_fila']} bytes")
    print(f"Cota máxima del prompt few-shot por conjunto (bytes UTF-8 >= fichas): {pub['cota_maxima']}")
    print(f"Límite few-shot: múltiplo de {MULTIPLO_LARGO} >= cota máxima + {NUEVAS_BASE} = {pub['limite_fewshot']}")
    print("0-shot, filas con la cota confirmada dentro de 256 (las demás las confirma el kernel con fichas reales):")
    for conj, v in pub["cero_shot_confirmadas"].items():
        print(f"  {conj}: {v}")
    print(f"Filas con '<|' en el nombre: {pub['filas_con_marca_especial']}")
    print(f"Llaves de test y hard presentes en train final: {pub['llaves_reservadas_en_train']}")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    from src.denue import REPO_RAIZ, sha256_archivo
    from src.evaluador_clasificador import SECTORES_2023
    from src.exploracion_clasificador import ErrorPrivacidad
    from src.humo_kaggle import MODELO_06B, MODELO_4B
    from src.scian import leer_manifest

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comprobar", action="store_true", help=f"compara lo calculado con {RUTA_CONFIG}")
    args = parser.parse_args(argv)

    manifest = leer_manifest()
    try:
        calc = calcular(manifest, REPO_RAIZ, SECTORES_2023)
    except ErrorPreregistro as e:
        sys.exit(f"Me detengo: {e}")
    _imprimir(calc)
    if not calc.coinciden:
        sys.exit(1)
    if not args.comprobar:
        print("No se escribió nada. Para comparar con el config: --comprobar.")
        return

    ruta = REPO_RAIZ / RUTA_CONFIG
    datos = ruta.read_bytes()
    if datos.startswith(b"\xef\xbb\xbf") or b"\r" in datos or not datos.endswith(b"\n"):
        sys.exit(f"{RUTA_CONFIG} debe ser UTF-8 sin BOM, con saltos \\n y uno al final.")
    config = json.loads(datos.decode("utf-8"))
    referencias = {"sectores": SECTORES_2023, "modelos_base": {"0.6B": MODELO_06B, "4B": MODELO_4B}}
    problemas = comprobar_config(config, calc, manifest, REPO_RAIZ, referencias)
    try:
        privacidad_config(config, calc.llaves)
    except ErrorPrivacidad as e:
        problemas.append(str(e))
    if problemas:
        sys.exit("El config no cuadra:\n  - " + "\n  - ".join(problemas))
    print(f"{RUTA_CONFIG}: todo cuadra (few-shot, largos, conjuntos, modelos, revisiones y archivos locales).")
    print("Privacidad del config: ningún texto es una llave de train final, validación, test ni hard.")
    print(f"commit_codigo: {config['registro']['commit_codigo']}")
    print(f"SHA-256 del config: {sha256_archivo(ruta)} ({len(datos):,} bytes)")


if __name__ == "__main__":
    main()
