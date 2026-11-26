"""Auditoría de privacidad de todo el historial de git y de lo que quedaría público (paso 20).

Recorre cada blob de cada commit alcanzable (`git rev-list --all`), los mensajes de commit y los
archivos del árbol de trabajo que quedarían públicos (no versionados ni ignorados, y versionados con
cambios). Busca, línea por línea (las líneas idénticas se revisan una sola vez):

a. Nombres: la regla acotada de nombres (método del paso 6c indexado, con la precisión del paso 11)
   contra train final, validación, test y hard (test y hard sin etiquetas, del dataset de evaluación).
   Pasada extra, reportada aparte: contra todos los `nom_estab` del DENUE.
b. Ids: corridas aisladas de 5 a 8 dígitos iguales a un id del DENUE.
c. Secretos: patrones de tokens y llaves, y rutas de archivos de credenciales.
d. Datos personales: correos, teléfonos con formato y rutas de carpetas de usuario.
Además: metadatos de texto de las imágenes PNG, rutas prohibidas en algún commit, autores y commits
que solo están en el reflog.

La terminal solo recibe commit, ruta, línea, tipo, largo y categoría de cada hallazgo, nunca el
nombre, la llave ni el id. El detalle (con las llaves, para revisarlo a mano) va a `data/auditoria/`,
ignorado por git, como el archivo de la regla 7. La clasificación manual se lee de
`data/auditoria/clasificacion_manual.json` ({id del hallazgo: categoría}).

Uso:
  .venv/Scripts/python -m src.auditoria_publicacion                  # historial completo; solo imprime y escribe el detalle
  .venv/Scripts/python -m src.auditoria_publicacion --escribir       # además, el registro de conteos en results/
  .venv/Scripts/python -m src.auditoria_publicacion --archivos RUTA [RUTA ...]   # texto nuevo, antes de un commit

El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import hashlib
import json
import re
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluar_corrida_clasificador import LARGO_MAXIMO_DENTRO_DE_TOKEN, buscar_nombres_indexada, indice_de_nombres, solo_digitos
from src.evaluar_humo import referencias_de_nombres, textos_de_json
from src.normalizacion import normalizar_nombre

SCRIPT = "src/auditoria_publicacion.py"
DIR_DETALLE = "data/auditoria"
ARCHIVO_MANUAL = "clasificacion_manual.json"
EXPERIMENTO = "paso20_auditoria-privacidad"

PERMITIDOS_DATA = frozenset({"data/.gitkeep", "data/manifest.json", "data/scian_2023_clases.csv", "data/scian_2023_vocabulario.csv"})
ARCHIVOS_CATALOGO = frozenset({"data/scian_2023_clases.csv", "data/scian_2023_vocabulario.csv"})
EXT_PROHIBIDAS = (".jsonl", ".joblib", ".safetensors", ".gguf", ".bin", ".pt", ".parquet", ".zip", ".pkl", ".xlsx", ".csv")
# Archivos que escribe un script (su texto fijo viene del código versionado).
PREFIJOS_GENERADOS = ("results/", "notebooks/", "evals/")
GENERADOS = frozenset({"data/manifest.json"})

# --- Patrones ---

RE_ID = re.compile(r"(?<![0-9A-Za-z])([0-9]{5,8})(?![0-9A-Za-z])")
RE_HEX_LARGO = re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{40,64}(?![0-9A-Fa-f])")
RE_UNIDAD = re.compile(
    r"^\s*(bytes|filas|pasos|fichas|par[aá]metros|MiB|MB|KB|GB|ms|s\b|segundos|llaves|ids\b|nombres|registros|"
    r"caracteres|tokens|ejemplos|rasgos|iteraciones)", re.IGNORECASE)
RE_CLAVE_ID = re.compile(r"(^|_)(id|ids|clee)(_|$)", re.IGNORECASE)
SECRETOS: tuple[tuple[str, re.Pattern], ...] = (
    ("token_hf", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("token_github", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("token_kaggle", re.compile(r"\bKGAT_[A-Za-z0-9]{16,}")),
    ("llave_kaggle_json", re.compile(r"\"key\"\s*:\s*\"[0-9a-f]{32}\"")),
    ("variable_con_secreto", re.compile(
        r"\b(?:KAGGLE_KEY|KAGGLE_API_TOKEN|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|GITHUB_TOKEN|GH_TOKEN|OPENAI_API_KEY|"
        r"ANTHROPIC_API_KEY)\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{16,}")),
    ("llave_privada", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("llave_aws", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("llave_sk", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}")),
    ("llave_google", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("token_slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("hex_32", re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{32}(?![0-9A-Fa-f])")),
)
NOMBRES_DE_CREDENCIALES = frozenset({".env", "kaggle.json", "access_token", "id_rsa", "id_ed25519", "settings.local.json",
                                     ".netrc", ".pypirc", ".git-credentials"})
EXT_CREDENCIALES = (".pem", ".key", ".p12", ".pfx")
RE_CORREO = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
CORREOS_ACEPTADOS = (re.compile(r"^noreply@anthropic\.com$", re.IGNORECASE),
                     re.compile(r"@users\.noreply\.github\.com$", re.IGNORECASE),
                     re.compile(r"@example\.(?:com|org|net|invalid)$", re.IGNORECASE))
RE_TELEFONO = re.compile(r"(?<![0-9])(?:\+?52[\s.\-]?)?\(?[0-9]{2,3}\)?[\s.\-][0-9]{3,4}[\s.\-][0-9]{4}(?![0-9])")
RE_RUTA_USUARIO = re.compile(r"(?:[A-Za-z]:[\\/]+|/[a-z]/|/)(?:Users|home)[\\/]+([^\\/\s\"'`:;,)]+)", re.IGNORECASE)
USUARIOS_ACEPTADOS = frozenset({"Invitado"})  # decisión de Samuel en el paso 11 (la captura de los baselines)
RE_UNIDAD_TEXTO = re.compile(r"\"([^\"\n]{1,300})\"|'([^'\n]{1,300})'|«([^»\n]{1,300})»|`([^`\n]{1,300})`|“([^”\n]{1,300})”")

DESCARTES_ID = ("codigo_scian", "prefijo_de_hash", "explicado_por_el_archivo")
CATEGORIAS_MANUALES = {
    "nombre": ("nombre_inventado", "identificador_de_codigo", "frase_comun_de_prosa", "posible_dato_real"),
    "id": ("numero_explicado_en_contexto", "id_inventado_de_prueba", "posible_dato_real"),
    "otro": ("falso_positivo", "posible_dato_real"),
}


class ErrorAuditoria(RuntimeError):
    """El mensaje nunca incluye un nom_estab, una llave, un id ni un secreto."""


# --- Git ---


def git(raiz: Path, *args: str, entrada: bytes | None = None) -> bytes:
    r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=raiz, input=entrada, capture_output=True)
    if r.returncode != 0:
        raise ErrorAuditoria(f"git {args[0]} falló con código {r.returncode}.")
    return r.stdout


@dataclass
class Documento:
    """Un texto a revisar: un blob, un archivo del árbol de trabajo o un mensaje de commit."""

    clave: str
    datos: bytes
    ubicaciones: list[tuple[str, str]] = field(default_factory=list)  # (commit o etiqueta, ruta)
    rutas_head: set[str] = field(default_factory=set)

    @property
    def ruta(self) -> str:
        return self.ubicaciones[0][1] if self.ubicaciones else ""


@dataclass
class Inventario:
    commits: list[str]
    head: str | None
    documentos: dict[str, Documento]
    rutas_por_commit: dict[str, set[str]]
    autores: Counter
    reflog_solo: list[str]
    no_versionados: list[str]
    modificados: list[str]
    rutas_todas: set[str]


def leer_blobs(raiz: Path, shas: Sequence[str]) -> dict[str, bytes]:
    if not shas:
        return {}
    salida = git(raiz, "cat-file", "--batch", entrada=("\n".join(shas) + "\n").encode())
    res: dict[str, bytes] = {}
    pos = 0
    for sha in shas:
        fin = salida.index(b"\n", pos)
        cab = salida[pos:fin].decode().split()
        if len(cab) != 3 or cab[0] != sha:
            raise ErrorAuditoria("Respuesta inesperada de git cat-file.")
        n = int(cab[2])
        res[sha] = salida[fin + 1:fin + 1 + n]
        pos = fin + 1 + n + 1
    return res


def inventario(raiz: Path) -> Inventario:
    """Commits alcanzables (padres primero), cada (ruta, blob) con el primer commit que lo trae, mensajes y árbol."""
    commits = git(raiz, "rev-list", "--all", "--reverse", "--topo-order").decode().split()
    head = git(raiz, "rev-parse", "HEAD").decode().strip() if commits else None
    primero: dict[tuple[str, str], str] = {}
    rutas_por_commit: dict[str, set[str]] = {}
    for c in commits:
        rutas = set()
        for ent in git(raiz, "ls-tree", "-r", "-z", c).split(b"\0"):
            if not ent:
                continue
            meta, ruta = ent.split(b"\t", 1)
            _modo, tipo, obj = meta.decode().split()
            ruta_s = ruta.decode("utf-8")
            rutas.add(ruta_s)
            if tipo != "blob":
                raise ErrorAuditoria(f"Objeto de tipo {tipo} en {ruta_s} (commit {c[:7]}).")
            primero.setdefault((ruta_s, obj), c)
        rutas_por_commit[c] = rutas
    head_blobs: dict[str, str] = {}
    if head:
        for ent in git(raiz, "ls-tree", "-r", "-z", head).split(b"\0"):
            if ent:
                meta, ruta = ent.split(b"\t", 1)
                head_blobs[ruta.decode("utf-8")] = meta.decode().split()[2]
    shas = sorted({b for (_r, b) in primero})
    datos = leer_blobs(raiz, shas)
    docs: dict[str, Documento] = {}
    orden = {c: i for i, c in enumerate(commits)}
    for (ruta, blob), c in sorted(primero.items(), key=lambda x: (orden[x[1]], x[0][0])):
        d = docs.setdefault(blob, Documento(clave=blob, datos=datos[blob]))
        d.ubicaciones.append((c, ruta))
        if head_blobs.get(ruta) == blob:
            d.rutas_head.add(ruta)
    # Mensajes de commit
    autores: Counter = Counter()
    if commits:
        crudo = git(raiz, "log", "--all", "--format=%H%x00%an <%ae>%x00%cn <%ce>%x00%B%x1e")
        for reg in crudo.split(b"\x1e"):
            reg = reg.strip(b"\n")
            if not reg:
                continue
            c, autor, committer, cuerpo = reg.split(b"\x00", 3)
            c_s = c.decode()
            autores[f"autor: {autor.decode('utf-8')}"] += 1
            autores[f"committer: {committer.decode('utf-8')}"] += 1
            docs[f"msg:{c_s}"] = Documento(clave=f"msg:{c_s}", datos=cuerpo, ubicaciones=[(c_s, "(mensaje del commit)")])
    # Árbol de trabajo: no versionados ni ignorados, y versionados con cambios
    no_vers = [p for p in git(raiz, "ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0") if p]
    modif = [p for p in git(raiz, "diff", "--name-only", "-z", "HEAD").decode("utf-8").split("\0") if p] if head else []
    for p in sorted(set(no_vers) | set(modif)):
        ruta = Path(raiz) / p
        if ruta.is_file():
            docs[f"wt:{p}"] = Documento(clave=f"wt:{p}", datos=ruta.read_bytes(), ubicaciones=[("(árbol de trabajo)", p)])
    try:
        con_reflog = set(git(raiz, "rev-list", "--reflog").decode().split())
    except ErrorAuditoria:
        con_reflog = set()
    reflog_solo = sorted(con_reflog - set(commits))
    rutas_todas = {r for rs in rutas_por_commit.values() for r in rs}
    return Inventario(commits, head, docs, rutas_por_commit, autores, reflog_solo, no_vers, modif, rutas_todas)


# --- Referencias ---


@dataclass
class Referencias:
    """Lo que se busca; nunca se imprime."""

    idx_4: Mapping            # índice de nombres de los 4 conjuntos
    idx_denue: Mapping | None  # índice de todo el DENUE (pasada extra)
    ids: frozenset[str]
    codigos_scian: frozenset[str]
    titulos_normalizados: str  # todos los títulos SCIAN normalizados, " T " separados por "\n"
    hashes: dict[str, str]     # SHA-256 de las referencias, para el registro


def texto_de_busqueda(lineas: Iterable[str]) -> str:
    """Líneas normalizadas con un espacio a cada lado, para buscar " LLAVE " como palabras completas."""
    return "\n".join(f" {normalizar_nombre(x)} " for x in lineas)


def contiene_frase(texto_busqueda: str, llave: str) -> bool:
    return bool(llave) and f" {llave} " in texto_busqueda


def cargar_referencias(raiz: Path, extra: bool = True) -> Referencias:
    """Los 4 conjuntos (con su SHA-256 del manifest), el DENUE completo (cargador único) y el SCIAN 2023 y 2018."""
    from src import salida_fase4
    from src.denue import cargar_denue
    from src.scian import CANDADO, CATALOGO, FUENTE_ESTRUCTURA, archivo_verificado, extraer_estructura, leer_catalogo, leer_manifest, texto_de_paginas

    manifest = leer_manifest(Path(raiz) / "data" / "manifest.json")
    conjuntos = salida_fase4.referencias(manifest, raiz)
    filas = [r for filas in conjuntos.values() for r in filas]
    idx_4 = indice_de_nombres(referencias_de_nombres(filas))
    df = cargar_denue(["id", "nom_estab"])
    ids = frozenset(df["id"].tolist())
    idx_denue = indice_de_nombres(referencias_de_nombres(df.to_dict("records"))) if extra else None
    codigos: set[str] = set()
    titulos: list[str] = []
    for version in ("2023", "2018"):
        est = extraer_estructura(texto_de_paginas(archivo_verificado(manifest["fuentes"][FUENTE_ESTRUCTURA[version]])),
                                 CANDADO[version])
        codigos |= set(est.categorias)
        titulos += list(est.categorias.values())
        cat = leer_catalogo(archivo_verificado(manifest["catalogos"][CATALOGO[version]]))
        codigos |= set(cat)
        titulos += list(cat.values())
    hashes = {"denue": manifest["fuentes"]["denue_cdmx"]["sha256"]}
    for nombre in ("denue_scian_clasificador", "denue_scian_evaluacion"):
        for archivo, ent in manifest["kaggle_datasets"][nombre]["archivos"].items():
            hashes[f"{nombre}/{archivo}"] = ent["sha256"]
    return Referencias(idx_4, idx_denue, ids, frozenset(codigos), texto_de_busqueda(titulos), hashes)


# --- Análisis de una línea (con caché: las líneas repetidas se revisan una vez) ---


@dataclass
class Hallazgo:
    tipo: str              # nombre_crudo, llave_varias_palabras, id, secreto:*, correo, telefono, ruta_de_usuario
    pasada: str            # "4_conjuntos", "denue_completo" o "general"
    valor: str             # lo encontrado (nunca se imprime)
    linea_texto: str       # la línea donde está (nunca se imprime)
    aislado: bool = True   # nombre crudo fuera de un token más largo
    documento: str = ""
    linea: int = 0
    categoria: str | None = None
    completo: bool = False   # pasada extra: nombre completo
    cerca_de_id: bool = False

    @property
    def largo(self) -> str:
        palabras = len(normalizar_nombre(self.valor).split()) if self.tipo.startswith(("nombre", "llave")) else 1
        return f"{len(self.valor)}c/{palabras}p"

    @property
    def es_nombre(self) -> bool:
        return self.tipo in ("nombre_crudo", "llave_varias_palabras")

    def ident(self) -> str:
        return hashlib.sha256(f"{self.pasada}\0{self.tipo}\0{self.valor}\0{self.linea_texto}".encode()).hexdigest()[:16]


def nombres_en_linea(linea: str, idx: Mapping, pasada: str) -> tuple[list[Hallazgo], dict[str, int]]:
    """Regla acotada sobre una línea: los hallazgos y los conteos que no son hallazgo."""
    _res, enc = buscar_nombres_indexada(linea, idx)
    hall: list[Hallazgo] = []
    cuenta = Counter()
    for n in sorted(enc["crudos"]):
        if solo_digitos(n):
            cuenta["crudos_solo_digitos"] += 1
        elif len(n) <= LARGO_MAXIMO_DENTRO_DE_TOKEN and n not in enc["crudos_aislados"]:
            cuenta["crudos_cortos_dentro_de_token"] += 1
        else:
            hall.append(Hallazgo("nombre_crudo", pasada, n, linea, aislado=n in enc["crudos_aislados"]))
    for k in sorted(enc["varias"]):
        if solo_digitos(k):
            cuenta["llaves_solo_digitos"] += 1
        else:
            hall.append(Hallazgo("llave_varias_palabras", pasada, k, linea))
    cuenta["llaves_una_palabra"] += sum(1 for k in enc["una"] if not solo_digitos(k))
    return hall, dict(cuenta)


def otros_en_linea(linea: str, ids: frozenset[str]) -> list[Hallazgo]:
    hall = [Hallazgo("id", "general", m.group(1), linea) for m in RE_ID.finditer(linea) if m.group(1) in ids]
    for nombre, patron in SECRETOS:
        hall += [Hallazgo(f"secreto:{nombre}", "general", m.group(0), linea) for m in patron.finditer(linea)]
    for m in RE_CORREO.finditer(linea):
        if not any(p.search(m.group(0)) for p in CORREOS_ACEPTADOS):
            hall.append(Hallazgo("correo", "general", m.group(0), linea))
    hall += [Hallazgo("telefono", "general", m.group(0), linea) for m in RE_TELEFONO.finditer(linea)]
    for m in RE_RUTA_USUARIO.finditer(linea):
        if m.group(1) not in USUARIOS_ACEPTADOS:
            hall.append(Hallazgo("ruta_de_usuario", "general", m.group(0), linea))
    return hall


def es_nombre_completo(h: Hallazgo) -> bool:
    """Pasada extra: nombre crudo de 5+ caracteres fuera de un token, o llave igual a una unidad entera de texto."""
    if h.tipo == "nombre_crudo":
        return h.aislado and len(h.valor) >= 5
    unidades = [h.linea_texto, *h.linea_texto.split("|")]
    for m in RE_UNIDAD_TEXTO.finditer(h.linea_texto):
        unidades.append(next(g for g in m.groups() if g is not None))
    return any(normalizar_nombre(u) == h.valor for u in unidades)


# --- Documentos ---


def lineas_de(doc: Documento) -> list[tuple[int, str]]:
    """(número, texto) de cada línea; en un JSON, también sus valores decodificados que no están tal cual en el texto."""
    texto = decodificar(doc.datos)
    lineas = [(i, x.rstrip("\r")) for i, x in enumerate(texto.split("\n"), 1)]
    if doc.ruta.endswith(".json"):
        try:
            obj = json.loads(texto)
        except ValueError:
            return lineas
        for v in textos_de_json(obj).split("\n"):
            if v and v not in texto:
                lineas.append((0, v))
    return lineas


def decodificar(datos: bytes) -> str:
    try:
        return datos.decode("utf-8")
    except UnicodeDecodeError:
        return datos.decode("latin-1")


def es_binario(datos: bytes) -> bool:
    return b"\0" in datos[:8192] or datos.startswith(b"\x89PNG")


def numeros_explicados(doc: Documento) -> set[str]:
    """Dígitos de valores numéricos de un JSON (enteros y partes de decimales) cuya clave no es de identificador."""
    if not doc.ruta.endswith(".json"):
        return set()
    try:
        obj = json.loads(decodificar(doc.datos))
    except ValueError:
        return set()
    res: set[str] = set()

    def recorrer(o, clave: str) -> None:
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            if not RE_CLAVE_ID.search(clave):
                res.update(RE_ID.findall(repr(o)))
            return
        if isinstance(o, Mapping):
            for k, v in o.items():
                recorrer(v, str(k))
        elif isinstance(o, list):
            for v in o:
                recorrer(v, clave)

    recorrer(obj, "")
    return res


def chunks_png(datos: bytes) -> list[tuple[str, int]]:
    """(tipo, largo) de cada bloque de un PNG."""
    if not datos.startswith(b"\x89PNG\r\n\x1a\n"):
        return []
    res, pos = [], 8
    while pos + 8 <= len(datos):
        n = int.from_bytes(datos[pos:pos + 4], "big")
        tipo = datos[pos + 4:pos + 8].decode("latin-1")
        res.append((tipo, n))
        pos += 12 + n
        if tipo == "IEND":
            break
    return res


def texto_de_chunks_png(datos: bytes) -> list[str]:
    """Textos de los bloques tEXt, zTXt e iTXt (para pasarlos por las reglas)."""
    res, pos = [], 8
    while pos + 8 <= len(datos):
        n = int.from_bytes(datos[pos:pos + 4], "big")
        tipo = datos[pos + 4:pos + 8]
        cuerpo = datos[pos + 8:pos + 8 + n]
        if tipo == b"tEXt":
            res.append(cuerpo.replace(b"\0", b": ").decode("latin-1"))
        elif tipo == b"zTXt":
            k, _, resto = cuerpo.partition(b"\0")
            res.append(k.decode("latin-1") + ": " + zlib.decompress(resto[1:]).decode("latin-1"))
        elif tipo == b"iTXt":
            partes = cuerpo.split(b"\0", 3)
            if len(partes) == 4:
                texto = partes[3].split(b"\0", 1)[-1]
                comprimido = partes[1][:1] == b"\x01"
                res.append(partes[0].decode("latin-1") + ": " + (zlib.decompress(texto) if comprimido else texto).decode("utf-8", "replace"))
        pos += 12 + n
        if tipo == b"IEND":
            break
    return res


# --- Revisión ---


@dataclass
class Resultado:
    hallazgos: list[Hallazgo]
    conteos_no_hallazgo: dict[str, Counter]
    imagenes: list[dict]
    binarios: list[str]
    rutas_prohibidas: list[dict]
    rutas_credenciales: list[dict]
    documentos_texto: int
    lineas_unicas: int


def revisar(inv: Inventario, refs: Referencias, avance: Callable[[str], None] = lambda s: None) -> Resultado:
    cache: dict[str, tuple] = {}  # línea -> (hallazgos 4 conjuntos, DENUE, generales, conteos 4, conteos DENUE)
    conteos = {"4_conjuntos": Counter(), "denue_completo": Counter()}
    hallazgos: list[Hallazgo] = []
    imagenes, binarios = [], []
    textos_docs = 0
    hexes = sorted({h for d in inv.documentos.values() if not es_binario(d.datos)
                    for h in RE_HEX_LARGO.findall(decodificar(d.datos))} | set(inv.commits) | set(inv.reflog_solo))
    for n_doc, doc in enumerate(sorted(inv.documentos.values(), key=lambda d: d.ruta), 1):
        if n_doc % 50 == 0:
            avance(f"  {n_doc:,} de {len(inv.documentos):,} documentos; {len(cache):,} líneas únicas revisadas")
        if es_binario(doc.datos):
            ch = chunks_png(doc.datos)
            if ch:
                textos = texto_de_chunks_png(doc.datos)
                imagenes.append({"documento": doc.clave, "ruta": doc.ruta, "bytes": len(doc.datos),
                                 "bloques_de_texto": sorted({t for t, _ in ch if t in ("tEXt", "zTXt", "iTXt", "eXIf")}),
                                 "textos": textos})
                lineas = [(0, t) for t in textos]
            else:
                binarios.append(doc.ruta)
                continue
        else:
            textos_docs += 1
            lineas = lineas_de(doc)
        explicados = numeros_explicados(doc)
        por_linea: dict[int, list[Hallazgo]] = defaultdict(list)
        for num, linea in lineas:
            if not linea.strip():
                continue
            if linea not in cache:
                h4, c4 = nombres_en_linea(linea, refs.idx_4, "4_conjuntos")
                hd, cd = nombres_en_linea(linea, refs.idx_denue, "denue_completo") if refs.idx_denue else ([], {})
                cache[linea] = (h4, hd, otros_en_linea(linea, refs.ids), c4, cd)
            h4, hd, ho, c4, cd = cache[linea]
            conteos["4_conjuntos"].update(c4)
            conteos["denue_completo"].update(cd)
            vistos4 = {(h.tipo, h.valor) for h in h4}
            for h in h4 + [x for x in hd if (x.tipo, x.valor) not in vistos4] + ho:
                nuevo = Hallazgo(**{**h.__dict__, "documento": doc.clave, "linea": num})
                if nuevo.tipo == "id":
                    nuevo.categoria = categoria_id(nuevo, refs, hexes, explicados)
                por_linea[num].append(nuevo)
        # Pasada extra: cerca de un id (misma línea o a 2 líneas o menos de un id no descartado)
        lineas_id = {n for n, hs in por_linea.items() for h in hs if h.tipo == "id" and h.categoria is None and n > 0}
        for n, hs in por_linea.items():
            for h in hs:
                if h.pasada == "denue_completo":
                    h.completo = es_nombre_completo(h)
                    h.cerca_de_id = n > 0 and any(abs(n - m) <= 2 for m in lineas_id)
                hallazgos.append(h)
    prohibidas = []
    for r in sorted(inv.rutas_todas):
        if r.startswith("kaggle-output/") or (r.startswith("data/") and r not in PERMITIDOS_DATA) or \
                (r.lower().endswith(EXT_PROHIBIDAS) and r not in PERMITIDOS_DATA):
            prohibidas.append({"ruta": r, "commits": sum(r in rs for rs in inv.rutas_por_commit.values())})
    credenciales = []
    for r in sorted(inv.rutas_todas | set(inv.no_versionados)):
        base = r.rsplit("/", 1)[-1]
        if base in NOMBRES_DE_CREDENCIALES or base.lower().endswith(EXT_CREDENCIALES) or base.lower().startswith("credentials"):
            credenciales.append({"ruta": r, "versionado": r in inv.rutas_todas})
    return Resultado(hallazgos, conteos, imagenes, sorted(set(binarios)), prohibidas, credenciales, textos_docs, len(cache))


def categoria_id(h: Hallazgo, refs: Referencias, hexes: Sequence[str], explicados: set[str]) -> str | None:
    """Descarte automático de un id: código SCIAN, prefijo de hash o valor que el mismo archivo explica."""
    if h.valor in refs.codigos_scian:
        return "codigo_scian"
    i = bisect.bisect_left(hexes, h.valor)
    if i < len(hexes) and hexes[i].startswith(h.valor):
        return "prefijo_de_hash"
    if h.valor in explicados:
        return "explicado_por_el_archivo"
    pos = h.linea_texto.find(h.valor)
    while pos >= 0:
        if RE_UNIDAD.match(h.linea_texto[pos + len(h.valor):]):
            return "explicado_por_el_archivo"
        pos = h.linea_texto.find(h.valor, pos + 1)
    return None


def clasificar_nombres(res: Resultado, inv: Inventario, refs: Referencias, src_busqueda: str) -> None:
    """Categorías automáticas de los hallazgos de nombres (falsos positivos ya aceptados)."""
    for h in res.hallazgos:
        if not h.es_nombre:
            continue
        doc = inv.documentos[h.documento]
        rutas = {r for _c, r in doc.ubicaciones}
        llave = h.valor if h.tipo == "llave_varias_palabras" else normalizar_nombre(h.valor)
        if rutas <= ARCHIVOS_CATALOGO or contiene_frase(refs.titulos_normalizados, llave):
            h.categoria = "catalogo"
        elif all(r.startswith("tests/") for r in rutas):
            h.categoria = "prueba"
        elif all(r.startswith(PREFIJOS_GENERADOS) or r in GENERADOS for r in rutas) and contiene_frase(src_busqueda, llave):
            h.categoria = "texto_fijo"


def busqueda_de_codigo(inv: Inventario) -> str:
    """Texto normalizado de los archivos de src/ en HEAD (el código versionado)."""
    lineas = []
    for d in inv.documentos.values():
        if any(r.startswith("src/") and r.endswith(".py") for r in d.rutas_head):
            lineas += decodificar(d.datos).split("\n")
    return texto_de_busqueda(lineas)


def aplicar_manual(hallazgos: Sequence[Hallazgo], manual: Mapping[str, str]) -> None:
    for h in hallazgos:
        if h.categoria is None and h.ident() in manual:
            cat = manual[h.ident()]
            grupo = "nombre" if h.es_nombre else "id" if h.tipo == "id" else "otro"
            if cat not in CATEGORIAS_MANUALES[grupo]:
                raise ErrorAuditoria(f"Categoría manual {cat!r} no válida para un hallazgo de tipo {h.tipo}.")
            h.categoria = cat


# --- Agrupación, reporte y veredicto ---


def agrupar(hallazgos: Sequence[Hallazgo]) -> list[list[Hallazgo]]:
    """Un grupo por hallazgo distinto (pasada, tipo, valor y texto de la línea), con todas sus ubicaciones."""
    g: dict[str, list[Hallazgo]] = defaultdict(list)
    for h in hallazgos:
        g[h.ident()].append(h)
    return list(g.values())


def ubicacion(grupo: Sequence[Hallazgo], inv: Inventario) -> dict:
    h = grupo[0]
    doc = inv.documentos[h.documento]
    commit, ruta = doc.ubicaciones[0]
    en_head = sorted({(r, x.linea) for x in grupo for r in inv.documentos[x.documento].rutas_head})
    return {"commit": commit[:7] if len(commit) == 40 else commit, "ruta": ruta, "linea": h.linea,
            "en_head": [f"{r}:{n}" for r, n in en_head], "ubicaciones": sum(len(inv.documentos[x.documento].ubicaciones) for x in grupo)}


def detiene(h: Hallazgo) -> bool:
    """Un hallazgo que detiene (PARADA A): sin categoría o posible dato real; en la pasada extra, solo si es
    nombre completo o está cerca de un id."""
    if h.categoria not in (None, "posible_dato_real"):
        return False
    if h.pasada == "denue_completo":
        return h.completo or h.cerca_de_id
    return True


def nombre_grupo(h: Hallazgo) -> str:
    return "nombre" if h.es_nombre else "id" if h.tipo == "id" else "otro"


def escribir_detalle(grupos: Sequence[Sequence[Hallazgo]], inv: Inventario, carpeta: Path, fecha: str) -> Path:
    """Detalle con las llaves (ignorado por git), para la revisión manual de Claude Code y Samuel."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{fecha}_detalle.md"
    partes = ["# Detalle de la auditoría (con llaves: no sale de data/)\n"]
    for g in sorted(grupos, key=lambda g: (g[0].categoria is not None, g[0].pasada, g[0].tipo, ubicacion(g, inv)["ruta"])):
        h, u = g[0], ubicacion(g, inv)
        partes.append(f"## {h.ident()} · {h.pasada} · {h.tipo} · {h.categoria or 'SIN CATEGORÍA'}"
                      f"{' · completo' if h.completo else ''}{' · cerca de id' if h.cerca_de_id else ''}\n")
        partes.append(f"- {u['commit']} {u['ruta']}:{u['linea']} · en HEAD: {', '.join(u['en_head']) or 'no'} · "
                      f"ubicaciones: {u['ubicaciones']}\n- valor: «{h.valor}»\n- línea: «{h.linea_texto[:400]}»\n")
    ruta.write_text("\n".join(partes), encoding="utf-8")
    return ruta


def clave_de_valor(h: Hallazgo) -> str:
    """Índice estable de un valor (sin la llave): agrupa los hallazgos del mismo valor para revisarlos juntos."""
    valor = normalizar_nombre(h.valor) if h.es_nombre else h.valor
    return "V" + hashlib.sha256(f"{h.pasada}\0{nombre_grupo(h)}\0{valor}".encode()).hexdigest()[:10]


def pendiente_de_revision(h: Hallazgo) -> bool:
    """Sin categoría y, en la pasada extra, solo si puede detener (las parciales quedan para mañana)."""
    return h.categoria is None and (h.pasada != "denue_completo" or h.completo or h.cerca_de_id)


def escribir_revision(grupos: Sequence[Sequence[Hallazgo]], inv: Inventario, carpeta: Path, fecha: str) -> tuple[Path, Path]:
    """Revisión manual agrupada por valor (con llaves; no sale de data/) y su índice {V…: [ids de hallazgo]}.

    Las decisiones se escriben por índice V… en `decisiones.json` y `expandir_decisiones` las pasa a los ids de
    hallazgo de ese momento: un hallazgo nuevo del mismo valor vuelve a quedar sin categoría."""
    carpeta.mkdir(parents=True, exist_ok=True)
    por_valor: dict[str, list[Sequence[Hallazgo]]] = defaultdict(list)
    for g in grupos:
        if pendiente_de_revision(g[0]):
            por_valor[clave_de_valor(g[0])].append(g)
    partes = ["# Revisión manual por valor (con llaves: no sale de data/)\n"]
    indice: dict[str, list[str]] = {}
    orden = sorted(por_valor.items(), key=lambda kv: (kv[1][0][0].pasada, nombre_grupo(kv[1][0][0]), -len(kv[1]), kv[0]))
    for v, gs in orden:
        h0 = gs[0][0]
        indice[v] = sorted(g[0].ident() for g in gs)
        partes.append(f"## {v} · {h0.pasada} · {nombre_grupo(h0)} · «{h0.valor}» · {len(gs)} hallazgos\n")
        for g in sorted(gs, key=lambda g: ubicacion(g, inv)["ruta"]):
            h, u = g[0], ubicacion(g, inv)
            marca = h.linea_texto.replace(h.valor, f"«{h.valor}»", 1)
            partes.append(f"- {h.tipo} · {u['commit']} {u['ruta']}:{u['linea']}{' · completo' if h.completo else ''}"
                          f"{' · cerca de id' if h.cerca_de_id else ''} · HEAD: {', '.join(u['en_head']) or 'no'}\n"
                          f"  > {marca[:300]}")
        partes.append("")
    ruta = carpeta / f"{fecha}_revision.md"
    ruta.write_text("\n".join(partes), encoding="utf-8")
    ruta_i = carpeta / "revision_indice.json"
    ruta_i.write_text(json.dumps(indice, indent=1), encoding="utf-8")
    return ruta, ruta_i


def expandir_decisiones(carpeta: Path) -> int:
    """decisiones.json {V…: categoría} + revision_indice.json → clasificacion_manual.json {id de hallazgo: categoría}."""
    decisiones = json.loads((carpeta / "decisiones.json").read_text(encoding="utf-8"))
    indice = json.loads((carpeta / "revision_indice.json").read_text(encoding="utf-8"))
    ruta = carpeta / ARCHIVO_MANUAL
    manual = json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else {}
    faltan = sorted(set(decisiones) - set(indice))
    if faltan:
        raise ErrorAuditoria(f"Decisiones con índices que no están en la revisión: {faltan}")
    n = 0
    for v, cat in decisiones.items():
        if not any(cat in c for c in CATEGORIAS_MANUALES.values()):
            raise ErrorAuditoria(f"Categoría desconocida en {v}: {cat!r}")
        for i in indice[v]:
            manual[i] = cat
            n += 1
    ruta.write_text(json.dumps(manual, indent=1, sort_keys=True), encoding="utf-8")
    return n


def resumen(res: Resultado, inv: Inventario) -> dict:
    """Solo conteos (sin rutas, líneas ni largos): es lo que puede ir a results/."""
    grupos = agrupar(res.hallazgos)
    por = Counter((g[0].pasada, g[0].tipo.split(":")[0], g[0].categoria or "sin_categoria") for g in grupos)
    tabla: dict = {}
    for (pasada, tipo, cat), n in sorted(por.items()):
        tabla.setdefault(pasada, {}).setdefault(tipo, {})[cat] = n
    return {
        "commits_alcanzables": len(inv.commits),
        "commits_solo_en_el_reflog": len(inv.reflog_solo),
        "documentos": len(inv.documentos),
        "documentos_de_texto": res.documentos_texto,
        "lineas_unicas": res.lineas_unicas,
        "imagenes_png": len(res.imagenes),
        "imagenes_con_bloques_de_texto": sum(bool(i["bloques_de_texto"]) for i in res.imagenes),
        "otros_binarios": len(res.binarios),
        "archivos_no_versionados_revisados": len(inv.no_versionados),
        "rutas_prohibidas_en_algun_commit": len(res.rutas_prohibidas),
        "rutas_de_credenciales": len(res.rutas_credenciales),
        "hallazgos_distintos": tabla,
        "no_hallazgos": {p: dict(sorted(c.items())) for p, c in res.conteos_no_hallazgo.items()},
        "detienen": sum(1 for g in grupos if detiene(g[0])),
    }


def imprimir(res: Resultado, inv: Inventario, salida=print) -> None:
    grupos = agrupar(res.hallazgos)
    salida(f"Commits alcanzables: {len(inv.commits)}; solo en el reflog (no se publican): {len(inv.reflog_solo)} "
           f"({', '.join(c[:7] for c in inv.reflog_solo) or '-'})")
    salida(f"Documentos: {len(inv.documentos):,} ({res.documentos_texto:,} de texto, {len(res.imagenes)} PNG, "
           f"{len(res.binarios)} otros binarios); líneas únicas: {res.lineas_unicas:,}")
    salida(f"No versionados revisados: {inv.no_versionados}; versionados con cambios: {inv.modificados}")
    for a, n in sorted(inv.autores.items()):
        salida(f"  {a}: {n} commits")
    salida(f"Rutas prohibidas en algún commit: {res.rutas_prohibidas or 'ninguna'}")
    salida(f"Rutas de credenciales: {res.rutas_credenciales or 'ninguna'}")
    for i in res.imagenes:
        salida(f"  PNG {i['ruta']}: {i['bytes']:,} bytes; bloques de texto: {i['bloques_de_texto'] or 'ninguno'}")
    for pasada in ("4_conjuntos", "denue_completo", "general"):
        gs = [g for g in grupos if g[0].pasada == pasada]
        salida(f"\n== Pasada {pasada}: {len(gs)} hallazgos distintos")
        salida(f"   Conteos que no son hallazgo: {dict(sorted(res.conteos_no_hallazgo.get(pasada, {}).items()))}")
        for cat, n in sorted(Counter(g[0].categoria or "SIN CATEGORÍA" for g in gs).items()):
            salida(f"   {cat}: {n}")
        listar = [g for g in gs if not (g[0].tipo == "id" and g[0].categoria in DESCARTES_ID)]
        if len(listar) < len(gs):
            salida(f"   (ids descartados solos, solo conteo: {len(gs) - len(listar)})")
        for n_g, g in enumerate(sorted(listar, key=lambda g: (g[0].categoria or "", ubicacion(g, inv)["ruta"], g[0].linea)), 1):
            h, u = g[0], ubicacion(g, inv)
            extra = (" · completo" if h.completo else "") + (" · cerca de id" if h.cerca_de_id else "")
            salida(f"   #{n_g:>3} {u['commit']:>18} {u['ruta']}:{u['linea']} · {h.tipo} · {h.largo} · "
                   f"{h.categoria or 'SIN CATEGORÍA'}{extra} · HEAD: {', '.join(u['en_head']) or 'no'} · ubic. {u['ubicaciones']}"
                   f"{' · DETIENE' if detiene(h) else ''}")


# --- Registro en results/ ---


def registro(res: Resultado, inv: Inventario, refs_hashes: Mapping[str, str], fecha: str, arbol_limpio: bool) -> dict:
    return {
        "fecha": fecha,
        "experimento": EXPERIMENTO,
        "commit": inv.head,
        "arbol_limpio": arbol_limpio,
        "modelo": "no aplica (auditoría de privacidad del repo)",
        "configuracion": {
            "script": SCRIPT,
            "metodo_nombres": "regla acotada (método del paso 6c indexado, con la precisión del paso 11), por línea "
                              "única y sobre los valores de cada JSON",
            "pasadas": {"4_conjuntos": "train final, validación, test y hard (test y hard sin etiquetas)",
                        "denue_completo": "todos los nom_estab del DENUE; solo detiene un nombre completo o uno cerca de un id"},
            "ids": "corridas aisladas de 5 a 8 dígitos iguales a un id del DENUE",
            "detalle": f"{DIR_DETALLE}/ (ignorado por git)",
        },
        "semilla": 42,
        "split": "no aplica",
        "hash_split": dict(refs_hashes),
        "tamano_muestra": {"commits": len(inv.commits), "documentos": len(inv.documentos)},
        "resultado": resumen(res, inv),
    }


def arbol_limpio(raiz: Path) -> bool:
    return not git(raiz, "status", "--porcelain", "--untracked-files=no").strip()


# --- Archivos sueltos (texto nuevo antes de un commit) ---


def revisar_archivos(rutas: Sequence[Path], refs: Referencias, raiz: Path, salida=print) -> list[Hallazgo]:
    """Las mismas reglas sobre archivos sueltos; imprime solo ruta, línea, tipo, largo y categoría."""
    docs = {}
    for r in rutas:
        rel = Path(r).resolve().relative_to(Path(raiz).resolve()).as_posix() if Path(r).resolve().is_relative_to(Path(raiz).resolve()) else str(r)
        docs[f"wt:{rel}"] = Documento(clave=f"wt:{rel}", datos=Path(r).read_bytes(), ubicaciones=[("(archivo)", rel)])
    inv = Inventario([], None, docs, {}, Counter(), [], [], [], set())
    res = revisar(inv, refs)
    src = texto_de_busqueda(l for p in sorted(Path(raiz, "src").glob("*.py")) for l in p.read_text(encoding="utf-8").split("\n"))
    clasificar_nombres(res, inv, refs, src)
    for g in agrupar(res.hallazgos):
        h, u = g[0], ubicacion(g, inv)
        if h.pasada == "denue_completo" and not detiene(h):
            continue
        salida(f"  {u['ruta']}:{u['linea']} · {h.pasada} · {h.tipo} · {h.largo} · {h.categoria or 'SIN CATEGORÍA'}"
               f"{' · completo' if h.completo else ''}{' · cerca de id' if h.cerca_de_id else ''}")
    carpeta = Path(raiz) / DIR_DETALLE
    if carpeta.is_dir():  # detalle con llaves, para reescribir el texto (ignorado por git)
        escribir_detalle(agrupar(res.hallazgos), inv, carpeta, "archivos")
    return res.hallazgos


# --- Programa principal ---


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument("--escribir", action="store_true", help="escribe el registro de conteos en results/")
    p.add_argument("--archivos", nargs="+", type=Path, help="revisa solo estos archivos")
    p.add_argument("--sin-extra", action="store_true", help="sin la pasada contra todo el DENUE")
    p.add_argument("--expandir", action="store_true",
                   help="pasa data/auditoria/decisiones.json a clasificacion_manual.json y termina")
    a = p.parse_args(argv)
    from src.denue import REPO_RAIZ

    raiz = REPO_RAIZ
    if a.expandir:
        print(f"Decisiones expandidas a {expandir_decisiones(raiz / DIR_DETALLE):,} hallazgos.")
        return 0
    print("Cargando referencias (4 conjuntos, DENUE completo y SCIAN 2023 y 2018)...", flush=True)
    refs = cargar_referencias(raiz, extra=not a.sin_extra)
    if a.archivos:
        hall = revisar_archivos(a.archivos, refs, raiz)
        n = sum(1 for g in agrupar(hall) if detiene(g[0]))
        print(f"Hallazgos que detienen (sin categoría automática): {n}")
        return 1 if n else 0
    fecha = datetime.date.today().isoformat()
    print("Inventario de git...", flush=True)
    inv = inventario(raiz)
    print(f"Revisando {len(inv.documentos):,} documentos...", flush=True)
    res = revisar(inv, refs, lambda s: print(s, flush=True))
    clasificar_nombres(res, inv, refs, busqueda_de_codigo(inv))
    ruta_manual = raiz / DIR_DETALLE / ARCHIVO_MANUAL
    if ruta_manual.is_file():
        aplicar_manual(res.hallazgos, json.loads(ruta_manual.read_text(encoding="utf-8")))
    detalle = escribir_detalle(agrupar(res.hallazgos), inv, raiz / DIR_DETALLE, fecha)
    revision, _ = escribir_revision(agrupar(res.hallazgos), inv, raiz / DIR_DETALLE, fecha)
    print(f"Revisión manual pendiente por valor: {revision.relative_to(raiz).as_posix()}")
    imprimir(res, inv)
    r = resumen(res, inv)
    print(f"\nDetalle (con llaves, ignorado por git): {detalle.relative_to(raiz).as_posix()}")
    print(f"Hallazgos que detienen: {r['detienen']}")
    if a.escribir:
        if r["detienen"]:
            print("No se escribe el registro: hay hallazgos que detienen.")
            return 1
        reg = registro(res, inv, refs.hashes, fecha, arbol_limpio(raiz))
        texto = json.dumps(reg, ensure_ascii=False, indent=2) + "\n"
        ruta = raiz / "results" / f"{fecha}_{EXPERIMENTO}.json"
        tmp = raiz / DIR_DETALLE / "registro_candidato.json"
        tmp.write_text(texto, encoding="utf-8", newline="\n")
        if any(detiene(g[0]) for g in agrupar(revisar_archivos([tmp], refs, raiz, salida=lambda s: None))):
            print("No se escribe el registro: la regla acotada encuentra algo en su propio texto.")
            return 1
        ruta.write_bytes(texto.encode("utf-8"))
        print(f"Registro escrito: {ruta.relative_to(raiz).as_posix()}")
    return 1 if r["detienen"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
