"""Subida de los modelos a Hugging Face (paso 20). Por omisión es un dry-run: lista y verifica, no sube nada.

Dos repos de modelo, con el usuario de Hugging Face como parámetro:
- <usuario>/denue-scian-qwen3-4b-lora: card, LICENSE y el adaptador LoRA del 4B (`4b-lr2e-4-r32`);
- <usuario>/denue-scian-qwen3-0.6b-lora-gguf: card, LICENSE, el adaptador LoRA del 0.6B (`06b-lr2e-4-r32`) y su
  GGUF BF16 (el que usa la demo).
Ningún dato del DENUE: nada de splits, predicciones, errores ni nombres.

Antes de subir verifica cada archivo: bytes y SHA-256 contra `results/` (adaptadores y GGUF) o contra el blob de
HEAD (cards y LICENSE, que además deben estar sin cambios); que ningún archivo tenga un SHA-256 de
`data/manifest.json` ni extensión de datos; y que la card, con el usuario ya puesto, no deje marcadores.

La subida real (con su bandera) solo la corre Samuel en su terminal:
- se niega si existe la variable CLAUDECODE (la definen las shells de Claude Code), antes de pedir nada;
- el token sale de HF_TOKEN o se pide con getpass (sin eco ni historial); nunca se imprime ni se escribe, y se
  tapa en cualquier mensaje de error;
- `whoami` debe coincidir con el usuario; un `create_commit` por repo; después compara bytes, SHA-256 de LFS y
  `blob_id` de git de cada archivo contra lo local.

Solo usa la biblioteca estándar; `huggingface_hub` se importa solo en la subida real, con el Python de
`venv-tokenizador` (huggingface_hub 1.33.0 y hf_xet 1.8.0). Las cards se versionan con el marcador del usuario.

Uso (desde la raíz del repo):
  .venv/Scripts/python -m src.subir_hf --usuario USUARIO            # dry-run

El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MARCA_USUARIO = "{{usuario_hf}}"
RE_USUARIO = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
EXT_DATOS = (".jsonl", ".csv", ".tsv", ".parquet", ".joblib", ".zip", ".pkl", ".xlsx")
VARIABLE_CLAUDE = "CLAUDECODE"
TAMANO_LFS = 10 * 1024 * 1024  # de aquí para arriba, un archivo sin SHA-256 remoto solo se avisa
CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Archivo:
    en_repo: str
    origen: str          # relativo a la raíz del repo (el GGUF vive fuera, con ../)
    esperado: tuple      # ("head",) | ("archivos_bajados", resultado, ruta) | ("gguf", resultado, nombre)
    plantilla: bool = False


R_4B = "results/2026-11-24_fase2_4b-lr2e-4-r32.json"
R_06B = "results/2026-11-24_fase2_06b-lr2e-4-r32.json"
R_GGUF = "results/2026-11-25_fase4_gguf-concordancia.json"
GGUF = "qwen3-0.6b-06b-lr2e-4-r32-BF16.gguf"
REPOS: dict[str, dict] = {
    "4b": {
        "nombre": "denue-scian-qwen3-4b-lora",
        "archivos": (
            Archivo("README.md", "hf/denue-scian-qwen3-4b-lora/README.md", ("head",), plantilla=True),
            Archivo("LICENSE", "LICENSE", ("head",)),
            Archivo("adapter_config.json", "kaggle-output/4b/4b-lr2e-4-r32/adaptador/adapter_config.json",
                    ("archivos_bajados", R_4B, "4b-lr2e-4-r32/adaptador/adapter_config.json")),
            Archivo("adapter_model.safetensors", "kaggle-output/4b/4b-lr2e-4-r32/adaptador/adapter_model.safetensors",
                    ("archivos_bajados", R_4B, "4b-lr2e-4-r32/adaptador/adapter_model.safetensors")),
        ),
    },
    "06b": {
        "nombre": "denue-scian-qwen3-0.6b-lora-gguf",
        "archivos": (
            Archivo("README.md", "hf/denue-scian-qwen3-0.6b-lora-gguf/README.md", ("head",), plantilla=True),
            Archivo("LICENSE", "LICENSE", ("head",)),
            Archivo("adapter_config.json", "kaggle-output/06b-barrido/06b-lr2e-4-r32/adaptador/adapter_config.json",
                    ("archivos_bajados", R_06B, "06b-lr2e-4-r32/adaptador/adapter_config.json")),
            Archivo("adapter_model.safetensors",
                    "kaggle-output/06b-barrido/06b-lr2e-4-r32/adaptador/adapter_model.safetensors",
                    ("archivos_bajados", R_06B, "06b-lr2e-4-r32/adaptador/adapter_model.safetensors")),
            Archivo(GGUF, f"../modelos/gguf/{GGUF}", ("gguf", R_GGUF, GGUF)),
        ),
    },
}


class ErrorSubida(RuntimeError):
    """Detiene todo antes de subir (o después, si la comprobación remota falla). Nunca lleva el token."""


# --- Utilidades ---


def sha256_de(fuente: bytes | Path) -> str:
    if isinstance(fuente, bytes):
        return hashlib.sha256(fuente).hexdigest()
    h = hashlib.sha256()
    with open(fuente, "rb") as f:
        while bloque := f.read(CHUNK):
            h.update(bloque)
    return h.hexdigest()


def sha1_git(datos: bytes) -> str:
    """El `blob_id` que da git (y Hugging Face) a un archivo que no va por LFS."""
    return hashlib.sha1(b"blob %d\0" % len(datos) + datos).hexdigest()


def git(raiz: Path, *args: str) -> bytes:
    r = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=raiz, capture_output=True)
    if r.returncode != 0:
        raise ErrorSubida(f"git {' '.join(args[:2])} falló (código {r.returncode}).")
    return r.stdout


def hashes_del_manifest(obj) -> set[str]:
    """Todos los valores de claves `sha256` o terminadas en `_sha256` del manifest."""
    res: set[str] = set()
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(v, str) and (k == "sha256" or str(k).endswith("_sha256")):
                res.add(v)
            else:
                res |= hashes_del_manifest(v)
    elif isinstance(obj, list):
        for v in obj:
            res |= hashes_del_manifest(v)
    return res


def tapar(texto: str, token: str | None) -> str:
    return texto.replace(token, "[token oculto]") if token else texto


# --- Plan y verificación ---


@dataclass
class Entrada:
    en_repo: str
    fuente: bytes | Path   # bytes para lo que sale de HEAD; ruta para los pesos
    bytes: int
    sha256: str
    origen: str            # para mostrar
    esperado_de: str       # de dónde sale el SHA-256 esperado
    sha256_plantilla: str | None = None


def esperado(raiz: Path, a: Archivo, cache: dict) -> tuple[int, str, str]:
    """(bytes, sha256, descripción) esperados de un peso, leídos de results/."""
    tipo, resultado, clave = a.esperado
    if resultado not in cache:
        cache[resultado] = json.loads((raiz / resultado).read_text(encoding="utf-8"))
    r = cache[resultado]
    if tipo == "archivos_bajados":
        ent = [x for x in r["corrida_kaggle"]["archivos_bajados"] if x["ruta"] == clave]
        if len(ent) != 1:
            raise ErrorSubida(f"{resultado} no registra una sola vez {clave}.")
        return ent[0]["bytes"], ent[0]["sha256"], f"{resultado} (corrida_kaggle.archivos_bajados)"
    if tipo == "gguf":
        g = r["herramientas_y_modelos"]["gguf"][clave]
        elegida = r["cuantizaciones"]["BF16"]["gguf"]
        if r["regla_demo"]["elegida"] != "BF16" or elegida["sha256"] != g["sha256"] or elegida["archivo"] != clave:
            raise ErrorSubida(f"{resultado}: el GGUF no es el BF16 que eligió la regla de la demo.")
        return g["bytes"], g["sha256"], f"{resultado} (herramientas_y_modelos.gguf)"
    raise ErrorSubida(f"Tipo de esperado desconocido: {tipo}.")


def de_head(raiz: Path, ruta: str) -> bytes:
    """Bytes versionados en HEAD; el archivo no puede tener cambios sin commit."""
    if git(raiz, "status", "--porcelain", "--", ruta).strip():
        raise ErrorSubida(f"{ruta} tiene cambios sin commit: se sube solo lo versionado.")
    try:
        return git(raiz, "show", f"HEAD:{ruta}")
    except ErrorSubida as e:
        raise ErrorSubida(f"{ruta} no está versionado en HEAD.") from e


def plan(usuario: str, repos: Sequence[str], raiz: Path = RAIZ) -> list[tuple[str, list[Entrada]]]:
    """Lo que se subiría, con todo verificado; lanza ErrorSubida ante cualquier diferencia."""
    if not RE_USUARIO.fullmatch(usuario):
        raise ErrorSubida("El usuario de Hugging Face solo puede tener letras, dígitos y guiones (sin guion en las orillas).")
    prohibidos = hashes_del_manifest(json.loads((raiz / "data" / "manifest.json").read_text(encoding="utf-8")))
    cache: dict = {}
    res = []
    for clave in repos:
        spec = REPOS[clave]
        entradas = []
        for a in spec["archivos"]:
            if a.en_repo.lower().endswith(EXT_DATOS) or a.origen.lower().endswith(EXT_DATOS):
                raise ErrorSubida(f"{a.en_repo}: extensión de datos, no se sube.")
            if a.esperado[0] == "head":
                datos = de_head(raiz, a.origen)
                sha_p = None
                if a.plantilla:
                    texto = datos.decode("utf-8")
                    if MARCA_USUARIO not in texto:
                        raise ErrorSubida(f"{a.origen} no tiene el marcador del usuario.")
                    sha_p = sha256_de(datos)
                    datos = texto.replace(MARCA_USUARIO, usuario).encode("utf-8")
                    if "{{" in datos.decode("utf-8") or "}}" in datos.decode("utf-8"):
                        raise ErrorSubida(f"{a.origen}: quedan marcadores después de poner el usuario.")
                e = Entrada(a.en_repo, datos, len(datos), sha256_de(datos), a.origen, f"HEAD:{a.origen}", sha_p)
            else:
                ruta = (raiz / a.origen).resolve()
                if not ruta.is_file():
                    raise ErrorSubida(f"Falta {ruta}.")
                n, sha, de = esperado(raiz, a, cache)
                tam = ruta.stat().st_size
                if tam != n:
                    raise ErrorSubida(f"{a.en_repo}: {tam:,} bytes y se esperaban {n:,} ({de}).")
                calc = sha256_de(ruta)
                if calc != sha:
                    raise ErrorSubida(f"{a.en_repo}: SHA-256 {calc} distinto del esperado {sha} ({de}).")
                e = Entrada(a.en_repo, ruta, n, calc, str(ruta), de)
            if e.sha256 in prohibidos:
                raise ErrorSubida(f"{a.en_repo} tiene un SHA-256 de data/manifest.json: no se sube.")
            entradas.append(e)
        res.append((f"{usuario}/{spec['nombre']}", entradas))
    return res


def imprimir_plan(p: Sequence[tuple[str, Sequence[Entrada]]], privado: bool, salida: Callable[[str], None]) -> int:
    total = 0
    for repo, entradas in p:
        salida(f"Repo {repo} ({'privado' if privado else 'público'} al crearse)")
        for e in entradas:
            salida(f"  {e.en_repo} · {e.bytes:,} bytes · SHA-256 {e.sha256}")
            salida(f"      origen: {e.origen}")
            if e.sha256_plantilla:
                salida(f"      card con el usuario puesto; plantilla {e.esperado_de}, SHA-256 {e.sha256_plantilla}")
            else:
                salida(f"      verificado contra {e.esperado_de}")
            total += e.bytes
    n = sum(len(x) for _r, x in p)
    salida(f"Total: {n} archivos, {total:,} bytes ({total / 1e9:.2f} GB). Todos los SHA-256 verificados.")
    return total


# --- Cliente de Hugging Face (solo en la subida real) ---


class ClienteHF:
    """Lo único que se usa de huggingface_hub; se importa aquí y no antes."""

    def __init__(self, token: str):
        from huggingface_hub import HfApi

        self._api = HfApi(token=token)

    def usuario(self) -> str:
        return self._api.whoami()["name"]

    def crear_repo(self, repo_id: str, privado: bool) -> None:
        self._api.create_repo(repo_id=repo_id, repo_type="model", private=privado, exist_ok=True)

    def archivos(self, repo_id: str) -> list[str]:
        return list(self._api.list_repo_files(repo_id=repo_id, repo_type="model"))

    def commit(self, repo_id: str, archivos: Sequence[tuple[str, bytes | Path]], mensaje: str) -> str:
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=f if isinstance(f, bytes) else str(f)) for p, f in archivos]
        return self._api.create_commit(repo_id=repo_id, repo_type="model", operations=ops, commit_message=mensaje).commit_url

    def info(self, repo_id: str, rutas: Sequence[str]) -> dict[str, dict]:
        res = {}
        for x in self._api.get_paths_info(repo_id=repo_id, paths=list(rutas), repo_type="model", expand=True):
            lfs = getattr(x, "lfs", None)
            res[x.path] = {"bytes": x.size, "sha256": getattr(lfs, "sha256", None), "blob_id": x.blob_id}
        return res


def comprobar_remoto(remoto: Mapping[str, Mapping], entradas: Sequence[Entrada]) -> list[str]:
    """Errores de la comparación remota y avisos (los avisos empiezan con 'aviso:')."""
    res = []
    for e in entradas:
        r = remoto.get(e.en_repo)
        if r is None:
            res.append(f"{e.en_repo}: no está en el repo.")
        elif r["bytes"] != e.bytes:
            res.append(f"{e.en_repo}: {r['bytes']} bytes en el repo, {e.bytes} locales.")
        elif r.get("sha256"):
            if r["sha256"] != e.sha256:
                res.append(f"{e.en_repo}: SHA-256 remoto distinto del local.")
        elif isinstance(e.fuente, bytes):
            if r.get("blob_id") != sha1_git(e.fuente):
                res.append(f"{e.en_repo}: blob_id remoto distinto del local.")
        elif e.bytes >= TAMANO_LFS:
            res.append(f"aviso: {e.en_repo} no trae SHA-256 remoto; compáralo en la web con {e.sha256}.")
        elif r.get("blob_id") != sha1_git(Path(e.fuente).read_bytes()):
            res.append(f"{e.en_repo}: blob_id remoto distinto del local.")
    return res


def subir(p: Sequence[tuple[str, Sequence[Entrada]]], usuario: str, privado: bool, token: str,
          cliente: Callable[[str], object], salida: Callable[[str], None]) -> None:
    api = cliente(token)
    quien = api.usuario()
    if quien != usuario:
        raise ErrorSubida(f"El token es de {quien!r}, no de {usuario!r}: no se sube nada.")
    salida(f"Token de {quien}: correcto.")
    for repo, entradas in p:
        api.crear_repo(repo, privado)
        extra = sorted(set(api.archivos(repo)) - {e.en_repo for e in entradas} - {".gitattributes"})
        if extra:
            raise ErrorSubida(f"{repo} ya tiene otros archivos ({extra}): no se mezcla.")
        url = api.commit(repo, [(e.en_repo, e.fuente) for e in entradas],
                         "Subida del paso 20 (src/subir_hf.py), con SHA-256 verificados")
        salida(f"{repo}: commit {url}")
        problemas = comprobar_remoto(api.info(repo, [e.en_repo for e in entradas]), entradas)
        for x in problemas:
            salida(f"  {x}")
        errores = [x for x in problemas if not x.startswith("aviso:")]
        if errores:
            raise ErrorSubida(f"{repo}: la comprobación después de subir falló en {len(errores)} archivos.")
        salida(f"  {len(entradas)} archivos comprobados en el repo.")


# --- Programa principal ---


def main(argv: Sequence[str] | None = None, *, entorno: Mapping[str, str] = os.environ,
         pedir_token: Callable[[str], str] = getpass.getpass, cliente: Callable[[str], object] = ClienteHF,
         raiz: Path = RAIZ, salida: Callable[[str], None] = print) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument("--usuario", required=True, help="usuario de Hugging Face dueño de los dos repos")
    p.add_argument("--repo", choices=sorted(REPOS), action="append", help="solo este repo (se puede repetir)")
    p.add_argument("--publico", action="store_true", help="crea los repos públicos (por omisión, privados)")
    p.add_argument("--subir", action="store_true", help="subida real; solo Samuel, desde su terminal")
    a = p.parse_args(argv)
    if a.subir and VARIABLE_CLAUDE in entorno:
        salida("La subida real solo la corre Samuel desde su terminal (existe CLAUDECODE): no se pidió token ni se subió nada.")
        return 2
    repos = a.repo or list(REPOS)
    try:
        pl = plan(a.usuario, repos, raiz)
    except ErrorSubida as e:
        salida(f"Detenido antes de subir: {e}")
        return 1
    imprimir_plan(pl, not a.publico, salida)
    if not a.subir:
        salida("Dry-run: nada se subió.")
        return 0
    token = entorno.get("HF_TOKEN") or pedir_token("Token de escritura de Hugging Face (no se muestra): ")
    if not token or not token.strip():
        salida("Sin token: no se subió nada.")
        return 1
    token = token.strip()
    try:
        subir(pl, a.usuario, not a.publico, token, cliente, salida)
    except Exception as e:  # noqa: BLE001  (cualquier error se reporta sin el token)
        salida(f"Detenido: {tapar(f'{type(e).__name__}: {e}', token)}")
        return 1
    salida("Subida terminada. Revisa las páginas y cambia la visibilidad en la web si hace falta.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
