"""Herramientas, fusión del adaptador y GGUF del 0.6B afinado (Fase 4, paso 19, Tema A).

Todo lo pesado vive fuera del repo (decisión D9 del plan): herramientas/ (llama.cpp b12359 y las ruedas de venv-gguf),
venv-gguf/ y modelos/ (caché de Hugging Face, carpetas para convertir y GGUF). En el repo solo se escribe el registro
data/fase4/gguf/registro_gguf.json (ignorado por git), con versiones, URLs, comandos, bytes, SHA-256 y comprobaciones.

Modos (con el entorno venv-gguf; torch, transformers, peft y safetensors se importan dentro de las funciones):
  ../venv-gguf/Scripts/python -m src.fusion_gguf --herramientas
      Verifica los zips de llama.cpp (binarios Windows CPU x64 contra el digest de GitHub; fuente de la misma etiqueta),
      los extrae cada uno en su carpeta nueva y registra los SHA-256 de llama-server, llama-quantize y el convertidor.
  ../venv-gguf/Scripts/python -m src.fusion_gguf --entorno
      Registra los SHA-256 de las ruedas de herramientas/wheels-gguf/ y el pip freeze de venv-gguf.
  ../venv-gguf/Scripts/python -m src.fusion_gguf --base
      Espejo unsloth/Qwen3-0.6B (db4e90b8) contra el oficial Qwen/Qwen3-0.6B (c1899de2): SHA-256 de cada archivo
      contra la API de Hugging Face y pesos tensor por tensor (uno a la vez); lm_head del oficial igual a embed_tokens.
  ../venv-gguf/Scripts/python -m src.fusion_gguf --fusionar
      Espejo en bf16 + adaptador con peft (merge_and_unload) en CPU, con la prueba de equivalencia en las capas 0, 13
      y 27; arma las carpetas del base y del afinado con la config y el tokenizador de la revisión oficial (D4).
  ../venv-gguf/Scripts/python -m src.fusion_gguf --convertir [--base-extra Q8_0]
      convert_hf_to_gguf.py a BF16 (D5) para los dos y llama-quantize: Q8_0 y Q4_K_M del afinado, Q4_K_M del base.

Antes de cada carga de pesos: `exigir_memoria` (3 GB libres o me detengo). En la terminal solo salen conteos, hashes
y tiempos. El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from src import llama_servidor as ls
from src.gguf_fase4 import (
    CARPETA,
    CUANTIZACIONES,
    DIR_GGUF,
    DIR_LLAMA,
    DIR_PROYECTO,
    MODELO,
    REGISTRO,
    REPO_RAIZ,
    REVISION_OFICIAL,
    SNAPSHOT_TOKENIZADOR,
    TAG_LLAMA,
    ErrorGguf,
    a_bytes,
    escribir_en_data,
    nombre_gguf,
    sha256_archivo,
)

# --- llama.cpp b12359 (la build de la release estable v1.0.0, decisión D1) ---

URL_NIGHTLY_V1 = "https://github.com/ggml-org/llama.cpp/releases/download/v1.0.0/nightly-tag.txt"
URL_BIN = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG_LLAMA}/llama-{TAG_LLAMA}-bin-win-cpu-x64.zip"
URL_FUENTE = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{TAG_LLAMA}.zip"
SHA_BIN = "837c1655e87df3b66989f4be33a792a1399c785bf259038dc75c900096683e69"  # digest publicado por GitHub
SHA_FUENTE = "247865b712dbfaff227bf93bf71d247de896b991b59c0ed67e8eaf76285203e5"  # calculado al bajarlo (2026-11-25)
COMMIT_TAG = "b1b770d90d94cad9e9ea5b993bdbc383a522fb28"  # target_commitish de la release b12359 en la API de GitHub
DIR_HERRAMIENTAS = DIR_PROYECTO / "herramientas"
DIR_DESCARGAS = DIR_HERRAMIENTAS / "descargas"
ZIP_BIN = DIR_DESCARGAS / "bin" / f"llama-{TAG_LLAMA}-bin-win-cpu-x64.zip"
ZIP_FUENTE = DIR_DESCARGAS / "fuente" / f"llama.cpp-{TAG_LLAMA}.zip"
NIGHTLY = DIR_DESCARGAS / "v1.0.0" / "nightly-tag.txt"
DIR_BIN = DIR_LLAMA / "bin-win-cpu-x64"
DIR_FUENTE = DIR_LLAMA / "fuente"
DIR_RUEDAS = DIR_HERRAMIENTAS / "wheels-gguf"
VENV_GGUF = DIR_PROYECTO / "venv-gguf"

# --- Modelos ---

ESPEJO = {"repo": "unsloth/Qwen3-0.6B", "revision": "db4e90b8359f816f09bf5c944797ce116f26da3c"}
OFICIAL = {"repo": "Qwen/Qwen3-0.6B", "revision": REVISION_OFICIAL}
ARCHIVOS_BASE = ("config.json", "generation_config.json", "model.safetensors")
ARCHIVOS_TOKENIZADOR = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
DIR_MODELOS = DIR_PROYECTO / "modelos"
HF_CACHE = DIR_MODELOS / "hf-cache"
DIR_HF = {"base": DIR_MODELOS / "hf-base-06b", "afinado": DIR_MODELOS / f"hf-afinado-{MODELO}"}
ADAPTADOR = {"carpeta": "kaggle-output/06b-barrido/06b-lr2e-4-r32/adaptador",
             "adapter_model.safetensors": "19d9f15ecb1ef403ed1bb341dcbac27a6c56967f2f23a9907e1ab082164967b0",
             "adapter_config.json": "d3f6f71c7dac7e35d46d59a62f0aa658486414903ace93d671805f400bbf7419"}
CAPAS = 28
MODULOS = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
           "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
CAPAS_EQUIVALENCIA = (0, 13, 27)
TENSOR_ENTRADA = "model.embed_tokens.weight"
TENSOR_SALIDA = "lm_head.weight"
CUANT_DEL_BASE = ("Q4_K_M",)


# --- Registro ---


def leer_registro_o_vacio(raiz: Path = REPO_RAIZ) -> dict:
    ruta = Path(raiz) / REGISTRO
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else {"version": 1}


def actualizar_registro(seccion: str, datos, *, raiz: Path = REPO_RAIZ,
                        escribir: Callable[[bytes, str], object] = escribir_en_data) -> dict:
    reg = leer_registro_o_vacio(raiz)
    if seccion == "gguf":
        reg.setdefault("gguf", {}).update(datos)
    else:
        reg[seccion] = datos
    escribir(a_bytes(reg), REGISTRO)
    return reg


# --- Funciones puras (se prueban en .venv) ---


def git_blob_sha1(datos: bytes) -> str:
    """El id de git de un archivo chico (lo que la API de Hugging Face da como blob_id de un archivo sin LFS)."""
    return hashlib.sha1(b"blob %d\0" % len(datos) + datos).hexdigest()


def extraer_zip(zip_ruta: Path, destino: Path) -> list[str]:
    """Extrae en una carpeta nueva o vacía; rechaza rutas absolutas o con '..'. Devuelve los nombres extraídos."""
    destino = Path(destino)
    if destino.exists() and any(destino.iterdir()):
        raise ErrorGguf(f"{destino} ya existe y no está vacía.")
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_ruta) as z:
        nombres = z.namelist()
        for n in nombres:
            p = Path(n)
            if p.is_absolute() or ".." in p.parts or n.startswith(("/", "\\")):
                raise ErrorGguf(f"Ruta peligrosa dentro de {Path(zip_ruta).name}.")
        z.extractall(destino)
    return nombres


def huella_de_carpeta(carpeta: Path, hash_archivo: Callable[[Path], str] = sha256_archivo) -> dict:
    """Archivos y SHA-256 de las líneas "ruta_relativa sha256\\n" de todos los archivos, en orden."""
    carpeta = Path(carpeta)
    archivos = sorted(p for p in carpeta.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    lineas = "".join(f"{p.relative_to(carpeta).as_posix()} {hash_archivo(p)}\n" for p in archivos)
    return {"archivos": len(archivos), "sha256": hashlib.sha256(lineas.encode("utf-8")).hexdigest(),
            "definicion": "SHA-256 de las líneas 'ruta_relativa sha256' de cada archivo, en orden, con \\n."}


def comparar_tensores(claves_a: Sequence[str], claves_b: Sequence[str], iguales: Callable[[str], bool]) -> dict:
    """Conteos de la comparación tensor por tensor de dos archivos (a: espejo, b: oficial)."""
    a, b = set(claves_a), set(claves_b)
    comunes = sorted(a & b)
    distintos = [k for k in comunes if not iguales(k)]
    return {"tensores_a": len(a), "tensores_b": len(b), "comunes": len(comunes), "distintos": len(distintos),
            "lista_distintos": distintos[:20], "solo_a": sorted(a - b), "solo_b": sorted(b - a)}


def diferencias(esperado, obtenido) -> dict:
    """Elementos distintos y diferencia absoluta máxima (sirve con arreglos de numpy o tensores de torch, en fp32)."""
    if tuple(esperado.shape) != tuple(obtenido.shape):
        raise ErrorGguf("Las formas no coinciden en la prueba de equivalencia.")
    return {"elementos": int(esperado.numel() if hasattr(esperado, "numel") else esperado.size),
            "distintos": int((esperado != obtenido).sum()), "max_abs": float(abs(esperado - obtenido).max())}


def clave_lora_en_modelo(clave_archivo: str) -> str:
    """'...q_proj.lora_A.weight' del archivo → '...q_proj.lora_A.default.weight' del modelo de peft."""
    if not clave_archivo.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ErrorGguf("Llave del adaptador inesperada.")
    return clave_archivo[: -len(".weight")] + ".default.weight"


def peso_de(capa: int, modulo: str) -> str:
    return f"model.layers.{capa}.{MODULOS[modulo]}.{modulo}.weight"


def diferencias_de_config(a: Mapping, b: Mapping) -> dict:
    claves = sorted(set(a) | set(b))
    return {k: {"espejo": a.get(k), "oficial": b.get(k)} for k in claves if a.get(k) != b.get(k)}


def nombre_corto(cmd: Sequence[str]) -> list[str]:
    """Un comando con las rutas reducidas a su nombre de archivo (sin rutas de la laptop en el registro)."""
    return [Path(x).name if ("\\" in x or "/" in x) else x for x in map(str, cmd)]


# --- Modos ---


def correr_herramientas() -> dict:
    if sha256_archivo(ZIP_BIN) != SHA_BIN:
        raise ErrorGguf("El zip de binarios no tiene el SHA-256 del digest de GitHub.")
    if sha256_archivo(ZIP_FUENTE) != SHA_FUENTE:
        raise ErrorGguf("El zip fuente no tiene el SHA-256 registrado al bajarlo.")
    nightly = NIGHTLY.read_text(encoding="utf-8").strip()
    if nightly != TAG_LLAMA:
        raise ErrorGguf(f"nightly-tag.txt de v1.0.0 dice {nightly}, no {TAG_LLAMA}.")
    with zipfile.ZipFile(ZIP_FUENTE) as z:
        comentario = z.comment.decode("ascii", "replace").strip()
    if comentario != COMMIT_TAG:
        raise ErrorGguf(f"El zip fuente no es del commit {COMMIT_TAG[:8]} (comentario del zip: {comentario[:12]}).")
    if not DIR_BIN.is_dir():
        extraer_zip(ZIP_BIN, DIR_BIN)
    if not DIR_FUENTE.is_dir():
        extraer_zip(ZIP_FUENTE, DIR_FUENTE)
    server = DIR_BIN / "llama-server.exe"
    quantize = DIR_BIN / "llama-quantize.exe"
    convert = convertidor()
    version = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=60)
    texto_version = (version.stdout + version.stderr).strip().splitlines()
    datos = {
        "release_estable": "v1.0.0",
        "nightly_tag_de_v1": {"url": URL_NIGHTLY_V1, "contenido": nightly, "sha256": sha256_archivo(NIGHTLY)},
        "etiqueta": TAG_LLAMA,
        "commit": COMMIT_TAG,
        "binarios": {"url": URL_BIN, "bytes": ZIP_BIN.stat().st_size, "sha256": SHA_BIN,
                     "verificado_contra": "digest sha256 de la API de GitHub"},
        "fuente": {"url": URL_FUENTE, "bytes": ZIP_FUENTE.stat().st_size, "sha256": SHA_FUENTE,
                   "comentario_del_zip": comentario},
        "llama_server_exe": {"bytes": server.stat().st_size, "sha256": sha256_archivo(server),
                             "nota": "Lanzador: el servidor está en llama-server-impl.dll (ver binarios_extraidos)."},
        "llama_quantize_exe": {"bytes": quantize.stat().st_size, "sha256": sha256_archivo(quantize)},
        "binarios_extraidos": {p.name: sha256_archivo(p) for p in sorted(DIR_BIN.iterdir()) if p.is_file()},
        "convert_hf_to_gguf_py": {"bytes": convert.stat().st_size, "sha256": sha256_archivo(convert)},
        "conversion_py": huella_de_carpeta(convert.parent / "conversion"),
        "gguf_py": huella_de_carpeta(convert.parent / "gguf-py"),
        "llama_server_version": [l for l in texto_version if "version" in l.lower() or "built" in l.lower()][:3],
    }
    actualizar_registro("llama_cpp", datos)
    return datos


def convertidor() -> Path:
    hay = sorted(DIR_FUENTE.glob("*/convert_hf_to_gguf.py"))
    if len(hay) != 1:
        raise ErrorGguf("No encuentro un solo convert_hf_to_gguf.py en la fuente extraída.")
    return hay[0]


def correr_entorno() -> dict:
    ruedas = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_archivo(p)}
              for p in sorted(DIR_RUEDAS.iterdir()) if p.is_file()}
    python = VENV_GGUF / "Scripts" / "python.exe"
    freeze = subprocess.run([str(python), "-m", "pip", "freeze", "--all"], capture_output=True, text=True,
                            check=True).stdout.splitlines()
    version = subprocess.run([str(python), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    datos = {"python": version, "pip_freeze": freeze, "ruedas": ruedas,
             # Redacción revisada con la regla acotada de nombres: la primera versión tenía frases comunes de dos
             # palabras que coinciden con llaves del DENUE y detenía la escritura del resultado (paso 19).
             "instalacion": "Ruedas bajadas con pip download a herramientas/wheels-gguf/ según "
                            "requirements-convert_hf_to_gguf.txt de b12359 (torch 2.11.0+cpu, transformers 4.57.6, "
                            "numpy 2.2.6, sentencepiece, protobuf <5), más peft 0.21.3 y safetensors; instaladas con "
                            "pip install --no-index --find-links. gguf: no se instala; convert_hf_to_gguf.py usa gguf-py "
                            "de b12359 desde sys.path (ver llama_cpp.gguf_py)."}
    actualizar_registro("venv_gguf", datos)
    return datos


def _hf_api_archivos(repo: str, revision: str) -> dict[str, dict]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ErrorGguf(f"La API de Hugging Face no devolvió la revisión fijada de {repo}.")
    res = {}
    for s in info.siblings:
        lfs = getattr(s, "lfs", None)
        res[s.rfilename] = {"size": s.size, "blob_id": s.blob_id,
                            "lfs_sha256": (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))}
    return res


def ruta_hf(repo: str, revision: str, archivo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=archivo, revision=revision, cache_dir=str(HF_CACHE)))


def _verificar_hf(modelo: Mapping) -> dict:
    api = _hf_api_archivos(modelo["repo"], modelo["revision"])
    res = {}
    for a in ARCHIVOS_BASE:
        p = ruta_hf(modelo["repo"], modelo["revision"], a)
        e = api.get(a)
        if e is None:
            raise ErrorGguf(f"{a} no está en {modelo['repo']}@{modelo['revision'][:8]}.")
        if e["lfs_sha256"]:
            sha = sha256_archivo(p)
            ok = sha == e["lfs_sha256"] and p.stat().st_size == e["size"]
            res[a] = {"bytes": p.stat().st_size, "sha256": sha, "igual_a_la_api": ok}
        else:
            datos = p.read_bytes()
            ok = git_blob_sha1(datos) == e["blob_id"]
            res[a] = {"bytes": len(datos), "sha256": hashlib.sha256(datos).hexdigest(), "git_blob_sha1": e["blob_id"],
                      "igual_a_la_api": ok}
        if not ok:
            raise ErrorGguf(f"{a} de {modelo['repo']} no coincide con la API de Hugging Face.")
    return res


def _iguales_torch(a, b) -> bool:
    import torch

    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    return bool(torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)))


def correr_base() -> dict:
    from safetensors import safe_open

    ls.exigir_memoria()
    archivos = {"espejo": _verificar_hf(ESPEJO), "oficial": _verificar_hf(OFICIAL)}
    pe = ruta_hf(ESPEJO["repo"], ESPEJO["revision"], "model.safetensors")
    po = ruta_hf(OFICIAL["repo"], OFICIAL["revision"], "model.safetensors")
    with safe_open(str(pe), framework="pt") as fe, safe_open(str(po), framework="pt") as fo:
        comp = comparar_tensores(list(fe.keys()), list(fo.keys()),
                                 lambda k: _iguales_torch(fe.get_tensor(k), fo.get_tensor(k)))
        lm_head_atado = (TENSOR_SALIDA in fo.keys() and TENSOR_ENTRADA in fo.keys()
                         and _iguales_torch(fo.get_tensor(TENSOR_SALIDA), fo.get_tensor(TENSOR_ENTRADA)))
    ok = comp["distintos"] == 0 and not comp["solo_a"] and comp["solo_b"] in ([], [TENSOR_SALIDA]) and \
        (TENSOR_SALIDA not in comp["solo_b"] or lm_head_atado)
    configs = {}
    for a in ("config.json", "generation_config.json"):
        ca = json.loads(ruta_hf(ESPEJO["repo"], ESPEJO["revision"], a).read_text(encoding="utf-8"))
        co = json.loads(ruta_hf(OFICIAL["repo"], OFICIAL["revision"], a).read_text(encoding="utf-8"))
        configs[a] = diferencias_de_config(ca, co)
    datos = {"espejo": ESPEJO, "oficial": OFICIAL, "archivos": archivos, "tensores": comp,
             "lm_head_oficial_igual_a_embed_tokens": lm_head_atado, "pesos_iguales": ok,
             "diferencias_de_config_espejo_contra_oficial": configs,
             "metodo": "safetensors safe_open, un tensor a la vez: mismo dtype, misma forma y mismos bytes."}
    actualizar_registro("base", datos)
    if not ok:
        raise ErrorGguf("Los pesos del espejo no coinciden con los del oficial: me detengo.")
    return datos


def _copiar_config_y_tokenizador(destino: Path) -> dict:
    res = {}
    for a in ("config.json", "generation_config.json"):
        p = ruta_hf(OFICIAL["repo"], OFICIAL["revision"], a)
        shutil.copyfile(p, destino / a)
        res[a] = sha256_archivo(destino / a)
    for a in ARCHIVOS_TOKENIZADOR:
        shutil.copyfile(SNAPSHOT_TOKENIZADOR / a, destino / a)
        res[a] = sha256_archivo(destino / a)
    return res


def correr_fusionar() -> dict:
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    reg = leer_registro_o_vacio()
    if not reg.get("base", {}).get("pesos_iguales"):
        raise ErrorGguf("Primero --base: el espejo debe estar verificado contra el oficial.")
    for d in DIR_HF.values():
        if d.exists():
            raise ErrorGguf(f"{d} ya existe: no se sobrescribe.")
    carpeta = REPO_RAIZ / ADAPTADOR["carpeta"]
    for a in ("adapter_model.safetensors", "adapter_config.json"):
        if sha256_archivo(carpeta / a) != ADAPTADOR[a]:
            raise ErrorGguf(f"{a} no tiene el SHA-256 del pre-registro.")
    cfg = json.loads((carpeta / "adapter_config.json").read_text(encoding="utf-8"))
    escala = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("modules_to_save") or set(cfg["target_modules"]) != set(MODULOS):
        raise ErrorGguf("El adaptador no es la LoRA simple que se espera.")

    libre = ls.exigir_memoria()
    t0 = time.perf_counter()
    pe = ruta_hf(ESPEJO["repo"], ESPEJO["revision"], "model.safetensors")
    model = AutoModelForCausalLM.from_pretrained(str(pe.parent), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    originales = {(c, m): model.get_parameter(peso_de(c, m)).detach().clone() for c in CAPAS_EQUIVALENCIA for m in MODULOS}
    model = PeftModel.from_pretrained(model, str(carpeta), is_trainable=False)
    nombres_mod = [n for n, _ in model.named_modules()]
    modulos_lora = sum(n.endswith(".lora_A.default") for n in nombres_mod)
    escalas = {float(mod.scaling["default"]) for n, mod in model.named_modules() if hasattr(mod, "scaling")
               and isinstance(getattr(mod, "scaling"), dict) and "default" in mod.scaling}
    estado = model.state_dict()
    iguales_lora = total_lora = 0
    dtypes_lora = set()
    with safe_open(str(carpeta / "adapter_model.safetensors"), framework="pt") as fa:
        llaves_adaptador = list(fa.keys())
        lora = {k: fa.get_tensor(k) for k in llaves_adaptador}
    for k, t in lora.items():
        total_lora += 1
        cargado = estado.get(clave_lora_en_modelo(k))
        if cargado is not None:
            dtypes_lora.add(str(cargado.dtype))
            iguales_lora += bool(torch.equal(cargado.float(), t.float()))
    if modulos_lora != CAPAS * len(MODULOS) or total_lora != 2 * modulos_lora or iguales_lora != total_lora:
        raise ErrorGguf(f"El adaptador no entró completo: {modulos_lora} módulos LoRA, {iguales_lora} de {total_lora} "
                        "tensores iguales al archivo.")
    if escalas != {escala}:
        raise ErrorGguf(f"Escalas de la LoRA inesperadas: {sorted(escalas)}.")
    del estado

    model = model.merge_and_unload()
    quedan_lora = sum("lora_" in n for n, _ in model.named_modules())
    equivalencia = {}
    for (c, m), w0 in originales.items():
        a = lora[f"base_model.model.{peso_de(c, m)[: -len('.weight')]}.lora_A.weight"].float()
        b = lora[f"base_model.model.{peso_de(c, m)[: -len('.weight')]}.lora_B.weight"].float()
        esperado = (w0.float() + escala * (b @ a)).to(torch.bfloat16).float()
        obtenido = model.get_parameter(peso_de(c, m)).detach().float()
        equivalencia[f"capa {c} {m}"] = diferencias(esperado, obtenido)
    # Los tensores que la LoRA no toca deben seguir idénticos al espejo.
    tocados = {peso_de(c, m) for c in range(CAPAS) for m in MODULOS}
    no_tocados_iguales = no_tocados = 0
    estado = model.state_dict()
    with safe_open(str(pe), framework="pt") as fe:
        for k in fe.keys():
            if k in tocados:
                continue
            no_tocados += 1
            no_tocados_iguales += _iguales_torch(fe.get_tensor(k), estado[k].detach())
    tocados_cambiados = sum(not torch.equal(estado[k], originales[(c, m)]) for (c, m) in originales
                            for k in [peso_de(c, m)])
    del estado

    guardado = {}
    for rol in ("afinado", "base"):
        DIR_HF[rol].mkdir(parents=True)
    model.save_pretrained(str(DIR_HF["afinado"]), safe_serialization=True, max_shard_size="5GB")
    del model
    shutil.copyfile(pe, DIR_HF["base"] / "model.safetensors")
    for rol, d in DIR_HF.items():
        cfg_guardada = json.loads((d / "config.json").read_text(encoding="utf-8")) if (d / "config.json").is_file() else {}
        archivos = _copiar_config_y_tokenizador(d)
        cfg_oficial = json.loads((d / "config.json").read_text(encoding="utf-8"))
        guardado[rol] = {"model_safetensors": {"bytes": (d / "model.safetensors").stat().st_size,
                                               "sha256": sha256_archivo(d / "model.safetensors")},
                         "config_y_tokenizador_oficiales_sha256": archivos,
                         "claves_de_config_guardada_distintas_de_la_oficial": sorted(diferencias_de_config(cfg_guardada, cfg_oficial)) if cfg_guardada else None,
                         "archivos": sorted(p.name for p in d.iterdir())}
    maximo = max(v["max_abs"] for v in equivalencia.values())
    distintos = sum(v["distintos"] for v in equivalencia.values())
    import peft
    import transformers

    datos = {
        "metodo": "transformers AutoModelForCausalLM en bf16 (dtype nativo) + peft PeftModel.from_pretrained + "
                  "merge_and_unload, en CPU; se guarda con save_pretrained (safetensors). Config y tokenizador: los de "
                  "la revisión oficial (D4).",
        "versiones": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "memoria_libre_antes_mb": libre,
        "dtypes_del_base": dtypes,
        "adaptador": {"r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "escala": escala, "modulos_lora": modulos_lora,
                      "tensores_iguales_al_archivo": iguales_lora, "tensores": total_lora, "dtypes_cargados": sorted(dtypes_lora),
                      "sha256": {a: ADAPTADOR[a] for a in ("adapter_model.safetensors", "adapter_config.json")}},
        "modulos_lora_despues_de_fusionar": quedan_lora,
        "prueba_de_equivalencia": {
            "definicion": "W' fusionado contra bf16(W + escala·B·A, calculado en fp32), en las capas 0, 13 y 27 con sus "
                          "7 módulos.",
            "por_modulo": equivalencia, "elementos_distintos": distintos, "max_abs": maximo,
            "pesos_que_cambiaron_con_la_fusion": tocados_cambiados, "pesos_revisados": len(originales)},
        "tensores_no_tocados_por_la_lora": {"tensores": no_tocados, "identicos_al_espejo": no_tocados_iguales},
        "carpetas": guardado,
        "segundos": round(time.perf_counter() - t0, 1),
    }
    actualizar_registro("fusion", datos)
    if quedan_lora or no_tocados_iguales != no_tocados or distintos:
        raise ErrorGguf("La fusión no pasó sus comprobaciones (ver registro_gguf.json, sección fusion).")
    return datos


def _correr(cmd: Sequence[str], log: Path) -> float:
    t0 = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as f:
        r = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise ErrorGguf(f"{Path(cmd[1] if Path(cmd[0]).name.startswith('python') else cmd[0]).name} terminó con "
                        f"código {r.returncode}; ver {log.name}.")
    return round(time.perf_counter() - t0, 1)


def correr_convertir(base_extra: Sequence[str] = ()) -> dict:
    reg = leer_registro_o_vacio()
    if not reg.get("fusion"):
        raise ErrorGguf("Primero --fusionar.")
    DIR_GGUF.mkdir(parents=True, exist_ok=True)
    logs = REPO_RAIZ / CARPETA / "logs"
    quantize = DIR_BIN / "llama-quantize.exe"
    nuevos = {}
    planes = {"afinado": [q for q in CUANTIZACIONES if q != "BF16"], "base": [*CUANT_DEL_BASE, *base_extra]}
    for rol, cuants in planes.items():
        bf16 = DIR_GGUF / nombre_gguf(rol, "BF16")
        if not bf16.is_file():
            ls.exigir_memoria()
            cmd = [sys.executable, convertidor(), DIR_HF[rol], "--outtype", "bf16", "--outfile", bf16]
            s = _correr(cmd, logs / f"convertir_{rol}.log")
            nuevos[bf16.name] = {"bytes": bf16.stat().st_size, "sha256": sha256_archivo(bf16), "segundos": s,
                                 "comando": nombre_corto(cmd)}
        for q in dict.fromkeys(cuants):
            salida = DIR_GGUF / nombre_gguf(rol, q)
            if salida.exists():
                continue
            ls.exigir_memoria()
            cmd = [quantize, bf16, salida, q, str(ls.HILOS)]
            s = _correr(cmd, logs / f"cuantizar_{rol}_{q}.log")
            nuevos[salida.name] = {"bytes": salida.stat().st_size, "sha256": sha256_archivo(salida), "segundos": s,
                                   "comando": nombre_corto(cmd), "desde": bf16.name}
    actualizar_registro("gguf", nuevos)
    return nuevos


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    for m in ("herramientas", "entorno", "base", "fusionar", "convertir"):
        g.add_argument(f"--{m}", action="store_true")
    p.add_argument("--base-extra", nargs="*", default=[], choices=[q for q in CUANTIZACIONES if q != "Q4_K_M"])
    args = p.parse_args(argv)
    try:
        if args.herramientas:
            r = correr_herramientas()
        elif args.entorno:
            r = correr_entorno()
            r = {"python": r["python"], "ruedas": len(r["ruedas"]), "paquetes": len(r["pip_freeze"])}
        elif args.base:
            r = correr_base()
            r = {k: r[k] for k in ("tensores", "lm_head_oficial_igual_a_embed_tokens", "pesos_iguales",
                                   "diferencias_de_config_espejo_contra_oficial")}
        elif args.fusionar:
            r = correr_fusionar()
            r = {k: r[k] for k in ("versiones", "adaptador", "modulos_lora_despues_de_fusionar",
                                   "tensores_no_tocados_por_la_lora", "segundos")} | {
                "equivalencia": {k: v for k, v in r["prueba_de_equivalencia"].items() if k != "por_modulo"}}
        else:
            r = correr_convertir(args.base_extra)
        print(json.dumps(r, ensure_ascii=False, indent=2))
    except (ErrorGguf, ls.ErrorServidor) as e:
        sys.exit(f"ME DETENGO: {e}")


if __name__ == "__main__":
    main()
