"""GGUF del 0.6B afinado en la laptop: tokens, concordancia con Kaggle y latencia (Fase 4, paso 19, Tema A).

El pre-registro (evals/fase4_preregistro.json, gguf.concordancia) manda: la concordancia del GGUF con el modelo de
Kaggle se mide en validación, nunca en test ni en hard. Este módulo nunca abre test ni hard.

Modos (el chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó):

  ../venv-tokenizador/Scripts/python -m src.gguf_fase4 --tokens ETIQUETA
      Con el llama-server de ese GGUF arriba: en las 1,000 filas de validación, los ids de /tokenize (add_special,
      lo mismo que hace /completion con un prompt de texto) contra los del tokenizador de Hugging Face; cada token
      especial suelto es un solo id igual al de Hugging Face; el largo de cada prompt es el `fichas_prompt` de Kaggle.
      Escribe solo conteos en data/fase4/gguf/<etiqueta>_tokens.json.
  ../venv-tokenizador/Scripts/python -m src.gguf_fase4 --generar ETIQUETA
      Las 1,000 respuestas de validación del afinado en esa cuantización (decisión D6 del plan): prompt crudo de
      prompt_generacion, greedy, 16 fichas nuevas, corte de Kaggle y decodificación con el tokenizador de Hugging Face
      (skip_special_tokens=False). Escribe data/fase4/gguf/<etiqueta>_validacion.jsonl (con ids: ignorado por git) y
      <etiqueta>_generacion.json (conteos de las comprobaciones).
  .venv/Scripts/python -m src.gguf_fase4 --evaluar [--escribir]
      Concordancia de BF16, Q8_0 y Q4_K_M con Kaggle, métricas del evaluador único y la regla de la demo
      (`elegir_cuantizacion`, fijada en código antes de medir). Con --escribir: results/<fecha>_fase4_gguf-concordancia.json.
  .venv/Scripts/python -m src.gguf_fase4 --latencia
      Los dos modelos de la demo (afinado y base en la cuantización elegida) arriba a la vez; 210 filas de validación
      sorteadas con default_rng(42): las 10 primeras de calentamiento y 200 medidas. Guarda las mediciones, sin ids, en
      data/fase4/gguf/latencia_mediciones.json.
  .venv/Scripts/python -m src.gguf_fase4 --escribir-latencia
      results/<fecha>_fase4_latencia.json, desde esas mediciones.

En la terminal solo salen conteos, hashes, tiempos y porcentajes: nunca nombres, llaves, ids ni respuestas.
Arriba solo se importan la biblioteca estándar y módulos de src/ que tampoco traen nada más, para que el entorno del
tokenizador (sin pandas) pueda correr --tokens y --generar; el evaluador y numpy se importan dentro de las funciones.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from src import llama_servidor as ls
from src.humo_kaggle import AbortoHumo, leer_jsonl_verificado
from src.prompt_clasificador import FIN_DE_TURNO, INICIO_DE_TURNO, prompt_generacion

REPO_RAIZ = Path(__file__).resolve().parents[1]
DIR_PROYECTO = REPO_RAIZ.parent
SCRIPT = "src/gguf_fase4.py"
SEMILLA = 42

TAG_LLAMA = "b12359"
DIR_LLAMA = DIR_PROYECTO / "herramientas" / f"llama.cpp-{TAG_LLAMA}"
LLAMA_SERVER = DIR_LLAMA / "bin-win-cpu-x64" / "llama-server.exe"
DIR_GGUF = DIR_PROYECTO / "modelos" / "gguf"
REVISION_OFICIAL = "c1899de289a04d12100db370d81485cdf75e47ca"
SNAPSHOT_TOKENIZADOR = (DIR_PROYECTO / "venv-tokenizador" / "hf-cache" / "models--Qwen--Qwen3-0.6B" / "snapshots"
                        / REVISION_OFICIAL)

CARPETA = "data/fase4/gguf"
REGISTRO = f"{CARPETA}/registro_gguf.json"
MEDICIONES_LATENCIA = f"{CARPETA}/latencia_mediciones.json"
MODELO = "06b-lr2e-4-r32"
CUANTIZACIONES = ("Q4_K_M", "Q8_0", "BF16")  # orden de generación: la más lenta al final (ajuste 4 de Samuel)
CUANTIZACION_DE_RESPALDO = "Q4_K_M"  # la de la regla: gana si la mejor no le saca al menos 1 punto
ROLES = ("base", "afinado")  # orden de las preguntas en la latencia y en la demo
PUERTOS = {"base": 8081, "afinado": 8082}
NUEVAS = {"afinado": ls.NUEVAS_AFINADO, "base": ls.NUEVAS_BASE}

REFERENCIA = {"ruta": "kaggle-output/fase4/fase4/predicciones/06b-lr2e-4-r32/validacion.jsonl",
              "sha256": "9ca2ad286a57b4f428ee3ffdf7db7ecc357d4455121c80c743742614259c1069"}
FASE2 = {"ruta": "kaggle-output/06b-barrido/06b-lr2e-4-r32/predicciones.jsonl",
         "sha256": "9e6b429cfd97ff5db451ef9a9ac16ec4af995ab620203803a8473b91ec0a8ed1"}
CAMPOS_REFERENCIA = ("id", "respuesta", "fichas_prompt", "fichas_generadas", "al_tope", "segundos")
CAMPOS_FASE2 = ("id", "respuesta")
CAMPOS_PREDICCION = ("id", "respuesta", "fichas_prompt", "fichas_generadas", "al_tope", "parada", "fin_visible", "ms")
ENTRADA_PAQUETE = "denue_scian_clasificador"
ARCHIVO_VAL_PAQUETE = "val.jsonl"
ARCHIVO_TRAIN_PAQUETE = "train_final.jsonl"
ENTRADA_VAL = "clasificador_validacion"
RUTA_CONFIG = "evals/fase4_preregistro.json"
FILAS_VAL = 1000

N_LATENCIA = 210  # ajuste 2 de Samuel: perm[:210]; las 10 primeras de calentamiento y 200 medidas
CALENTAMIENTO = 10
PATRON_CONCORDANCIA = "*_fase4_gguf-concordancia.json"
PATRON_LATENCIA = "*_fase4_latencia.json"
MINIMO_PROGRAMA_MB = 100

REGLA_DEMO = (
    "Fijada antes de medir (paso 19): se usa la cuantización con mayor concordancia (% de respuestas idénticas, "
    "carácter por carácter, a las de Kaggle en las 1,000 filas de validación); si su diferencia con Q4_K_M es menor "
    "de 1 punto, se usa Q4_K_M. Empate en concordancia: gana la de archivo más chico (D7). Con conteos enteros: "
    "Q4_K_M gana si 100·(n_mejor − n_Q4_K_M) < total."
)
TOKENS_ESPECIALES = (INICIO_DE_TURNO, FIN_DE_TURNO, "<|endoftext|>")


class ErrorGguf(RuntimeError):
    """PARADA del paso 19; el mensaje nunca incluye nombres, llaves, ids, prompts ni respuestas."""


# --- Nombres de archivos ---


def nombre_gguf(rol: str, cuant: str) -> str:
    if rol not in ROLES or cuant not in CUANTIZACIONES:
        raise ErrorGguf(f"Modelo o cuantización desconocidos: {rol}, {cuant}.")
    return f"qwen3-0.6b-{MODELO if rol == 'afinado' else 'base'}-{cuant}.gguf"


def etiqueta(rol: str, cuant: str) -> str:
    nombre_gguf(rol, cuant)
    return f"{rol}-{cuant}"


def de_etiqueta(texto: str) -> tuple[str, str]:
    rol, _, cuant = texto.partition("-")
    nombre_gguf(rol, cuant)
    return rol, cuant


# --- Lectura verificada (solo biblioteca estándar) ---


def sha256_bytes(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def sha256_archivo(ruta: str | Path, bloque: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while b := f.read(bloque):
            h.update(b)
    return h.hexdigest()


def leer_verificado(ruta: Path, sha_esperado: str) -> bytes:
    datos = Path(ruta).read_bytes()
    sha = sha256_bytes(datos)
    if sha != sha_esperado:
        raise ErrorGguf(f"El SHA-256 de {Path(ruta).name} es {sha}, no {sha_esperado}.")
    return datos


def leer_jsonl(datos: bytes, campos: Sequence[str]) -> list[dict]:
    """Registros con las claves exactas, en orden; UTF-8 sin BOM, cortando solo en "\\n"; ids de dígitos sin repetir."""
    if datos.startswith(b"\xef\xbb\xbf"):
        raise ErrorGguf("Un JSONL tiene BOM.")
    texto = datos.decode("utf-8")
    if not texto.endswith("\n"):
        raise ErrorGguf("Un JSONL está vacío o no termina en salto de línea.")
    regs = []
    for i, linea in enumerate(texto[:-1].split("\n"), start=1):
        r = json.loads(linea)
        if not isinstance(r, dict) or tuple(r) != tuple(campos):
            raise ErrorGguf(f"Línea {i}: las claves no son las esperadas.")
        if not isinstance(r["id"], str) or not r["id"].isascii() or not r["id"].isdigit():
            raise ErrorGguf(f"Línea {i}: el id no es solo dígitos ASCII.")
        if not isinstance(r["respuesta"], str):
            raise ErrorGguf(f"Línea {i}: la respuesta no es texto.")
        regs.append(r)
    if len({r["id"] for r in regs}) != len(regs):
        raise ErrorGguf("Un JSONL tiene ids repetidos.")
    return regs


def leer_manifest(raiz: Path = REPO_RAIZ) -> dict:
    return json.loads((Path(raiz) / "data" / "manifest.json").read_text(encoding="utf-8"))


def _paquete(manifest: Mapping, archivo: str) -> Mapping:
    return manifest["kaggle_datasets"][ENTRADA_PAQUETE]["archivos"][archivo]


def val_del_paquete(raiz: Path = REPO_RAIZ, manifest: Mapping | None = None) -> list[dict]:
    """La validación que vio Kaggle (id, nom_estab y codigo_act), con su SHA-256 del manifest, en orden de int(id)."""
    manifest = manifest if manifest is not None else leer_manifest(raiz)
    e = _paquete(manifest, ARCHIVO_VAL_PAQUETE)
    try:
        regs = leer_jsonl_verificado(Path(raiz) / e["ruta"], e["sha256"])
    except AbortoHumo as err:
        raise ErrorGguf(str(err)) from err
    if len(regs) != FILAS_VAL:
        raise ErrorGguf(f"La validación del paquete tiene {len(regs)} filas, no {FILAS_VAL}.")
    return sorted(regs, key=lambda r: int(r["id"]))


def leer_referencia(raiz: Path = REPO_RAIZ) -> dict[str, dict]:
    """Respuestas de validación del afinado en el kernel de la Fase 4 (SHA-256 fijo), por id."""
    regs = leer_jsonl(leer_verificado(Path(raiz) / REFERENCIA["ruta"], REFERENCIA["sha256"]), CAMPOS_REFERENCIA)
    for r in regs:
        if not (isinstance(r["al_tope"], bool) and all(isinstance(r[k], int) and not isinstance(r[k], bool)
                                                         for k in ("fichas_prompt", "fichas_generadas"))):
            raise ErrorGguf("La referencia de Kaggle tiene un campo con un tipo inesperado.")
    if len(regs) != FILAS_VAL:
        raise ErrorGguf(f"La referencia de Kaggle tiene {len(regs)} filas, no {FILAS_VAL}.")
    return {r["id"]: r for r in regs}


def leer_fase2(raiz: Path = REPO_RAIZ) -> dict[str, str]:
    regs = leer_jsonl(leer_verificado(Path(raiz) / FASE2["ruta"], FASE2["sha256"]), CAMPOS_FASE2)
    return {r["id"]: r["respuesta"] for r in regs}


# --- Git y archivos ignorados (sin pandas: el entorno del tokenizador también los usa) ---


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True).stdout.strip()


def estado_git_simple() -> dict:
    return {"commit": _git("rev-parse", "HEAD"),
            "arbol_limpio": _git("status", "--porcelain", "--untracked-files=no") == ""}


def ignorado(ruta: str | Path) -> bool:
    p = Path(ruta)
    p = p if p.is_absolute() else REPO_RAIZ / p
    rel = p.resolve().relative_to(REPO_RAIZ.resolve()).as_posix()
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO_RAIZ, capture_output=True)
    if r.returncode not in (0, 1):
        raise ErrorGguf(f"git check-ignore falló con código {r.returncode}.")
    return r.returncode == 0


def escribir_en_data(datos: bytes, ruta: str | Path) -> Path:
    """Escribe solo en una ruta que git ignora (data/fase4/gguf/)."""
    p = Path(ruta)
    p = p if p.is_absolute() else REPO_RAIZ / p
    p.parent.mkdir(parents=True, exist_ok=True)
    if not ignorado(p):
        raise ErrorGguf(f"{p.name} no está ignorado por git: no se escribe.")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(datos)
    tmp.replace(p)
    return p


def a_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(regs: Sequence[Mapping]) -> bytes:
    return "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


# --- Tokens ---


def comprobar_tokens(cliente, fichas_hf: Callable[[str], list[int]], id_hf: Callable[[str], int],
                     filas: Sequence[Mapping], referencia: Mapping[str, Mapping]) -> dict:
    """Conteos: ids del servidor contra Hugging Face en todas las filas, tokens especiales y largos contra Kaggle."""
    especiales = {}
    for t in TOKENS_ESPECIALES:
        srv = cliente.tokenizar(t, add_special=False)
        especiales[t] = {"un_solo_id": len(srv) == 1, "igual_a_hf": srv == [id_hf(t)]}
    prueba = "x"
    bos = cliente.tokenizar(prueba, add_special=True) != cliente.tokenizar(prueba, add_special=False)
    iguales = largo_kaggle = 0
    distintas = []
    for i, f in enumerate(filas, start=1):
        p = prompt_generacion(f["nom_estab"])
        a, b = cliente.tokenizar(p, add_special=True), fichas_hf(p)
        if a == b:
            iguales += 1
        else:
            distintas.append(i)
        largo_kaggle += len(b) == referencia[f["id"]]["fichas_prompt"]
    n = len(filas)
    ok = (iguales == n and largo_kaggle == n and not bos
          and all(e["un_solo_id"] and e["igual_a_hf"] for e in especiales.values()))
    return {
        "filas": n,
        "ids_iguales_a_hf": iguales,
        "posiciones_distintas": distintas[:20],
        "largo_hf_igual_a_fichas_prompt_de_kaggle": largo_kaggle,
        "agrega_bos": bos,
        "especiales": especiales,
        "ok": ok,
    }


# --- Generación ---


def generar(cliente, filas: Sequence[Mapping], n_predict: int, fin: Sequence[int],
            decodificar: Callable[[list[int]], str] | None = None, fichas_hf: Callable[[str], list[int]] | None = None,
            avance: Callable[[str], None] | None = None) -> tuple[list[dict], dict]:
    """Respuestas por fila (con ids) y conteos de las comprobaciones (sin ids).

    `decodificar`: la decodificación de Kaggle (Hugging Face, skip_special_tokens=False); sin ella se usa /detokenize.
    """
    regs = []
    detok_igual = fichas_prompt_igual = fin_visible = eos_sin_fin = 0
    t0 = time.perf_counter()
    for i, f in enumerate(filas, start=1):
        prompt = prompt_generacion(f["nom_estab"])
        c = cliente.completar(prompt, n_predict)
        r = ls.respuesta_de(c, fin)
        det = cliente.detokenizar(r["fichas"])
        texto = decodificar(r["fichas"]) if decodificar else det
        detok_igual += det == texto
        if fichas_hf is not None:
            fichas_prompt_igual += c.get("tokens_evaluated") == len(fichas_hf(prompt))
        fin_visible += r["fin_visible"]
        eos_sin_fin += r["stop_type"] == "eos" and not r["fin_visible"]
        regs.append({
            "id": f["id"],
            "respuesta": texto,
            "fichas_prompt": c.get("tokens_evaluated"),
            "fichas_generadas": len(r["fichas"]) + (0 if r["al_tope"] else 1),  # como Kaggle: incluye el token de fin
            "al_tope": r["al_tope"],
            "parada": r["stop_type"],
            "fin_visible": r["fin_visible"],
            "ms": round(c["ms"], 3),
        })
        if avance and i % 100 == 0:
            avance(f"{i:,} de {len(filas):,} filas en {time.perf_counter() - t0:.0f} s")
    n = len(filas)
    return regs, {
        "filas": n,
        "segundos": round(time.perf_counter() - t0, 1),
        "detokenize_igual_a_la_decodificacion": detok_igual,
        "tokens_evaluated_igual_a_hf": fichas_prompt_igual if fichas_hf is not None else None,
        "parada_con_token_de_fin_visible": fin_visible,
        "parada_eos_sin_token_de_fin_de_kaggle_visible": eos_sin_fin,
        "al_tope": sum(r["al_tope"] for r in regs),
    }


# --- Concordancia y regla de la demo ---


def comparar_con_kaggle(preds: Mapping[str, Mapping], referencia: Mapping[str, Mapping]) -> dict:
    """Conteos de respuestas idénticas (carácter por carácter, por id) y de las comprobaciones contra Kaggle."""
    if set(preds) != set(referencia):
        raise ErrorGguf(f"Los ids no son los de la referencia: faltan {len(set(referencia) - set(preds))} y "
                        f"sobran {len(set(preds) - set(referencia))}.")
    n = len(referencia)
    ident = sum(preds[i]["respuesta"] == referencia[i]["respuesta"] for i in referencia)
    return {
        "filas": n,
        "identicas": ident,
        "distintas": n - ident,
        "pct_identicas": round(100 * ident / n, 2),
        "al_tope_igual": sum(preds[i]["al_tope"] == referencia[i]["al_tope"] for i in referencia),
        "fichas_prompt_igual": sum(preds[i]["fichas_prompt"] == referencia[i]["fichas_prompt"] for i in referencia),
        "fichas_generadas_igual": sum(preds[i]["fichas_generadas"] == referencia[i]["fichas_generadas"]
                                      for i in referencia),
    }


def elegir_cuantizacion(identicas: Mapping[str, int], tamanos: Mapping[str, int], total: int) -> dict:
    """La regla de la demo (REGLA_DEMO), con conteos enteros."""
    if set(identicas) != set(CUANTIZACIONES) or set(tamanos) != set(CUANTIZACIONES):
        raise ErrorGguf("La regla necesita las tres cuantizaciones.")
    if total <= 0 or any(not 0 <= v <= total for v in identicas.values()):
        raise ErrorGguf("Conteos de concordancia fuera de rango.")
    mejor = min(CUANTIZACIONES, key=lambda q: (-identicas[q], tamanos[q]))
    ventaja = identicas[mejor] - identicas[CUANTIZACION_DE_RESPALDO]
    elegida = CUANTIZACION_DE_RESPALDO if 100 * ventaja < total else mejor
    return {"mejor_concordancia": mejor, "elegida": elegida,
            "ventaja_sobre_Q4_K_M_filas": ventaja, "ventaja_sobre_Q4_K_M_puntos": round(100 * ventaja / total, 2)}


def tabla_aciertos(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """2×2 de aciertos de clase: Kaggle (a) contra GGUF (b), sobre las mismas filas."""
    if len(a) != len(b):
        raise ErrorGguf("Los aciertos pareados deben tener el mismo largo.")
    return {
        "los_dos_aciertan": sum(x and y for x, y in zip(a, b)),
        "solo_kaggle_acierta": sum(x and not y for x, y in zip(a, b)),
        "solo_gguf_acierta": sum(y and not x for x, y in zip(a, b)),
        "ninguno_acierta": sum(not x and not y for x, y in zip(a, b)),
    }


# --- Latencia ---


def percentil(valores: Sequence[float], q: float) -> float:
    """Percentil con interpolación lineal, igual a numpy.percentile por omisión."""
    v = sorted(valores)
    if not v:
        raise ErrorGguf("No hay valores para el percentil.")
    pos = (len(v) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def muestra_latencia(filas: Sequence[Mapping], n: int = N_LATENCIA, calentamiento: int = CALENTAMIENTO
                     ) -> tuple[list[Mapping], list[Mapping]]:
    """perm = default_rng(42).permutation(len(filas)) sobre las filas en orden de int(id); perm[:n]: las primeras
    `calentamiento` se descartan y el resto se mide."""
    import numpy as np

    ordenadas = sorted(filas, key=lambda f: int(f["id"]))
    perm = np.random.default_rng(SEMILLA).permutation(len(ordenadas))[:n]
    elegidas = [ordenadas[int(i)] for i in perm]
    return elegidas[:calentamiento], elegidas[calentamiento:]


def resumen_latencia(medidas: Sequence[Mapping]) -> dict:
    """p50 y p95 de punta a punta, fichas por segundo (mediana por fila y agregado) y fichas, desde las mediciones."""
    e2e = [m["ms_punta_a_punta"] for m in medidas]
    srv = [m["ms_completion"] for m in medidas]
    pn = [m["fichas_prompt"] for m in medidas]
    gn = [m["fichas_generadas"] for m in medidas]
    pms = [m["prompt_ms"] for m in medidas]
    gms = [m["generacion_ms"] for m in medidas]
    por_fila_p = [m["prompt_por_s"] for m in medidas]
    por_fila_g = [m["generacion_por_s"] for m in medidas if m["generacion_por_s"] is not None]
    r = lambda x, d=1: round(x, d)  # noqa: E731
    return {
        "n": len(medidas),
        "ms_punta_a_punta": {"p50": r(percentil(e2e, 50)), "p95": r(percentil(e2e, 95)),
                             "min": r(min(e2e)), "max": r(max(e2e))},
        "ms_completion_en_el_cliente": {"p50": r(percentil(srv, 50)), "p95": r(percentil(srv, 95))},
        "fichas_por_s": {
            "prompt_mediana_por_fila": r(percentil(por_fila_p, 50)),
            "prompt_agregado": r(1000 * sum(pn) / sum(pms)),
            "generacion_mediana_por_fila": r(percentil(por_fila_g, 50)) if por_fila_g else None,
            "generacion_agregado": r(1000 * sum(gn) / sum(gms)) if sum(gms) else None,
        },
        "fichas_prompt": {"p50": percentil(pn, 50), "max": max(pn)},
        "fichas_generadas": {"p50": percentil(gn, 50), "max": max(gn)},
        "al_tope": sum(m["al_tope"] for m in medidas),
    }


def programas_abiertos(minimo_mb: int = MINIMO_PROGRAMA_MB) -> list[dict]:
    """Nombres de proceso que usan al menos `minimo_mb` en total (nunca títulos de ventana)."""
    cmd = ("Get-Process | Group-Object ProcessName | ForEach-Object { '{0}|{1}' -f $_.Name, "
           "($_.Group | Measure-Object WorkingSet64 -Sum).Sum }")
    salida = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                            check=True).stdout
    res = []
    for linea in salida.splitlines():
        nombre, _, mb = linea.strip().rpartition("|")
        if nombre and mb.isdigit() and int(mb) >= minimo_mb * 1024 * 1024:
            res.append({"proceso": nombre, "mb": round(int(mb) / (1024 * 1024))})
    return sorted(res, key=lambda x: -x["mb"])


def nombre_cpu() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except OSError:
        return platform.processor()


# --- Registro de la fusión y de los GGUF (lo escribe src/fusion_gguf.py) ---


def leer_registro(raiz: Path = REPO_RAIZ) -> dict:
    ruta = Path(raiz) / REGISTRO
    if not ruta.is_file():
        raise ErrorGguf(f"Falta {REGISTRO}: corre antes src/fusion_gguf.py.")
    return json.loads(ruta.read_text(encoding="utf-8"))


def gguf_registrado(registro: Mapping, rol: str, cuant: str, verificar: bool = True) -> dict:
    """La entrada del GGUF en el registro; con `verificar`, exige que el archivo tenga esos bytes y ese SHA-256."""
    nombre = nombre_gguf(rol, cuant)
    e = registro.get("gguf", {}).get(nombre)
    if not e:
        raise ErrorGguf(f"{nombre} no está en el registro.")
    ruta = DIR_GGUF / nombre
    if verificar:
        if not ruta.is_file() or ruta.stat().st_size != e["bytes"] or sha256_archivo(ruta) != e["sha256"]:
            raise ErrorGguf(f"{nombre} no tiene los bytes o el SHA-256 del registro.")
    return {"archivo": nombre, "bytes": e["bytes"], "sha256": e["sha256"], "ruta": ruta}


# --- Modos del entorno del tokenizador ---


def tokenizador_hf():
    from transformers import AutoTokenizer

    if not SNAPSHOT_TOKENIZADOR.is_dir():
        raise ErrorGguf(f"Falta el tokenizador de la revisión {REVISION_OFICIAL[:8]} en venv-tokenizador.")
    return AutoTokenizer.from_pretrained(str(SNAPSHOT_TOKENIZADOR))


def _servidor(rol: str, cuant: str, registro: Mapping, sufijo: str) -> ls.Servidor:
    g = gguf_registrado(registro, rol, cuant)
    return ls.Servidor(LLAMA_SERVER, g["ruta"], PUERTOS[rol], REPO_RAIZ / CARPETA / "logs" / f"{etiqueta(rol, cuant)}_{sufijo}.log")


def correr_tokens(etiq: str) -> dict:
    rol, cuant = de_etiqueta(etiq)
    registro = leer_registro()
    filas = val_del_paquete()
    referencia = leer_referencia()
    if {f["id"] for f in filas} != set(referencia):
        raise ErrorGguf("Los ids de la validación del paquete no son los de la referencia de Kaggle.")
    tok = tokenizador_hf()
    fichas_hf = lambda t: list(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    with _servidor(rol, cuant, registro, "tokens") as s:
        res = comprobar_tokens(s.cliente, fichas_hf, tok.convert_tokens_to_ids, filas, referencia)
        res["servidor"] = {"arranque_s": round(s.arranque_s, 2), "memoria_libre_antes_mb": s.memoria_libre_antes_mb}
    res.update({"etiqueta": etiq, "gguf": nombre_gguf(rol, cuant), "git": estado_git_simple(),
                "fecha": datetime.date.today().isoformat()})
    escribir_en_data(a_bytes(res), f"{CARPETA}/{etiq}_tokens.json")
    return res


def correr_generar(etiq: str) -> dict:
    rol, cuant = de_etiqueta(etiq)
    if rol != "afinado":
        raise ErrorGguf("La concordancia se mide solo con el afinado: el base no tiene respuestas de validación en Kaggle.")
    git = estado_git_simple()
    if not git["arbol_limpio"]:
        raise ErrorGguf("El árbol de git no está limpio: el commit registrado no sería el código exacto.")
    registro = leer_registro()
    filas = val_del_paquete()
    referencia = leer_referencia()
    if {f["id"] for f in filas} != set(referencia):
        raise ErrorGguf("Los ids de la validación del paquete no son los de la referencia de Kaggle.")
    tok = tokenizador_hf()
    fichas_hf = lambda t: list(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    decodificar = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    with _servidor(rol, cuant, registro, "generar") as s:
        fin = ls.ids_fin(s.cliente)
        if fin != [tok.convert_tokens_to_ids(t) for t in ls.TOKENS_FIN]:
            raise ErrorGguf("Los ids de los tokens de fin del servidor no son los de Hugging Face.")
        regs, conteos = generar(s.cliente, filas, NUEVAS[rol], fin, decodificar, fichas_hf,
                                avance=lambda m: print(f"  {etiq}: {m}", flush=True))
        servidor = {"arranque_s": round(s.arranque_s, 2), "memoria_libre_antes_mb": s.memoria_libre_antes_mb,
                    "memoria_proceso": ls.memoria_proceso(s.pid)}
    datos = jsonl_bytes(regs)
    ruta_pred = escribir_en_data(datos, f"{CARPETA}/{etiq}_validacion.jsonl")
    meta = {"etiqueta": etiq, "gguf": nombre_gguf(rol, cuant), "git": git, "fecha": datetime.date.today().isoformat(),
            "predicciones": {"ruta": f"{CARPETA}/{ruta_pred.name}", "bytes": len(datos), "sha256": sha256_bytes(datos)},
            "n_predict": NUEVAS[rol], "ids_fin": fin, "parametros": ls.PARAMETROS_GREEDY,
            "comando_servidor": [Path(x).name if i in (0, 2) else x for i, x in enumerate(s.comando)],
            "servidor": servidor, "comprobaciones": conteos,
            "transformers": __import__("transformers").__version__}
    escribir_en_data(a_bytes(meta), f"{CARPETA}/{etiq}_generacion.json")
    return meta


# --- Evaluación y resultado (entorno del proyecto) ---


def _compacto(m: Mapping) -> dict:
    return {
        "regla5": m["regla5"],
        "ic95_accuracy": m["ic95_accuracy"],
        "accuracy_por_nivel": {k: v["accuracy"] for k, v in m["por_nivel"].items()},
        "macro_f1_sector": m["macro_f1_sector"]["valor"],
    }


def evaluar(raiz: Path = REPO_RAIZ) -> dict:
    """Todo el cálculo de la concordancia, sin escribir (lo usan la terminal y --escribir)."""
    from src.evaluador_clasificador import (
        aciertos_por_nivel,
        cargar_codigos,
        cargar_referencias,
        diferencia_pareada_bootstrap,
        evaluar_validacion,
    )
    from src.scian import leer_manifest as leer_manifest_scian

    manifest = leer_manifest_scian()
    filas = sorted(cargar_referencias(manifest), key=lambda f: int(f["id"]))
    codigos = cargar_codigos(manifest)
    referencia = leer_referencia(raiz)
    fase2 = leer_fase2(raiz)
    ids = [f["id"] for f in filas]
    if set(ids) != set(referencia) or set(fase2) != set(referencia):
        raise ErrorGguf("Los ids de la referencia, de la Fase 2 y de data/splits/val.jsonl no coinciden.")
    iguales_fase2 = sum(fase2[i] == referencia[i]["respuesta"] for i in ids)
    if iguales_fase2 != len(ids):
        raise ErrorGguf(f"La referencia de la Fase 4 no es idéntica a la de la Fase 2 ({iguales_fase2} de {len(ids)}).")
    pred_ref = {i: referencia[i]["respuesta"] for i in ids}
    met_ref = evaluar_validacion(pred_ref, filas, codigos)
    ac_ref = aciertos_por_nivel(pred_ref, filas, codigos)["clase"]

    registro = leer_registro(raiz)
    cuants, tokens = {}, {}
    for q in CUANTIZACIONES:
        et = etiqueta("afinado", q)
        g = gguf_registrado(registro, "afinado", q)
        meta = json.loads((Path(raiz) / CARPETA / f"{et}_generacion.json").read_text(encoding="utf-8"))
        tk = json.loads((Path(raiz) / CARPETA / f"{et}_tokens.json").read_text(encoding="utf-8"))
        if meta["gguf"] != g["archivo"] or tk["gguf"] != g["archivo"]:
            raise ErrorGguf(f"Las salidas de {et} no son de {g['archivo']}.")
        if not tk["ok"]:
            raise ErrorGguf(f"La comprobación de tokens de {et} no pasó.")
        datos = leer_verificado(Path(raiz) / meta["predicciones"]["ruta"], meta["predicciones"]["sha256"])
        preds = {r["id"]: r for r in leer_jsonl(datos, CAMPOS_PREDICCION)}
        comp = comparar_con_kaggle(preds, referencia)
        p = {i: preds[i]["respuesta"] for i in ids}
        met = evaluar_validacion(p, filas, codigos)
        ac = aciertos_por_nivel(p, filas, codigos)["clase"]
        cuants[q] = {
            "gguf": {"archivo": g["archivo"], "bytes": g["bytes"], "sha256": g["sha256"]},
            "concordancia": comp,
            "metricas": _compacto(met),
            "diferencia_contra_kaggle": diferencia_pareada_bootstrap(ac_ref, ac),
            "aciertos_kaggle_contra_gguf": tabla_aciertos(ac_ref, ac),
            "generacion": {"commit": meta["git"]["commit"], "arbol_limpio": meta["git"]["arbol_limpio"],
                           "segundos": meta["comprobaciones"]["segundos"], "comprobaciones": meta["comprobaciones"],
                           "predicciones_sha256": meta["predicciones"]["sha256"],
                           "memoria_proceso": meta["servidor"]["memoria_proceso"],
                           "transformers_decodificacion": meta["transformers"]},
        }
        tokens[et] = {k: v for k, v in tk.items() if k not in ("git", "fecha", "posiciones_distintas")}
    for rol_base in ("base",):
        for q in CUANTIZACIONES:
            ruta = Path(raiz) / CARPETA / f"{etiqueta(rol_base, q)}_tokens.json"
            if ruta.is_file():
                tk = json.loads(ruta.read_text(encoding="utf-8"))
                tokens[etiqueta(rol_base, q)] = {k: v for k, v in tk.items() if k not in ("git", "fecha", "posiciones_distintas")}
    regla = elegir_cuantizacion({q: cuants[q]["concordancia"]["identicas"] for q in CUANTIZACIONES},
                                {q: cuants[q]["gguf"]["bytes"] for q in CUANTIZACIONES}, len(ids))
    return {"filas": filas, "manifest": manifest, "referencia_metricas": _compacto(met_ref),
            "iguales_fase2": iguales_fase2, "cuantizaciones": cuants, "tokens": tokens, "regla": regla,
            "registro": registro}


def _sin_rutas_locales(obj):
    """El registro sin rutas absolutas de la laptop (solo nombres de archivo)."""
    if isinstance(obj, Mapping):
        return {k: _sin_rutas_locales(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sin_rutas_locales(v) for v in obj]
    if isinstance(obj, str) and (":\\" in obj or obj.startswith("C:/") or obj.startswith("/c/")):
        return Path(obj.replace("\\", "/")).name
    return obj


def _preregistro(raiz: Path) -> dict:
    datos = (Path(raiz) / RUTA_CONFIG).read_bytes()
    return {"ruta": RUTA_CONFIG, "sha256": sha256_bytes(datos)}


def armar_concordancia(ev: Mapping, *, fecha: str, git: Mapping, raiz: Path = REPO_RAIZ) -> dict:
    val = ev["manifest"]["splits"][ENTRADA_VAL]
    return {
        "fecha": fecha,
        "experimento": "fase4_paso19_gguf_concordancia",
        "commit": git["commit"],
        "arbol_limpio": git["arbol_limpio"],
        "modelo": MODELO,
        "configuracion": {
            "script": SCRIPT,
            "python": platform.python_version(),
            "corre_en": "laptop (CPU, sin GPU)",
            "llama_cpp": TAG_LLAMA,
            "banderas_servidor": list(ls.BANDERAS_SERVIDOR),
            "prompt": "prompt_generacion de src/prompt_clasificador.py, como texto crudo a /completion, sin la "
                      "plantilla de chat de llama.cpp.",
            "decodificacion": {"parametros": ls.PARAMETROS_GREEDY, "n_predict": ls.NUEVAS_AFINADO,
                               "tokens_de_fin": list(ls.TOKENS_FIN),
                               "corte": "cortar_respuesta de src/humo_kaggle.py sobre las fichas devueltas "
                                        "(return_tokens), antes del primer token de fin de Kaggle.",
                               "texto": "Las fichas cortadas, decodificadas con el tokenizador de Hugging Face de la "
                                        "revisión oficial (skip_special_tokens=False), como en Kaggle."},
            "concordancia": "Respuesta idéntica carácter por carácter a la de Kaggle, emparejada por id.",
        },
        "semilla": SEMILLA,
        "split": "validacion",
        "hash_split": {"validacion": val["sha256"]},
        "hash_split_origen": f"data/manifest.json (splits.{ENTRADA_VAL})",
        "tamano_muestra": len(ev["filas"]),
        "horas_gpu": 0.0,
        "preregistro": _preregistro(raiz),
        "referencia_kaggle": {
            "ruta": REFERENCIA["ruta"], "sha256": REFERENCIA["sha256"],
            "identica_a_la_fase_2": {"ruta": FASE2["ruta"], "sha256": FASE2["sha256"],
                                     "respuestas_identicas": ev["iguales_fase2"]},
            "carga_en_kaggle": "Base Qwen/Qwen3-0.6B en NF4 (bitsandbytes) con la LoRA sin fusionar.",
            "metricas": ev["referencia_metricas"],
        },
        "herramientas_y_modelos": _sin_rutas_locales(ev["registro"]),
        "tokens": ev["tokens"],
        "cuantizaciones": ev["cuantizaciones"],
        "regla_demo": {"texto": REGLA_DEMO, "implementacion": f"{SCRIPT} (elegir_cuantizacion)", **ev["regla"]},
        "autoria": "El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.",
        "advertencias": [
            "La referencia de Kaggle salió de la base en NF4 con la LoRA sin fusionar; los GGUF salen de la base en "
            "bf16 con la LoRA fusionada. Por eso no se espera 100 % de respuestas idénticas ni en BF16.",
            "Solo validación: nunca test ni hard (pre-registro, gguf.concordancia). La accuracy de validación del "
            "06b-lr2e-4-r32 es algo optimista, porque la configuración se eligió con esta misma validación.",
        ],
    }


def nombres_para_candados(raiz: Path = REPO_RAIZ) -> tuple[set[str], set[str], list[dict]]:
    """Llaves, ids y filas de la validación y el train final del paquete (lo que vio el kernel); nunca test ni hard."""
    from src.normalizacion import normalizar_nombre

    manifest = leer_manifest(raiz)
    filas = []
    for archivo in (ARCHIVO_VAL_PAQUETE, ARCHIVO_TRAIN_PAQUETE):
        e = _paquete(manifest, archivo)
        filas += leer_jsonl_verificado(Path(raiz) / e["ruta"], e["sha256"])
    llaves = {k for k in (normalizar_nombre(r["nom_estab"]) for r in filas) if k}
    return llaves, {r["id"] for r in filas}, filas


def escribir_resultado(construir: Callable[[], dict], patron: str, nombre: str, *, raiz: Path = REPO_RAIZ,
                       versionado: Callable[[str], bool] | None = None, nombres=nombres_para_candados,
                       git: Callable[[], dict] = estado_git_simple, avance: Callable[[str], None] = print) -> Path:
    """Una sola vez: árbol limpio, código versionado, doble cálculo con los mismos bytes y candados de privacidad."""
    from src.evaluacion_final_fase4 import candados
    from src.muestra_humo import codigo_versionado

    versionado = versionado or codigo_versionado
    hay = sorted(p.name for p in (Path(raiz) / "results").glob(patron))
    if hay:
        raise ErrorGguf(f"Ya existe {hay}: no se repite ni se sobrescribe.")
    inicial = git()
    if not (inicial["arbol_limpio"] and versionado(SCRIPT)):
        raise ErrorGguf(f"{SCRIPT} debe estar en git y el árbol limpio: el commit registrado no sería el código exacto.")
    a, b = a_bytes(construir()), a_bytes(construir())
    if a != b:
        raise ErrorGguf("El doble cálculo no da los mismos bytes; no se escribe nada.")
    llaves, ids, filas = nombres(raiz)
    try:
        rev = candados(json.loads(a), {"resultado": a.decode("utf-8")}, llaves, ids, filas)
    except Exception as e:  # los candados solo citan rutas de campos o nombres de archivo, nunca el texto
        raise ErrorGguf(f"Candado de privacidad ({type(e).__name__}): {e}") from e
    avance(f"Candados limpios: privacidad ({len(llaves):,} llaves), sin ids ({len(ids):,}) y regla acotada "
           f"({sum(c['llaves_una_palabra_con_letras'] for c in rev['archivos'].values())} coincidencias de una palabra, "
           "solo contadas).")
    if git() != inicial:
        raise ErrorGguf("El repo cambió durante el cálculo: no se escribe en results/.")
    ruta = Path(raiz) / "results" / nombre
    tmp = Path(raiz) / CARPETA / (nombre + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(a)
    tmp.replace(ruta)
    avance(f"Escrito results/{nombre}: {len(a):,} bytes, SHA-256 {sha256_bytes(a)}.")
    return ruta


def imprimir_evaluacion(ev: Mapping) -> None:
    m = ev["referencia_metricas"]
    print(f"Kaggle (NF4 + LoRA): accuracy {m['regla5']['accuracy']:.3f}; respuestas idénticas a la Fase 2: "
          f"{ev['iguales_fase2']:,}")
    for q in CUANTIZACIONES:
        c = ev["cuantizaciones"][q]
        r5 = c["metricas"]["regla5"]
        d = c["diferencia_contra_kaggle"]
        print(f"{q:>7}: idénticas {c['concordancia']['identicas']:,} ({c['concordancia']['pct_identicas']:.1f} %), "
              f"accuracy {r5['accuracy']:.3f}, formato inválido {r5['porcentajes']['formato_invalido']:.2f} %, "
              f"inexistentes {r5['pct_codigos_inexistentes']:.2f} %, GGUF − Kaggle {d['diferencia']:+.3f}, "
              f"{c['gguf']['bytes'] / 1e6:,.0f} MB")
    r = ev["regla"]
    print(f"Regla de la demo: mejor concordancia {r['mejor_concordancia']}, ventaja sobre Q4_K_M "
          f"{r['ventaja_sobre_Q4_K_M_puntos']} puntos → elegida {r['elegida']}")


# --- Latencia (entorno del proyecto) ---


def elegida_de_resultado(raiz: Path = REPO_RAIZ) -> str:
    hay = sorted((Path(raiz) / "results").glob(PATRON_CONCORDANCIA))
    if len(hay) != 1:
        raise ErrorGguf("Se necesita exactamente un resultado de concordancia antes de medir la latencia.")
    return json.loads(hay[0].read_text(encoding="utf-8"))["regla_demo"]["elegida"]


def medir_fila(cliente, prompt: str, n_predict: int, fin: Sequence[int]) -> dict:
    """Una pregunta como la de la demo: /completion + /detokenize, medida de punta a punta en el cliente."""
    t0 = time.perf_counter()
    c = cliente.completar(prompt, n_predict)
    r = ls.respuesta_de(c, fin)
    cliente.detokenizar(r["fichas"])
    ms = (time.perf_counter() - t0) * 1000
    t = c.get("timings") or {}
    return {
        "ms_punta_a_punta": round(ms, 3),
        "ms_completion": round(c["ms"], 3),
        "fichas_prompt": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_por_s": t.get("prompt_per_second"),
        "fichas_generadas": t.get("predicted_n"),
        "generacion_ms": t.get("predicted_ms"),
        "generacion_por_s": t.get("predicted_per_second"),
        "al_tope": r["al_tope"],
    }


def correr_latencia(raiz: Path = REPO_RAIZ) -> dict:
    cuant = elegida_de_resultado(raiz)
    registro = leer_registro(raiz)
    filas = val_del_paquete(raiz)
    calentar, medir = muestra_latencia(filas)
    memoria_inicio = ls.estado_memoria()
    programas_inicio = programas_abiertos()
    git = estado_git_simple()
    servidores = {rol: _servidor(rol, cuant, registro, "latencia") for rol in ROLES}
    medidas = {rol: [] for rol in ROLES}
    info = {}
    try:
        for s in servidores.values():
            s.__enter__()
        fin = {rol: ls.ids_fin(s.cliente) for rol, s in servidores.items()}
        t0 = time.perf_counter()
        for i, f in enumerate(calentar + medir):
            p = prompt_generacion(f["nom_estab"])
            for rol in ROLES:
                m = medir_fila(servidores[rol].cliente, p, NUEVAS[rol], fin[rol])
                if i >= len(calentar):
                    medidas[rol].append(m)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} de {len(calentar) + len(medir)} filas en {time.perf_counter() - t0:.0f} s", flush=True)
        for rol, s in servidores.items():
            g = gguf_registrado(registro, rol, cuant, verificar=False)
            info[rol] = {"gguf": {"archivo": g["archivo"], "bytes": g["bytes"], "sha256": g["sha256"]},
                         "arranque_s": round(s.arranque_s, 2), "memoria_proceso": ls.memoria_proceso(s.pid),
                         "hilos": ls.HILOS, "contexto": ls.CONTEXTO, "n_predict": NUEVAS[rol], "puerto": PUERTOS[rol]}
    finally:
        for s in servidores.values():
            s.cerrar()
    res = {
        "fecha": datetime.date.today().isoformat(),
        "git": git,
        "cuantizacion": cuant,
        "muestra": {"sorteadas": N_LATENCIA, "calentamiento": len(calentar), "medidas": len(medir),
                    "regla": "perm = numpy.random.default_rng(42).permutation(1000) sobre la validación en orden de "
                             "int(id); filas perm[:210]: las 10 primeras de calentamiento (descartadas) y 200 medidas. "
                             "En cada fila se pregunta primero al base y después al afinado, con los dos servidores arriba."},
        "sistema": {"cpu": nombre_cpu(), "nucleos_logicos": __import__("os").cpu_count(),
                    "ram_total_mb": memoria_inicio["total_mb"], "ram_libre_inicio_mb": memoria_inicio["libre_mb"],
                    "ram_libre_fin_mb": ls.estado_memoria()["libre_mb"], "programas_abiertos_inicio": programas_inicio,
                    "windows": platform.platform()},
        "modelos": {rol: {**info[rol], "mediciones": medidas[rol]} for rol in ROLES},
    }
    escribir_en_data(a_bytes(res), MEDICIONES_LATENCIA)
    return res


def armar_latencia(med: Mapping, *, fecha: str, git: Mapping, raiz: Path = REPO_RAIZ) -> dict:
    val = leer_manifest(raiz)["splits"][ENTRADA_VAL]
    modelos = {}
    for rol in ROLES:
        m = med["modelos"][rol]
        modelos[f"{rol}-{med['cuantizacion']}"] = {k: v for k, v in m.items() if k != "mediciones"} | {
            "resumen": resumen_latencia(m["mediciones"])}
    return {
        "fecha": fecha,
        "experimento": "fase4_paso19_latencia",
        "commit": git["commit"],
        "arbol_limpio": git["arbol_limpio"],
        "medido_con_commit": med["git"]["commit"],
        "modelo": MODELO,
        "configuracion": {
            "script": SCRIPT,
            "python": platform.python_version(),
            "corre_en": "laptop (CPU, sin GPU)",
            "llama_cpp": TAG_LLAMA,
            "banderas_servidor": list(ls.BANDERAS_SERVIDOR),
            "parametros": ls.PARAMETROS_GREEDY,
            "punta_a_punta": "/completion + /detokenize, medido con time.perf_counter en el cliente (como la demo).",
            "percentiles": "Interpolación lineal (como numpy.percentile por omisión).",
            "fichas_por_s": "De los timings del servidor: mediana por fila y agregado (fichas totales / ms totales).",
        },
        "semilla": SEMILLA,
        "split": "validacion",
        "hash_split": {"validacion": val["sha256"]},
        "tamano_muestra": med["muestra"]["medidas"],
        "muestra": med["muestra"],
        "horas_gpu": 0.0,
        "preregistro": _preregistro(raiz),
        "sistema": med["sistema"],
        "modelos": modelos,
        "autoria": "El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.",
    }


def imprimir_latencia(med: Mapping) -> None:
    for rol in ROLES:
        m = med["modelos"][rol]
        r = resumen_latencia(m["mediciones"])
        print(f"{rol}-{med['cuantizacion']}: p50 {r['ms_punta_a_punta']['p50']} ms, p95 {r['ms_punta_a_punta']['p95']} ms, "
              f"prompt {r['fichas_por_s']['prompt_agregado']} fichas/s, generación {r['fichas_por_s']['generacion_agregado']} "
              f"fichas/s, RAM {m['memoria_proceso']['en_uso_mb']} MB (pico {m['memoria_proceso']['pico_mb']}), "
              f"arranque {m['arranque_s']} s")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tokens", metavar="ETIQUETA")
    g.add_argument("--generar", metavar="ETIQUETA")
    g.add_argument("--evaluar", action="store_true")
    g.add_argument("--latencia", action="store_true")
    g.add_argument("--escribir-latencia", action="store_true")
    p.add_argument("--escribir", action="store_true", help="con --evaluar: escribe el resultado en results/")
    args = p.parse_args(argv)
    hoy = datetime.date.today().isoformat()
    try:
        if args.tokens:
            r = correr_tokens(args.tokens)
            print(f"{args.tokens}: ids iguales a Hugging Face {r['ids_iguales_a_hf']:,} de {r['filas']:,}; largo igual a "
                  f"Kaggle {r['largo_hf_igual_a_fichas_prompt_de_kaggle']:,}; agrega BOS: {r['agrega_bos']}; especiales "
                  f"{r['especiales']}; posiciones distintas {r['posiciones_distintas']}; ok: {r['ok']}")
        elif args.generar:
            m = correr_generar(args.generar)
            print(f"{args.generar}: {json.dumps(m['comprobaciones'], ensure_ascii=False)}; servidor {m['servidor']}")
        elif args.evaluar:
            ev = evaluar()
            imprimir_evaluacion(ev)
            if args.escribir:
                escribir_resultado(lambda: armar_concordancia(ev, fecha=hoy, git=estado_git_simple()),
                                   PATRON_CONCORDANCIA, f"{hoy}_fase4_gguf-concordancia.json")
        elif args.latencia:
            imprimir_latencia(correr_latencia())
        else:
            med = json.loads((REPO_RAIZ / MEDICIONES_LATENCIA).read_text(encoding="utf-8"))
            imprimir_latencia(med)
            escribir_resultado(lambda: armar_latencia(med, fecha=hoy, git=estado_git_simple()),
                               PATRON_LATENCIA, f"{hoy}_fase4_latencia.json")
    except (ErrorGguf, ls.ErrorServidor, AbortoHumo) as e:
        sys.exit(f"ME DETENGO: {e}")


if __name__ == "__main__":
    main()
