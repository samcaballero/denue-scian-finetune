"""Procesos del kernel de la Fase 4 (paso 15): las comprobaciones iniciales y una etapa de inferencia.

El orquestador (`src/inferencia_fase4_kaggle.py`) lanza cada uno en su propio proceso de Python:

  python -m src.etapa_fase4_kaggle --comprobaciones <entrada.json>
  python -m src.etapa_fase4_kaggle --etapa <entrada.json>

Comprobaciones (antes de cargar ningún modelo; si algo falla, el kernel se detiene):
- las copias de test y hard solo con id y nom_estab, y train final y validación, con su SHA-256;
- la selección few-shot rehecha sobre la copia de train final (hash de los ids, códigos y sectores),
  sin ids ni llaves de los ejemplos en validación, test ni hard, y sin nombres con "<|";
- la descarga verificada de las dos bases en su revisión fijada;
- con los tokenizadores reales: plantilla 0-shot y few-shot y largos de TODAS las filas de cada
  modelo (`src/plantilla_fase4.py`).

Etapa (un modelo en un conjunto):
1. Datos con su SHA-256 y, en el few-shot, la selección rehecha y comparada con el config.
2. Imports, una sola T4 y semillas (`hk.importar`); la base en NF4 desde su revisión fijada con el
   max_seq_length del config (`kc.cargar`).
3. Afinados: el adaptador. Camino principal: la LoRA de su adapter_config con la misma llamada de
   Unsloth del entrenamiento (`kc.aplicar_lora`) y los pesos del safetensors con
   `set_peft_model_state_dict`. Respaldo, si el principal falla por un error o por la comparación de
   tensores: `PeftModel.from_pretrained` y `patch_peft_model`, como el cargador de Unsloth. Cada
   camino compara cada tensor del archivo, convertido al dtype del parámetro, con el del modelo
   (igualdad exacta). Si los dos fallan, la etapa falla sin generar. Test y hard usan el método con
   que pasó validación.
4. Antes de la primera respuesta, con el tokenizador del modelo: tokens especiales, plantilla y
   fichas de todas las filas que responde ese modelo; si una no cabe, la etapa falla sin generar.
5. Generación greedy de una fila a la vez (`inferir_filas`, la lógica de `hk.inferir` con mediciones
   por fila y un límite de tiempo). Al terminar, el JSONL se escribe en un temporal y se renombra; si
   la etapa se corta, no deja predicciones.

En el log y en los JSON solo van estados, conteos, tiempos y hashes; los JSON tapan cualquier texto
que, normalizado, sea la llave de una fila leída.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import sys
import time
import traceback
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import src.clasificador_kaggle as kc
import src.humo_kaggle as hk
import src.inferencia_fase4_kaggle as fo
from src.normalizacion import normalizar_nombre
from src.plantilla_fase4 import (
    CAMPOS_CON_CODIGO,
    CAMPOS_RESERVADOS,
    MODOS,
    comprobar_tokenizadores,
    prompt_de,
    revisar_modo,
    seleccion_fewshot,
    tokens_especiales,
)
from src.preregistro_fase4 import parsear_jsonl
from src.prompt_clasificador import tiene_marca_especial

CAMPOS_DE_ARCHIVO = {
    "test.jsonl": CAMPOS_RESERVADOS,
    "hard.jsonl": CAMPOS_RESERVADOS,
    "val.jsonl": CAMPOS_CON_CODIGO,
    "train_final.jsonl": CAMPOS_CON_CODIGO,
}
METODOS_ADAPTADOR = fo.METODOS_ADAPTADOR
ARCHIVO_PESOS = "adapter_model.safetensors"
ARCHIVO_CONFIG_ADAPTADOR = "adapter_config.json"
RE_LORA = re.compile(r"\.(lora_[AB])\.default\.")
AVANCE_CADA = 250


class ErrorEtapa(RuntimeError):
    """La etapa no puede seguir; el mensaje nunca incluye nombres, ids ni respuestas."""


def informar(mensaje: str) -> None:
    print(f"[fase4-etapa {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


# --- Datos ---


def leer_conjuntos(rutas: Mapping[str, str], esperados: Mapping[str, str], config: Mapping,
                   conjuntos: Sequence[str]) -> dict[str, list[dict]]:
    """Cada conjunto con su SHA-256 verificado antes de parsear y sus filas del config.

    Test y hard se parsean solo con id y nom_estab: cualquier otra clave detiene la etapa.
    """
    res = {}
    for conj in conjuntos:
        archivo = fo.ARCHIVO_DE_CONJUNTO[conj]
        datos = Path(rutas[archivo]).read_bytes()
        if hk.sha256_bytes(datos) != esperados[archivo]:
            raise ErrorEtapa(f"El SHA-256 de {archivo} no es el del manifest.")
        filas = parsear_jsonl(datos, CAMPOS_DE_ARCHIVO[archivo])
        if len(filas) != config["conjuntos"][conj]["filas"]:
            raise ErrorEtapa(f"{archivo} tiene {len(filas)} filas; el config dice {config['conjuntos'][conj]['filas']}.")
        res[conj] = filas
    return res


# --- Adaptador ---


def lora_de_config(cfg: Mapping, repo: str) -> dict:
    """La LoRA del entrenamiento a partir de su adapter_config.json (ya verificado por SHA-256)."""
    problemas = []
    if cfg.get("peft_type") != "LORA" or cfg.get("use_dora") or cfg.get("use_rslora") or cfg.get("modules_to_save"):
        problemas.append("debe ser LoRA sin DoRA, sin rsLoRA y sin modules_to_save")
    if cfg.get("bias") != "none" or cfg.get("base_model_name_or_path") != repo:
        problemas.append("bias 'none' y la base del modelo")
    if set(cfg.get("target_modules") or []) != set(hk.LORA["target_modules"]):
        problemas.append("los 7 módulos del entrenamiento")
    if not all(isinstance(cfg.get(k), int) and cfg[k] > 0 for k in ("r", "lora_alpha")) or cfg.get("lora_dropout") != 0.0:
        problemas.append("r y alpha enteros positivos y dropout 0")
    if problemas:
        raise ErrorEtapa("adapter_config inesperado: " + "; ".join(problemas) + ".")
    return {
        "r": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "bias": "none",
        "target_modules": list(hk.LORA["target_modules"]),
        "use_gradient_checkpointing": hk.LORA["use_gradient_checkpointing"],
        "random_state": kc.SEMILLA,
    }


def nombre_en_archivo(nombre_parametro: str) -> str | None:
    """Nombre del tensor en adapter_model.safetensors de un parámetro LoRA del modelo (sin el adaptador "default")."""
    return RE_LORA.sub(r".\1.", nombre_parametro) if RE_LORA.search(nombre_parametro) else None


def comparar_lora(parametros: Mapping[str, object], archivo: Mapping[str, object],
                  iguales: Callable[[object, object], bool], dtype: Callable[[object], str]) -> dict:
    """¿Cada tensor del archivo, convertido al dtype de su parámetro, es igual al del modelo? Solo conteos."""
    mapa = {}
    for nombre in parametros:
        k = nombre_en_archivo(nombre)
        if k is not None:
            mapa[k] = nombre
    comunes = sorted(set(mapa) & set(archivo))
    distintos = 0
    dtypes: dict[str, int] = {}
    for k in comunes:
        p, t = parametros[mapa[k]], archivo[k]
        clave = f"{dtype(t)}->{dtype(p)}"
        dtypes[clave] = dtypes.get(clave, 0) + 1
        distintos += not iguales(t, p)
    res = {
        "tensores_archivo": len(archivo),
        "parametros_lora_modelo": len(mapa),
        "comparados": len(comunes),
        "distintos": distintos,
        "faltan_en_el_modelo": len(set(archivo) - set(mapa)),
        "sobran_en_el_modelo": len(set(mapa) - set(archivo)),
        "dtypes_archivo_a_modelo": dict(sorted(dtypes.items())),
    }
    res["ok"] = bool(archivo) and distintos == 0 and res["faltan_en_el_modelo"] == 0 and res["sobran_en_el_modelo"] == 0
    return res


def cargar_con_adaptador(motor, mods, etapa: Mapping, carpeta: Path, metodos: Sequence[str]) -> tuple[object, object, dict, dict]:
    """La base y el adaptador por el primer camino que pase la comparación de tensores; si ninguno, ErrorEtapa."""
    cfg = json.loads((carpeta / ARCHIVO_CONFIG_ADAPTADOR).read_text(encoding="utf-8"))
    intentos = []
    for metodo in metodos:
        if metodo not in METODOS_ADAPTADOR:
            raise ErrorEtapa(f"Método de adaptador desconocido: {metodo}.")
        model = None
        try:
            model, tok, info = motor.cargar(mods, etapa["base"], etapa["max_seq_length"])
            model, extra = motor.poner_adaptador(mods, model, metodo, carpeta, cfg, etapa["base"]["repo"])
            comp = comparar_lora(motor.parametros(model), motor.tensores_archivo(carpeta / ARCHIVO_PESOS),
                                 motor.igual_tras_convertir, motor.dtype)
            intentos.append({"metodo": metodo, "comparacion": comp, **extra})
            if comp["ok"]:
                informar(f"adaptador cargado por {metodo}: {comp['comparados']} tensores idénticos")
                return model, tok, info, {"metodo": metodo, "metodo_indice": METODOS_ADAPTADOR.index(metodo), "intentos": intentos}
            informar(f"adaptador por {metodo}: {comp['distintos']} tensores distintos; se prueba el siguiente camino")
        except Exception as e:
            intentos.append({"metodo": metodo, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            informar(f"adaptador por {metodo}: falló ({type(e).__name__}); se prueba el siguiente camino")
        model = None
        motor.liberar(mods)
    raise ErrorEtapa(f"Ningún camino cargó el adaptador con sus pesos exactos: {intentos}")


# --- Inferencia ---


def inferir_filas(mods, model, tok, prompts: Sequence[tuple[str, str]], ids_fin: Sequence[int], max_new_tokens: int, *,
                  limite: float | None = None, reloj: Callable[[], float] = time.perf_counter,
                  avance: Callable[[int, int, float], None] | None = None) -> tuple[list[dict], dict]:
    """Greedy, lote de 1: la generación de `hk.inferir` con el prompt y el tope de cada modo.

    Igual que `hk.inferir`: `for_inference`, el mismo generation_config (sin muestreo, parámetros de
    muestreo en None, tokens de fin y relleno), la misma tokenización, `generate`, el corte con
    `hk.cortar_respuesta` y la decodificación con skip_special_tokens=False, y el mismo respaldo si
    `generate` no acepta None en el muestreo. Además mide cada fila y, si el reloj pasa de `limite`
    antes de una fila, para y devuelve la etapa como incompleta.
    """
    torch = mods.torch
    mods.FastLanguageModel.for_inference(model)
    gen = copy.deepcopy(model.generation_config)
    gen.do_sample = False
    gen.num_beams = 1
    for atributo in ("temperature", "top_p", "top_k", "min_p"):
        if hasattr(gen, atributo):
            setattr(gen, atributo, None)
    gen.max_new_tokens = max_new_tokens
    gen.eos_token_id = list(ids_fin)
    gen.pad_token_id = tok.pad_token_id
    torch.cuda.reset_peak_memory_stats()
    filas: list[dict] = []
    motivos: dict[str, int] = {}
    respaldo = False
    completa = True
    t = reloj()
    for i, (id_, prompt) in enumerate(prompts, start=1):
        if limite is not None and reloj() > limite:
            completa = False
            break
        t_fila = reloj()
        enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            try:
                salida = model.generate(**enc, generation_config=gen)
            except TypeError:
                if respaldo or i != 1:
                    raise
                respaldo = True
                gen.temperature, gen.top_p, gen.top_k = 1.0, 1.0, 50
                salida = model.generate(**enc, generation_config=gen)
        n_prompt = enc["input_ids"].shape[1]
        nuevos = salida[0, n_prompt:].tolist()
        corte, motivo = hk.cortar_respuesta(nuevos, ids_fin)
        filas.append({
            "id": id_,
            "respuesta": tok.decode(corte, skip_special_tokens=False),
            "fichas_prompt": int(n_prompt),
            "fichas_generadas": len(nuevos),
            "al_tope": motivo == "limite_de_tokens",
            "segundos": round(reloj() - t_fila, 4),
        })
        motivos[motivo] = motivos.get(motivo, 0) + 1
        if avance is not None and i % AVANCE_CADA == 0:
            avance(i, len(prompts), reloj() - t)
    torch.cuda.synchronize()
    segundos = reloj() - t
    seg = [f["segundos"] for f in filas]
    n = len(filas)
    resumen = {
        "completa": completa and n == len(prompts),
        "filas": n,
        "filas_esperadas": len(prompts),
        "lote": 1,
        "segundos": round(segundos, 2),
        "s_por_fila": round(segundos / max(n, 1), 4),
        "s_por_fila_mediana": round(hk.percentil(seg, 50), 4) if seg else None,
        "s_por_fila_p90": round(hk.percentil(seg, 90), 4) if seg else None,
        "terminacion": motivos,
        "al_tope": sum(f["al_tope"] for f in filas),
        "pct_al_tope": round(100 * sum(f["al_tope"] for f in filas) / n, 2) if n else None,
        "fichas_prompt_max": max((f["fichas_prompt"] for f in filas), default=0),
        "fichas_prompt_media": round(sum(f["fichas_prompt"] for f in filas) / n, 2) if n else None,
        "fichas_generadas_max": max((f["fichas_generadas"] for f in filas), default=0),
        "fichas_generadas_media": round(sum(f["fichas_generadas"] for f in filas) / n, 2) if n else None,
        "respaldo_parametros_de_muestreo": respaldo,
        "generation_config": {
            k: getattr(gen, k, None)
            for k in ("do_sample", "num_beams", "temperature", "top_p", "top_k", "max_new_tokens", "eos_token_id", "pad_token_id")
        },
        "vram_pico_mib": {
            "asignada": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "reservada": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        },
    }
    return filas, resumen


def serializar_filas(filas: Sequence[Mapping]) -> bytes:
    """El JSONL de predicciones: las claves de CAMPOS_PREDICCION en orden; UTF-8 sin BOM, "\\n" al final de cada línea."""
    lineas = []
    for f in filas:
        if not isinstance(f["id"], str) or not fo.RE_ID.fullmatch(f["id"]) or not isinstance(f["respuesta"], str):
            raise ValueError("Cada predicción necesita un id de dígitos y una respuesta de texto.")
        lineas.append(json.dumps({k: f[k] for k in fo.CAMPOS_PREDICCION}, ensure_ascii=False) + "\n")
    return "".join(lineas).encode("utf-8")


def escribir_atomico(ruta: Path, datos: bytes) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_bytes(datos)
    temporal.replace(ruta)


# --- Motor (la parte de GPU; las pruebas lo reemplazan) ---


class MotorKaggle:
    """Imports, carga de modelos y adaptadores y tensores, con torch, Unsloth y peft (solo en Kaggle)."""

    def importar(self, reg: hk.Registro):
        reg.datos.setdefault("gpu", {})
        mods = hk.importar(reg)
        self.torch = mods.torch
        return mods

    def importar_ligero(self):
        import huggingface_hub
        from transformers import AutoTokenizer

        return types.SimpleNamespace(hf=huggingface_hub, AutoTokenizer=AutoTokenizer)

    def descargar(self, mods, repo: str, revision: str):
        return hk.descargar_verificado(mods, repo, revision)

    def tokenizador(self, mods, ruta: Path):
        return mods.AutoTokenizer.from_pretrained(str(ruta))

    def cargar(self, mods, base: Mapping[str, str], max_seq: int):
        return kc.cargar(mods, base, max_seq)

    def poner_adaptador(self, mods, model, metodo: str, carpeta: Path, cfg: Mapping, repo: str):
        if metodo == "estado_lora":
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            lora = lora_de_config(cfg, repo)
            model, info = kc.aplicar_lora(mods, model, lora)
            resultado = set_peft_model_state_dict(model, load_file(str(carpeta / ARCHIVO_PESOS)))
            inesperadas = list(getattr(resultado, "unexpected_keys", None) or [])
            if inesperadas:
                raise ErrorEtapa(f"{len(inesperadas)} tensores del archivo no entraron al modelo.")
            return model, {"lora": lora, **info}
        if metodo == "peft_from_pretrained":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(carpeta), is_trainable=True)
            model = mods.FastLanguageModel.patch_peft_model(model, hk.LORA["use_gradient_checkpointing"])
            return model, {}
        raise ErrorEtapa(f"Método de adaptador desconocido: {metodo}.")

    def parametros(self, model) -> dict:
        return dict(model.named_parameters())

    def tensores_archivo(self, ruta: Path) -> dict:
        from safetensors.torch import load_file

        return load_file(str(ruta))

    def igual_tras_convertir(self, tensor, parametro) -> bool:
        p = parametro.detach()
        return bool(self.torch.equal(tensor.to(device=p.device, dtype=p.dtype), p))

    def dtype(self, x) -> str:
        return str(x.dtype).replace("torch.", "")

    def liberar(self, mods) -> None:
        hk.liberar(mods)


# --- Comprobaciones iniciales ---


def _bases(plan: Mapping) -> dict[str, dict]:
    res: dict[str, dict] = {}
    for e in plan["etapas"]:
        res.setdefault(e["tamano"], dict(e["base"]))
    return res


def _comprobaciones(reg: hk.Registro, entrada: Mapping, motor) -> bool:
    config, huella = fo.leer_config(entrada["raiz_fuentes"], entrada["fuentes"].get(fo.RUTA_CONFIG))
    plan = fo.validar_config_kernel(config)
    reg.datos["config"] = huella
    conjuntos = leer_conjuntos(entrada["rutas"], entrada["esperados"], config, ("validacion", "test", "hard", "train_final"))
    reservados = ("validacion", "test", "hard")
    reg.nombres = tuple(r["nom_estab"] for c in reservados for r in conjuntos[c])
    reg.llaves = frozenset(k for k in map(normalizar_nombre, reg.nombres) if k)
    reg.datos["datos"] = {c: {"filas": len(rs), "sha256_verificado": True} for c, rs in conjuntos.items()}

    sel = seleccion_fewshot(conjuntos["train_final"], config["few_shot"])
    nombres_ej = [n for n, _ in sel["ejemplos"]]
    reg.nombres = reg.nombres + tuple(nombres_ej)
    reg.llaves = reg.llaves | {k for k in map(normalizar_nombre, nombres_ej) if k}
    ids = {c: {r["id"] for r in conjuntos[c]} for c in reservados}
    llaves = {c: {normalizar_nombre(r["nom_estab"]) for r in conjuntos[c]} for c in reservados}
    llaves_ej = [normalizar_nombre(n) for n in nombres_ej]
    fs = {
        **sel["publico"],
        "ids_ejemplos_en": {c: sum(i in ids[c] for i in sel["ids"]) for c in reservados},
        "llaves_ejemplos_en": {c: sum(k in llaves[c] for k in llaves_ej) for c in reservados},
        "ejemplos_con_marca_especial": sum(tiene_marca_especial(n) for n in nombres_ej),
    }
    fs["ok"] = (fs["coincide_con_el_config"] and not any(fs["ids_ejemplos_en"].values())
                and not any(fs["llaves_ejemplos_en"].values()) and fs["ejemplos_con_marca_especial"] == 0)
    reg.datos["few_shot"] = fs
    marcas = {c: sum(tiene_marca_especial(r["nom_estab"]) for r in conjuntos[c]) for c in reservados}
    reg.datos["nombres_con_marca_especial"] = marcas
    reg.escribir()
    informar(f"datos y few-shot: {'bien' if fs['ok'] else 'MAL'}; nombres con '<|': {marcas}")

    mods = motor.importar_ligero()
    toks, descargas = {}, {}
    for tam, base in _bases(plan).items():
        t = time.perf_counter()
        ruta, verif = motor.descargar(mods, base["repo"], base["revision"])
        descargas[tam] = {**base, "revision_descargada": Path(ruta).name, "archivos_lfs_verificados": len(verif),
                          "bytes_lfs": sum(v["bytes"] for v in verif.values()), "segundos": round(time.perf_counter() - t, 1)}
        toks[tam] = motor.tokenizador(mods, ruta)
        informar(f"{base['repo']}: revisión fijada descargada y verificada ({len(verif)} archivos LFS)")
    reg.datos["descargas"] = descargas
    nombres = {c: [r["nom_estab"] for r in conjuntos[c]] for c in reservados}
    tk = comprobar_tokenizadores(toks, nombres, sel["ejemplos"], config["largo_maximo"])
    reg.datos["tokenizadores"] = tk
    informar(f"tokenizadores: especiales {tk['especiales_ok']}, (b) 0-shot {tk['b_0shot_ok']}, (c) few-shot "
             f"{tk['c_fewshot_ok']}, (d) largos {tk['d_largos_ok']}; (a) idénticos "
             f"{tk.get('a_tokenizadores', {}).get('identicos')}")
    return fs["ok"] and not any(marcas.values()) and tk["ok"]


def correr_comprobaciones(entrada: Mapping, motor) -> int:
    salida = Path(entrada["salida"])
    salida.mkdir(parents=True, exist_ok=True)
    reg = hk.Registro(salida / fo.ARCHIVO_COMPROBACIONES, time.time())
    reg.datos["descripcion"] = "Comprobaciones iniciales del kernel de la Fase 4, antes de cargar ningún modelo."
    reg.escribir()
    try:
        ok = _comprobaciones(reg, entrada, motor)
    except Exception as e:
        ok = False
        reg.falla("comprobaciones", e)
        reg.datos["error"] = hk.tapar_nombres(traceback.format_exc(), reg.nombres)[-3000:]
        print(reg.datos["error"], flush=True)
    reg.datos["ok"] = ok
    reg.datos["estado"] = "completo" if ok else "fallido"
    reg.escribir()
    informar(f"comprobaciones: {'todo pasa' if ok else 'FALLAN'}")
    return 0 if ok else 1


# --- Etapa ---


def _etapa(reg: hk.Registro, entrada: Mapping, motor, reloj: Callable[[], float], t0: float) -> None:
    et = entrada["etapa"]
    config, huella = fo.leer_config(entrada["raiz_fuentes"], entrada["fuentes"].get(fo.RUTA_CONFIG))
    reg.datos["config"] = huella
    plan = {e["id"]: e for e in fo.validar_config_kernel(config)["etapas"]}
    if plan.get(et["id"]) != et:
        raise ErrorEtapa("La etapa recibida no es la del plan del config.")
    fewshot = MODOS[et["modo"]]["fewshot"]
    necesarios = list(et["conjuntos_del_modelo"]) + (["train_final"] if fewshot else [])
    conjuntos = leer_conjuntos(entrada["rutas"], entrada["esperados"], config, necesarios)
    reg.nombres = tuple(r["nom_estab"] for c in et["conjuntos_del_modelo"] for r in conjuntos[c])
    reg.llaves = frozenset(k for k in map(normalizar_nombre, reg.nombres) if k)
    marcas = sum(tiene_marca_especial(n) for n in reg.nombres)
    if marcas:
        raise ErrorEtapa(f"{marcas} nombres contienen '<|'.")
    ejemplos: list[tuple[str, str]] = []
    if fewshot:
        sel = seleccion_fewshot(conjuntos["train_final"], config["few_shot"])
        reg.datos["few_shot"] = sel["publico"]
        if not sel["publico"]["coincide_con_el_config"]:
            raise ErrorEtapa("La selección few-shot no es la del config.")
        ejemplos = sel["ejemplos"]
        reg.nombres = reg.nombres + tuple(n for n, _ in ejemplos)
        reg.llaves = reg.llaves | {k for k in (normalizar_nombre(n) for n, _ in ejemplos) if k}
    reg.datos["datos"] = {c: len(conjuntos[c]) for c in necesarios}
    reg.escribir()
    informar(f"{et['id']}: datos verificados ({', '.join(f'{c} {len(conjuntos[c])}' for c in necesarios)})")

    mods = motor.importar(reg)
    if et["tipo"] == "afinado":
        metodos = [entrada["metodo_adaptador"]] if entrada.get("metodo_adaptador") else list(METODOS_ADAPTADOR)
        carpeta = Path(entrada["afinados"][et["modelo"]]["adaptador"]).parent
        for clave in ("adaptador", "adaptador_config"):
            ruta = Path(entrada["afinados"][et["modelo"]][clave])
            if hk.sha256_archivo(ruta) != et["insumos_afinado"][clave]["sha256"] or ruta.parent != carpeta:
                raise ErrorEtapa(f"El {clave} no es el del config.")
        model, tok, info, info_ad = cargar_con_adaptador(motor, mods, et, carpeta, metodos)
        reg.datos["adaptador"] = info_ad
    else:
        model, tok, info = motor.cargar(mods, et["base"], et["max_seq_length"])
    reg.datos["modelo"] = info
    reg.escribir()

    especiales = tokens_especiales(tok)
    nombres = {c: [r["nom_estab"] for r in conjuntos[c]] for c in et["conjuntos_del_modelo"]}
    rev = revisar_modo(tok, et["modo"], nombres, ejemplos, config["largo_maximo"])
    reg.datos["antes_de_generar"] = {"tokens_especiales": especiales, "plantilla_y_largos": rev}
    if rev["limite"] != et["max_seq_length"] or rev["nuevas"] != et["max_new_tokens"]:
        raise ErrorEtapa("El largo máximo o las fichas nuevas no son los de la etapa.")
    if not especiales["ok"] or not rev["ok"]:
        reg.datos["motivo"] = "plantilla_o_largo"
        raise ErrorEtapa("La plantilla, los tokens especiales o los largos no pasan; no se genera ninguna respuesta.")
    informar(f"{et['id']}: plantilla y largos de {sum(len(v) for v in nombres.values())} filas del modelo bien; "
             f"fichas máx {max(c['fichas_max'] for c in rev['conjuntos'].values())} + {et['max_new_tokens']} <= {et['max_seq_length']}")

    prompts = [(r["id"], prompt_de(et["modo"], r["nom_estab"], ejemplos)) for r in conjuntos[et["conjunto"]]]
    limite = t0 + float(entrada["limite_s"])

    def avance(i, n, s):
        informar(f"{et['id']}: {i} de {n} filas; {s / i:.3f} s/fila; faltan ~{(n - i) * s / i / 60:.0f} min")

    filas, resumen = inferir_filas(mods, model, tok, prompts, [especiales["im_end"], especiales["endoftext"]],
                                   et["max_new_tokens"], limite=limite, reloj=reloj, avance=avance)
    reg.datos["inferencia"] = resumen
    if not resumen["completa"]:
        reg.datos["motivo"] = "tiempo"
        raise ErrorEtapa(f"Se llegó al límite de tiempo con {resumen['filas']} de {resumen['filas_esperadas']} filas.")
    datos = serializar_filas(filas)
    ruta = fo.ruta_prediccion(Path(entrada["salida"]), et)
    escribir_atomico(ruta, datos)
    reg.datos["archivo"] = {"ruta": ruta.relative_to(Path(entrada["salida"])).as_posix(), "bytes": len(datos),
                            "sha256": hk.sha256_bytes(datos), "filas": len(filas)}
    reg.datos["estado"] = "completa"
    reg.datos["completa"] = True  # el orquestador decide con este booleano (los textos del JSON se tapan)
    informar(f"{et['id']}: {len(filas)} filas en {resumen['segundos'] / 60:.1f} min ({resumen['s_por_fila']} s/fila); "
             f"{resumen['al_tope']} al tope de {et['max_new_tokens']} fichas")


def correr_etapa(entrada: Mapping, motor, reloj: Callable[[], float] = time.perf_counter) -> int:
    """Una etapa; 0 si dejó sus predicciones completas, 1 si no (sin predicciones)."""
    t0 = reloj()
    et = entrada["etapa"]
    salida = Path(entrada["salida"])
    (salida / fo.CARPETA_ETAPAS).mkdir(parents=True, exist_ok=True)
    reg = hk.Registro(salida / fo.CARPETA_ETAPAS / f"{et['id']}.json", time.time())
    reg.datos["descripcion"] = f"Etapa {et['id']} del kernel de la Fase 4: un modelo en un conjunto."
    reg.datos["etapa"] = {k: et[k] for k in ("id", "modelo", "conjunto", "tipo", "tamano", "modo", "max_seq_length",
                                             "max_new_tokens", "filas")}
    reg.datos["limite_s"] = entrada["limite_s"]
    reg.datos["completa"] = False
    reg.escribir()
    try:
        _etapa(reg, entrada, motor, reloj, t0)
    except Exception as e:
        reg.datos["estado"] = "incompleta"
        reg.datos["completa"] = False
        reg.datos.setdefault("motivo", "error")
        reg.falla("etapa", e)
        tb = hk.tapar_nombres(traceback.format_exc(), reg.nombres)
        reg.datos["error"] = tb[-3000:]
        print(tb, flush=True)
    reg.datos["segundos"] = round(reloj() - t0, 1)
    reg.escribir()
    informar(f"{et['id']}: estado {reg.datos['estado']}")
    return 0 if reg.datos["estado"] == "completa" else 1


def _cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Proceso del kernel de la Fase 4.")
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--comprobaciones", help="JSON de entrada de las comprobaciones iniciales")
    modo.add_argument("--etapa", help="JSON de entrada de una etapa")
    args = parser.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    hk.informar = informar
    ruta = args.comprobaciones or args.etapa
    entrada = json.loads(Path(ruta).read_text(encoding="utf-8"))
    motor = MotorKaggle()
    sys.exit(correr_comprobaciones(entrada, motor) if args.comprobaciones else correr_etapa(entrada, motor))


if __name__ == "__main__":
    _cli()
