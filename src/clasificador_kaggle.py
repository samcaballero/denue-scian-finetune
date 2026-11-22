"""Kernel de Kaggle de las corridas del clasificador (Fase 2, paso 9 en adelante).

Entrena con QLoRA sobre `train_final.jsonl` (train final con tope de 1,000 filas por clase),
mide la pérdida de validación durante el entrenamiento, guarda el adaptador LoRA y hace la
inferencia greedy en `val.jsonl`. Las métricas de la regla 5 y de los subgrupos de la regla 8 se
calculan en la laptop a partir de `predicciones.jsonl`. El kernel solo lee los dos archivos del
dataset del clasificador; los conjuntos reservados para la Fase 4 no suben a Kaggle.

La configuración de cada corrida (modelo, lote, tasa, LoRA, largo máximo...) llega como el
diccionario `CORRIDA` que `src/construir_kernel.py` incrusta en el kernel, y `validar_corrida()`
la revisa antes de empezar.

Reutiliza lo validado en la prueba de punta a punta (`src/humo_kaggle.py`): verificación de la
T4, lectura de los datos con su SHA-256, instalación de versiones fijadas, imports y semillas,
descarga verificada, colador, inferencia y registro de mediciones. Lo que necesita parámetros
(largo máximo, LoRA, validación durante el entrenamiento) se reescribe aquí partiendo de ese
código. `src/humo_kaggle.py` no cambia: va incrustado en el kernel de la prueba, que ya corrió.

Arriba solo importa la biblioteca estándar y los módulos incrustados; torch, Unsloth y
transformers se importan dentro de las funciones, después de instalar las versiones fijadas.

Orden de las etapas (lo que puede fallar barato va antes de instalar):
 1. nvidia-smi: toda GPU debe ser T4; se usa una sola (CUDA_VISIBLE_DEVICES=0).
 2. Datos: en /kaggle/input solo los dos archivos; SHA-256 de cada uno contra el manifest.
 3. Instalación con versiones fijadas (pip freeze antes y después).
 4. Imports, una sola GPU visible y semillas.
 5. Descarga y carga del modelo en la revisión fijada (NF4 de bitsandbytes).
 6. Plantilla y tokens: comprobaciones (a)-(d), etiquetas solo en la respuesta, largo máximo.
 7. LoRA.
 8. Entrenamiento: pérdida de validación en el paso 0, cada N pasos y al final; orden barajado.
 9. Adaptador LoRA.
10. Inferencia greedy en validación -> predicciones.jsonl (id y respuesta cruda).
11. Cierre: archivos de salida con su SHA-256.

Ningún nom_estab va al log ni a `mediciones.json`: el log solo lleva conteos, hashes y tiempos;
los errores se imprimen con los nombres tapados, y cada escritura del JSON tapa cualquier texto
que, normalizado, sea la llave de una fila de los datos.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import src.humo_kaggle as hk
from src.normalizacion import normalizar_nombre
from src.prompt_clasificador import (
    FIN_DE_TURNO,
    IGNORAR,
    INICIO_DE_TURNO,
    INSTRUCCION,
    ErrorPrompt,
    etiquetas,
    mensajes,
    prompt_generacion,
    texto_entrenamiento,
    tiene_marca_especial,
)

# --- Configuración fija ---

SEMILLA = 42  # regla 1
ARCHIVO_TRAIN = "train_final.jsonl"
ARCHIVO_VAL = "val.jsonl"
ARCHIVOS_ENTRADA = (ARCHIVO_TRAIN, ARCHIVO_VAL)
ARCHIVO_MEDICIONES = "mediciones.json"
CARPETA_ADAPTADOR = "adaptador"

LARGO_GUIA = 128  # largo máximo original de la guía: solo se cuenta cuántas filas lo pasan
AVANCE_CADA = 250  # una línea de avance en el log cada tantos pasos
PASOS_REVISION_ORDEN = 20  # al llegar a este paso se revisa que el train no vaya en orden de id

CLAVES_CORRIDA = (
    "nombre",
    "descripcion",
    "modelo",
    "max_seq_length",
    "lora",
    "lote",
    "gradient_accumulation_steps",
    "learning_rate",
    "epocas",
    "optim",
    "lr_scheduler_type",
    "fraccion_warmup",
    "eval_cada_pasos",
    "lote_eval",
    "semilla",
)
CLAVES_LORA = (
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "target_modules",
    "use_gradient_checkpointing",
    "random_state",
)
RE_REVISION = re.compile(r"[0-9a-f]{40}")


class AbortoCorrida(hk.AbortoHumo):
    """Condición que detiene el kernel; el mensaje nunca incluye un nom_estab."""


def informar(mensaje: str) -> None:
    print(f"[clasificador {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


# --- Funciones puras (se prueban en la laptop) ---


def _entero(valor, minimo: int) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool) and valor >= minimo


def _numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def validar_corrida(corrida: Mapping) -> dict:
    """La corrida incrustada por el generador, revisada clave por clave; falla con ValueError."""
    if not isinstance(corrida, Mapping):
        raise ValueError("La corrida debe ser un diccionario.")
    if set(corrida) != set(CLAVES_CORRIDA):
        faltan = sorted(set(CLAVES_CORRIDA) - set(corrida))
        sobran = sorted(set(corrida) - set(CLAVES_CORRIDA))
        raise ValueError(f"Claves de la corrida: faltan {faltan} y sobran {sobran}.")
    errores = []
    for clave in ("nombre", "descripcion", "optim", "lr_scheduler_type"):
        if not isinstance(corrida[clave], str) or not corrida[clave].strip():
            errores.append(f"{clave} debe ser un texto no vacío")
    modelo = corrida["modelo"]
    if (
        not isinstance(modelo, Mapping)
        or set(modelo) != {"repo", "revision"}
        or not isinstance(modelo["repo"], str)
        or "/" not in modelo["repo"]
        or not isinstance(modelo["revision"], str)
        or not RE_REVISION.fullmatch(modelo["revision"])
    ):
        errores.append("modelo debe tener repo (dueño/nombre) y revision (40 hexadecimales)")
    if not _entero(corrida["max_seq_length"], LARGO_GUIA):
        errores.append(f"max_seq_length debe ser un entero de al menos {LARGO_GUIA}")
    for clave in ("lote", "gradient_accumulation_steps", "epocas", "eval_cada_pasos", "lote_eval"):
        if not _entero(corrida[clave], 1):
            errores.append(f"{clave} debe ser un entero positivo")
    lr = corrida["learning_rate"]
    if not _numero(lr) or not 0 < lr <= 1e-2:
        errores.append("learning_rate debe estar entre 0 (sin incluir) y 0.01")
    fw = corrida["fraccion_warmup"]
    if not _numero(fw) or not 0 <= fw < 0.5:
        errores.append("fraccion_warmup debe estar en [0, 0.5)")
    if corrida["semilla"] != SEMILLA or isinstance(corrida["semilla"], bool):
        errores.append(f"semilla debe ser {SEMILLA} (regla 1)")
    lora = corrida["lora"]
    if not isinstance(lora, Mapping) or set(lora) != set(CLAVES_LORA):
        errores.append(f"lora debe tener exactamente las claves {list(CLAVES_LORA)}")
    else:
        if not _entero(lora["r"], 1) or not _entero(lora["lora_alpha"], 1):
            errores.append("lora: r y lora_alpha deben ser enteros positivos")
        if not _numero(lora["lora_dropout"]) or not 0 <= lora["lora_dropout"] < 1:
            errores.append("lora: lora_dropout debe estar en [0, 1)")
        if lora["bias"] != "none":
            errores.append("lora: bias debe ser 'none'")
        modulos = lora["target_modules"]
        if (
            not isinstance(modulos, (list, tuple))
            or not modulos
            or not all(isinstance(m, str) and m for m in modulos)
            or len(set(modulos)) != len(modulos)
        ):
            errores.append("lora: target_modules debe ser una lista de nombres distintos")
        if lora["use_gradient_checkpointing"] not in ("unsloth", True, False):
            errores.append("lora: use_gradient_checkpointing debe ser 'unsloth', True o False")
        if lora["random_state"] != SEMILLA or isinstance(lora["random_state"], bool):
            errores.append(f"lora: random_state debe ser {SEMILLA} (regla 1)")
    if errores:
        raise ValueError("Corrida inválida: " + "; ".join(errores) + ".")
    return dict(corrida)


def pasos_por_epoca(filas: int, lote: int, acumulacion: int = 1) -> int:
    """Pasos de optimización de una época, con el último lote incompleto (como Trainer)."""
    if not (_entero(filas, 1) and _entero(lote, 1) and _entero(acumulacion, 1)):
        raise ValueError("filas, lote y acumulación deben ser enteros positivos.")
    return math.ceil(math.ceil(filas / lote) / acumulacion)


def pasos_warmup(pasos: int, fraccion: float) -> int:
    """Pasos de calentamiento de la tasa: ceil(pasos × fracción), la regla de transformers."""
    return math.ceil(pasos * fraccion)


def argumentos_entrenamiento(corrida: Mapping, pasos: int, bf16: bool, carpeta: str | Path) -> dict:
    """Argumentos de TrainingArguments para la corrida.

    El train se recorre en orden aleatorio: `train_sampling_strategy="random"` da un
    RandomSampler, que accelerate cambia por un SeedableRandomSampler con `data_seed`.
    """
    return {
        "output_dir": str(carpeta),
        "per_device_train_batch_size": corrida["lote"],
        "per_device_eval_batch_size": corrida["lote_eval"],
        "gradient_accumulation_steps": corrida["gradient_accumulation_steps"],
        "num_train_epochs": corrida["epocas"],
        "max_steps": -1,
        "learning_rate": corrida["learning_rate"],
        "lr_scheduler_type": corrida["lr_scheduler_type"],
        "warmup_steps": pasos_warmup(pasos, corrida["fraccion_warmup"]),
        "optim": corrida["optim"],
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "fp16": not bf16,
        "bf16": bf16,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": corrida["eval_cada_pasos"],
        "prediction_loss_only": True,  # sin esto el Trainer juntaría los logits de validación
        "save_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
        "dataloader_drop_last": False,
        "remove_unused_columns": False,  # el colador necesita `conjunto` e `indice`
        "train_sampling_strategy": "random",
        "seed": corrida["semilla"],
        "data_seed": corrida["semilla"],
    }


def columnas(ejemplos: Sequence[Mapping], conjunto: str) -> dict[str, list]:
    """Columnas del Dataset; `indice` es la posición de la fila en su archivo (orden de id)."""
    return {
        "input_ids": [list(e["input_ids"]) for e in ejemplos],
        "labels": [list(e["labels"]) for e in ejemplos],
        "conjunto": [conjunto] * len(ejemplos),
        "indice": list(range(len(ejemplos))),
    }


class ColadorPorConjunto:
    """Colador que registra, por conjunto y en orden, las medidas y los índices de cada lote.

    Train y validación pasan por el mismo colador del Trainer; separarlos por la columna
    `conjunto` permite emparejar cada paso de entrenamiento con su lote y revisar el orden.
    """

    def __init__(self, a_tensor: Callable[[list], object], pad_id: int):
        self.a_tensor = a_tensor
        self.pad_id = pad_id
        self.lotes: dict[str, list[dict]] = {"train": [], "val": []}
        self.indices: dict[str, list[int]] = {"train": [], "val": []}

    def __call__(self, ejemplos):
        conjuntos = {e["conjunto"] for e in ejemplos}
        if len(conjuntos) != 1 or not conjuntos <= set(self.lotes):
            raise ValueError("Cada lote debe ser de un solo conjunto (train o val).")
        conjunto = conjuntos.pop()
        lote, medidas = hk.colar(ejemplos, self.pad_id)
        self.lotes[conjunto].append(medidas)
        self.indices[conjunto].extend(int(e["indice"]) for e in ejemplos)
        return {k: self.a_tensor(v) for k, v in lote.items()}


class Cronometro:
    """Tiempo entre fines de paso consecutivos, sin el tiempo de las evaluaciones.

    Cada evaluación periódica ocurre entre el fin de un paso y el inicio del siguiente; su
    duración se registra aparte y el reloj se reinicia al terminarla.
    """

    def __init__(self, reloj: Callable[[], float] = time.perf_counter, sincronizar: Callable[[], None] = lambda: None):
        self.reloj = reloj
        self.sincronizar = sincronizar
        self.tiempos: list[float] = []
        self.evaluaciones: list[dict] = []
        self._t: float | None = None

    def _ahora(self) -> float:
        self.sincronizar()
        return self.reloj()

    def inicio(self) -> None:
        self._t = self._ahora()

    def fin_de_paso(self) -> None:
        t = self._ahora()
        if self._t is not None:
            self.tiempos.append(t - self._t)
        self._t = t

    def fin_de_evaluacion(self, paso: int) -> None:
        if self._t is None:  # evaluación fuera del ciclo de entrenamiento (paso 0): se mide aparte
            return
        t = self._ahora()
        self.evaluaciones.append({"paso": paso, "s": round(t - self._t, 3)})
        self._t = t


def orden_secuencial(prefijo: Sequence[int]) -> bool:
    """True si los primeros índices vistos van en orden creciente (el train no se barajó)."""
    return len(prefijo) > 1 and list(prefijo) == sorted(prefijo)


def revisar_orden(indices: Sequence[int], n: int) -> dict:
    """¿El train se recorrió completo, una vez cada fila y en orden barajado?"""
    indices = [int(i) for i in indices]
    distintos = set(indices)
    completa = len(indices) == n and distintos == set(range(n))
    en_orden = indices == sorted(indices)
    return {
        "filas_esperadas": n,
        "filas_vistas": len(indices),
        "repetidas": len(indices) - len(distintos),
        "faltantes": len(set(range(n)) - distintos),
        "fuera_de_rango": len(distintos - set(range(n))),
        "permutacion_completa": completa,
        "en_orden_de_id": en_orden,
        "barajado": completa and not en_orden,
        "posiciones_en_su_lugar": sum(1 for p, i in enumerate(indices) if p == i),
        "sha256_orden": hashlib.sha256(",".join(map(str, indices)).encode("ascii")).hexdigest(),
    }


def curvas(historial: Sequence[Mapping]) -> dict:
    """Pérdida de train por paso y de validación por evaluación, del log_history del Trainer."""
    train, val = [], []
    for h in historial:
        if "loss" in h:
            train.append(
                {"paso": h.get("step"), "loss": h.get("loss"), "grad_norm": h.get("grad_norm"), "learning_rate": h.get("learning_rate")}
            )
        if "eval_loss" in h:
            val.append({"paso": h.get("step"), "eval_loss": h.get("eval_loss"), "eval_runtime_s": h.get("eval_runtime")})
    return {"train": train, "validacion": val}


def linea_avance(paso: int, pasos: int, perdidas: Sequence[float], segundos: Sequence[float]) -> str:
    """Línea del log: paso, pérdida media y s/paso recientes y minutos restantes estimados."""
    texto = f"paso {paso} de {pasos}"
    if perdidas:
        texto += f"; pérdida media de los últimos {len(perdidas)} pasos registrados: {sum(perdidas) / len(perdidas):.4f}"
    if segundos:
        sp = sum(segundos) / len(segundos)
        texto += f"; {sp:.3f} s/paso; faltan ~{(pasos - paso) * sp / 60:.0f} min de entrenamiento"
    return texto


def listar_entrada(raiz: str | Path) -> list[dict]:
    """Archivos bajo /kaggle/input: ruta relativa (POSIX) y bytes, ordenados."""
    raiz = Path(raiz)
    return [
        {"ruta": p.relative_to(raiz).as_posix(), "bytes": p.stat().st_size}
        for p in sorted(raiz.rglob("*"))
        if p.is_file()
    ]


def revisar_entrada(listado: Sequence[Mapping], esperados: Sequence[str] = ARCHIVOS_ENTRADA) -> dict[str, str]:
    """Ruta relativa de cada archivo esperado; aborta si falta, se repite o hay cualquier otro archivo."""
    por_nombre: dict[str, list[str]] = {}
    otros = []
    for a in listado:
        nombre = a["ruta"].rsplit("/", 1)[-1]
        if nombre in esperados:
            por_nombre.setdefault(nombre, []).append(a["ruta"])
        else:
            otros.append(a["ruta"])
    problemas = [f"{n}: {len(por_nombre.get(n, []))} copias" for n in esperados if len(por_nombre.get(n, [])) != 1]
    if otros:
        problemas.append(f"{len(otros)} archivos de más: {otros[:20]}")
    if problemas:
        raise AbortoCorrida(f"En la entrada deben estar solo {list(esperados)}, una vez cada uno. " + "; ".join(problemas) + ".")
    return {n: por_nombre[n][0] for n in esperados}


def archivos_salida(carpeta: str | Path, excluir: Sequence[str]) -> list[dict]:
    """Archivos de la salida con bytes y SHA-256, menos los excluidos (el JSON de mediciones)."""
    carpeta = Path(carpeta)
    return [
        {"ruta": p.relative_to(carpeta).as_posix(), "bytes": p.stat().st_size, "sha256": hk.sha256_archivo(p)}
        for p in sorted(carpeta.rglob("*"))
        if p.is_file() and p.name not in excluir
    ]


def configuracion(corrida: Mapping) -> dict:
    """Configuración completa de la corrida, para el JSON de mediciones."""
    return {
        "corrida": dict(corrida),
        "paquetes_fijados": list(hk.PAQUETES),
        "use_exact_model_name": True,
        "load_in_4bit": True,
        "inferencia": {"greedy": True, "lote": 1, "max_new_tokens": hk.MAX_NEW_TOKENS},
        "instruccion": INSTRUCCION,
        "machine_shape": hk.MACHINE_SHAPE,
        "archivos_entrada": list(ARCHIVOS_ENTRADA),
        "largo_guia": LARGO_GUIA,
        "avance_cada_pasos": AVANCE_CADA,
        "pasos_revision_orden": PASOS_REVISION_ORDEN,
        "pasos_excluidos_de_las_estadisticas_de_tiempo": hk.PASOS_CALENTAMIENTO,
    }


# --- Etapas en Kaggle (no se prueban en la laptop: necesitan GPU, torch y Unsloth) ---


def cargar(mods, modelo: Mapping[str, str], max_seq: int) -> tuple[object, object, dict]:
    """Como hk.cargar, con el largo máximo de la corrida: NF4 desde el repo exacto y la revisión fijada."""
    torch = mods.torch
    torch.cuda.reset_peak_memory_stats()
    model, tok = mods.FastLanguageModel.from_pretrained(
        model_name=modelo["repo"],
        revision=modelo["revision"],
        use_exact_model_name=True,
        load_in_4bit=True,
        max_seq_length=max_seq,
        dtype=None,
        random_state=SEMILLA,
    )
    qc = getattr(model.config, "quantization_config", None)
    qd = qc if isinstance(qc, Mapping) else (qc.to_dict() if qc is not None else {})
    if not qd.get("load_in_4bit") or qd.get("bnb_4bit_quant_type") != "nf4":
        raise AbortoCorrida(f"{modelo['repo']}: no se cargó en NF4 de 4 bits ({qd.get('bnb_4bit_quant_type')}).")
    if tok.pad_token_id is None:
        raise AbortoCorrida(f"{modelo['repo']}: el tokenizador no tiene token de relleno.")
    info = {
        "repo_pedido": modelo["repo"],
        "revision_fijada": modelo["revision"],
        "name_or_path": model.config._name_or_path,
        "dtype": str(getattr(model, "dtype", None)),
        "max_seq_length_pedido": max_seq,
        "max_seq_length_del_modelo": getattr(model, "max_seq_length", None),
        "cuantizacion": {k: qd.get(k) for k in sorted(qd) if k.startswith(("load_in", "bnb_", "quant_method"))},
        "capas_linear4bit": sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit"),
        "vram_tras_carga_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "tokenizador": {
            "eos": tok.eos_token,
            "pad": tok.pad_token,
            "pad_id": tok.pad_token_id,
            "padding_side": getattr(tok, "padding_side", None),
            "plantilla_sha256": hk.sha256_bytes(str(tok.chat_template).encode("utf-8")),
        },
    }
    return model, tok, info


def aplicar_lora(mods, model, lora: Mapping) -> tuple[object, dict]:
    mods.transformers.set_seed(SEMILLA)
    model = mods.FastLanguageModel.get_peft_model(model, **dict(lora))
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, {"parametros_entrenables": entrenables, "parametros_totales": total}


def preparar_tokens(reg: hk.Registro, tok, train: Sequence[Mapping], val: Sequence[Mapping], max_seq: int) -> dict:
    """Como hk.preparar_tokens, con el largo máximo de la corrida y los ejemplos de validación.

    (a), (c), etiquetas incorrectas y filas de más de `max_seq` fichas abortan; (b) solo se
    registra; (d) se revisa en la etapa de datos. Las filas de más de LARGO_GUIA solo se cuentan.
    """
    comp = reg.datos["comprobaciones"]
    id_inicio = tok.convert_tokens_to_ids(INICIO_DE_TURNO)
    id_fin = tok.convert_tokens_to_ids(FIN_DE_TURNO)
    id_fin_texto = tok.convert_tokens_to_ids(hk.TOKEN_FIN_DE_TEXTO)
    c_ok = (
        tok(INICIO_DE_TURNO, add_special_tokens=False)["input_ids"] == [id_inicio]
        and tok(FIN_DE_TURNO, add_special_tokens=False)["input_ids"] == [id_fin]
        and tok(hk.TOKEN_FIN_DE_TEXTO, add_special_tokens=False)["input_ids"] == [id_fin_texto]
        and len({id_inicio, id_fin, id_fin_texto}) == 3
    )
    comp["c_tokens_especiales_un_solo_token"] = {
        "ok": c_ok, "im_start": id_inicio, "im_end": id_fin, "endoftext": id_fin_texto,
    }
    if not c_ok:
        raise AbortoCorrida("(c) <|im_start|>, <|im_end|> o <|endoftext|> no son un solo token.")

    filas = [("train", r) for r in train] + [("val", r) for r in val]
    # (a) prompt de generación de la plantilla == prompt_generacion, byte a byte, en todas las filas.
    nombres = [r["nom_estab"] for _, r in filas] + [hk.NOMBRE_INVENTADO]
    distintos_a = sum(
        tok.apply_chat_template(mensajes(n), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        != prompt_generacion(n)
        for n in nombres
    )
    comp["a_prompt_generacion_igual_a_plantilla"] = {"textos": len(nombres), "distintos": distintos_a}
    if distintos_a:
        raise AbortoCorrida(f"(a) La plantilla no da prompt_generacion en {distintos_a} de {len(nombres)} textos.")

    # (b) plantilla con la respuesta frente al texto de entrenamiento construido: solo se registra.
    cuenta_b = {"igual": 0, "igual_mas_salto_final": 0, "otro": 0}
    for _, r in filas:
        plantilla = tok.apply_chat_template(
            mensajes(r["nom_estab"]) + [{"role": "assistant", "content": r["codigo_act"]}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        texto = texto_entrenamiento(r["nom_estab"], r["codigo_act"])
        clave = "igual" if plantilla == texto else ("igual_mas_salto_final" if plantilla == texto + "\n" else "otro")
        cuenta_b[clave] += 1
    comp["b_texto_entrenamiento_frente_a_plantilla"] = {"solo_registro": True, "conteo": cuenta_b}

    # Etiquetas y largos.
    ejemplos: dict[str, list[dict]] = {"train": [], "val": []}
    largos: dict[str, list[int]] = {"train": [], "val": []}
    no_prefijo = etiquetas_mal = 0
    etiquetados: dict[int, int] = {}
    for conjunto, r in filas:
        ids_p = tok(prompt_generacion(r["nom_estab"]), add_special_tokens=False)["input_ids"]
        ids_c = tok(texto_entrenamiento(r["nom_estab"], r["codigo_act"]), add_special_tokens=False)["input_ids"]
        largos[conjunto].append(len(ids_c))
        try:
            etq = etiquetas(ids_p, ids_c)
        except ErrorPrompt:
            no_prefijo += 1
            continue
        con_etiqueta = [t for t, e in zip(ids_c, etq) if e != IGNORAR]
        etiquetados[len(con_etiqueta)] = etiquetados.get(len(con_etiqueta), 0) + 1
        if tok.decode(con_etiqueta, skip_special_tokens=False) != r["codigo_act"] + FIN_DE_TURNO or con_etiqueta[-1] != id_fin:
            etiquetas_mal += 1
        ejemplos[conjunto].append({"input_ids": ids_c, "labels": etq})
    todos = largos["train"] + largos["val"]
    mayores = sum(1 for n in todos if n > max_seq)
    reg.datos["tokens"] = {
        "filas": len(filas),
        "max_seq_length": max_seq,
        "largo_guia": LARGO_GUIA,
        "por_conjunto": {
            c: {
                "filas": len(v),
                "largo_max": max(v),
                "largo_medio": round(sum(v) / len(v), 2),
                "mayores_a_largo_guia": sum(1 for n in v if n > LARGO_GUIA),
                "mayores_a_max_seq": sum(1 for n in v if n > max_seq),
            }
            for c, v in largos.items()
        },
        "prompt_no_es_prefijo": no_prefijo,
        "etiquetas_distintas_de_codigo_y_fin": etiquetas_mal,
        "fichas_etiquetadas_por_fila": {str(k): v for k, v in sorted(etiquetados.items())},
    }
    if no_prefijo or etiquetas_mal:
        raise AbortoCorrida(f"Etiquetas: {no_prefijo} filas sin prefijo y {etiquetas_mal} con etiquetas incorrectas.")
    if mayores:
        raise AbortoCorrida(f"{mayores} filas pasan de {max_seq} fichas; no se trunca.")

    # Ejemplo inventado: tabla de etiquetas en el log y en el JSON.
    texto_inv = texto_entrenamiento(hk.NOMBRE_INVENTADO, hk.CODIGO_INVENTADO)
    ids_p = tok(prompt_generacion(hk.NOMBRE_INVENTADO), add_special_tokens=False)["input_ids"]
    ids_c = tok(texto_inv, add_special_tokens=False)["input_ids"]
    etq = etiquetas(ids_p, ids_c)
    piezas = [tok.decode([t], skip_special_tokens=False) for t in ids_c]
    tabla = hk.tabla_etiquetas(piezas, ids_c, etq)
    print("\n".join(["Tokens del ejemplo inventado (etiqueta -100 = no entra a la pérdida):", *tabla]), flush=True)
    con_etiqueta = [p for p, e in zip(piezas, etq) if e != IGNORAR]
    reg.datos["ejemplo_etiquetas"] = {"codigo": hk.CODIGO_INVENTADO, "fichas": len(ids_c), "fichas_con_etiqueta": con_etiqueta}
    return {"ejemplos_train": ejemplos["train"], "ejemplos_val": ejemplos["val"], "ids_fin": [id_fin, id_fin_texto]}


def _ganchos(mods, cron: Cronometro, colador: ColadorPorConjunto, reg: hk.Registro, pasos: int, lote: int):
    """Callback del Trainer: cronómetro, revisión temprana del orden, avance y curva parcial."""

    class Ganchos(mods.TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            cron.inicio()

        def on_step_end(self, args, state, control, **kwargs):
            cron.fin_de_paso()
            paso = state.global_step
            if paso == PASOS_REVISION_ORDEN:
                prefijo = colador.indices["train"][: PASOS_REVISION_ORDEN * lote]
                secuencial = len(prefijo) == PASOS_REVISION_ORDEN * lote and orden_secuencial(prefijo)
                reg.datos["entrenamiento"]["revision_temprana_del_orden"] = {
                    "paso": paso, "filas_revisadas": len(prefijo), "en_orden_creciente": secuencial,
                }
                if secuencial:
                    raise AbortoCorrida(
                        f"El train no se está barajando: las primeras {len(prefijo)} filas llegaron en orden de id."
                    )
            if paso % AVANCE_CADA == 0 or paso == pasos:
                perdidas = [h["loss"] for h in state.log_history if "loss" in h][-AVANCE_CADA:]
                informar(linea_avance(paso, pasos, perdidas, cron.tiempos[-AVANCE_CADA:]))

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            cron.fin_de_evaluacion(state.global_step)
            parcial = reg.datos["entrenamiento"]
            parcial["paso_actual"] = state.global_step
            parcial["curvas"] = curvas(state.log_history)
            parcial["evaluaciones_s"] = list(cron.evaluaciones)
            reg.escribir()  # si la corrida se corta, la curva hasta aquí queda en el JSON
            perdida = (metrics or {}).get("eval_loss")
            informar(f"pérdida de validación en el paso {state.global_step}: {perdida}")

    return Ganchos()


def entrenar(reg: hk.Registro, mods, model, tok, prep: Mapping, corrida: Mapping) -> dict:
    """Entrena una corrida con la pérdida de validación en el paso 0, cada N pasos y al final."""
    torch = mods.torch
    n_train = len(prep["ejemplos_train"])
    pasos = pasos_por_epoca(n_train, corrida["lote"], corrida["gradient_accumulation_steps"]) * corrida["epocas"]
    bf16 = hk.bf16_nativo(torch)
    argumentos = argumentos_entrenamiento(corrida, pasos, bf16, hk.DIR_TRABAJO / f"trainer_{corrida['nombre']}")
    args = mods.TrainingArguments(**argumentos)
    colador = ColadorPorConjunto(lambda v: torch.tensor(v, dtype=torch.long), tok.pad_token_id)
    cron = Cronometro(sincronizar=torch.cuda.synchronize)
    ds_train = mods.datasets.Dataset.from_dict(columnas(prep["ejemplos_train"], "train"))
    ds_val = mods.datasets.Dataset.from_dict(columnas(prep["ejemplos_val"], "val"))
    reg.datos["entrenamiento"] = {"en_curso": True, "pasos_esperados": pasos, "pasos_warmup": argumentos["warmup_steps"]}
    reg.escribir()

    mods.transformers.set_seed(SEMILLA)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer = mods.Trainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val, data_collator=colador,
        callbacks=[_ganchos(mods, cron, colador, reg, pasos, corrida["lote"])],
    )
    # Sin las líneas de log del Trainer en cada paso: el avance lo imprime el callback propio.
    quitados = []
    for cb in list(trainer.callback_handler.callbacks):
        if type(cb).__name__ == "PrinterCallback":
            trainer.remove_callback(cb)
            quitados.append(type(cb).__name__)

    t = time.perf_counter()
    trainer.evaluate()  # paso 0: antes de entrenar; si la validación falla, falla aquí
    torch.cuda.synchronize()
    segundos_eval_0 = time.perf_counter() - t
    informar(f"evaluación del paso 0 en {segundos_eval_0:.1f} s; empieza el entrenamiento de {pasos} pasos")

    t = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    segundos = time.perf_counter() - t
    estado = trainer.state
    if not any("eval_loss" in h and h.get("step") == estado.global_step for h in estado.log_history):
        trainer.evaluate()  # respaldo: transformers 5.17 ya evalúa en el último paso

    hechos = len(cron.tiempos)
    lotes = colador.lotes["train"][:hechos]
    res = {
        "lote": corrida["lote"],
        "fp16": not bf16,
        "argumentos": argumentos,
        "pasos_esperados": pasos,
        "pasos_hechos": estado.global_step,
        "max_steps_del_trainer": estado.max_steps,
        "pasos_warmup": argumentos["warmup_steps"],
        "segundos_train_total": round(segundos, 2),
        "segundos_evaluacion_paso_0": round(segundos_eval_0, 2),
        "evaluaciones_s": cron.evaluaciones,
        "callbacks_quitados": quitados,
        "lotes_registrados_por_el_colador": {c: len(v) for c, v in colador.lotes.items()},
        "vram_pico_mib": {
            "asignada": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "reservada": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        },
        "revision_temprana_del_orden": reg.datos["entrenamiento"].get("revision_temprana_del_orden"),
        "orden_train": revisar_orden(colador.indices["train"], n_train),
        "curvas": curvas(estado.log_history),
        "resumen_del_trainer": [h for h in estado.log_history if "train_runtime" in h],
    }
    if len(lotes) == hechos:
        res["estadisticas"] = hk.estadisticas_pasos(
            cron.tiempos, [x["ejemplos"] for x in lotes], [x["fichas_reales"] for x in lotes]
        )
        res["pasos"] = [{"paso": i + 1, "s": round(s, 5), **x} for i, (s, x) in enumerate(zip(cron.tiempos, lotes))]
    else:
        res["estadisticas"] = {"error": "El colador registró menos lotes de train que pasos; no se emparejan."}
        res["pasos_s"] = [round(s, 5) for s in cron.tiempos]
    del trainer
    return res


def _revisar_entrenamiento(reg: hk.Registro, res: Mapping) -> None:
    """Fallas (no abortos: el adaptador ya está entrenado) si los pasos o el orden no son los esperados."""
    if not res["pasos_hechos"] == res["max_steps_del_trainer"] == res["pasos_esperados"]:
        reg.falla(
            "entrenamiento",
            f"Pasos: esperados {res['pasos_esperados']}, del Trainer {res['max_steps_del_trainer']}, hechos {res['pasos_hechos']}.",
        )
    if not res["orden_train"]["barajado"]:
        o = res["orden_train"]
        reg.falla(
            "orden_train",
            f"El train no se recorrió como una permutación barajada: {o['filas_vistas']} filas vistas de "
            f"{o['filas_esperadas']}, {o['repetidas']} repetidas, en orden de id = {o['en_orden_de_id']}.",
        )


def _correr(reg: hk.Registro, esperados: Mapping[str, str], corrida: Mapping) -> None:
    with reg.etapa("gpu"):
        gpus = hk.consultar_gpus()
        hk.verificar_t4(gpus)
        reg.datos["gpu"] = {
            "nvidia_smi": gpus,
            "machine_shape_pedido": hk.MACHINE_SHAPE,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus_usadas": 1,
        }
        informar(f"GPU: {[g['nombre'] for g in gpus]}; se usa solo la 0 (CUDA_VISIBLE_DEVICES=0)")

    with reg.etapa("datos"):
        listado = listar_entrada(hk.DIR_ENTRADA)
        reg.datos["datos"] = {"entrada": listado}
        rutas = revisar_entrada(listado)
        leidos = {a: hk.leer_jsonl_verificado(hk.DIR_ENTRADA / rutas[a], esperados[a]) for a in ARCHIVOS_ENTRADA}
        train, val = leidos[ARCHIVO_TRAIN], leidos[ARCHIVO_VAL]
        if {r["id"] for r in train} & {r["id"] for r in val}:
            raise AbortoCorrida("Train y validación comparten id.")
        reg.nombres = tuple(r["nom_estab"] for r in train + val)
        reg.llaves = frozenset(k for k in map(normalizar_nombre, reg.nombres) if k)
        reg.datos["datos"].update(
            {
                a: {"sha256": esperados[a], "sha256_verificado": True, "filas": len(rs), "clases": len({r["codigo_act"] for r in rs})}
                for a, rs in leidos.items()
            }
        )
        marcas = sum(tiene_marca_especial(n) for n in reg.nombres)
        reg.datos["comprobaciones"] = {"d_nombres_con_marca_especial": marcas}
        if marcas:
            raise AbortoCorrida(f"(d) {marcas} nombres contienen '<|'.")
        informar(f"datos verificados: {len(train)} filas de train y {len(val)} de validación; {len(reg.llaves)} llaves")

    with reg.etapa("instalacion"):
        hk.instalar(reg)

    with reg.etapa("imports"):
        mods = hk.importar(reg)

    modelo = corrida["modelo"]
    with reg.etapa("descarga"):
        ruta, verif = hk.descargar_verificado(mods, modelo["repo"], modelo["revision"])
        reg.datos["modelo"] = {"revision_descargada": ruta.name, "lfs_verificado": verif}

    with reg.etapa("carga"):
        model, tok, info = cargar(mods, modelo, corrida["max_seq_length"])
        reg.datos["modelo"].update(info)

    with reg.etapa("tokens_y_plantilla"):
        prep = preparar_tokens(reg, tok, train, val, corrida["max_seq_length"])

    with reg.etapa("lora"):
        model, info_lora = aplicar_lora(mods, model, corrida["lora"])
        reg.datos["modelo"]["lora"] = info_lora

    with reg.etapa("entrenamiento"):
        res = entrenar(reg, mods, model, tok, prep, corrida)
        reg.datos["entrenamiento"] = res
        _revisar_entrenamiento(reg, res)

    with reg.etapa("adaptador"):
        model.save_pretrained(str(hk.DIR_SALIDA / CARPETA_ADAPTADOR))

    with reg.etapa("inferencia"):
        hk.inferir(reg, mods, model, tok, val, prep["ids_fin"])

    del model
    hk.liberar(mods)

    with reg.etapa("cierre"):
        reg.datos["descargas_hf"] = hk.descargas_hf(mods)
        reg.datos["archivos_salida"] = archivos_salida(hk.DIR_SALIDA, (ARCHIVO_MEDICIONES, ARCHIVO_MEDICIONES + ".tmp"))
        reg.datos["estado"] = "completo" if not reg.datos["fallas"] else "completo_con_fallas"


def main(
    esperados: Mapping[str, str],
    t_arranque: float | None = None,
    fuentes: Mapping[str, str] | None = None,
    corrida: Mapping | None = None,
) -> None:
    """Punto de entrada en Kaggle. `esperados` y `corrida` los incrusta el generador."""
    t_arranque = time.time() if t_arranque is None else t_arranque
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # antes de importar torch: una sola T4
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    hk.informar = informar  # las funciones reutilizadas de la prueba imprimen con el prefijo de esta corrida
    corrida = validar_corrida(corrida)
    if set(esperados) != set(ARCHIVOS_ENTRADA):
        raise AbortoCorrida(f"Los SHA-256 esperados deben ser exactamente los de {list(ARCHIVOS_ENTRADA)}.")
    hk.DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    hk.DIR_TRABAJO.mkdir(parents=True, exist_ok=True)
    os.chdir(hk.DIR_TRABAJO)  # Unsloth deja su caché compilada en la carpeta actual
    reg = hk.Registro(hk.DIR_SALIDA / ARCHIVO_MEDICIONES, t_arranque)
    reg.datos["descripcion"] = f"Corrida {corrida['nombre']} del clasificador. {corrida['descripcion']}"
    reg.datos["configuracion"] = configuracion(corrida)
    reg.datos["fuentes_incrustadas_sha256"] = dict(fuentes or {})
    reg.datos["datos_sha256_esperados"] = dict(esperados)
    reg.escribir()
    try:
        _correr(reg, esperados, corrida)
    except hk.AbortoHumo as e:
        mensaje = hk.tapar_nombres(str(e), reg.nombres)
        reg.datos["estado"] = "abortado"
        reg.datos["aborto"] = mensaje
        reg.escribir()
        informar(f"ABORTO: {mensaje}")
        sys.exit(1)
    except Exception:
        tb = hk.tapar_nombres(traceback.format_exc(), reg.nombres)
        reg.datos["estado"] = "error"
        reg.datos["error"] = tb[-4000:]
        reg.escribir()
        print(tb, flush=True)
        sys.exit(1)
    reg.escribir()
    informar(f"estado final: {reg.datos['estado']}; fallas: {len(reg.datos['fallas'])}")
