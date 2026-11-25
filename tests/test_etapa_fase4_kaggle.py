"""Pruebas de los procesos del kernel de la Fase 4 (src/etapa_fase4_kaggle.py), en CPU.

Usan un motor falso (sin torch, Unsloth ni peft), un tokenizador y un modelo falsos y los datos de
juguete de tests/juguete_fase4.py: nada sale del DENUE.
"""

import contextlib
import json
import re
import types
from pathlib import Path

import pytest

from src import etapa_fase4_kaggle as ef
from src import humo_kaggle as hk
from src import inferencia_fase4_kaggle as fo
from src.preregistro_fase4 import prompt_fewshot
from src.prompt_clasificador import prompt_generacion
from tests import juguete_fase4 as jg

ESPECIALES = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643}
RE_ESPECIAL = re.compile("(" + "|".join(re.escape(t) for t in ESPECIALES) + ")")
DIGITO = 70_000  # id de la ficha del dígito d: DIGITO + d
RESPUESTA = "999111"  # no es código de ningún ejemplo few-shot de juguete


# --- Falsos ---


class Fila(list):
    def tolist(self):
        return list(self)


class Mat:
    def __init__(self, filas):
        self.filas = [list(f) for f in filas]

    @property
    def shape(self):
        return (len(self.filas), len(self.filas[0]))

    def __getitem__(self, idx):
        i, corte = idx
        return Fila(self.filas[i][corte])


class Enc(dict):
    def to(self, dispositivo):
        assert dispositivo == "cuda"
        return self


class TokFalso:
    """Plantilla de Qwen3 (sin <think> en el historial); una ficha por especial y por cada 2 bytes de texto."""

    pad_token_id = ESPECIALES["<|endoftext|>"]
    chat_template = "plantilla falsa"

    def __init__(self, bytes_por_ficha: int = 2, extra: str = "", repetir: int = 1):
        self.bytes_por_ficha = bytes_por_ficha
        self.extra = extra
        self.repetir = repetir  # cada ficha de texto cuenta `repetir` veces: prompts más largos
        self.added_tokens_decoder = {i: types.SimpleNamespace(content=t, special=True) for t, i in ESPECIALES.items()}

    def get_vocab(self):
        return {"x": 1}

    def convert_tokens_to_ids(self, t):
        return ESPECIALES[t]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        texto = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs) + self.extra
        if add_generation_prompt:
            texto += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "")
        return texto

    def __call__(self, texto, add_special_tokens=False, return_tensors=None):
        assert add_special_tokens is False
        ids = []
        for trozo in RE_ESPECIAL.split(texto):
            if trozo in ESPECIALES:
                ids.append(ESPECIALES[trozo])
            else:
                b = trozo.encode("utf-8")
                ids += [1000 + int.from_bytes(b[i:i + self.bytes_por_ficha], "big")
                        for i in range(0, len(b), self.bytes_por_ficha) for _ in range(self.repetir)]
        if return_tensors == "pt":
            return Enc(input_ids=Mat([ids]), attention_mask=Mat([[1] * len(ids)]))
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        inv = {v: k for k, v in ESPECIALES.items()}
        return "".join(inv.get(i, str(i - DIGITO) if DIGITO <= i < DIGITO + 10 else "?") for i in ids)


class ModeloFalso:
    def __init__(self, respuesta=RESPUESTA, fin=True, error_de_tipo=False):
        self.respuesta, self.fin, self.error_de_tipo = respuesta, fin, error_de_tipo
        self.generation_config = types.SimpleNamespace(do_sample=True, num_beams=4, temperature=0.6, top_p=0.95, top_k=20,
                                                       min_p=0.0, max_new_tokens=99, eos_token_id=1, pad_token_id=2)
        self.llamadas = []

    def generate(self, input_ids, attention_mask, generation_config):
        g = generation_config
        if self.error_de_tipo and g.temperature is None:
            raise TypeError("None no se acepta")
        ids = input_ids.filas[0]
        self.llamadas.append((tuple(ids), {k: getattr(g, k) for k in ("do_sample", "num_beams", "temperature", "top_p", "top_k",
                                                                        "max_new_tokens", "eos_token_id", "pad_token_id")}))
        nuevos = [DIGITO + int(d) for d in self.respuesta]
        nuevos = nuevos + [ESPECIALES["<|im_end|>"]] if self.fin else (nuevos * 20)
        return Mat([ids + nuevos[: g.max_new_tokens]])


class _Cuda:
    def reset_peak_memory_stats(self):
        pass

    def synchronize(self):
        pass

    def max_memory_allocated(self):
        return 100 * 2**20

    def max_memory_reserved(self):
        return 200 * 2**20


def mods_falsos():
    torch = types.SimpleNamespace(cuda=_Cuda(), inference_mode=contextlib.nullcontext)
    flm = types.SimpleNamespace(for_inference=lambda model: setattr(model, "para_inferencia", True))
    return types.SimpleNamespace(torch=torch, FastLanguageModel=flm)


PARAMS = [f"base_model.model.model.layers.{i}.self_attn.q_proj.lora_{ab}.default.weight" for i in range(2) for ab in "AB"]


class MotorFalso:
    def __init__(self, tok=None, modelo=None, fallan=(), distintos=()):
        self.tok = tok or TokFalso()
        self.modelo = modelo
        self.fallan, self.distintos = set(fallan), set(distintos)
        self.cargas, self.metodos, self.liberaciones, self.modelos = 0, [], 0, []

    def importar(self, reg):
        reg.datos.setdefault("gpu", {})
        return mods_falsos()

    def importar_ligero(self):
        return types.SimpleNamespace(hf=None, AutoTokenizer=None)

    def descargar(self, mods, repo, revision):
        return Path("/cache") / revision, {"model.safetensors": {"sha256": "0" * 64, "bytes": 10}}

    def tokenizador(self, mods, ruta):
        return self.tok

    def cargar(self, mods, base, max_seq):
        self.cargas += 1
        m = self.modelo or ModeloFalso()
        self.modelos.append(m)
        return m, self.tok, {"repo_pedido": base["repo"], "max_seq_length_pedido": max_seq}

    def poner_adaptador(self, mods, model, metodo, carpeta, cfg, repo):
        self.metodos.append(metodo)
        if metodo in self.fallan:
            raise RuntimeError("camino roto")
        model.metodo = metodo
        return model, {}

    def parametros(self, model):
        valor = 2.0 if getattr(model, "metodo", None) in self.distintos else 1.0
        return {**{p: valor for p in PARAMS}, "base_model.model.model.embed_tokens.weight": 5.0}

    def tensores_archivo(self, ruta):
        assert ruta.name == "adapter_model.safetensors"
        return {ef.nombre_en_archivo(p): 1.0 for p in PARAMS}

    def igual_tras_convertir(self, t, p):
        return t == p

    def dtype(self, x):
        return "float32"

    def liberar(self, mods):
        self.liberaciones += 1


# --- Inferencia: equivalencia con hk.inferir ---


def _val(n=5):
    return [{"id": str(jg.IDS["validacion"] + i), "nom_estab": f"Café {i}", "codigo_act": "461110"} for i in range(n)]


@pytest.mark.parametrize("error_de_tipo", [False, True])
def test_inferir_filas_equivale_a_hk_inferir(tmp_path, monkeypatch, error_de_tipo):
    monkeypatch.setattr(hk, "DIR_SALIDA", tmp_path)
    val = _val()
    ids_fin = [ESPECIALES["<|im_end|>"], ESPECIALES["<|endoftext|>"]]
    m1, m2 = ModeloFalso(error_de_tipo=error_de_tipo), ModeloFalso(error_de_tipo=error_de_tipo)
    reg = hk.Registro(tmp_path / "m.json", 0.0)
    hk.inferir(reg, mods_falsos(), m1, TokFalso(), val, ids_fin)
    filas, res = ef.inferir_filas(mods_falsos(), m2, TokFalso(), [(r["id"], prompt_generacion(r["nom_estab"])) for r in val],
                                  ids_fin, hk.MAX_NEW_TOKENS)
    pares = hk.serializar_predicciones([(f["id"], f["respuesta"]) for f in filas])
    assert pares == (tmp_path / hk.ARCHIVO_PREDICCIONES).read_bytes()
    assert m1.llamadas == m2.llamadas and len(m1.llamadas) == len(val)  # el intento con TypeError no llega a generar
    assert res["generation_config"] == reg.datos["inferencia"]["generation_config"]
    assert res["terminacion"] == reg.datos["inferencia"]["terminacion"] == {"token_de_fin": len(val)}
    assert res["respaldo_parametros_de_muestreo"] is error_de_tipo
    assert res["fichas_generadas_max"] == reg.datos["inferencia"]["fichas_generadas_max"] == 7
    assert m2.para_inferencia and res["completa"] and [f["respuesta"] for f in filas] == [RESPUESTA] * len(val)


def test_inferir_filas_mide_cada_fila_y_el_tope():
    prompts = [(r["id"], prompt_generacion(r["nom_estab"])) for r in _val(3)]
    tok = TokFalso()
    filas, res = ef.inferir_filas(mods_falsos(), ModeloFalso(fin=False), tok, prompts, [151645, 151643], 32)
    assert list(filas[0]) == list(fo.CAMPOS_PREDICCION)
    assert all(f["al_tope"] and f["fichas_generadas"] == 32 for f in filas)
    assert filas[0]["fichas_prompt"] == len(tok(prompts[0][1])["input_ids"])
    assert res["al_tope"] == 3 and res["pct_al_tope"] == 100.0 and res["terminacion"] == {"limite_de_tokens": 3}
    assert res["generation_config"]["max_new_tokens"] == 32 and res["vram_pico_mib"]["asignada"] == 100.0
    datos = ef.serializar_filas(filas)
    assert fo.leer_jsonl(datos, fo.CAMPOS_PREDICCION)[0]["respuesta"] == RESPUESTA * 5 + RESPUESTA[:2]


def test_inferir_filas_para_al_limite_de_tiempo():
    t = iter(range(0, 10_000, 10))
    prompts = [(r["id"], prompt_generacion(r["nom_estab"])) for r in _val(5)]
    modelo = ModeloFalso()
    filas, res = ef.inferir_filas(mods_falsos(), modelo, TokFalso(), prompts, [151645], 16, limite=35, reloj=lambda: next(t))
    assert not res["completa"] and 0 < res["filas"] < 5 and len(modelo.llamadas) == res["filas"]


# --- Adaptador ---


def test_comparar_lora_y_nombres():
    assert ef.nombre_en_archivo(PARAMS[0]) == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert ef.nombre_en_archivo("base_model.model.model.embed_tokens.weight") is None
    archivo = {ef.nombre_en_archivo(p): 1.0 for p in PARAMS}
    iguales, dt = (lambda a, b: a == b), (lambda x: "float32")
    r = ef.comparar_lora({p: 1.0 for p in PARAMS}, archivo, iguales, dt)
    assert r["ok"] and r["comparados"] == 4 and r["dtypes_archivo_a_modelo"] == {"float32->float32": 4}
    assert not ef.comparar_lora({**{p: 1.0 for p in PARAMS}, PARAMS[0]: 3.0}, archivo, iguales, dt)["ok"]
    r = ef.comparar_lora({p: 1.0 for p in PARAMS[:3]}, archivo, iguales, dt)
    assert not r["ok"] and r["faltan_en_el_modelo"] == 1
    assert not ef.comparar_lora({}, {}, iguales, dt)["ok"]


def test_lora_de_config():
    lora = ef.lora_de_config(jg.adapter_config("Qwen/Qwen3-4B"), "Qwen/Qwen3-4B")
    assert lora == {**hk.LORA, "r": 32, "lora_alpha": 32}
    for cambio in ({"use_dora": True}, {"lora_dropout": 0.1}, {"target_modules": ["q_proj"]}, {"r": 0},
                   {"base_model_name_or_path": "otro/modelo"}, {"modules_to_save": ["lm_head"]}):
        with pytest.raises(ef.ErrorEtapa):
            ef.lora_de_config({**jg.adapter_config("Qwen/Qwen3-4B"), **cambio}, "Qwen/Qwen3-4B")


# --- Etapa completa con el motor falso ---


@pytest.fixture
def juguete(tmp_path):
    return jg.armar(tmp_path)


def _entrada(j, tmp_path, id_etapa, metodo=None, limite_s=1000.0):
    plan = {e["id"]: e for e in fo.validar_config_kernel(j["config"])["etapas"]}
    listado = fo.kc.listar_entrada(j["entrada"])
    hashes = {a["ruta"]: hk.sha256_archivo(j["entrada"] / a["ruta"]) for a in listado}
    rev = fo.revisar_entrada(listado, hashes, {"etapas": list(plan.values())}, j["esperados"], j["config"])
    dep = types.SimpleNamespace(dir_entrada=j["entrada"], dir_salida=tmp_path / "working", raiz_fuentes=j["raiz"])
    ctx = fo._contexto(rev, dep, j["fuentes"])
    return {**ctx, "etapa": plan[id_etapa], "esperados": j["esperados"], "limite_s": limite_s, "metodo_adaptador": metodo}


def _medicion(tmp_path, id_etapa):
    return json.loads((tmp_path / "working" / fo.CARPETA_ETAPAS / f"{id_etapa}.json").read_text(encoding="utf-8"))


def _sin_privados(texto):
    return not any(t in texto for t in jg.textos_privados()) and RESPUESTA not in texto


def test_etapa_base_0shot_completa(juguete, tmp_path, capsys):
    motor = MotorFalso()
    e = _entrada(juguete, tmp_path, "06b-base-0shot__test")
    assert ef.correr_etapa(e, motor) == 0
    m = _medicion(tmp_path, "06b-base-0shot__test")
    assert m["estado"] == "completa" and m["archivo"]["filas"] == jg.FILAS["test"]
    pred = tmp_path / "working" / m["archivo"]["ruta"]
    assert m["archivo"]["ruta"] == "predicciones/06b-base-0shot/test.jsonl" and hk.sha256_archivo(pred) == m["archivo"]["sha256"]
    regs = fo.leer_jsonl(pred.read_bytes(), fo.CAMPOS_PREDICCION)
    assert [r["id"] for r in regs] == [r["id"] for r in jg.filas("test")] and {r["respuesta"] for r in regs} == {RESPUESTA}
    llamadas = motor.modelos[0].llamadas
    assert [ids for ids, _ in llamadas] == [tuple(motor.tok(prompt_generacion(r["nom_estab"]))["input_ids"]) for r in jg.filas("test")]
    assert llamadas[0][1]["max_new_tokens"] == 32 and llamadas[0][1]["do_sample"] is False
    rev = m["antes_de_generar"]["plantilla_y_largos"]
    assert set(rev["conjuntos"]) == {"test", "hard"} and rev["limite"] == 256 and rev["nuevas"] == 32
    assert "adaptador" not in m and m["config"]["sha256"] == juguete["fuentes"][fo.RUTA_CONFIG]
    texto = (tmp_path / "working" / fo.CARPETA_ETAPAS / "06b-base-0shot__test.json").read_text(encoding="utf-8")
    assert _sin_privados(texto) and _sin_privados(capsys.readouterr().out)


def test_etapa_fewshot_usa_los_cinco_ejemplos(juguete, tmp_path):
    motor = MotorFalso()
    assert ef.correr_etapa(_entrada(juguete, tmp_path, "4b-base-fewshot__hard"), motor) == 0
    m = _medicion(tmp_path, "4b-base-fewshot__hard")
    assert m["few_shot"]["coincide_con_el_config"] and m["antes_de_generar"]["plantilla_y_largos"]["limite"] == 1600
    ids = motor.modelos[0].llamadas[0][0]
    assert ids.count(ESPECIALES["<|im_start|>"]) == 12  # 5 pares de ejemplo, la fila y el asistente
    sel = ef.seleccion_fewshot(jg.filas("train_final"), juguete["config"]["few_shot"])
    assert ids == tuple(motor.tok(prompt_fewshot(sel["ejemplos"], jg.filas("hard")[0]["nom_estab"]))["input_ids"])


def test_etapa_afinado_carga_el_adaptador_por_el_camino_principal(juguete, tmp_path):
    motor = MotorFalso()
    assert ef.correr_etapa(_entrada(juguete, tmp_path, "06b-lr2e-4-r32__validacion"), motor) == 0
    m = _medicion(tmp_path, "06b-lr2e-4-r32__validacion")
    assert m["adaptador"]["metodo"] == "estado_lora" and motor.metodos == ["estado_lora"] and motor.cargas == 1
    assert m["adaptador"]["metodo_indice"] == 0 and m["completa"] is True
    assert m["adaptador"]["intentos"][0]["comparacion"]["comparados"] == len(PARAMS)
    assert m["antes_de_generar"]["plantilla_y_largos"]["nuevas"] == 16
    assert set(m["antes_de_generar"]["plantilla_y_largos"]["conjuntos"]) == {"validacion", "test", "hard"}


@pytest.mark.parametrize("motor_kw", [{"fallan": ["estado_lora"]}, {"distintos": ["estado_lora"]}])
def test_etapa_afinado_usa_el_respaldo(juguete, tmp_path, motor_kw):
    motor = MotorFalso(**motor_kw)
    assert ef.correr_etapa(_entrada(juguete, tmp_path, "4b-lr2e-4-r32__validacion"), motor) == 0
    m = _medicion(tmp_path, "4b-lr2e-4-r32__validacion")
    assert m["adaptador"]["metodo"] == "peft_from_pretrained" and motor.metodos == list(ef.METODOS_ADAPTADOR)
    assert motor.cargas == 2 and motor.liberaciones == 1 and len(m["adaptador"]["intentos"]) == 2


def test_etapa_afinado_sin_camino_valido_no_genera(juguete, tmp_path):
    motor = MotorFalso(fallan=["peft_from_pretrained"], distintos=["estado_lora"])
    assert ef.correr_etapa(_entrada(juguete, tmp_path, "4b-lr2e-4-r32__validacion"), motor) == 1
    m = _medicion(tmp_path, "4b-lr2e-4-r32__validacion")
    assert m["estado"] == "incompleta" and m["completa"] is False and "inferencia" not in m
    assert all(not mo.llamadas for mo in motor.modelos)
    assert not (tmp_path / "working" / fo.CARPETA_PREDICCIONES).exists()


def test_etapa_de_test_usa_el_metodo_de_validacion(juguete, tmp_path):
    motor = MotorFalso(fallan=["estado_lora"])
    e = _entrada(juguete, tmp_path, "06b-lr2e-4-r32__test", metodo="peft_from_pretrained")
    assert ef.correr_etapa(e, motor) == 0 and motor.metodos == ["peft_from_pretrained"]
    motor = MotorFalso(fallan=["peft_from_pretrained"])
    e = _entrada(juguete, tmp_path, "06b-lr2e-4-r32__hard", metodo="peft_from_pretrained")
    assert ef.correr_etapa(e, motor) == 1 and motor.metodos == ["peft_from_pretrained"]  # sin probar otro camino


@pytest.mark.parametrize("tok,motivo", [(TokFalso(repetir=3), "plantilla_o_largo"), (TokFalso(extra="\n"), "plantilla_o_largo")])
def test_largo_o_plantilla_mal_no_genera_ninguna_respuesta(juguete, tmp_path, tok, motivo):
    motor = MotorFalso(tok=tok)
    assert ef.correr_etapa(_entrada(juguete, tmp_path, "06b-base-0shot__hard"), motor) == 1
    m = _medicion(tmp_path, "06b-base-0shot__hard")
    assert m["estado"] == "incompleta" and m["motivo"] == motivo and not motor.modelos[0].llamadas
    assert not (tmp_path / "working" / fo.CARPETA_PREDICCIONES).exists()


def test_etapa_cortada_por_tiempo_no_deja_predicciones(juguete, tmp_path):
    t = iter(range(0, 100_000, 5))
    e = _entrada(juguete, tmp_path, "06b-base-0shot__test", limite_s=12)
    assert ef.correr_etapa(e, MotorFalso(), reloj=lambda: next(t)) == 1
    m = _medicion(tmp_path, "06b-base-0shot__test")
    assert m["estado"] == "incompleta" and m["motivo"] == "tiempo" and m["inferencia"]["filas"] < jg.FILAS["test"]
    assert not (tmp_path / "working" / fo.CARPETA_PREDICCIONES).exists()


def test_datos_alterados_o_etapa_ajena_detienen_la_etapa(juguete, tmp_path):
    e = _entrada(juguete, tmp_path, "06b-base-0shot__test")
    assert ef.correr_etapa({**e, "esperados": {**e["esperados"], "hard.jsonl": "0" * 64}}, MotorFalso()) == 1
    assert ef.correr_etapa({**e, "etapa": {**e["etapa"], "max_new_tokens": 64}}, MotorFalso()) == 1
    assert "no es la del plan" in _medicion(tmp_path, "06b-base-0shot__test")["error"]


def test_test_con_etiquetas_no_se_lee(juguete, tmp_path):
    """Una copia de test con codigo_act no pasa el parseo estricto (además de no tener el SHA-256 del manifest)."""
    ruta = juguete["entrada"] / "denue-scian-evaluacion" / "test.jsonl"
    datos = jg.jsonl({**r, "codigo_act": "461110"} for r in jg.filas("test"))
    ruta.write_bytes(datos)
    e = _entrada({**juguete, "esperados": {**juguete["esperados"], "test.jsonl": jg.sha(datos)}}, tmp_path, "06b-base-0shot__test")
    assert ef.correr_etapa(e, MotorFalso()) == 1
    assert "claves" in _medicion(tmp_path, "06b-base-0shot__test")["error"]


# --- Comprobaciones iniciales ---


def test_comprobaciones_pasan_y_no_traen_nombres(juguete, tmp_path, capsys):
    e = {k: v for k, v in _entrada(juguete, tmp_path, "06b-base-0shot__test").items() if k not in ("etapa", "limite_s")}
    assert ef.correr_comprobaciones(e, MotorFalso()) == 0
    texto = (tmp_path / "working" / fo.ARCHIVO_COMPROBACIONES).read_text(encoding="utf-8")
    c = json.loads(texto)
    assert c["ok"] and c["few_shot"]["ok"] and c["tokenizadores"]["ok"] and c["tokenizadores"]["a_tokenizadores"]["identicos"]
    assert set(c["descargas"]) == {"0.6B", "4B"} and c["datos"]["train_final"]["filas"] == jg.FILAS["train_final"]
    assert c["few_shot"]["ids_ejemplos_en"] == {"validacion": 0, "test": 0, "hard": 0}
    assert _sin_privados(texto) and _sin_privados(capsys.readouterr().out)


def test_comprobaciones_fallan_con_otra_seleccion_o_plantilla(juguete, tmp_path):
    e = {k: v for k, v in _entrada(juguete, tmp_path, "06b-base-0shot__test").items() if k not in ("etapa", "limite_s")}
    assert ef.correr_comprobaciones(e, MotorFalso(tok=TokFalso(extra="\n"))) == 1
    assert json.loads((tmp_path / "working" / fo.ARCHIVO_COMPROBACIONES).read_bytes())["tokenizadores"]["c_fewshot_ok"] is False
    config = json.loads(json.dumps(juguete["config"]))
    config["few_shot"]["codigos"] = list(reversed(config["few_shot"]["codigos"]))
    texto = jg.escribir_config(juguete["raiz"], config)
    e["fuentes"] = {fo.RUTA_CONFIG: jg.sha(texto)}
    assert ef.correr_comprobaciones(e, MotorFalso()) == 1
    assert json.loads((tmp_path / "working" / fo.ARCHIVO_COMPROBACIONES).read_bytes())["few_shot"]["ok"] is False
