"""Pruebas de las comprobaciones de plantilla y largos de la Fase 4 (paso 15).

Usan un tokenizador falso que imita la plantilla de Qwen3 y nombres, ids y códigos inventados.
"""

import re

import pytest

from src.evaluador_clasificador import SECTORES_2023
from src.plantilla_fase4 import (
    MODOS,
    ErrorPlantilla,
    comprobar_tokenizadores,
    mensajes_fewshot,
    plantilla_de,
    prompt_de,
    revisar_modo,
    seleccion_fewshot,
    tokenizadores_identicos,
    tokens_especiales,
)
from src.preregistro_fase4 import prompt_fewshot
from src.prompt_clasificador import prompt_generacion

ESPECIALES = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643}
RE_ESPECIAL = re.compile("(" + "|".join(re.escape(t) for t in ESPECIALES) + ")")
LARGOS = {"general": 256, "few_shot": 1600, "nuevas_base": 32, "nuevas_afinados": 16}
EJEMPLOS = [("Tlapalería inventada", "467111"), ("Estética de prueba", "812110"), ("Fonda ficticia", "722513"),
            ("Gimnasio imaginario", "713943"), ("Despacho supuesto", "541110")]
NOMBRES = {"validacion": ["Café inventado", "Taller ficticio"], "test": ["Papelería de prueba", "Ñandú & Cía."],
           "hard": ["Negocio raro"]}


class TokFalso:
    """Imita la plantilla de Qwen3 (sin <think> en el historial) y tokeniza por bytes."""

    def __init__(self, extra_fewshot: str = "", partir_im_end: bool = False, vocab_extra: int = 0):
        self.extra_fewshot = extra_fewshot
        self.partir_im_end = partir_im_end
        self.chat_template = "plantilla falsa"
        self.added_tokens_decoder = {i: type("T", (), {"content": t, "special": True})() for t, i in ESPECIALES.items()}
        self._vocab = {f"b{i}": i for i in range(256 + vocab_extra)}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_ids(self, t):
        return ESPECIALES[t]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        assert tokenize is False
        texto = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        if len(msgs) > 1:
            texto += self.extra_fewshot
        if add_generation_prompt:
            texto += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "")
        return texto

    def __call__(self, texto, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        for trozo in RE_ESPECIAL.split(texto):
            if trozo in ESPECIALES:
                ids += [ESPECIALES[trozo], 0] if (self.partir_im_end and trozo == "<|im_end|>") else [ESPECIALES[trozo]]
            else:
                ids += list(trozo.encode("utf-8"))
        return {"input_ids": ids}


def test_prompts_de_cada_modo():
    assert prompt_de("afinado", "X") == prompt_de("base-0shot", "X") == prompt_generacion("X")
    assert prompt_de("base-fewshot", "X", EJEMPLOS) == prompt_fewshot(EJEMPLOS, "X")
    msgs = mensajes_fewshot(EJEMPLOS, "X")
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 5 + ["user"]
    assert [m["content"] for m in msgs[1:10:2]] == [c for _, c in EJEMPLOS]
    tok = TokFalso()
    assert plantilla_de(tok, "base-fewshot", "X", EJEMPLOS) == prompt_fewshot(EJEMPLOS, "X")
    assert plantilla_de(tok, "base-0shot", "X") == prompt_generacion("X")
    assert set(MODOS) == {"afinado", "base-0shot", "base-fewshot"}
    assert MODOS["afinado"]["conjuntos"] == ("validacion", "test", "hard")


def test_todo_pasa_con_la_plantilla_de_qwen():
    res = comprobar_tokenizadores({"0.6B": TokFalso(), "4B": TokFalso()}, NOMBRES, EJEMPLOS, LARGOS)
    assert res["ok"] and res["a_tokenizadores"]["identicos"]
    assert res["a_tokenizadores"]["textos"] == 5 + 3 + 3  # afinado (val, test, hard), 0-shot y few-shot (test, hard)
    fs = res["modos"]["4B"]["base-fewshot"]
    assert fs["limite"] == 1600 and fs["nuevas"] == 32 and fs["conjuntos"]["test"]["filas"] == 2
    af = res["modos"]["0.6B"]["afinado"]
    assert af["limite"] == 256 and af["nuevas"] == 16 and set(af["conjuntos"]) == {"validacion", "test", "hard"}
    largos = [len(TokFalso()(prompt_generacion(n))["input_ids"]) for n in NOMBRES["test"]]
    assert res["modos"]["0.6B"]["base-0shot"]["conjuntos"]["test"]["fichas_max"] == max(largos)


def test_fewshot_distinto_de_la_plantilla_falla_solo_c():
    res = comprobar_tokenizadores({"0.6B": TokFalso(extra_fewshot="\n")}, NOMBRES, EJEMPLOS, LARGOS)
    assert res["b_0shot_ok"] and not res["c_fewshot_ok"] and res["d_largos_ok"] and not res["ok"]
    assert res["modos"]["0.6B"]["base-fewshot"]["conjuntos"]["test"]["plantilla_distinta"] == 2
    assert "a_tokenizadores" not in res


def test_largos_que_no_caben():
    largo = max(len(TokFalso()(prompt_generacion(n))["input_ids"]) for n in NOMBRES["test"] + NOMBRES["hard"])
    chico = dict(LARGOS, general=largo + 31)  # el 0-shot de esa fila no cabe con 32 nuevas; con 16, sí
    r = revisar_modo(TokFalso(), "base-0shot", NOMBRES, EJEMPLOS, chico)
    assert r["conjuntos"]["test"]["no_caben"] >= 1 and not r["largo_ok"] and r["plantilla_ok"]
    assert revisar_modo(TokFalso(), "afinado", NOMBRES, EJEMPLOS, chico)["largo_ok"]
    res = comprobar_tokenizadores({"0.6B": TokFalso()}, NOMBRES, EJEMPLOS, dict(LARGOS, few_shot=300))
    assert not res["d_largos_ok"] and res["b_0shot_ok"] and res["c_fewshot_ok"]


def test_tokens_especiales_y_tokenizadores_distintos():
    assert tokens_especiales(TokFalso())["ok"]
    assert not tokens_especiales(TokFalso(partir_im_end=True))["ok"]
    textos = [prompt_generacion(n) for n in NOMBRES["test"]]
    assert tokenizadores_identicos(TokFalso(), TokFalso(), textos)["identicos"]
    r = tokenizadores_identicos(TokFalso(), TokFalso(vocab_extra=1), textos)
    assert not r["identicos"] and not r["vocabulario_igual"] and r["textos_con_ids_distintos"] == 0
    r = tokenizadores_identicos(TokFalso(), TokFalso(partir_im_end=True), textos)
    assert r["textos_con_ids_distintos"] == 2 and not r["identicos"]
    assert not comprobar_tokenizadores({"0.6B": TokFalso(partir_im_end=True)}, NOMBRES, EJEMPLOS, LARGOS)["ok"]


def test_errores():
    with pytest.raises(ErrorPlantilla):
        revisar_modo(TokFalso(), "base-fewshot", NOMBRES, [], LARGOS)
    with pytest.raises(ErrorPlantilla):
        revisar_modo(TokFalso(), "afinado", {"test": ["X"], "hard": ["Y"]}, EJEMPLOS, LARGOS)
    with pytest.raises(ErrorPlantilla):
        revisar_modo(TokFalso(), "base-0shot", {"test": [], "hard": ["Y"]}, EJEMPLOS, LARGOS)


def test_seleccion_fewshot_coincide_con_el_config():
    codigos = ("461110", "811111", "722511", "311110", "321111", "541110", "431110", "221110", "332110")
    train = [{"id": str(10 + 3 * i), "nom_estab": f"Taller inventado {i}", "codigo_act": codigos[i % len(codigos)]}
             for i in range(60)]
    base = {"k": 5, "semilla": 42, "sectores_oficiales": list(SECTORES_2023)}
    primera = seleccion_fewshot(train, {**base, "sha256_ids": "", "codigos": [], "sectores": []})
    assert not primera["publico"]["coincide_con_el_config"] and len(primera["ejemplos"]) == 5
    config = {**base, **{k: primera["publico"][k] for k in ("sha256_ids", "codigos", "sectores")}}
    sel = seleccion_fewshot(train, config)
    assert sel["publico"]["coincide_con_el_config"] and len(set(sel["publico"]["sectores"])) == 5
    assert [c for _, c in sel["ejemplos"]] == config["codigos"] and len(sel["ids"]) == 5
    assert not seleccion_fewshot(train, {**config, "codigos": list(reversed(config["codigos"]))})["publico"][
        "coincide_con_el_config"]
