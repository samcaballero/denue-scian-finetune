"""Pruebas de src/gguf_fase4.py con datos inventados y un cliente falso (sin llama-server ni modelos)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import gguf_fase4 as gf
from src.prompt_clasificador import prompt_generacion

IM_END, ENDOFTEXT = 151645, 151643


# --- Nombres ---


def test_nombres_y_etiquetas():
    assert gf.nombre_gguf("afinado", "Q4_K_M") == "qwen3-0.6b-06b-lr2e-4-r32-Q4_K_M.gguf"
    assert gf.nombre_gguf("base", "BF16") == "qwen3-0.6b-base-BF16.gguf"
    assert gf.de_etiqueta(gf.etiqueta("afinado", "Q8_0")) == ("afinado", "Q8_0")
    with pytest.raises(gf.ErrorGguf):
        gf.nombre_gguf("afinado", "F16")


# --- Regla de la demo (fijada antes de medir) ---

TAMANOS = {"Q4_K_M": 400, "Q8_0": 640, "BF16": 1200}


@pytest.mark.parametrize("identicas,mejor,elegida", [
    ({"BF16": 990, "Q8_0": 985, "Q4_K_M": 980}, "BF16", "BF16"),     # 10 filas = 1 punto exacto: no es "menor de 1"
    ({"BF16": 990, "Q8_0": 985, "Q4_K_M": 981}, "BF16", "Q4_K_M"),   # 9 filas = 0.9 puntos: Q4_K_M
    ({"BF16": 970, "Q8_0": 975, "Q4_K_M": 990}, "Q4_K_M", "Q4_K_M"),
    ({"BF16": 990, "Q8_0": 990, "Q4_K_M": 970}, "Q8_0", "Q8_0"),     # empate: la de archivo más chico
    ({"BF16": 990, "Q8_0": 990, "Q4_K_M": 990}, "Q4_K_M", "Q4_K_M"),
])
def test_elegir_cuantizacion(identicas, mejor, elegida):
    r = gf.elegir_cuantizacion(identicas, TAMANOS, 1000)
    assert (r["mejor_concordancia"], r["elegida"]) == (mejor, elegida)


def test_elegir_cuantizacion_rechaza_entradas_malas():
    with pytest.raises(gf.ErrorGguf):
        gf.elegir_cuantizacion({"BF16": 1, "Q8_0": 1}, TAMANOS, 1000)
    with pytest.raises(gf.ErrorGguf):
        gf.elegir_cuantizacion({"BF16": 1001, "Q8_0": 1, "Q4_K_M": 1}, TAMANOS, 1000)


# --- Conteos ---


def test_percentil_igual_a_numpy():
    rng = np.random.default_rng(0)
    x = list(rng.uniform(10, 900, size=200))
    for q in (0, 5, 50, 95, 100):
        assert gf.percentil(x, q) == pytest.approx(float(np.percentile(x, q)))
    with pytest.raises(gf.ErrorGguf):
        gf.percentil([], 50)


def _ref(i, resp, tope=False, fp=60, fg=7):
    return {"id": i, "respuesta": resp, "fichas_prompt": fp, "fichas_generadas": fg, "al_tope": tope, "segundos": 0.1}


def test_comparar_con_kaggle():
    ref = {"1": _ref("1", "461110"), "2": _ref("2", "722511"), "3": _ref("3", "abc", tope=True, fg=16)}
    preds = {"1": _ref("1", "461110"), "2": _ref("2", "722512"), "3": _ref("3", "abc", tope=True, fg=16)}
    r = gf.comparar_con_kaggle(preds, ref)
    assert (r["identicas"], r["distintas"], r["al_tope_igual"], r["fichas_generadas_igual"]) == (2, 1, 3, 3)
    with pytest.raises(gf.ErrorGguf):
        gf.comparar_con_kaggle({"1": preds["1"]}, ref)


def test_tabla_aciertos():
    t = gf.tabla_aciertos([True, True, False, False], [True, False, True, False])
    assert t == {"los_dos_aciertan": 1, "solo_kaggle_acierta": 1, "solo_gguf_acierta": 1, "ninguno_acierta": 1}


def test_muestra_latencia():
    filas = [{"id": str(i), "nom_estab": f"n{i}"} for i in range(1000, 0, -1)]  # desordenadas a propósito
    cal, med = gf.muestra_latencia(filas)
    assert (len(cal), len(med)) == (10, 200)
    perm = np.random.default_rng(42).permutation(1000)[:210]
    esperadas = [str(int(i) + 1) for i in perm]  # en orden de int(id), la posición i es el id i + 1
    assert [f["id"] for f in cal + med] == esperadas
    assert len({f["id"] for f in cal + med}) == 210


def test_resumen_latencia():
    medidas = [{"ms_punta_a_punta": 100.0 + i, "ms_completion": 90.0 + i, "fichas_prompt": 60, "prompt_ms": 30.0,
                "prompt_por_s": 2000.0, "fichas_generadas": 7, "generacion_ms": 70.0, "generacion_por_s": 100.0,
                "al_tope": False} for i in range(200)]
    r = gf.resumen_latencia(medidas)
    assert r["n"] == 200 and r["ms_punta_a_punta"]["p50"] == pytest.approx(199.5 - 100 + 100, abs=0.1)
    assert r["fichas_por_s"]["prompt_agregado"] == pytest.approx(2000.0) and r["al_tope"] == 0


# --- Tokens y generación con un cliente falso ---


class ClienteFalso:
    def __init__(self, bos=False, fin_visible=True, salidas=None):
        self.bos, self.fin_visible, self.salidas = bos, fin_visible, salidas or {}

    def tokenizar(self, texto, add_special=True):
        especiales = {"<|im_start|>": [151644], "<|im_end|>": [IM_END], "<|endoftext|>": [ENDOFTEXT]}
        ids = especiales.get(texto) or [ord(c) for c in texto]
        return ([ENDOFTEXT] if (self.bos and add_special) else []) + ids

    def detokenizar(self, ids):
        return "".join(chr(48 + t) if t < 10 else "?" for t in ids)

    def completar(self, prompt, n_predict):
        fichas = self.salidas.get(prompt, [4, 6, 1, 1, 1, 0])
        tope = len(fichas) >= n_predict
        fichas = fichas[:n_predict]
        if not tope and self.fin_visible:
            fichas = fichas + [IM_END]
        return {"tokens": fichas, "stop_type": "limit" if tope else "eos", "tokens_evaluated": len(prompt), "ms": 5.0,
                "timings": {}}


def _fichas_hf(texto):
    especiales = {"<|im_start|>": [151644], "<|im_end|>": [IM_END], "<|endoftext|>": [ENDOFTEXT]}
    return especiales.get(texto) or [ord(c) for c in texto]


def _id_hf(t):
    return _fichas_hf(t)[0]


FILAS = [{"id": str(i), "nom_estab": f"Negocio Inventado {i}", "codigo_act": "461110"} for i in (3, 7, 11)]


def test_comprobar_tokens_ok_y_con_bos():
    ref = {f["id"]: _ref(f["id"], "x", fp=len(prompt_generacion(f["nom_estab"]))) for f in FILAS}
    r = gf.comprobar_tokens(ClienteFalso(), _fichas_hf, _id_hf, FILAS, ref)
    assert r["ok"] and r["ids_iguales_a_hf"] == 3 and r["largo_hf_igual_a_fichas_prompt_de_kaggle"] == 3
    r = gf.comprobar_tokens(ClienteFalso(bos=True), _fichas_hf, _id_hf, FILAS, ref)
    assert not r["ok"] and r["agrega_bos"] and r["ids_iguales_a_hf"] == 0 and r["posiciones_distintas"] == [1, 2, 3]


@pytest.mark.parametrize("visible", [True, False])
def test_generar(visible):
    largo = {prompt_generacion(FILAS[2]["nom_estab"]): list(range(1, 10)) * 2}  # 18 fichas: llega al tope de 16
    regs, c = gf.generar(ClienteFalso(fin_visible=visible, salidas=largo), FILAS, 16, [IM_END, ENDOFTEXT])
    assert [r["respuesta"] for r in regs[:2]] == ["461110", "461110"]
    assert [r["fichas_generadas"] for r in regs] == [7, 7, 16]  # como Kaggle: con el token de fin
    assert [r["al_tope"] for r in regs] == [False, False, True]
    assert c["parada_con_token_de_fin_visible"] == (2 if visible else 0)
    assert c["parada_eos_sin_token_de_fin_de_kaggle_visible"] == (0 if visible else 2)
    assert tuple(regs[0]) == gf.CAMPOS_PREDICCION


def test_leer_jsonl():
    bueno = b'{"id": "1", "respuesta": "461110"}\n'
    assert gf.leer_jsonl(bueno, ("id", "respuesta")) == [{"id": "1", "respuesta": "461110"}]
    for malo in (b"\xef\xbb\xbf" + bueno, bueno[:-1], b'{"respuesta": "x", "id": "1"}\n', bueno + bueno,
                 b'{"id": "1a", "respuesta": "x"}\n'):
        with pytest.raises(gf.ErrorGguf):
            gf.leer_jsonl(malo, ("id", "respuesta"))


def test_sin_rutas_locales():
    r = gf._sin_rutas_locales({"a": ["C:\\sca\\x\\llama-server.exe", "C:/sca/y/m.gguf", "-c"], "b": 3})
    assert r == {"a": ["llama-server.exe", "m.gguf", "-c"], "b": 3}


# --- Escritura en results/ con candados ---

FILAS_NOMBRES = [{"id": "987654321", "nom_estab": "Taller Inventado Zafiro", "codigo_act": "811111"}]


def _nombres(_raiz):
    return {"TALLER INVENTADO ZAFIRO"}, {"987654321"}, FILAS_NOMBRES


def _escribir(tmp_path, obj):
    (tmp_path / "results").mkdir(exist_ok=True)
    git = lambda: {"commit": "a" * 40, "arbol_limpio": True}  # noqa: E731
    return gf.escribir_resultado(lambda: obj, "*_x.json", "2026-11-25_x.json", raiz=tmp_path,
                                 versionado=lambda s: True, nombres=_nombres, git=git, avance=lambda m: None)


def test_escribir_resultado_una_sola_vez(tmp_path):
    ruta = _escribir(tmp_path, {"accuracy": 0.687, "nota": "texto fijo"})
    assert json.loads(ruta.read_text(encoding="utf-8"))["accuracy"] == 0.687
    with pytest.raises(gf.ErrorGguf, match="Ya existe"):
        _escribir(tmp_path, {"accuracy": 0.687})


@pytest.mark.parametrize("obj", [{"id": "987654321"}, {"nota": "Taller Inventado Zafiro"},
                                 {"nota": "dice taller inventado zafiro aquí"}])
def test_escribir_resultado_candados(tmp_path, obj):
    with pytest.raises(gf.ErrorGguf, match="Candado"):
        _escribir(tmp_path, obj)
    assert not list((tmp_path / "results").glob("*.json"))


def test_escribir_resultado_exige_arbol_limpio(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(gf.ErrorGguf, match="árbol limpio"):
        gf.escribir_resultado(lambda: {}, "*_x.json", "x.json", raiz=tmp_path, versionado=lambda s: True,
                              nombres=_nombres, git=lambda: {"commit": "a", "arbol_limpio": False}, avance=lambda m: None)
