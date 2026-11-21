"""Pruebas de las funciones puras del kernel de Kaggle (Fase 2, paso 6b).

Solo nombres, ids y códigos inventados en carpetas temporales; nunca leen el DENUE ni los
splits. Las etapas que necesitan GPU, torch o Unsloth se prueban en la corrida de Kaggle.
"""

import hashlib
import json
import re
import struct
from pathlib import Path

import pytest

from src import humo_kaggle as hk
from src.humo_kaggle import (
    AbortoHumo,
    Registro,
    buscar_unico,
    colar,
    comparar_tensores,
    cortar_respuesta,
    diferencias_freeze,
    estadisticas_pasos,
    hashes_safetensors,
    leer_jsonl_verificado,
    parsear_nvidia_smi,
    percentil,
    serializar_predicciones,
    tabla_etiquetas,
    tapar_llaves,
    tapar_nombres,
    verificar_t4,
)
from src.prompt_clasificador import IGNORAR
from src.scian import leer_catalogo

NOMBRES = ["Taller Inventado Uno", "Fonda Ficticia Dos", "Estética Imaginaria"]


def _jsonl(regs) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def _regs():
    return [{"id": str(10 + i), "nom_estab": n, "codigo_act": f"46711{i}"} for i, n in enumerate(NOMBRES)]


def _escribir(tmp_path: Path, datos: bytes, nombre: str = "humo_train.jsonl") -> tuple[Path, str]:
    ruta = tmp_path / nombre
    ruta.write_bytes(datos)
    return ruta, hashlib.sha256(datos).hexdigest()


# --- Configuración aprobada ---


def test_configuracion_aprobada():
    assert hk.SEMILLA == 42
    assert hk.MODELO_06B["repo"] == "Qwen/Qwen3-0.6B"
    assert hk.MODELO_4B["repo"] == "Qwen/Qwen3-4B"
    assert hk.BASE_16_ESPEJO["repo"] == "unsloth/Qwen3-0.6B"
    for m in (hk.MODELO_06B, hk.MODELO_4B, hk.BASE_16_ESPEJO):
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"])
    assert hk.LORA["r"] == 16 and hk.LORA["lora_alpha"] == 16
    assert set(hk.LORA["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert hk.LORA["random_state"] == 42
    assert hk.ENTRENAMIENTO["learning_rate"] == 2e-4
    assert hk.ENTRENAMIENTO["num_train_epochs"] == 1
    assert hk.ENTRENAMIENTO["seed"] == hk.ENTRENAMIENTO["data_seed"] == 42
    assert hk.ENTRENAMIENTO["gradient_accumulation_steps"] == 1
    assert hk.LOTE == 16 and hk.MAX_SEQ == 128 and hk.MAX_NEW_TOKENS == 16
    assert hk.LOTES_06B == (32, 64) and hk.LOTES_4B_EXTRA == (32,)
    assert hk.PASOS_4B == 50 and hk.PASOS_LOTE == hk.PASOS_CALENTAMIENTO + 20
    assert hk.PRESUPUESTO_4B_S == 3600
    assert hk.MACHINE_SHAPE == "NvidiaTeslaT4"
    assert all(re.fullmatch(r"[A-Za-z0-9_]+==[0-9.]+", p) for p in hk.PAQUETES)
    assert json.dumps(hk.configuracion())  # serializable


def test_ejemplo_inventado():
    catalogo = leer_catalogo(Path(hk.__file__).resolve().parents[1] / "data" / "scian_2023_clases.csv")
    assert hk.CODIGO_INVENTADO in catalogo
    assert not hk.NOMBRE_INVENTADO.isascii()  # prueba la codificación UTF-8 en la laptop (6d)


# --- Datos ---


def test_leer_jsonl_verificado(tmp_path):
    ruta, sha = _escribir(tmp_path, _jsonl(_regs()))
    assert leer_jsonl_verificado(ruta, sha) == _regs()


def test_leer_jsonl_hash_distinto_sin_nombres(tmp_path):
    ruta, _ = _escribir(tmp_path, _jsonl(_regs()))
    with pytest.raises(AbortoHumo) as e:
        leer_jsonl_verificado(ruta, "0" * 64)
    assert not any(n in str(e.value) for n in NOMBRES)


@pytest.mark.parametrize(
    "mutar",
    [
        lambda b: b"\xef\xbb\xbf" + b,
        lambda b: b[:-1],
        lambda b: b.replace(b'"id"', b'"ID"', 1),
        lambda b: b.replace(b'"10"', b'"1x"', 1),
        lambda b: b.replace(b'"467110"', b'"46711"', 1),
        lambda b: b + b.split(b"\n")[0] + b"\n",
    ],
    ids=["bom", "sin_salto_final", "claves", "id", "codigo", "id_repetido"],
)
def test_leer_jsonl_rechaza(tmp_path, mutar):
    ruta, sha = _escribir(tmp_path, mutar(_jsonl(_regs())))
    with pytest.raises(AbortoHumo) as e:
        leer_jsonl_verificado(ruta, sha)
    assert not any(n in str(e.value) for n in NOMBRES)


def test_buscar_unico(tmp_path):
    with pytest.raises(AbortoHumo):
        buscar_unico(tmp_path, "humo_train.jsonl")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "humo_train.jsonl").write_bytes(b"x")
    assert buscar_unico(tmp_path, "humo_train.jsonl") == tmp_path / "a" / "humo_train.jsonl"
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "humo_train.jsonl").write_bytes(b"x")
    with pytest.raises(AbortoHumo):
        buscar_unico(tmp_path, "humo_train.jsonl")


# --- GPU ---


def test_nvidia_smi_t4_x2():
    gpus = parsear_nvidia_smi("0, Tesla T4, 15360, 560.35.03\n1, Tesla T4, 15360, 560.35.03\n")
    assert [g["indice"] for g in gpus] == [0, 1]
    assert gpus[0] == {"indice": 0, "nombre": "Tesla T4", "memoria_mib": 15360, "driver": "560.35.03"}
    verificar_t4(gpus)


@pytest.mark.parametrize(
    "salida",
    ["0, Tesla P100-PCIE-16GB, 16384, 560.35.03\n", "0, NVIDIA L4, 23034, 560.35.03\n",
     "0, Tesla T4, 15360, 560\n1, NVIDIA A100, 40960, 560\n", ""],
)
def test_nvidia_smi_no_t4(salida):
    with pytest.raises(AbortoHumo):
        verificar_t4(parsear_nvidia_smi(salida))


def test_nvidia_smi_formato_raro():
    with pytest.raises(AbortoHumo):
        parsear_nvidia_smi("No devices were found")


class _CudaFalsa:
    def __init__(self, capacidad):
        self.capacidad = capacidad

    def get_device_capability(self, i=0):
        return self.capacidad

    def is_bf16_supported(self, including_emulation=True):
        raise AssertionError("no debe usarse: cuenta la emulación y en una T4 puede dar True")


@pytest.mark.parametrize("capacidad,esperado", [((7, 5), False), ((6, 0), False), ((8, 0), True), ((9, 0), True)])
def test_bf16_nativo_por_capacidad(capacidad, esperado):
    torch = type("T", (), {"cuda": _CudaFalsa(capacidad)})
    assert hk.bf16_nativo(torch) is esperado


# --- Instalación ---


def test_diferencias_freeze():
    antes = "torch==2.14.0\nTransformers==5.1.0\nfoo==1.0\n"
    despues = "torch==2.14.0\ntransformers==5.17.0\nunsloth==2026.11.9\npkg @ file:///x\n"
    d = diferencias_freeze(antes, despues)
    assert d["cambiados"] == {"transformers": ["5.1.0", "5.17.0"]}
    assert d["agregados"] == {"pkg": "file:///x", "unsloth": "2026.11.9"}
    assert d["quitados"] == {"foo": "1.0"}


# --- Mediciones ---


def test_percentil_como_numpy():
    assert percentil([4, 1, 3, 2], 50) == 2.5
    assert percentil([1, 2, 3, 4], 90) == pytest.approx(3.7)
    assert percentil([5], 90) == 5
    assert percentil([], 50) is None


def test_estadisticas_sin_calentamiento():
    tiempos = [10.0, 9.0, 8.0, 7.0, 6.0, 1.0, 1.0, 2.0]
    ejemplos = [16] * 8
    fichas = [1000] * 8
    s = estadisticas_pasos(tiempos, ejemplos, fichas, calentamiento=5)
    assert s["pasos_total"] == 8 and s["pasos_medidos"] == 3 and s["pasos_calentamiento"] == 5
    assert s["segundos_medidos"] == 4.0
    assert s["s_por_paso_media"] == pytest.approx(4 / 3, abs=1e-5)
    assert s["ejemplos_por_s"] == pytest.approx(48 / 4, abs=1e-3)
    assert s["fichas_por_s"] == pytest.approx(3000 / 4, abs=0.1)
    assert s["segundos_con_calentamiento"] == 44.0


def test_estadisticas_solo_calentamiento_y_largos():
    s = estadisticas_pasos([1.0, 2.0], [1, 1], [1, 1], calentamiento=5)
    assert s["pasos_medidos"] == 0 and "s_por_paso_media" not in s
    with pytest.raises(ValueError):
        estadisticas_pasos([1.0], [1, 2], [1])


def test_colar_relleno_a_la_derecha():
    ejemplos = [
        {"input_ids": [1, 2, 3, 4], "labels": [IGNORAR, IGNORAR, 3, 4]},
        {"input_ids": [5, 6], "labels": [IGNORAR, 6]},
    ]
    lote, medidas = colar(ejemplos, pad_id=0)
    assert lote["input_ids"] == [[1, 2, 3, 4], [5, 6, 0, 0]]
    assert lote["attention_mask"] == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert lote["labels"] == [[IGNORAR, IGNORAR, 3, 4], [IGNORAR, 6, IGNORAR, IGNORAR]]
    assert medidas == {"ejemplos": 2, "fichas_reales": 6, "fichas_con_relleno": 8, "fichas_etiquetadas": 3}
    with pytest.raises(ValueError):
        colar([{"input_ids": [1, 2], "labels": [1]}], 0)
    with pytest.raises(ValueError):
        colar([], 0)


def test_cortar_respuesta():
    assert cortar_respuesta([4, 6, 7, 1, 1, 1, 99, 5], [99, 98]) == ([4, 6, 7, 1, 1, 1], "token_de_fin")
    assert cortar_respuesta([98], [99, 98]) == ([], "token_de_fin")
    assert cortar_respuesta([1, 2, 3], [99]) == ([1, 2, 3], "limite_de_tokens")


def test_predicciones_solo_id_y_respuesta():
    datos = serializar_predicciones([("10", "467111"), ("11", " 46711x\ncódigo")])
    lineas = datos.decode("utf-8").split("\n")
    assert lineas[-1] == ""
    regs = [json.loads(x) for x in lineas[:-1]]
    assert [tuple(r) for r in regs] == [("id", "respuesta"), ("id", "respuesta")]
    assert regs[1]["respuesta"] == " 46711x\ncódigo"
    with pytest.raises(ValueError):
        serializar_predicciones([("x1", "467111")])


# --- Privacidad ---


def test_tapar_nombres():
    texto = "Traceback: KeyError en 'Fonda Ficticia Dos' cerca de Taller Inventado Uno"
    tapado = tapar_nombres(texto, NOMBRES + ["AB", "  "])
    assert not any(n in tapado for n in NOMBRES)
    assert tapado.count("[nombre]") == 2
    assert "Traceback" in tapado  # los nombres de 1 o 2 letras no se tapan


def test_tapar_llaves():
    llaves = frozenset({"FONDA FICTICIA DOS", "ESTETICA IMAGINARIA"})
    obj = {
        "a": "fonda   ficticia, dos",
        "b": ["ok", {"c": "Estética Imaginaria"}],
        "d": 467111,
        "e": "467111",
        "f": "Qwen/Qwen3-0.6B",
    }
    limpio, rutas = tapar_llaves(obj, llaves)
    assert limpio["a"] == hk.TAPADO and limpio["b"][1]["c"] == hk.TAPADO
    assert limpio["d"] == 467111 and limpio["e"] == "467111" and limpio["f"] == "Qwen/Qwen3-0.6B"
    assert rutas == ["$.a", "$.b[1].c"]
    assert obj["a"] == "fonda   ficticia, dos"  # el original no cambia


def test_tabla_etiquetas():
    lineas = tabla_etiquetas(["<|im_start|>", "4", "<|im_end|>"], [151644, 19, 151645], [IGNORAR, 19, 151645])
    assert lineas[0].split() == ["i", "id", "etiqueta", "token"]
    assert lineas[1].split()[2] == "-100"
    assert lineas[3].endswith("'<|im_end|>'")


def test_registro_escribe_tras_cada_etapa_y_tapa(tmp_path):
    ruta = tmp_path / "humo_mediciones.json"
    reg = Registro(ruta, t_arranque=0.0)
    reg.llaves = frozenset({"TALLER INVENTADO UNO"})
    reg.nombres = tuple(NOMBRES)
    with reg.etapa("uno"):
        reg.datos["x"] = "Taller Inventado Uno"
    escrito = json.loads(ruta.read_text(encoding="utf-8"))
    assert "uno" in escrito["duraciones_s"] and escrito["x"] == hk.TAPADO and escrito["tapados"] == ["$.x"]
    assert reg.datos["x"] == "Taller Inventado Uno"  # solo se tapa lo escrito
    with pytest.raises(KeyError):
        with reg.etapa("dos"):
            raise KeyError("Fonda Ficticia Dos")
    assert "dos" in json.loads(ruta.read_text(encoding="utf-8"))["duraciones_s"]
    reg.falla("tres", KeyError("Fonda Ficticia Dos"))
    assert "Fonda" not in reg.datos["fallas"][0]["error"]
    assert not (tmp_path / "humo_mediciones.json.tmp").exists()


# --- Tensores ---

DTYPE_BYTES = {"BF16": 2, "F32": 4}


def _safetensors(ruta: Path, tensores: dict[str, tuple[str, list[int], bytes]]) -> None:
    """Safetensors mínimo: cabecera JSON con offsets y los bytes de cada tensor."""
    cab, datos, pos = {"__metadata__": {"format": "pt"}}, b"", 0
    for nombre, (dtype, forma, b) in tensores.items():
        cab[nombre] = {"dtype": dtype, "shape": forma, "data_offsets": [pos, pos + len(b)]}
        datos += b
        pos += len(b)
    h = json.dumps(cab).encode("utf-8")
    ruta.write_bytes(struct.pack("<Q", len(h)) + h + datos)


def _par(tmp_path, cambios_espejo=None, lm_head=b"\x01\x02\x03\x04"):
    emb = b"\x01\x02\x03\x04"
    oficial = {
        "model.embed_tokens.weight": ("BF16", [2, 1], emb),
        "model.layers.0.mlp.down_proj.weight": ("BF16", [1, 2], b"\x09\x08\x07\x06"),
        "lm_head.weight": ("BF16", [2, 1], lm_head),
    }
    espejo = {k: v for k, v in oficial.items() if k != "lm_head.weight"}
    espejo.update(cambios_espejo or {})
    _safetensors(tmp_path / "oficial.safetensors", oficial)
    _safetensors(tmp_path / "espejo.safetensors", espejo)
    return hashes_safetensors([tmp_path / "oficial.safetensors"]), hashes_safetensors([tmp_path / "espejo.safetensors"])


def test_hashes_safetensors(tmp_path):
    o, _ = _par(tmp_path)
    assert o["model.embed_tokens.weight"] == {
        "dtype": "BF16", "shape": [2, 1], "sha256": hashlib.sha256(b"\x01\x02\x03\x04").hexdigest()
    }
    assert set(o) == {"model.embed_tokens.weight", "model.layers.0.mlp.down_proj.weight", "lm_head.weight"}


def test_tensores_identicos_con_lm_head_atado(tmp_path):
    o, e = _par(tmp_path)
    r = comparar_tensores(o, e, True, True)
    assert r["identicos"] and r["solo_oficial"] == ["lm_head.weight"] and r["distintos"] == []
    assert r["solo_un_lado_aceptado_por_atadura"]


def test_tensores_sin_atadura_no_son_identicos(tmp_path):
    o, e = _par(tmp_path)
    assert not comparar_tensores(o, e, True, False)["identicos"]
    assert not comparar_tensores(o, e, False, True)["identicos"]


def test_lm_head_distinto_de_embeddings(tmp_path):
    o, e = _par(tmp_path, lm_head=b"\x00\x00\x00\x00")
    assert not comparar_tensores(o, e, True, True)["identicos"]


def test_tensor_distinto_se_lista(tmp_path):
    o, e = _par(tmp_path, {"model.layers.0.mlp.down_proj.weight": ("BF16", [1, 2], b"\x09\x08\x07\x00")})
    r = comparar_tensores(o, e, True, True)
    assert not r["identicos"] and r["distintos"] == ["model.layers.0.mlp.down_proj.weight"]


def test_tensor_de_mas_en_el_espejo(tmp_path):
    o, e = _par(tmp_path, {"model.extra.weight": ("F32", [1], b"\x00\x00\x00\x00")})
    r = comparar_tensores(o, e, True, True)
    assert not r["identicos"] and r["solo_espejo"] == ["model.extra.weight"]


def test_forma_distinta(tmp_path):
    o, e = _par(tmp_path, {"model.layers.0.mlp.down_proj.weight": ("BF16", [2, 1], b"\x09\x08\x07\x06")})
    assert not comparar_tensores(o, e, True, True)["identicos"]


def test_safetensors_truncado(tmp_path):
    _par(tmp_path)
    ruta = tmp_path / "oficial.safetensors"
    ruta.write_bytes(ruta.read_bytes()[:-2])
    with pytest.raises(AbortoHumo):
        hashes_safetensors([ruta])
