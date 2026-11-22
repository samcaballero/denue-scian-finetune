"""Pruebas de las funciones puras del kernel de las corridas del clasificador (Fase 2, paso 9).

Solo nombres, ids y códigos inventados en carpetas temporales; nunca leen el DENUE ni los
splits. Las etapas que necesitan GPU, torch o Unsloth se prueban en la corrida de Kaggle.
"""

import copy
import hashlib
import json
import random
import types
from pathlib import Path

import pytest

from src import clasificador_kaggle as ck
from src import construir_kernel as gen
from src import humo_kaggle as hk
from src.prompt_clasificador import IGNORAR

NOMBRES = ["Taller Inventado Uno", "Fonda Ficticia Dos", "Estética Imaginaria"]


@pytest.fixture()
def corrida():
    return copy.deepcopy(dict(gen.CLASIFICADOR_06B_INICIAL.corrida))


# --- Configuración aprobada ---


def test_corrida_06b_inicial_aprobada(corrida):
    assert corrida["modelo"] == hk.MODELO_06B
    assert corrida["max_seq_length"] == 256
    assert corrida["lora"] == {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "use_gradient_checkpointing": "unsloth",
        "random_state": 42,
    }
    assert corrida["lote"] == 32 and corrida["gradient_accumulation_steps"] == 1
    assert corrida["learning_rate"] == 2e-4 and corrida["epocas"] == 1
    assert corrida["optim"] == "adamw_8bit" and corrida["lr_scheduler_type"] == "linear"
    assert corrida["fraccion_warmup"] == 0.03
    assert corrida["eval_cada_pasos"] == 500 and corrida["lote_eval"] == 32
    assert corrida["semilla"] == 42
    assert ck.validar_corrida(corrida) == corrida
    assert json.dumps(ck.configuracion(corrida))  # serializable


def test_largo_maximo_256_es_el_estandar():
    """256 es el valor por omisión de todas las corridas del clasificador (aprobado en el paso 9)."""
    assert gen.LARGO_MAXIMO_CLASIFICADOR == 256
    assert gen.CORRIDA_POR_OMISION["max_seq_length"] == 256
    assert gen.corrida("x", "y", hk.MODELO_06B, lote=8, learning_rate=1e-4)["max_seq_length"] == 256
    for espec in gen.KERNELS.values():
        if espec.corrida is not None:
            assert espec.corrida["max_seq_length"] == 256, espec.clave


def test_corrida_rechaza_claves_desconocidas():
    with pytest.raises(gen.ErrorKernel):
        gen.corrida("x", "y", hk.MODELO_06B, lote=8, learning_rate=1e-4, max_seq=128)


def test_pasos_y_warmup_de_la_corrida():
    assert ck.pasos_por_epoca(187_350, 32) == 5_855
    assert ck.pasos_warmup(5_855, 0.03) == 176
    assert ck.pasos_por_epoca(64, 32) == 2 and ck.pasos_por_epoca(65, 32) == 3
    assert ck.pasos_por_epoca(65, 32, 2) == 2
    assert ck.pasos_warmup(100, 0) == 0
    for malo in [(0, 32), (10, 0), (10, 2, 0), (10.0, 2)]:
        with pytest.raises(ValueError):
            ck.pasos_por_epoca(*malo)


@pytest.mark.parametrize(
    "mutar",
    [
        lambda c: c.update(semilla=41),
        lambda c: c.update(semilla=True),
        lambda c: c["lora"].update(random_state=7),
        lambda c: c.pop("lote"),
        lambda c: c.update(extra=1),
        lambda c: c["modelo"].update(revision="abc"),
        lambda c: c.update(learning_rate=0),
        lambda c: c.update(max_seq_length=64),
        lambda c: c.update(lote=0),
        lambda c: c.update(lote=True),
        lambda c: c.update(fraccion_warmup=0.6),
        lambda c: c["lora"].update(bias="all"),
        lambda c: c["lora"].update(target_modules=["q_proj", "q_proj"]),
        lambda c: c["lora"].pop("r"),
        lambda c: c.update(nombre=""),
    ],
)
def test_validar_corrida_rechaza(corrida, mutar):
    mutar(corrida)
    with pytest.raises(ValueError):
        ck.validar_corrida(corrida)


# --- Barajado del train ---


def test_argumentos_piden_orden_aleatorio_con_semilla(corrida):
    """El sampler del Trainer debe ser aleatorio con data_seed=42 (accelerate lo vuelve SeedableRandomSampler)."""
    a = ck.argumentos_entrenamiento(corrida, 5_855, False, "salida")
    assert a["train_sampling_strategy"] == "random"
    assert a["seed"] == a["data_seed"] == 42
    assert a["dataloader_drop_last"] is False and a["dataloader_num_workers"] == 0
    assert "group_by_length" not in a and "accelerator_config" not in a
    assert a["warmup_steps"] == 176 and a["eval_steps"] == 500 and a["eval_strategy"] == "steps"
    assert a["prediction_loss_only"] is True and a["remove_unused_columns"] is False
    assert (a["fp16"], a["bf16"]) == (True, False)
    assert a["per_device_train_batch_size"] == a["per_device_eval_batch_size"] == 32
    assert a["logging_steps"] == 1 and a["max_steps"] == -1 and a["save_strategy"] == "no"
    assert ck.argumentos_entrenamiento(corrida, 10, True, "s")["bf16"] is True


def test_el_modulo_no_usa_caminos_de_orden_secuencial():
    """Ni el ciclo propio de Unsloth (`unsloth_train`, con SequentialSampler) ni muestreos no aleatorios."""
    texto = Path(ck.__file__).read_text(encoding="utf-8")
    for prohibido in ("unsloth_train", "SequentialSampler", "group_by_length", '"sequential"', "batch_rebalance"):
        assert prohibido not in texto, prohibido


def test_columnas_indice_en_orden_de_archivo():
    ejemplos = [{"input_ids": [1, 2], "labels": [IGNORAR, 2]}, {"input_ids": [3], "labels": [3]}]
    c = ck.columnas(ejemplos, "train")
    assert c == {"input_ids": [[1, 2], [3]], "labels": [[IGNORAR, 2], [3]], "conjunto": ["train", "train"], "indice": [0, 1]}


def _ejemplo(i, conjunto="train"):
    return {"input_ids": [10 + i, 20, 30], "labels": [IGNORAR, 20, 30], "conjunto": conjunto, "indice": i}


def test_colador_separa_conjuntos_y_registra_indices():
    col = ck.ColadorPorConjunto(lambda v: v, pad_id=0)
    lote = col([_ejemplo(5), _ejemplo(2)])
    assert lote["input_ids"] == [[15, 20, 30], [12, 20, 30]] and set(lote) == {"input_ids", "attention_mask", "labels"}
    col([_ejemplo(0, "val"), _ejemplo(1, "val")])
    col([_ejemplo(7)])
    assert col.indices == {"train": [5, 2, 7], "val": [0, 1]}
    assert [m["ejemplos"] for m in col.lotes["train"]] == [2, 1] and len(col.lotes["val"]) == 1
    with pytest.raises(ValueError):
        col([_ejemplo(1), _ejemplo(2, "val")])
    with pytest.raises(ValueError):
        col([_ejemplo(1, "otro")])


def test_revisar_orden():
    n = 100
    barajado = random.Random(42).sample(range(n), n)
    r = ck.revisar_orden(barajado, n)
    assert r["permutacion_completa"] and r["barajado"] and not r["en_orden_de_id"]
    assert r["sha256_orden"] == hashlib.sha256(",".join(map(str, barajado)).encode()).hexdigest()
    r = ck.revisar_orden(list(range(n)), n)
    assert r["permutacion_completa"] and r["en_orden_de_id"] and not r["barajado"] and r["posiciones_en_su_lugar"] == n
    r = ck.revisar_orden(barajado[:-1] + [barajado[0]], n)
    assert not r["permutacion_completa"] and r["repetidas"] == 1 and r["faltantes"] == 1 and not r["barajado"]
    r = ck.revisar_orden(barajado + [5], n)
    assert r["filas_vistas"] == n + 1 and not r["barajado"]
    assert ck.revisar_orden([0, 1, 200], 3)["fuera_de_rango"] == 1


def test_orden_secuencial():
    assert ck.orden_secuencial([0, 1, 2, 3]) and ck.orden_secuencial([3, 8, 9])
    assert not ck.orden_secuencial([0, 2, 1]) and not ck.orden_secuencial([5])


def _ganchos(tmp_path, lote=4):
    reg = hk.Registro(tmp_path / "mediciones.json", t_arranque=0.0)
    reg.datos["entrenamiento"] = {}
    col = ck.ColadorPorConjunto(lambda v: v, pad_id=0)
    cron = ck.Cronometro(reloj=iter(range(10_000)).__next__)
    mods = types.SimpleNamespace(TrainerCallback=object)
    return reg, col, cron, ck._ganchos(mods, cron, col, reg, pasos=50, lote=lote)


def _pasar(ganchos, col, orden, lote, pasos):
    estado = types.SimpleNamespace(global_step=0, log_history=[])
    ganchos.on_train_begin(None, estado, None)
    for k in range(pasos):
        col([_ejemplo(i) for i in orden[k * lote:(k + 1) * lote]])
        estado.global_step += 1
        ganchos.on_step_end(None, estado, None)
        estado.log_history.append({"step": estado.global_step, "loss": 1.0})


def test_ganchos_abortan_si_el_train_va_en_orden_de_id(tmp_path, capsys):
    reg, col, cron, g = _ganchos(tmp_path)
    with pytest.raises(ck.AbortoCorrida, match="no se está barajando"):
        _pasar(g, col, list(range(200)), 4, ck.PASOS_REVISION_ORDEN)
    assert reg.datos["entrenamiento"]["revision_temprana_del_orden"]["en_orden_creciente"] is True


def test_ganchos_siguen_si_el_train_esta_barajado(tmp_path, capsys):
    reg, col, cron, g = _ganchos(tmp_path)
    _pasar(g, col, random.Random(42).sample(range(200), 200), 4, 50)
    rev = reg.datos["entrenamiento"]["revision_temprana_del_orden"]
    assert rev == {"paso": ck.PASOS_REVISION_ORDEN, "filas_revisadas": ck.PASOS_REVISION_ORDEN * 4, "en_orden_creciente": False}
    assert len(cron.tiempos) == 50 and ck.revisar_orden(col.indices["train"], 200)["barajado"]


def test_ganchos_escriben_la_curva_parcial_al_evaluar(tmp_path, capsys):
    reg, col, cron, g = _ganchos(tmp_path)
    estado = types.SimpleNamespace(global_step=3, log_history=[{"step": 3, "loss": 0.5}, {"step": 3, "eval_loss": 0.7}])
    g.on_evaluate(None, estado, None, metrics={"eval_loss": 0.7})
    escrito = json.loads((tmp_path / "mediciones.json").read_text(encoding="utf-8"))
    assert escrito["entrenamiento"]["paso_actual"] == 3
    assert escrito["entrenamiento"]["curvas"]["validacion"] == [{"paso": 3, "eval_loss": 0.7, "eval_runtime_s": None}]


# --- Mediciones ---


def test_cronometro_descuenta_las_evaluaciones():
    tiempos = iter([0.0, 1.0, 2.0, 7.0, 7.5, 8.5])
    cron = ck.Cronometro(reloj=tiempos.__next__)
    cron.fin_de_evaluacion(0)  # paso 0, antes de entrenar: no cuenta
    cron.inicio()  # 0.0
    cron.fin_de_paso()  # 1.0
    cron.fin_de_paso()  # 2.0
    cron.fin_de_evaluacion(2)  # 7.0: 5 s de evaluación
    cron.fin_de_paso()  # 7.5
    cron.fin_de_paso()  # 8.5
    assert cron.tiempos == [1.0, 1.0, 0.5, 1.0]
    assert cron.evaluaciones == [{"paso": 2, "s": 5.0}]


def test_curvas_del_historial():
    hist = [
        {"eval_loss": 3.0, "eval_runtime": 2.0, "step": 0},
        {"loss": 2.5, "grad_norm": 1.0, "learning_rate": 1e-5, "step": 1},
        {"loss": 2.0, "grad_norm": 0.9, "learning_rate": 2e-5, "step": 2},
        {"eval_loss": 1.5, "eval_runtime": 2.1, "step": 2},
        {"train_runtime": 10.0, "train_loss": 2.2, "step": 2},
    ]
    c = ck.curvas(hist)
    assert [x["paso"] for x in c["train"]] == [1, 2] and c["train"][1]["loss"] == 2.0
    assert c["validacion"] == [
        {"paso": 0, "eval_loss": 3.0, "eval_runtime_s": 2.0},
        {"paso": 2, "eval_loss": 1.5, "eval_runtime_s": 2.1},
    ]


def test_linea_avance():
    linea = ck.linea_avance(250, 5_855, [2.0, 1.0], [0.6, 0.7])
    assert linea.startswith("paso 250 de 5855") and "1.5000" in linea and "0.650 s/paso" in linea and "faltan ~61 min" in linea
    assert ck.linea_avance(1, 10, [], []) == "paso 1 de 10"


# --- Entrada y salida ---


def _entrada(tmp_path, extra=None):
    d = tmp_path / "input" / "denue-scian-clasificador"
    d.mkdir(parents=True)
    (d / "train_final.jsonl").write_bytes(b"a\n")
    (d / "val.jsonl").write_bytes(b"bb\n")
    if extra:
        (tmp_path / "input" / extra).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "input" / extra).write_bytes(b"x")
    return tmp_path / "input"


def test_entrada_con_solo_los_dos_archivos(tmp_path):
    raiz = _entrada(tmp_path)
    listado = ck.listar_entrada(raiz)
    assert listado == [
        {"ruta": "denue-scian-clasificador/train_final.jsonl", "bytes": 2},
        {"ruta": "denue-scian-clasificador/val.jsonl", "bytes": 3},
    ]
    assert ck.revisar_entrada(listado) == {
        "train_final.jsonl": "denue-scian-clasificador/train_final.jsonl",
        "val.jsonl": "denue-scian-clasificador/val.jsonl",
    }


@pytest.mark.parametrize("extra", ["otro/otro.csv", "denue-scian-clasificador/x.txt", "copia/val.jsonl"])
def test_entrada_aborta_con_archivos_de_mas(tmp_path, extra):
    raiz = _entrada(tmp_path, extra)
    with pytest.raises(ck.AbortoCorrida):
        ck.revisar_entrada(ck.listar_entrada(raiz))


def test_entrada_aborta_si_falta_uno():
    with pytest.raises(ck.AbortoCorrida):
        ck.revisar_entrada([{"ruta": "d/train_final.jsonl", "bytes": 1}])


def test_archivos_salida_excluye_mediciones(tmp_path):
    (tmp_path / "adaptador").mkdir()
    (tmp_path / "adaptador" / "a.bin").write_bytes(b"123")
    (tmp_path / "predicciones.jsonl").write_bytes(b"{}\n")
    (tmp_path / "mediciones.json").write_bytes(b"{}")
    (tmp_path / "mediciones.json.tmp").write_bytes(b"{}")
    res = ck.archivos_salida(tmp_path, ("mediciones.json", "mediciones.json.tmp"))
    assert [a["ruta"] for a in res] == ["adaptador/a.bin", "predicciones.jsonl"]
    assert res[0]["sha256"] == hashlib.sha256(b"123").hexdigest()


# --- Privacidad y candado ---


def test_registro_de_la_corrida_tapa_nombres(tmp_path):
    reg = hk.Registro(tmp_path / ck.ARCHIVO_MEDICIONES, t_arranque=0.0)
    reg.llaves = frozenset({"FONDA FICTICIA DOS"})
    reg.nombres = tuple(NOMBRES)
    reg.datos["entrenamiento"] = {"x": "Fonda Ficticia Dos"}
    reg.escribir()
    escrito = (tmp_path / ck.ARCHIVO_MEDICIONES).read_text(encoding="utf-8")
    assert "Fonda" not in escrito and hk.TAPADO in escrito
    reg.falla("entrenamiento", KeyError("Taller Inventado Uno"))
    assert "Taller" not in reg.datos["fallas"][0]["error"]


def test_modulo_sin_nombres_de_archivos_reservados():
    texto = Path(ck.__file__).read_text(encoding="utf-8")
    assert "test.jsonl" not in texto and "hard.jsonl" not in texto
    assert set(ck.ARCHIVOS_ENTRADA) == {"train_final.jsonl", "val.jsonl"}
