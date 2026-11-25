"""Pruebas del pre-registro de la Fase 4 (paso 14).

Solo usan nombres, llaves, ids y códigos inventados en carpetas temporales; nunca leen el DENUE
ni los splits reales.
"""

import copy
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pytest

import src.preregistro_fase4 as pf
from src.evaluador_clasificador import SECTORES_2023
from src.exploracion_clasificador import ErrorPrivacidad, mapa_sectores
from src.preregistro_fase4 import (
    NUEVAS_AFINADOS,
    NUEVAS_BASE,
    ErrorPreregistro,
    calcular,
    comprobar_config,
    cota_fichas,
    elegir_fewshot,
    leer_conjuntos,
    limite_largo,
    parsear_jsonl,
    privacidad_config,
    prompt_fewshot,
    sector_de_prefijo,
    sha256_ids,
)
from src.normalizacion import normalizar_nombre
from src.prompt_clasificador import ErrorPrompt, prompt_generacion

SECTOR_DE = sector_de_prefijo(SECTORES_2023)
CAMPOS = ["id", "nom_estab", "codigo_act", "llave", "sin_nombre", "generico", "igual_titulo"]
CODIGOS_TRAIN = ("461110", "811111", "722511", "311110", "321111", "541110", "431110", "221110", "332110")


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _reg(i: int, nombre: str, codigo: str, **extra) -> dict:
    return {"id": str(i), "nom_estab": nombre, "codigo_act": codigo, "llave": normalizar_nombre(nombre),
            "sin_nombre": False, "generico": False, "igual_titulo": False, **extra}


def _jsonl(regs) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def _repo(tmp_path: Path, kaggle_codigo_cambiado: bool = False):
    """Repo de juguete: los cuatro conjuntos, la copia de Kaggle de train final y su manifest."""
    raiz = tmp_path / "repo"
    (raiz / "data" / "splits").mkdir(parents=True)
    train = [_reg(10 + 3 * i, f"Taller inventado {i}", CODIGOS_TRAIN[i % len(CODIGOS_TRAIN)]) for i in range(60)]
    conjuntos = {
        "train_final.jsonl": ("clasificador_train_final", train, CAMPOS),
        "val.jsonl": ("clasificador_validacion", [_reg(1000 + i, f"Café inventado {i}", "461110") for i in range(6)], CAMPOS),
        "test.jsonl": ("clasificador_test", [_reg(5000 + i, f"Prueba inventada número {i}", "811111") for i in range(5)], CAMPOS),
        "hard.jsonl": ("clasificador_hard", [_reg(6000 + i, f"Difícil {i}", "722511", tipo="ambiguo") for i in range(3)],
                       [*CAMPOS, "tipo"]),
    }
    splits = {}
    for archivo, (entrada, regs, campos) in conjuntos.items():
        datos = _jsonl(regs)
        (raiz / "data" / "splits" / archivo).write_bytes(datos)
        splits[entrada] = {"ruta": f"data/splits/{archivo}", "sha256": _sha(datos), "filas": len(regs), "campos": campos}
    splits["clasificador_hard"]["por_tipo"] = {"ambiguo": 3}
    copia = [{"id": r["id"], "nom_estab": r["nom_estab"], "codigo_act": r["codigo_act"]} for r in train]
    if kaggle_codigo_cambiado:
        for r in copia:
            r["codigo_act"] = "541110" if r["codigo_act"] == "461110" else r["codigo_act"]
    carpeta = tmp_path / "kaggle-datasets" / "denue-scian-clasificador"
    carpeta.mkdir(parents=True)
    datos = _jsonl(copia)
    (carpeta / "train_final.jsonl").write_bytes(datos)
    manifest = {
        "splits": splits,
        "kaggle_datasets": {"denue_scian_clasificador": {"archivos": {"train_final.jsonl": {
            "ruta": "../kaggle-datasets/denue-scian-clasificador/train_final.jsonl", "sha256": _sha(datos)}}}},
    }
    return raiz, manifest


# --- Funciones puras ---


def test_sector_de_prefijo_igual_al_del_evaluador_y_rechaza_prefijos_compartidos():
    assert SECTOR_DE == mapa_sectores(SECTORES_2023)
    assert SECTOR_DE["32"] == "31-33" and SECTOR_DE["49"] == "48-49" and SECTOR_DE["46"] == "46"
    with pytest.raises(ErrorPreregistro, match="prefijo"):
        sector_de_prefijo(["31-33", "32"])


def test_elegir_fewshot_sigue_la_permutacion_y_agrupa_sectores():
    codigos = list(CODIGOS_TRAIN) * 7
    pos = elegir_fewshot(codigos, SECTOR_DE)
    # Reimplementación directa de la regla.
    vistos, esperadas = set(), []
    for i in np.random.default_rng(42).permutation(len(codigos)):
        s = SECTOR_DE[codigos[i][:2]]
        if s not in vistos and len(esperadas) < 5:
            vistos.add(s)
            esperadas.append(int(i))
    assert pos == esperadas  # en el orden de elección, no ordenadas
    sectores = [SECTOR_DE[codigos[p][:2]] for p in pos]
    assert len(set(sectores)) == 5
    # 31, 32 y 33 son el mismo sector: con solo esos tres y otros dos sectores no alcanza para 5.
    with pytest.raises(ErrorPreregistro, match="3 sectores"):
        elegir_fewshot(["311110", "321111", "332110", "461110", "811111"] * 3, SECTOR_DE)
    with pytest.raises(ErrorPreregistro, match="seis dígitos"):
        elegir_fewshot(["46111", "811111"], SECTOR_DE, k=2)
    assert elegir_fewshot(codigos, SECTOR_DE, semilla=7) != pos  # la semilla importa


def test_sha256_ids():
    assert sha256_ids(["12", "3"]) == hashlib.sha256(b"12\n3\n").hexdigest()
    assert sha256_ids(["3", "12"]) != sha256_ids(["12", "3"])  # el orden cuenta
    for malo in ([], ["12a"], ["١٢"], [12]):
        with pytest.raises(ErrorPreregistro):
            sha256_ids(malo)


def test_prompt_fewshot_turnos_sin_think_y_termina_en_el_prompt_de_generacion():
    ejemplos = [("Tlapalería Inventada", "467111"), ("Fonda Inventada", "722511")]
    texto = prompt_fewshot(ejemplos, "Papelería Inventada")
    assert texto == (
        "<|im_start|>user\n" + "Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. "
        "Responde solo con el código de 6 dígitos, sin texto adicional.\nNombre: Tlapalería Inventada<|im_end|>\n"
        "<|im_start|>assistant\n467111<|im_end|>\n"
        "<|im_start|>user\n" + "Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. "
        "Responde solo con el código de 6 dígitos, sin texto adicional.\nNombre: Fonda Inventada<|im_end|>\n"
        "<|im_start|>assistant\n722511<|im_end|>\n"
        + prompt_generacion("Papelería Inventada")
    )
    assert texto.count("<think>") == 1 and texto.endswith("<think>\n\n</think>\n\n")
    with pytest.raises(ErrorPreregistro):
        prompt_fewshot([], "Papelería Inventada")
    with pytest.raises(ErrorPrompt):
        prompt_fewshot([("Fonda Inventada", "72251")], "Papelería Inventada")


def test_cota_fichas_bytes_utf8_y_forma_nfc():
    assert cota_fichas("abc") == 3
    assert cota_fichas("Café Ñandú") == len("Café Ñandú".encode("utf-8")) == 13
    descompuesto = unicodedata.normalize("NFD", "é")  # 3 bytes; su NFC tiene 2
    assert cota_fichas(descompuesto) == 3
    assert cota_fichas("<|im_end|>") == 10  # un token especial: 1 ficha, 10 bytes


def test_limite_largo():
    assert limite_largo(1526, NUEVAS_BASE, 64) == 1600
    assert limite_largo(32, 32, 64) == 64  # un múltiplo exacto se queda
    assert limite_largo(33, 32, 64) == 128
    with pytest.raises(ErrorPreregistro):
        limite_largo(0, 32, 64)


def test_parsear_jsonl_rechaza_formatos_malos():
    bueno = _jsonl([{"id": "1", "nom_estab": "A"}, {"id": "2", "nom_estab": "B"}])
    assert parsear_jsonl(bueno, ["id", "nom_estab"])[1]["id"] == "2"
    casos = {
        "BOM": b"\xef\xbb\xbf" + bueno,
        "salto": bueno[:-1],
        "claves": _jsonl([{"nom_estab": "A", "id": "1"}]),
        "dígitos": _jsonl([{"id": "1a", "nom_estab": "A"}]),
        "ordenado": _jsonl([{"id": "2", "nom_estab": "B"}, {"id": "1", "nom_estab": "A"}]),
        "JSON": b"{no es json}\n",
    }
    for motivo, datos in casos.items():
        with pytest.raises(ErrorPreregistro, match=motivo):
            parsear_jsonl(datos, ["id", "nom_estab"])


# --- Cálculo sobre el repo de juguete ---


def test_leer_conjuntos_descarta_codigo_y_tipo_de_test_y_hard(tmp_path):
    raiz, manifest = _repo(tmp_path)
    regs = leer_conjuntos(manifest, raiz)
    for conj in ("test", "hard"):
        assert all(tuple(r) == ("id", "nom_estab", "llave") for r in regs[conj])
    assert "codigo_act" in regs["train_final"][0] and "codigo_act" in regs["validacion"][0]
    ruta = raiz / "data" / "splits" / "test.jsonl"
    ruta.write_bytes(ruta.read_bytes().replace(b"811111", b"811112", 1))
    with pytest.raises(ErrorPreregistro, match="SHA-256"):
        leer_conjuntos(manifest, raiz)


def test_calcular_coincide_llaves_fuera_y_limite(tmp_path):
    raiz, manifest = _repo(tmp_path)
    calc = calcular(manifest, raiz, SECTORES_2023)
    assert calc.coinciden
    pub = calc.publico()
    assert pub["seleccion"]["kaggle"] == pub["seleccion"]["local"]
    assert len(set(pub["seleccion"]["kaggle"]["sectores"])) == 5
    assert pub["llaves_ejemplos_en"] == {"validacion": 0, "test": 0, "hard": 0}
    assert pub["llave_kaggle_igual_a_local"] is True
    assert pub["llaves_reservadas_en_train"] == {"test": 0, "hard": 0}
    # La cota máxima es la de la fila de nombre más largo de test, con el bloque de 5 ejemplos.
    sel = calc.seleccion["kaggle"]
    ejemplos = list(zip(sel.nombres, sel.codigos))
    esperada = cota_fichas(prompt_fewshot(ejemplos, "Prueba inventada número 4"))
    assert pub["cota_maxima"]["test"] == esperada
    assert pub["limite_fewshot"] == limite_largo(esperada, NUEVAS_BASE)
    cotas0 = [cota_fichas(prompt_generacion(f"Prueba inventada número {i}")) for i in range(5)]
    assert pub["cero_shot_confirmadas"]["test"] == {
        "filas": 5,
        "base_cota_mas_32_hasta_256": sum(c + 32 <= 256 for c in cotas0),
        "afinados_cota_mas_16_hasta_256": sum(c + 16 <= 256 for c in cotas0),
    }
    texto = json.dumps(pub, ensure_ascii=False)
    for nombre in ("Taller inventado", "Prueba inventada", "Difícil", "TALLER", "5000"):
        assert nombre not in texto  # lo público no trae nombres, llaves ni ids


def test_calcular_se_detiene_si_las_copias_no_coinciden(tmp_path):
    raiz, manifest = _repo(tmp_path, kaggle_codigo_cambiado=True)
    calc = calcular(manifest, raiz, SECTORES_2023)
    assert not calc.coinciden
    assert calc.limite_fewshot == 0 and not calc.cota_maxima  # no se calcula nada más
    assert comprobar_config({}, calc, manifest, raiz, {}) == [
        "La selección few-shot no coincide entre la copia de Kaggle y data/splits: PARADA."
    ]


# --- Comprobación del config ---


def _config(calc, manifest, raiz: Path) -> dict:
    """Config mínimo que cuadra con el repo de juguete."""
    adaptador = raiz / "kaggle-output" / "adaptador.safetensors"
    adaptador.parent.mkdir(parents=True)
    adaptador.write_bytes(b"pesos inventados")
    resultado = raiz / "results" / "afinado.json"
    resultado.parent.mkdir(parents=True)
    resultado.write_text(json.dumps({"modelo": {"adaptador_sha256": _sha(b"pesos inventados")}}), encoding="utf-8")
    referencia = raiz / "kaggle-output" / "predicciones.jsonl"
    referencia.write_bytes(b'{"id": "1000", "respuesta": "461110"}\n')
    base06 = {"repo": "Qwen/Qwen3-0.6B", "revision": "a" * 40}
    base4 = {"repo": "Qwen/Qwen3-4B", "revision": "b" * 40}
    pub = calc.publico()
    s = pub["seleccion"]["kaggle"]
    nombres = {"train_final": "clasificador_train_final", "validacion": "clasificador_validacion",
               "test": "clasificador_test", "hard": "clasificador_hard"}
    llm = [
        {"id": "afinado-06b", "tamano": "0.6B", "base": base06,
         "adaptador": {"ruta": "kaggle-output/adaptador.safetensors", "sha256": _sha(b"pesos inventados")},
         "predicciones_validacion": {"ruta": "kaggle-output/predicciones.jsonl",
                                     "sha256": _sha(b'{"id": "1000", "respuesta": "461110"}\n')},
         "resultado_validacion": "results/afinado.json"},
        {"id": "afinado-4b", "tamano": "4B", "base": base4},
        {"id": "base-06b-0", "tamano": "0.6B", "base": base06},
        {"id": "base-4b-0", "tamano": "4B", "base": base4},
        {"id": "base-06b-f", "tamano": "0.6B", "base": base06},
        {"id": "base-4b-f", "tamano": "4B", "base": base4},
    ]
    return {
        "registro": {"commit_codigo": "pendiente"},
        "conjuntos": {
            n: {"entrada_manifest": e, "sha256": manifest["splits"][e]["sha256"], "filas": manifest["splits"][e]["filas"]}
            for n, e in nombres.items()
        } | {"hard": {"entrada_manifest": "clasificador_hard", "sha256": manifest["splits"]["clasificador_hard"]["sha256"],
                      "filas": 3, "por_tipo": {"ambiguo": 3}}},
        "few_shot": {"k": 5, "semilla": 42, "sectores_oficiales": list(SECTORES_2023),
                     "fuente": {"sha256": calc.kaggle_sha256}, "sha256_ids": s["sha256_ids"], "codigos": s["codigos"],
                     "sectores": s["sectores"], "llaves_fuera_de_val_test_hard": True},
        "largo_maximo": {"general": 256, "few_shot": pub["limite_fewshot"],
                         "cota_maxima_few_shot": max(pub["cota_maxima"].values()), "nuevas_base": NUEVAS_BASE,
                         "nuevas_afinados": NUEVAS_AFINADOS, "multiplo": 64},
        "modelos": [{"id": "diccionario"}, {"id": "tfidf"}, *llm],
        "no_van_a_test": [{"modelo": f"descartado-{i}"} for i in range(7)],
        "orden_en_el_kernel": [m["id"] for m in llm],
        "comparaciones": {
            "principales": {"pares": [{"a": "tfidf", "b": "afinado-06b"}]},
            "secundarias": {"pares": [{"a": "base-06b-0", "b": "afinado-06b"}]},
            "secundarias_sector": {"pares": [{"a": "tfidf", "b": "afinado-06b"}, {"a": "base-06b-0", "b": "afinado-06b"}]},
            "hard": {"pares": ["principales", "secundarias", "secundarias_sector"]},
            "nota": "texto",
        },
        "uso_unico": {"reproduccion_validacion_en_el_kernel": {
            "modelos": ["afinado-06b"],
            "criterio_de_paro": {
                "comparacion": "respuesta cruda idéntica por id", "minimo_identicas": UMBRAL_JUGUETE,
                "filas": manifest["splits"]["clasificador_validacion"]["filas"], "si_cumple": "sigue",
                "si_no_cumple": "no corre", "validacion_incompleta": "no cumple",
            },
        }},
    }, {"sectores": SECTORES_2023, "modelos_base": {"0.6B": base06, "4B": base4}}


UMBRAL_JUGUETE = 5  # de las 6 filas de validación del repo de juguete


def _criterio(c):
    return c["uso_unico"]["reproduccion_validacion_en_el_kernel"]["criterio_de_paro"]


def test_comprobar_config_cuadra_y_detecta_cada_cambio(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "UMBRAL_REPRODUCCION", UMBRAL_JUGUETE)
    raiz, manifest = _repo(tmp_path)
    calc = calcular(manifest, raiz, SECTORES_2023)
    config, refs = _config(calc, manifest, raiz)
    assert comprobar_config(config, calc, manifest, raiz, refs) == []
    cambios = {
        "few_shot.sha256_ids": lambda c: c["few_shot"].update(sha256_ids="0" * 64),
        "few_shot.codigos": lambda c: c["few_shot"]["codigos"].reverse(),
        "few_shot.sectores_oficiales": lambda c: c["few_shot"]["sectores_oficiales"].pop(),
        "few_shot.fuente": lambda c: c["few_shot"]["fuente"].update(sha256="1" * 64),
        "largo_maximo.few_shot": lambda c: c["largo_maximo"].update(few_shot=256),
        "largo_maximo.general": lambda c: c["largo_maximo"].update(general=512),
        "conjuntos.test": lambda c: c["conjuntos"]["test"].update(filas=4),
        "por_tipo": lambda c: c["conjuntos"]["hard"].update(por_tipo={"ambiguo": 2, "indirecto": 1}),
        "8 modelos": lambda c: c["modelos"].pop(0),
        "revisión fijada": lambda c: c["modelos"][3].update(base={"repo": "Qwen/Qwen3-4B", "revision": "c" * 40}),
        "adaptador: el SHA-256": lambda c: c["modelos"][2]["adaptador"].update(sha256="2" * 64),
        "commit": lambda c: c["registro"].update(commit_codigo="no-es-commit"),
        "no_van_a_test": lambda c: c["no_van_a_test"].pop(),
        "no_van_a_test ": lambda c: c["no_van_a_test"].__setitem__(0, {"modelo": "tfidf"}),  # uno de los 8
        "comparaciones.principales": lambda c: c["comparaciones"]["principales"]["pares"].append(
            {"a": "tfidf", "b": "no-existe"}),
        "comparaciones.hard": lambda c: c["comparaciones"]["hard"]["pares"].append("no-existe"),
        "comparaciones.hard.pares": lambda c: c["comparaciones"]["hard"].update(pares="Los mismos de principales."),
        "secundarias_sector.pares": lambda c: c["comparaciones"]["secundarias_sector"]["pares"].reverse(),
        "reproduccion_validacion_en_el_kernel no está": lambda c: c.pop("uso_unico"),
        "criterio_de_paro debe tener": lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"].update(
            criterio_de_paro=None),
        "criterio_de_paro debe tener exactamente": lambda c: _criterio(c).pop("validacion_incompleta"),
        "criterio_de_paro: minimo_identicas": lambda c: _criterio(c).update(minimo_identicas=UMBRAL_JUGUETE - 1),
        "criterio_de_paro: minimo_identicas ": lambda c: _criterio(c).update(filas=7),
        "criterio_de_paro: minimo_identicas  ": lambda c: _criterio(c).update(minimo_identicas=float(UMBRAL_JUGUETE)),
        "criterio_de_paro: [": lambda c: _criterio(c).update(si_no_cumple=" "),
        "kernel.modelos": lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"].update(modelos=["base-06b-0"]),
        "kernel.modelos ": lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"].update(
            modelos=["afinado-06b", "afinado-06b"]),
        "kernel.modelos debe": lambda c: c["uso_unico"]["reproduccion_validacion_en_el_kernel"].update(modelos=[]),
    }
    for motivo, cambiar in cambios.items():
        c = copy.deepcopy(config)
        cambiar(c)
        problemas = comprobar_config(c, calc, manifest, raiz, refs)
        assert any(motivo.split(":")[0] in p for p in problemas), motivo


def test_privacidad_config_rechaza_una_llave(tmp_path):
    raiz, manifest = _repo(tmp_path)
    calc = calcular(manifest, raiz, SECTORES_2023)
    privacidad_config({"nota": "texto del config", "codigo": "461110"}, calc.llaves)
    with pytest.raises(ErrorPrivacidad):
        privacidad_config({"nota": "prueba inventada número 2"}, calc.llaves)
