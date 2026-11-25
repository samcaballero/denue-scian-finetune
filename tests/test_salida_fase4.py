"""Pruebas de la revisión de la salida del kernel de la Fase 4 (src/salida_fase4.py), con datos inventados."""

import json

import pytest

from src import salida_fase4 as sf
from src.evaluar_corrida_clasificador import indice_de_nombres
from src.evaluar_humo import referencias_de_nombres
from src.humo_kaggle import sha256_bytes


def _pred(ids, al_tope=None, respuesta="461110"):
    al_tope = al_tope or {}
    return "".join(json.dumps({"id": i, "respuesta": respuesta, "fichas_prompt": 60, "fichas_generadas": 7,
                               "al_tope": al_tope.get(i, False), "segundos": 0.2}, ensure_ascii=False) + "\n"
                   for i in ids).encode("utf-8")


def _armar(tmp_path, ids=("5", "12", "40")):
    """Una descarga mínima: una etapa completa, su JSON, mediciones con archivos_salida y el log de la CLI."""
    base = tmp_path / "kaggle-output" / "fase4"
    sal = base / sf.SUB
    (sal / "predicciones" / "m1").mkdir(parents=True)
    (sal / "etapas").mkdir()
    (sal / "logs").mkdir()
    datos = _pred(ids)
    (sal / "predicciones" / "m1" / "test.jsonl").write_bytes(datos)
    arch = {"ruta": "predicciones/m1/test.jsonl", "bytes": len(datos), "sha256": sha256_bytes(datos), "filas": len(ids)}
    etapa = {"estado": "completa", "archivo": arch, "inferencia": {"al_tope": 0, "s_por_fila": 0.2, "filas": len(ids)},
             "fallas": []}
    (sal / "etapas" / "m1__test.json").write_text(json.dumps(etapa), encoding="utf-8")
    (sal / "logs" / "m1__test.log").write_text("[fase4-etapa] 3 filas\n", encoding="utf-8")
    archivos = []
    for p in sorted(sal.rglob("*")):
        if p.is_file():
            b = p.read_bytes()
            archivos.append({"ruta": p.relative_to(sal).as_posix(), "bytes": len(b), "sha256": sha256_bytes(b)})
    med = {"estado": "completo", "fallas": [], "archivos_salida": archivos, "segundos_desde_arranque": 100.0,
           "etapas": [{"id": "m1__test", "modelo": "m1", "conjunto": "test", "estado": "completa", "segundos": 60.0,
                       "predicciones": dict(arch)}]}
    (sal / "mediciones.json").write_text(json.dumps(med), encoding="utf-8")
    (base / sf.LOG_CLI).write_text(json.dumps([{"stream_name": "stdout", "time": 1.0, "data": "hola\n"}]), encoding="utf-8")
    return tmp_path, base


def test_cargar_y_verificar_descarga(tmp_path):
    raiz, _ = _armar(tmp_path)
    ent = sf.cargar(raiz)
    etapas_json = {"m1__test": ent.jsons["fase4/etapas/m1__test.json"]}
    d = sf.verificar_descarga(ent.inventario, ent.mediciones, etapas_json, set())
    assert d["faltan"] == d["distintos"] == d["extra"] == []
    assert d["jsonl_de_predicciones"] == d["etapas_completas"] == 1
    assert ent.textos[sf.LOG_CLI] == "hola\n"
    assert [r["id"] for r in ent.predicciones[("m1", "test")]["registros"]] == ["5", "12", "40"]


def test_descarga_con_un_archivo_alterado_es_parada(tmp_path):
    raiz, base = _armar(tmp_path)
    (base / sf.SUB / "logs" / "m1__test.log").write_text("otro\n", encoding="utf-8")
    ent = sf.cargar(raiz)
    with pytest.raises(sf.ErrorSalida, match="distintos 1"):
        sf.verificar_descarga(ent.inventario, ent.mediciones, {"m1__test": ent.jsons["fase4/etapas/m1__test.json"]}, set())


def test_descarga_con_archivo_de_mas_o_hash_prohibido(tmp_path):
    raiz, base = _armar(tmp_path)
    (base / sf.SUB / "logs" / "extra.log").write_text("x\n", encoding="utf-8")
    ent = sf.cargar(raiz)
    ej = {"m1__test": ent.jsons["fase4/etapas/m1__test.json"]}
    with pytest.raises(sf.ErrorSalida, match="extra 1"):
        sf.verificar_descarga(ent.inventario, ent.mediciones, ej, set())
    (base / sf.SUB / "logs" / "extra.log").unlink()
    ent = sf.cargar(raiz)
    prohibido = ent.predicciones[("m1", "test")]["sha256"]
    with pytest.raises(sf.ErrorSalida, match="hash prohibido 1"):
        sf.verificar_descarga(ent.inventario, ent.mediciones, ej, {prohibido})


def test_predicciones_de_la_etapa_distintas_a_su_json(tmp_path):
    raiz, base = _armar(tmp_path)
    ent = sf.cargar(raiz)
    ej = json.loads(json.dumps(ent.jsons["fase4/etapas/m1__test.json"]))
    ej["archivo"]["sha256"] = "0" * 64
    with pytest.raises(sf.ErrorSalida, match="predicciones distintas 1"):
        sf.verificar_descarga(ent.inventario, ent.mediciones, {"m1__test": ej}, set())


def test_leer_predicciones_rechaza_tipos_y_claves():
    bien = _pred(["1", "2"])
    assert len(sf.leer_predicciones(bien)) == 2
    with pytest.raises(sf.ErrorSalida):
        sf.leer_predicciones(b'{"id": "1", "respuesta": "x"}\n')
    malo = bien.replace(b'"al_tope": false', b'"al_tope": 0', 1)
    with pytest.raises(sf.ErrorSalida):
        sf.leer_predicciones(malo)


def test_verificar_ids():
    preds = {("m1", "test"): {"registros": [{"id": "5"}, {"id": "12"}]}}
    assert sf.verificar_ids(preds, {"test": ["12", "5"]}) == {"m1/test": 2}
    with pytest.raises(sf.ErrorSalida):
        sf.verificar_ids(preds, {"test": ["12", "5", "7"]})
    desordenadas = {("m1", "test"): {"registros": [{"id": "12"}, {"id": "5"}]}}
    with pytest.raises(sf.ErrorSalida):
        sf.verificar_ids(desordenadas, {"test": ["12", "5"]})


def test_revisar_nombres_detecta_posible_fuga_y_acepta_cortos_dentro_de_token():
    filas = [{"id": "1", "nom_estab": "Tortillería La Inventada"}, {"id": "2", "nom_estab": "ABC"}]
    idx = indice_de_nombres(referencias_de_nombres(filas))
    limpio = sf.revisar_nombres({"a.log": "paquete XABCX instalado\n"}, {"b.json": {"k": "ok"}}, idx, "")
    assert not limpio["posible_fuga"]
    assert limpio["archivos"]["a.log"]["clasificacion"]["nombres_crudos_cortos_solo_dentro_de_un_token"] == 1
    fuga = sf.revisar_nombres({"a.log": "fila: Tortillería La Inventada\n"}, {}, idx, "")
    assert fuga["posible_fuga"] and fuga["archivos_con_posible_fuga"] == ["a.log"]
    en_json = sf.revisar_nombres({}, {"b.json": {"k": "tortilleria la inventada"}}, idx, "")
    assert en_json["archivos_con_posible_fuga"] == ["b.json (valores)"]


def test_compuertas_recalculadas():
    preds = {("m1", "validacion"): {"registros": [{"id": "1", "respuesta": "461110"}, {"id": "2", "respuesta": "722511"}]}}
    med = {"reproduccion_validacion": {"m1": {"identicas": 1, "cumple": True, "metodo_adaptador": "estado_lora"}}}
    g = sf.compuertas(med, preds, {"m1": {"1": "461110", "2": "311110"}}, minimo=1)["m1"]
    assert g["local_identicas"] == 1 and g["cumple"]
    g = sf.compuertas(med, preds, {"m1": {"1": "461110", "2": "311110"}}, minimo=2)["m1"]
    assert not g["cumple"]


def test_resumen_etapas_al_tope_y_costos():
    preds = {("m1", "test"): {"registros": [{"id": "1", "al_tope": True}, {"id": "2", "al_tope": False}]}}
    med = {"segundos_desde_arranque": 7200.0, "duraciones_s": {"gpu": 1.0, "etapa_m1__test": 3.0},
           "estimaciones": {"etapas": {"m1__test": {"mejor_s": 10, "peor_s": 20}}},
           "etapas": [{"id": "m1__test", "modelo": "m1", "conjunto": "test", "estado": "completa", "segundos": 3600.0}]}
    ej = {"m1__test": {"inferencia": {"al_tope": 1, "pct_al_tope": 50.0, "s_por_fila": 0.5, "filas": 2}}}
    etapas = sf.resumen_etapas(med, ej, preds)
    assert etapas[0]["al_tope_jsonl"] == 1 and etapas[0]["al_tope_coincide"]
    ej["m1__test"]["inferencia"]["al_tope"] = 0
    assert not sf.resumen_etapas(med, ej, preds)[0]["al_tope_coincide"]
    c = sf.costos(med, etapas)
    assert c["horas_gpu_kernel"] == 2.0
    assert c["por_modelo"]["m1"]["horas_gpu_procesos"] == 1.0
    assert c["segundos_fuera_de_etapas"] == 3600.0
    assert c["duraciones_orquestador_s"] == {"gpu": 1.0}
