"""Pruebas de los baselines en test y hard set (Fase 4, paso 16).

Solo usan nombres, llaves, ids y códigos inventados (el juguete de src/baselines_fase4.py) en carpetas
temporales; nunca leen el DENUE, los splits reales ni las copias reales. Del repo solo se lee
evals/fase4_preregistro.json, que no tiene nombres ni ids.
"""

import builtins
import io
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

import src.baselines as bl
import src.baselines_fase4 as bf
from src.baselines_fase4 import EstadoGit, ErrorFase4
from src.denue import REPO_RAIZ
from src.muestra_humo import ErrorMuestra
from src.splits import sha256_bytes

COMMIT = "a" * 40
OTRO_COMMIT = "b" * 40


class GitFalso:
    """Estado de git por llamada: la primera es la del arranque, la segunda la del paso e."""

    def __init__(self, commits=(COMMIT,), limpios=(True,), versionado=True):
        self.commits, self.limpios, self.versionado, self.n = list(commits), list(limpios), versionado, 0

    def __call__(self) -> EstadoGit:
        i = self.n
        self.n += 1
        return EstadoGit(self.commits[min(i, len(self.commits) - 1)], self.limpios[min(i, len(self.limpios) - 1)],
                         self.versionado)


def _juguete(tmp_path: Path) -> bf.Juguete:
    return bf.armar_juguete(tmp_path / "humo")


def _correr(j: bf.Juguete, *, modo="humo", git=None, memoria=lambda: 12.0, minimo=None, lineas=None, sha=None):
    lineas = [] if lineas is None else lineas
    return bf.correr(modo=modo, raiz=j.raiz, manifest=j.manifest, config_bytes=j.config_bytes,
                     sha_config=sha or sha256_bytes(j.config_bytes), carpeta_salida=j.salida,
                     carpeta_resultados=j.resultados, minimo_libre_gb=minimo, fecha="2026-11-25",
                     avance=lineas.append, git=git or GitFalso(), memoria=memoria, log="baselines_prueba.log")


def _espiar_open(monkeypatch) -> list[Path]:
    abiertos: list[Path] = []
    original = io.open

    def espia(archivo, *args, **kwargs):
        if not isinstance(archivo, int):
            abiertos.append(Path(os.fspath(archivo)).resolve())
        return original(archivo, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", espia)
    monkeypatch.setattr(io, "open", espia)
    return abiertos


def _splits_abiertos(j: bf.Juguete, abiertos: list[Path]) -> list[str]:
    carpeta = (j.raiz / "data" / "splits").resolve()
    return sorted({p.name for p in abiertos if p.parent == carpeta})


def _conjunto_de(j: bf.Juguete):
    filas = j.nombres()
    nombres = {c: {r["nom_estab"] for r in filas[c]} for c in bf.CONJUNTOS}

    def conjunto(lista) -> str:
        return next(c for c, n in nombres.items() if set(lista) == n)

    return conjunto


def _alterar_registro(j: bf.Juguete, id_modelo: str) -> None:
    """Cambia un acierto de validación del registro y ajusta su SHA-256 en el config del juguete."""
    rv = bf.modelo_config(j.config, id_modelo)["registro_validacion"]
    ruta = j.raiz / rv["ruta"]
    reg = json.loads(ruta.read_bytes())
    reg["aciertos"]["clase"][0] = not reg["aciertos"]["clase"][0]
    datos = bl.a_bytes(reg)
    ruta.write_bytes(datos)
    rv["sha256"] = sha256_bytes(datos)
    bf.guardar_config_juguete(j)


def _predicciones(j: bf.Juguete) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(j.salida.glob("*.jsonl"))} if j.salida.exists() else {}


def _resultados(j: bf.Juguete) -> list[Path]:
    return sorted(j.resultados.glob(bf.PATRON_RESULTADO))


# --- Flujo completo ---


def test_flujo_completo_en_orden_y_sin_abrir_test_ni_hard_con_etiquetas(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    conjunto = _conjunto_de(j)
    eventos: list = []
    pd, lr, gm, pt, ch, er = (bl.predecir_diccionario, bl.ajustar_lr, bf.guardar_modelo, bf.predecir_tfidf,
                              bf.comprobar_head, bf.escribir_resultado)
    monkeypatch.setattr(bl, "predecir_diccionario",
                        lambda d, n, llave=None: (eventos.append(("diccionario", conjunto(n))), pd(d, n, llave))[1])
    monkeypatch.setattr(bl, "ajustar_lr", lambda *a, **k: (eventos.append("ajuste"), lr(*a, **k))[1])
    monkeypatch.setattr(bf, "guardar_modelo", lambda *a, **k: (eventos.append("guardado"), gm(*a, **k))[1])
    monkeypatch.setattr(bf, "predecir_tfidf", lambda v, m, n: (eventos.append(("tfidf", conjunto(n))), pt(v, m, n))[1])
    monkeypatch.setattr(bf, "comprobar_head", lambda *a, **k: (eventos.append("head"), ch(*a, **k))[1])
    monkeypatch.setattr(bf, "escribir_resultado", lambda *a, **k: (eventos.append("results"), er(*a, **k))[1])
    abiertos = _espiar_open(monkeypatch)
    reg = _correr(j)
    monkeypatch.undo()

    assert eventos == [("diccionario", "validacion"), ("diccionario", "test"), ("diccionario", "hard"), "ajuste",
                       "guardado", ("tfidf", "validacion"), ("tfidf", "test"), ("tfidf", "hard"), "head", "results"]
    # Train, train antes del tope y validación sí se abren; test y hard con etiquetas, nunca.
    assert _splits_abiertos(j, abiertos) == ["train_candidatos_sin_hard.jsonl", "train_final.jsonl", "val.jsonl"]
    copias = (j.carpeta / "kaggle-datasets" / "denue-scian-evaluacion").resolve()
    assert {copias / "test.jsonl", copias / "hard.jsonl"} <= set(abiertos)

    esperados = {f"{i}_{c}.jsonl" for i in (bf.ID_DICC, bf.ID_TFIDF) for c in bf.CONJUNTOS}
    assert set(_predicciones(j)) == esperados
    assert (j.salida / "tfidf-lr-C10_modelo.joblib").exists() and (j.salida / "tfidf-lr-C10_modelo.json").exists()
    filas = j.nombres()
    for i in (bf.ID_DICC, bf.ID_TFIDF):
        for c in bf.CONJUNTOS:
            regs = [json.loads(x) for x in (j.salida / f"{i}_{c}.jsonl").read_text(encoding="utf-8").splitlines()]
            assert [tuple(r) for r in regs] == [bf.CAMPOS_PREDICCION] * len(regs)
            assert [r["id"] for r in regs] == sorted((r["id"] for r in filas[c]), key=int)
            assert all(re.fullmatch(r"[0-9]{6}", r["respuesta"]) for r in regs)

    [ruta] = _resultados(j)
    assert ruta.name == "2026-11-25_fase4_baselines-predicciones.json"
    res = json.loads(ruta.read_text(encoding="utf-8"))
    for campo in ("fecha", "commit", "modelo", "configuracion", "semilla", "split", "hash_split", "tamano_muestra",
                  "horas_gpu"):
        assert campo in res
    assert res == json.loads(json.dumps(reg)) and res["commit"] == COMMIT and res["head_sin_cambios"] is True and res["horas_gpu"] == 0
    assert res["configuracion"]["preregistro"]["sha256"] == sha256_bytes(j.config_bytes)
    assert set(res["configuracion"]["versiones"]) == {"numpy", "scipy", "scikit-learn", "joblib"}
    for i in (bf.ID_DICC, bf.ID_TFIDF):
        comp = res["reproduccion_validacion"][i]
        assert comp["cumple"] and comp["identicas"] == comp["filas"] == len(filas["validacion"])
        for c in bf.CONJUNTOS:
            datos = (j.salida / f"{i}_{c}.jsonl").read_bytes()
            assert res["predicciones"][i][c]["sha256"] == sha256_bytes(datos)
            assert res["predicciones"][i][c]["filas"] == len(filas[c])
    assert res["modelo_guardado"]["sha256"] == bf.sha256_archivo(j.salida / "tfidf-lr-C10_modelo.joblib")
    assert res["tfidf_ajuste"]["modelo_reusado"] is False and res["tfidf_ajuste"]["commit_entrenamiento"] == COMMIT
    assert "metricas" not in json.dumps(res)  # ninguna métrica de test ni de hard
    assert not list(j.resultados.glob("*.tmp")) and not list(j.salida.glob("*.tmp"))


def test_results_y_log_sin_ids_nombres_ni_llaves(tmp_path):
    j = _juguete(tmp_path)
    lineas: list[str] = []
    _correr(j, lineas=lineas)
    [ruta] = _resultados(j)
    texto = ruta.read_text(encoding="utf-8")
    log = "\n".join(lineas)
    filas = [r for regs in j.nombres().values() for r in regs]
    assert len(filas) > 100
    for r in filas:
        assert f'"{r["id"]}"' not in texto
        assert not re.search(rf"(?<![0-9a-f]){r['id']}(?![0-9a-f])", log)
        for t in (texto, log):
            assert r["nom_estab"] not in t and r["llave"] not in t


def test_el_chequeo_de_ids_detecta_un_id():
    bf.verificar_sin_ids({"a": ["x", 41000003]}, {"41000003"})  # un número no es un texto
    with pytest.raises(ErrorFase4, match=r"\$\.a\[1\]"):
        bf.verificar_sin_ids({"a": ["x", "41000003"]}, {"41000003"})


# --- Paros por validación que no reproduce ---


def test_si_el_diccionario_no_reproduce_no_predice_test_ni_hard(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    _alterar_registro(j, bf.ID_DICC)
    conjunto = _conjunto_de(j)
    vistos, ajustes = [], []
    pd = bl.predecir_diccionario
    monkeypatch.setattr(bl, "predecir_diccionario",
                        lambda d, n, llave=None: (vistos.append(conjunto(n)), pd(d, n, llave))[1])
    monkeypatch.setattr(bl, "ajustar_lr", lambda *a, **k: ajustes.append(1))
    with pytest.raises(ErrorFase4, match="diccionario no reproduce su validación"):
        _correr(j)
    assert vistos == ["validacion"] and not ajustes
    assert _predicciones(j) == {} and not _resultados(j)


def test_si_tfidf_no_reproduce_el_modelo_queda_guardado_y_no_predice_test_ni_hard(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    _alterar_registro(j, bf.ID_TFIDF)
    conjunto = _conjunto_de(j)
    vistos = []
    pt = bf.predecir_tfidf
    monkeypatch.setattr(bf, "predecir_tfidf", lambda v, m, n: (vistos.append(conjunto(n)), pt(v, m, n))[1])
    with pytest.raises(ErrorFase4, match="tfidf-lr-C10 no reproduce su validación"):
        _correr(j)
    assert vistos == ["validacion"]
    assert set(_predicciones(j)) == {f"{bf.ID_DICC}_{c}.jsonl" for c in bf.CONJUNTOS}
    meta = json.loads((j.salida / "tfidf-lr-C10_modelo.json").read_bytes())
    assert meta["sha256"] == bf.sha256_archivo(j.salida / "tfidf-lr-C10_modelo.joblib")
    assert meta["commit_entrenamiento"] == COMMIT and set(meta["versiones"]) == set(bf.PAQUETES_MODELO)
    assert not _resultados(j)


# --- Paso e: HEAD y árbol ---


@pytest.mark.parametrize("git", [GitFalso(commits=(COMMIT, OTRO_COMMIT)), GitFalso(limpios=(True, False))],
                         ids=["head-distinto", "arbol-sucio-al-final"])
def test_si_el_repo_cambia_durante_la_corrida_no_escribe_en_results(tmp_path, git):
    j = _juguete(tmp_path)
    with pytest.raises(ErrorFase4, match="cambió durante la corrida"):
        _correr(j, git=git)
    assert not _resultados(j) and len(_predicciones(j)) == 6


# --- Paros al arrancar o en la verificación, antes de leer splits o de entrenar ---


@pytest.mark.parametrize("git, patron", [
    (GitFalso(limpios=(False,)), "árbol limpio"),
    (GitFalso(versionado=False), "debe estar en git"),
])
def test_arbol_sucio_o_script_sin_versionar_se_detiene_al_arrancar(tmp_path, monkeypatch, git, patron):
    j = _juguete(tmp_path)
    abiertos = _espiar_open(monkeypatch)
    with pytest.raises(ErrorFase4, match=patron):
        _correr(j, git=git)
    monkeypatch.undo()
    assert _splits_abiertos(j, abiertos) == [] and _predicciones(j) == {}


def test_memoria_insuficiente_en_escribir_se_detiene_antes_de_leer_datos(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    abiertos = _espiar_open(monkeypatch)
    lineas: list[str] = []
    with pytest.raises(ErrorFase4) as e:
        _correr(j, modo="escribir", minimo=bf.MINIMO_LIBRE_GB, memoria=lambda: 3.2, lineas=lineas)
    monkeypatch.undo()
    assert str(e.value) == ("Memoria libre 3.2 GB, por debajo del mínimo de 5.0 GB: me detengo antes de leer datos. "
                            "Cierra programas (navegador, editores) hasta tener al menos 5.0 GB libres y vuelve a "
                            "lanzar el mismo comando.")
    assert lineas[-1].endswith("Memoria libre: 3.2 GB (mínimo para lanzar: 5.0 GB; pico estimado ~5.3 GB; si falta "
                               "memoria, Windows usa el archivo de paginación y la corrida solo se hace más lenta).")
    assert _splits_abiertos(j, abiertos) == []
    with pytest.raises(ErrorFase4, match="no se pudo medir"):
        _correr(_juguete(tmp_path / "otro"), modo="escribir", minimo=bf.MINIMO_LIBRE_GB, memoria=lambda: None)


def test_con_memoria_justo_en_el_minimo_si_arranca(tmp_path):
    j = _juguete(tmp_path)
    _correr(j, modo="escribir", minimo=bf.MINIMO_LIBRE_GB, memoria=lambda: 5.0)
    assert len(_resultados(j)) == 1


def test_minimo_de_memoria_aprobado():
    # Decisión de Samuel del 2026-11-25, antes de lanzar: 5.0 GB (el plan tenía 7.0; la laptop no pasa de ~5.6 GB).
    assert bf.MINIMO_LIBRE_GB == 5.0


def test_ya_hay_un_registro_en_results_se_detiene_al_arrancar(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    (j.resultados / "2026-11-24_fase4_baselines-predicciones.json").write_text("{}", encoding="utf-8")
    abiertos = _espiar_open(monkeypatch)
    with pytest.raises(ErrorFase4, match="ya se hizo"):
        _correr(j)
    monkeypatch.undo()
    assert _splits_abiertos(j, abiertos) == []


def test_config_y_registros_se_verifican_antes_de_leer_splits(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    abiertos = _espiar_open(monkeypatch)
    with pytest.raises(ErrorFase4, match="no el esperado"):
        _correr(j, sha="0" * 64)
    # Un registro con otro SHA-256 que el del config.
    rv = bf.modelo_config(j.config, bf.ID_TFIDF)["registro_validacion"]
    ruta = j.raiz / rv["ruta"]
    ruta.write_bytes(ruta.read_bytes() + b" ")
    with pytest.raises(ErrorMuestra, match="SHA-256"):
        _correr(j)
    monkeypatch.undo()
    assert _splits_abiertos(j, abiertos) == []


@pytest.mark.parametrize("cambio, patron", [
    (lambda m: m["configuracion"]["tfidf"].update(min_df=3), "configuración del TF-IDF"),
    (lambda m: m["configuracion"]["regresion_logistica"].update(max_iter=50), "regresión logística"),
    (lambda m: m["configuracion"]["regresion_logistica"].update(C=1.0), "huella"),
])
def test_config_de_tfidf_distinta_de_baselines_se_detiene(tmp_path, cambio, patron):
    j = _juguete(tmp_path)
    cambio(bf.modelo_config(j.config, bf.ID_TFIDF))
    bf.guardar_config_juguete(j)
    with pytest.raises(ErrorFase4, match=patron):
        _correr(j)
    assert _predicciones(j) == {}


def test_el_config_real_es_el_esperado_y_coincide_con_baselines():
    datos = (REPO_RAIZ / bf.RUTA_CONFIG).read_bytes()
    assert sha256_bytes(datos) == bf.SHA_CONFIG
    p = bf.verificar_config(datos, bf.SHA_CONFIG)
    assert p.C == 10.0
    assert p.huella_tfidf == bf.modelo_config(p.config, bf.ID_TFIDF)["configuracion"]["huella_config"]


# --- Modelo guardado: recarga y reúso ---


def test_el_modelo_recargado_predice_igual_que_el_de_memoria(tmp_path):
    j = _juguete(tmp_path)
    split = bl.cargar_split_nombre(j.manifest, j.raiz)
    v, X, _ = bl.ajustar_vectorizador([r["nom_estab"] for r in split.train])
    a = bl.ajustar_lr(X, [r["codigo_act"] for r in split.train], 10.0)
    carpeta = tmp_path / "modelo"
    meta = bf.guardar_modelo(v, a.modelo, carpeta, {"commit_entrenamiento": COMMIT})
    assert meta["sha256"] == bf.sha256_archivo(carpeta / "tfidf-lr-C10_modelo.joblib")
    v2, m2 = bf.cargar_modelo(carpeta, meta)
    nombres = [r["nom_estab"] for r in split.val]
    assert bf.predecir_tfidf(v, a.modelo, nombres) == bf.predecir_tfidf(v2, m2, nombres)
    assert np.array_equal(a.modelo.coef_, m2.coef_) and v.vocabulary_ == v2.vocabulary_
    with pytest.raises(ErrorFase4, match="ya existe"):
        bf.guardar_modelo(v, a.modelo, carpeta, {})
    with pytest.raises(ErrorFase4, match="no es el de su JSON"):
        bf.cargar_modelo(carpeta, {**meta, "sha256": "0" * 64})


def test_relanzar_con_otro_commit_reusa_el_modelo_y_acepta_salidas_identicas(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    _correr(j)
    antes = _predicciones(j)
    _resultados(j)[0].unlink()
    ajustes = []
    monkeypatch.setattr(bl, "ajustar_lr", lambda *a, **k: ajustes.append(1))
    reg = _correr(j, git=GitFalso(commits=(OTRO_COMMIT,)))
    assert not ajustes and _predicciones(j) == antes
    assert reg["commit"] == OTRO_COMMIT
    assert reg["tfidf_ajuste"]["modelo_reusado"] is True and reg["tfidf_ajuste"]["commit_entrenamiento"] == COMMIT


@pytest.mark.parametrize("campo", ["versiones.numpy", "versiones.joblib", "huella_config", "sha256_train_final", "C"])
def test_no_reusa_un_modelo_con_otra_configuracion_datos_o_versiones(tmp_path, campo):
    j = _juguete(tmp_path)
    _correr(j)
    _resultados(j)[0].unlink()
    ruta = j.salida / "tfidf-lr-C10_modelo.json"
    meta = json.loads(ruta.read_bytes())
    if campo.startswith("versiones."):
        meta["versiones"][campo.split(".")[1]] = "0.0.0"
    else:
        meta[campo] = 1.0 if campo == "C" else "0" * 64
    ruta.write_bytes(bl.a_bytes(meta))
    with pytest.raises(ErrorFase4, match=re.escape(f"no coincide en ['{campo}']")):
        _correr(j)


def test_no_reusa_un_modelo_cambiado_ni_incompleto_y_no_sobrescribe_predicciones(tmp_path):
    j = _juguete(tmp_path)
    _correr(j)
    _resultados(j)[0].unlink()
    modelo = j.salida / "tfidf-lr-C10_modelo.joblib"
    original = modelo.read_bytes()
    modelo.write_bytes(original + b"\0")
    with pytest.raises(ErrorFase4, match="no es el de su JSON"):
        _correr(j)
    modelo.write_bytes(original)
    pred = j.salida / f"{bf.ID_DICC}_test.jsonl"
    pred.write_bytes(pred.read_bytes().replace(b'"respuesta": "', b'"respuesta": "9', 1)[:-1])
    with pytest.raises(ErrorFase4, match="otro contenido"):
        _correr(j)
    (j.salida / "tfidf-lr-C10_modelo.json").unlink()
    with pytest.raises(ErrorFase4, match="Solo existe uno"):
        _correr(j)


# --- Punto de control y humo ---


def test_el_punto_de_control_no_escribe_ni_predice_test_ni_hard_y_no_exige_memoria(tmp_path, monkeypatch):
    j = _juguete(tmp_path)
    conjunto = _conjunto_de(j)
    vistos = []
    pd = bl.predecir_diccionario
    monkeypatch.setattr(bl, "predecir_diccionario",
                        lambda d, n, llave=None: (vistos.append(conjunto(n)), pd(d, n, llave))[1])
    lineas: list[str] = []
    res = _correr(j, modo="control", memoria=lambda: 0.5, lineas=lineas, git=GitFalso(limpios=(False,)))
    assert vistos == ["validacion"]
    assert not j.salida.exists() and not _resultados(j)
    assert res["pico_estimado_gb"] > 0 and res["minimo_recomendado_gb"] >= bf.MARGEN_GB
    assert any("no se exige mínimo" in x for x in lineas) and "no se escribió nada" in lineas[-1]


def test_humo_tambien_corre_con_poca_memoria(tmp_path):
    j = _juguete(tmp_path)
    _correr(j, modo="humo", memoria=lambda: 0.5)
    assert len(_resultados(j)) == 1


def test_la_carpeta_del_humo_queda_fuera_del_repo_y_vacia(tmp_path):
    for dentro in (REPO_RAIZ / "results" / "humo", REPO_RAIZ / "data" / "fase4" / "humo", REPO_RAIZ):
        with pytest.raises(ErrorFase4, match="dentro del repo"):
            bf.verificar_carpeta_humo(dentro)
    (tmp_path / "llena").mkdir()
    (tmp_path / "llena" / "x.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ErrorFase4, match="ya tiene archivos"):
        bf.verificar_carpeta_humo(tmp_path / "llena")
    assert bf.verificar_carpeta_humo(tmp_path / "nueva") == (tmp_path / "nueva").resolve()


# --- Bitácora y errores ---


def test_un_error_inesperado_se_registra_sin_su_mensaje():
    lineas: list[str] = []
    try:
        raise KeyError("41000003 Abarrotes único 3")
    except KeyError as e:
        bf.registrar_error(e, lineas.append)
    assert "KeyError" in lineas[0] and not any("41000003" in x or "Abarrotes" in x for x in lineas)
    assert any("test_baselines_fase4.py" in x for x in lineas[1:])
    lineas.clear()
    bf.registrar_error(ErrorFase4("tfidf-lr-C10 no reproduce"), lineas.append)
    assert lineas == ["ME DETENGO: tfidf-lr-C10 no reproduce"]


def test_la_bitacora_escribe_cada_linea_con_hora(tmp_path):
    eco: list[str] = []
    b = bf.Bitacora(tmp_path / "x.log", eco=eco.append)
    b("uno")
    b("dos")
    texto = (tmp_path / "x.log").read_text(encoding="utf-8")
    assert texto.splitlines() == eco == b.lineas
    assert re.fullmatch(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] dos", eco[1])


def test_memoria_libre_se_mide():
    libre = bf.memoria_libre_gb()
    assert libre is None or libre > 0
