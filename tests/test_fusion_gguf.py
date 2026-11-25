"""Pruebas de las partes puras de src/fusion_gguf.py (sin torch, peft ni modelos)."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from src import fusion_gguf as fg
from src.gguf_fase4 import REGISTRO, ErrorGguf


def test_git_blob_sha1():
    assert fg.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object


def test_extraer_zip(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("carpeta/archivo.txt", "hola")
    assert fg.extraer_zip(z, tmp_path / "nueva") == ["carpeta/archivo.txt"]
    assert (tmp_path / "nueva" / "carpeta" / "archivo.txt").read_text() == "hola"
    with pytest.raises(ErrorGguf, match="no está vacía"):
        fg.extraer_zip(z, tmp_path / "nueva")
    malo = tmp_path / "malo.zip"
    with zipfile.ZipFile(malo, "w") as f:
        f.writestr("../fuera.txt", "x")
    with pytest.raises(ErrorGguf, match="peligrosa"):
        fg.extraer_zip(malo, tmp_path / "otra")
    assert not (tmp_path / "fuera.txt").exists()


def test_huella_de_carpeta(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a.py").write_bytes(b"x")
    (tmp_path / "b" / "c.py").write_bytes(b"y")
    (tmp_path / "b" / "__pycache__").mkdir()
    (tmp_path / "b" / "__pycache__" / "c.pyc").write_bytes(b"z")
    h = fg.huella_de_carpeta(tmp_path, hash_archivo=lambda p: p.read_text())
    import hashlib

    assert h["archivos"] == 2 and h["sha256"] == hashlib.sha256(b"a.py x\nb/c.py y\n").hexdigest()


def test_comparar_tensores():
    a = {"x": np.ones(3), "y": np.zeros(2)}
    b = {"x": np.ones(3), "y": np.ones(2), "lm_head.weight": np.ones(1)}
    r = fg.comparar_tensores(list(a), list(b), lambda k: bool(np.array_equal(a[k], b[k])))
    assert (r["comunes"], r["distintos"], r["solo_a"], r["solo_b"]) == (2, 1, [], ["lm_head.weight"])


def test_diferencias():
    e = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    o = e.copy()
    o[1, 1] = 4.5
    assert fg.diferencias(e, o) == {"elementos": 4, "distintos": 1, "max_abs": 0.5}
    with pytest.raises(ErrorGguf):
        fg.diferencias(e, o[:1])


def test_claves_y_pesos():
    assert fg.clave_lora_en_modelo("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight") == \
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    with pytest.raises(ErrorGguf):
        fg.clave_lora_en_modelo("model.layers.0.self_attn.q_proj.weight")
    assert fg.peso_de(13, "down_proj") == "model.layers.13.mlp.down_proj.weight"
    assert len(fg.MODULOS) == 7


def test_diferencias_de_config_y_nombre_corto():
    assert fg.diferencias_de_config({"a": 1, "b": 2}, {"a": 1, "b": 3, "c": 4}) == {
        "b": {"espejo": 2, "oficial": 3}, "c": {"espejo": None, "oficial": 4}}
    assert fg.nombre_corto(["C:\\x\\llama-quantize.exe", "C:/m/a.gguf", "Q8_0", "4"]) == [
        "llama-quantize.exe", "a.gguf", "Q8_0", "4"]


def test_actualizar_registro(tmp_path):
    def escribir(datos, rel):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(datos)

    fg.actualizar_registro("llama_cpp", {"etiqueta": "b1"}, raiz=tmp_path, escribir=escribir)
    fg.actualizar_registro("gguf", {"a.gguf": {"bytes": 1}}, raiz=tmp_path, escribir=escribir)
    reg = fg.actualizar_registro("gguf", {"b.gguf": {"bytes": 2}}, raiz=tmp_path, escribir=escribir)
    assert reg["llama_cpp"] == {"etiqueta": "b1"} and set(reg["gguf"]) == {"a.gguf", "b.gguf"}
    assert json.loads((tmp_path / REGISTRO).read_text(encoding="utf-8")) == reg
