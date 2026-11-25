"""Pruebas de src/llama_servidor.py con un llama-server falso (http.server en un hilo); solo datos inventados."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src import llama_servidor as ls

IM_END, ENDOFTEXT = 151645, 151643


class _Falso(BaseHTTPRequestHandler):
    respuesta_completion: dict = {}
    ultimo_cuerpo: dict = {}

    def log_message(self, *a):  # sin ruido en la salida de pytest
        pass

    def _json(self, codigo: int, obj: dict) -> None:
        datos = json.dumps(obj).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        self._json(200, {"status": "ok"}) if self.path == "/health" else self._json(404, {})

    def do_POST(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        type(self).ultimo_cuerpo = cuerpo
        if self.path == "/tokenize":
            especiales = {"<|im_end|>": [IM_END], "<|endoftext|>": [ENDOFTEXT]}
            self._json(200, {"tokens": especiales.get(cuerpo["content"], [1, 2, 3])})
        elif self.path == "/detokenize":
            self._json(200, {"content": "".join(str(t) for t in cuerpo["tokens"])})
        elif self.path == "/completion":
            self._json(200, type(self).respuesta_completion)
        else:
            self._json(404, {})


@pytest.fixture()
def servidor():
    srv = ThreadingHTTPServer((ls.HOST, 0), _Falso)
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def test_salud_tokenizar_detokenizar(servidor):
    c = ls.Cliente(servidor)
    assert c.salud()
    assert c.tokenizar("Nombre inventado") == [1, 2, 3]
    assert c.detokenizar([4, 6]) == "46"
    assert ls.ids_fin(c) == [IM_END, ENDOFTEXT]


def test_completar_manda_greedy_y_mide(servidor):
    _Falso.respuesta_completion = {"tokens": [1, IM_END], "stop_type": "eos", "content": "x"}
    r = ls.Cliente(servidor).completar("prompt inventado", 16)
    cuerpo = _Falso.ultimo_cuerpo
    assert cuerpo["n_predict"] == 16 and cuerpo["temperature"] == 0.0 and cuerpo["top_k"] == 1
    assert cuerpo["cache_prompt"] is False and cuerpo["return_tokens"] is True
    assert r["ms"] >= 0


def test_completar_sin_fichas_falla(servidor):
    _Falso.respuesta_completion = {"stop_type": "eos", "content": "x"}
    with pytest.raises(ls.ErrorServidor):
        ls.Cliente(servidor).completar("prompt inventado", 16)


def test_cliente_solo_127():
    with pytest.raises(ls.ErrorServidor):
        ls.Cliente(8080, host="0.0.0.0")


@pytest.mark.parametrize("fichas,stop,esperado,visible,tope", [
    ([4, 6, 7, IM_END], "eos", [4, 6, 7], True, False),      # el servidor devuelve el token de fin
    ([4, 6, 7], "eos", [4, 6, 7], False, False),             # no lo devuelve: parada marcada como no visible
    ([4, ENDOFTEXT, 9], "eos", [4], True, False),
    (list(range(16)), "limit", list(range(16)), False, True),  # tope de fichas nuevas
])
def test_respuesta_de(fichas, stop, esperado, visible, tope):
    r = ls.respuesta_de({"tokens": fichas, "stop_type": stop}, [IM_END, ENDOFTEXT])
    assert r["fichas"] == esperado and r["fin_visible"] is visible and r["al_tope"] is tope


def test_respuesta_de_parada_rara():
    with pytest.raises(ls.ErrorServidor):
        ls.respuesta_de({"tokens": [1], "stop_type": "word"}, [IM_END])


def test_exigir_memoria():
    assert ls.exigir_memoria(3072, lambda: 4000) == 4000
    with pytest.raises(ls.MemoriaInsuficiente):
        ls.exigir_memoria(3072, lambda: 2999)


def test_comando_servidor():
    cmd = ls.comando_servidor("llama-server.exe", "m.gguf", 8082, hilos=2, contexto=256)
    assert cmd[:5] == ["llama-server.exe", "-m", "m.gguf", "--port", "8082"]
    assert cmd[cmd.index("-t") + 1] == "2" and cmd[cmd.index("-c") + 1] == "256"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1" and "--offline" in cmd


def test_servidor_no_arranca_con_poca_memoria(tmp_path):
    exe, gguf = tmp_path / "s.exe", tmp_path / "m.gguf"
    exe.write_bytes(b"")
    gguf.write_bytes(b"")
    llamado = []
    s = ls.Servidor(exe, gguf, 1, tmp_path / "log.txt", libre=lambda: 100, popen=lambda *a, **k: llamado.append(1))
    with pytest.raises(ls.MemoriaInsuficiente):
        s.__enter__()
    assert not llamado  # no se lanzó nada
