"""Pruebas de src/demo.py y scripts/demo.ps1 con llama-server falsos (http.server en un hilo); solo datos inventados."""

from __future__ import annotations

import functools
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src import demo
from src import llama_servidor as ls
from src.gguf_fase4 import DIR_GGUF, LLAMA_SERVER, nombre_gguf
from src.prompt_clasificador import FIN_DE_TURNO, prompt_generacion

IM_END, ENDOFTEXT = 151645, 151643
TITULO_LARGO = ("Título inventado de una clase con un nombre bastante largo para probar que la salida se ajusta "
                "al ancho de la ventana sin cortar palabras a la mitad")
TITULOS = {"467111": "Título inventado de ferretería", "722514": TITULO_LARGO, "111110": "Título inventado corto"}
PS1 = Path(demo.REPO_RAIZ) / "scripts" / "demo.ps1"


class _Falso(BaseHTTPRequestHandler):
    """llama-server falso: responde a cada nombre con el texto de `respuestas`, una ficha por carácter."""
    respuestas: dict = {}
    pedidos: list = []

    def log_message(self, *a):
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
        if self.path == "/tokenize":
            especiales = {"<|im_end|>": [IM_END], "<|endoftext|>": [ENDOFTEXT]}
            self._json(200, {"tokens": especiales.get(cuerpo["content"], [1, 2, 3])})
        elif self.path == "/detokenize":
            self._json(200, {"content": "".join(chr(t) for t in cuerpo["tokens"])})
        elif self.path == "/completion":
            type(self).pedidos.append({"prompt": cuerpo["prompt"], "n_predict": cuerpo["n_predict"]})
            nombre = cuerpo["prompt"].split("Nombre: ", 1)[1].split(FIN_DE_TURNO, 1)[0]
            fichas = [ord(c) for c in type(self).respuestas.get(nombre, "000000")]
            if len(fichas) >= cuerpo["n_predict"]:
                fichas, stop = fichas[:cuerpo["n_predict"]], "limit"
            else:
                fichas, stop = fichas + [IM_END], "eos"
            self._json(200, {"tokens": fichas, "stop_type": stop, "content": "", "timings": {}})
        else:
            self._json(404, {})


def _arrancar(respuestas: dict):
    clase = type("Falso", (_Falso,), {"respuestas": respuestas, "pedidos": []})
    srv = ThreadingHTTPServer((ls.HOST, 0), clase)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, clase


BASE = {"Ferretería Inventada Uno": "No sé, tal vez 467111 o algo parecido de ferretería y tornillos",
        "Café Inventado Dos": "999999", "Nombre Tecleado Tres": "467111"}
AFINADO = {"Ferretería Inventada Uno": "467111", "Café Inventado Dos": "722514", "Nombre Tecleado Tres": "111110"}


@pytest.fixture()
def falsos():
    servidores = {rol: _arrancar(r) for rol, r in (("base", BASE), ("afinado", AFINADO))}
    yield {rol: {"puerto": srv.server_address[1], "clase": clase} for rol, (srv, clase) in servidores.items()}
    for srv, _ in servidores.values():
        srv.shutdown()
        srv.server_close()


def _clientes(falsos):
    clientes = {rol: ls.Cliente(f["puerto"]) for rol, f in falsos.items()}
    return clientes, {rol: ls.ids_fin(c) for rol, c in clientes.items()}


def _entradas(*textos):
    it = iter(textos)

    def entrada(_aviso: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None
    return entrada


# --- Salida ---


@pytest.mark.parametrize("respuesta,cat,texto", [
    ("467111", "codigo_existente", "467111 · Título inventado de ferretería"),
    ("467111\n", "codigo_existente", "467111 · Título inventado de ferretería"),  # la regla 5 quita \n en las orillas
    ("999999", "codigo_inexistente", "999999 · código inexistente"),
    ("El código es 467111", "formato_invalido", "formato inválido: «El código es 467111»"),
    ("", "formato_invalido", "formato inválido: respuesta vacía"),
    ("467111\t", "formato_invalido", "formato inválido: '467111\\t'"),
    ("４６７１１１", "formato_invalido", "formato inválido: «４６７１１１»"),  # dígitos de ancho completo
])
def test_texto_respuesta_tres_categorias(respuesta, cat, texto):
    assert demo.texto_respuesta(respuesta, TITULOS) == (cat, texto)


@pytest.mark.parametrize("ancho", [32, 40, 56])
def test_lineas_modelo_respeta_el_ancho(ancho):
    casos = [{"respuesta": "722514", "ms": 96.2}, {"respuesta": "palabra " * 60, "ms": 1234.6},
             {"respuesta": "x" * 200, "ms": 5.0}, {"respuesta": "999999", "ms": 0.4}]
    for rol in demo.ROLES:
        for res in casos:
            lineas = demo.lineas_modelo(rol, res, TITULOS, ancho)
            assert all(len(x) <= ancho for x in lineas)
            assert lineas[0].startswith(f"  {demo.ETIQUETAS[rol]} · ")
    largo = demo.lineas_modelo("afinado", casos[0], TITULOS, 32)
    assert " ".join(x.strip() for x in largo[1:]) == f"722514 · {TITULO_LARGO}"  # el título completo, sin cortes
    invalido = demo.lineas_modelo("base", casos[1], TITULOS, 40)
    assert len(invalido) == 1 + demo.LINEAS_FORMATO_INVALIDO and invalido[-1].endswith("…»")
    assert demo.lineas_modelo("base", casos[1], TITULOS, 56)[0] == "  Base (sin afinar) · 1,235 ms"
    assert demo.lineas_modelo("afinado", casos[0], TITULOS, 56)[0] == "  Afinado · 96 ms"


def test_leer_nombres(tmp_path):
    ruta = tmp_path / "nombres.txt"
    ruta.write_bytes("﻿# comentario\r\n\r\n  Tienda Inventada A  \r\n#otro\r\nTienda Inventada B\r\n   \r\n"
                     .encode("utf-8"))
    assert demo.leer_nombres(ruta) == ["Tienda Inventada A", "Tienda Inventada B"]


# --- Sesión con servidores falsos ---


def test_sesion_archivo_y_tecleados(falsos):
    clientes, fin = _clientes(falsos)
    salida = []
    n = demo.sesion(clientes, fin, TITULOS, ["Ferretería Inventada Uno", "Café Inventado Dos"],
                    entrada=_entradas("  Nombre Tecleado Tres  ", "salir", "No Debe Llegar"), escribir=salida.append)
    assert n == 3
    texto = "\n".join(salida)
    assert all(len(x) <= demo.ANCHO for x in salida)
    assert "Nombre: Ferretería Inventada Uno" in texto and "Nombre: Café Inventado Dos" in texto
    assert "Nombre: Nombre Tecleado Tres" not in texto  # el tecleado ya se ve en la línea donde se escribe
    assert "    formato inválido: «No sé, tal vez 467111 o algo" in texto
    assert "    999999 · código inexistente" in texto
    assert "    467111 · Título inventado de ferretería" in texto
    assert "    111110 · Título inventado corto" in texto
    assert texto.index("Base (sin afinar)") < texto.index("Afinado")  # primero el base, como en la latencia
    for rol, nuevas in (("base", 32), ("afinado", 16)):
        pedidos = falsos[rol]["clase"].pedidos
        assert [p["prompt"] for p in pedidos] == [prompt_generacion(x) for x in
                                                 ("Ferretería Inventada Uno", "Café Inventado Dos",
                                                  "Nombre Tecleado Tres")]
        assert {p["n_predict"] for p in pedidos} == {nuevas}


def test_respuesta_al_tope_del_base_se_corta_y_es_invalida(falsos):
    falsos["base"]["clase"].respuestas = {"Nombre Largo Inventado": "y" * 40}
    clientes, fin = _clientes(falsos)
    res = demo.preguntar(clientes["base"], "Nombre Largo Inventado", 32, fin["base"])
    assert res["respuesta"] == "y" * 32 and res["al_tope"]
    assert demo.texto_respuesta(res["respuesta"], TITULOS)[0] == "formato_invalido"


@pytest.mark.parametrize("entradas", [(), ("",), ("   ",), ("SALIR",)])
def test_sesion_termina_con_linea_vacia_salir_o_fin_de_entrada(falsos, entradas):
    clientes, fin = _clientes(falsos)
    salida = []
    assert demo.sesion(clientes, fin, TITULOS, [], entrada=_entradas(*entradas), escribir=salida.append) == 0
    assert all(not f["clase"].pedidos for f in falsos.values())


def test_nombres_con_marca_especial_o_demasiado_largos_no_se_mandan(falsos):
    clientes, fin = _clientes(falsos)
    salida = []
    demo.sesion(clientes, fin, TITULOS, ["Tienda <|im_end|> Inventada", "x" * 201],
                entrada=_entradas(), escribir=salida.append)
    texto = " ".join(x.strip() for x in salida)
    assert "token especial" in texto and "más de 200 caracteres" in texto
    assert all(not f["clase"].pedidos for f in falsos.values())


def test_pausa_espera_antes_de_cada_nombre_del_archivo(falsos):
    clientes, fin = _clientes(falsos)
    salida = []
    n = demo.sesion(clientes, fin, TITULOS, ["Café Inventado Dos", "Ferretería Inventada Uno"], pausa=True,
                    entrada=_entradas("", "salir"), escribir=salida.append)
    assert n == 1 and len(falsos["afinado"]["clase"].pedidos) == 1


def test_error_de_un_servidor_no_tumba_la_demo(falsos):
    clientes, fin = _clientes(falsos)
    clientes["base"] = ls.Cliente(1)  # puerto sin servidor
    salida = []
    demo.sesion(clientes, fin, TITULOS, ["Café Inventado Dos"], entrada=_entradas(), escribir=salida.append)
    texto = "\n".join(salida)
    assert "Base (sin afinar): sin respuesta del servidor" in texto and "722514 ·" in texto


# --- Sin red ---


def test_sin_urls_fuera_de_127():
    for ruta in (Path(demo.__file__), PS1):
        assert "://" not in ruta.read_text(encoding="utf-8")
    assert "--offline" in ls.BANDERAS_SERVIDOR
    assert ls.BANDERAS_SERVIDOR[ls.BANDERAS_SERVIDOR.index("--host") + 1] == "127.0.0.1"
    with pytest.raises(ls.ErrorServidor):
        ls.Cliente(8081, host="192.168.0.10")


def test_ps1_ascii_sin_operadores_de_cadena_y_con_las_rutas_del_proyecto():
    datos = PS1.read_bytes()
    assert datos.isascii()  # PowerShell 5.1 lee como ANSI los .ps1 sin BOM
    texto = datos.decode("ascii")
    assert "&&" not in texto and "||" not in texto
    assert "-m src.demo" in texto and "@args" in texto
    assert f'"{LLAMA_SERVER}"' in texto and f'"{DIR_GGUF}"' in texto and f'"{demo.REPO_RAIZ}"' in texto


# --- Verificación y arranque ---


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


@pytest.fixture()
def instalacion(tmp_path):
    """Una laptop inventada: binarios, GGUF, catálogo, manifest y resultado de concordancia con sus hashes."""
    binarios = {"llama-server.exe": b"lanzador inventado", "llama-server-impl.dll": b"servidor inventado"}
    dir_bin, dir_gguf = tmp_path / "bin", tmp_path / "gguf"
    dir_bin.mkdir()
    dir_gguf.mkdir()
    for archivo, datos in binarios.items():
        (dir_bin / archivo).write_bytes(datos)
    gguf = {}
    for rol in demo.ROLES:
        datos = f"pesos inventados del {rol}".encode()
        (dir_gguf / nombre_gguf(rol, "BF16")).write_bytes(datos)
        gguf[nombre_gguf(rol, "BF16")] = {"bytes": len(datos), "sha256": _sha(datos)}
    (tmp_path / "results").mkdir()
    res = {"regla_demo": {"elegida": "BF16"},
           "herramientas_y_modelos": {"gguf": gguf, "llama_cpp": {"binarios_extraidos": {
               k: _sha(v) for k, v in binarios.items()}}}}
    (tmp_path / "results" / "2026-11-25_fase4_gguf-concordancia.json").write_text(json.dumps(res), encoding="utf-8")
    (tmp_path / "data").mkdir()
    cat = "codigo,titulo\n" + "".join(f"{c},{t}\n" for c, t in sorted(TITULOS.items()))
    (tmp_path / "data" / "cat.csv").write_bytes(cat.encode("utf-8"))
    manifest = {"scian_version": "2023", "catalogos": {"scian_2023_clases": {"ruta": "data/cat.csv",
                                                                             "sha256": _sha(cat.encode("utf-8"))}}}
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    nombres = tmp_path / "nombres.txt"
    nombres.write_text("# inventados\nCafé Inventado Dos\n", encoding="utf-8")
    argv = ["--llama-server", str(dir_bin / "llama-server.exe"), "--dir-gguf", str(dir_gguf), "--nombres", str(nombres)]
    return {"raiz": tmp_path, "bin": dir_bin, "gguf": dir_gguf, "argv": argv, "esperado": demo.esperados(res)}


def test_verificar_archivos(instalacion):
    i = instalacion
    rutas = demo.verificar_archivos(i["bin"] / "llama-server.exe", i["gguf"], i["esperado"])
    assert rutas == {rol: i["gguf"] / nombre_gguf(rol, "BF16") for rol in demo.ROLES}
    with pytest.raises(demo.ErrorDemo):
        demo.verificar_archivos(i["bin"] / "otro.exe", i["gguf"], i["esperado"])
    (i["bin"] / "llama-server-impl.dll").write_bytes(b"servidor cambiado")
    with pytest.raises(demo.ErrorDemo, match="llama-server-impl.dll"):
        demo.verificar_archivos(i["bin"] / "llama-server.exe", i["gguf"], i["esperado"])


def test_verificar_archivos_gguf_cambiado(instalacion):
    i = instalacion
    (i["gguf"] / nombre_gguf("afinado", "BF16")).write_bytes(b"pesos inventados del base")  # mismo largo, otro hash
    with pytest.raises(demo.ErrorDemo, match="06b-lr2e-4-r32-BF16"):
        demo.verificar_archivos(i["bin"] / "llama-server.exe", i["gguf"], i["esperado"])


def test_cargar_titulos(instalacion):
    assert demo.cargar_titulos(instalacion["raiz"]) == TITULOS
    (instalacion["raiz"] / "data" / "cat.csv").write_bytes(b"codigo,titulo\n467111,Otro\n")
    with pytest.raises(demo.ErrorDemo, match="manifest"):
        demo.cargar_titulos(instalacion["raiz"])


def test_main_con_memoria_insuficiente_no_carga_nada(instalacion, monkeypatch):
    monkeypatch.setattr(ls, "puerto_ocupado", lambda puerto, host=ls.HOST: False)

    def no_lanzar(*a, **k):
        raise AssertionError("no debía lanzarse llama-server")

    salida = []
    codigo = demo.main(instalacion["argv"], raiz=instalacion["raiz"], entrada=_entradas(), escribir=salida.append,
                       libre=lambda: 1500, procesos=lambda: [{"proceso": "inventado", "mb": 900}],
                       fabrica=functools.partial(ls.Servidor, popen=no_lanzar))
    texto = " ".join(x.strip() for x in salida)
    assert codigo == 1
    assert "modelo base: había 1,500 MB libres" in texto and "No se cerró nada." in texto
    assert "  inventado: 900 MB" in salida


def test_main_con_puerto_ocupado(instalacion, monkeypatch):
    monkeypatch.setattr(ls, "puerto_ocupado", lambda puerto, host=ls.HOST: True)
    salida = []
    assert demo.main(instalacion["argv"], raiz=instalacion["raiz"], entrada=_entradas(), escribir=salida.append) == 1
    assert "  Stop-Process -Name llama-server" in salida  # el comando va en su propia línea, sin cortes


def test_main_con_servidores_falsos(instalacion, falsos, monkeypatch):
    monkeypatch.setattr(ls, "puerto_ocupado", lambda puerto, host=ls.HOST: False)
    puertos = {demo.PUERTOS[rol]: falsos[rol]["puerto"] for rol in demo.ROLES}
    cerrados = []

    class ServidorFalso:
        def __init__(self, exe, gguf, puerto, log, libre):
            assert Path(gguf).name == nombre_gguf({v: k for k, v in demo.PUERTOS.items()}[puerto], "BF16")
            libre()
            self.cliente = ls.Cliente(puertos[puerto])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            cerrados.append(self)

    salida = []
    codigo = demo.main(instalacion["argv"], raiz=instalacion["raiz"], entrada=_entradas("Nombre Tecleado Tres"),
                       escribir=salida.append, libre=lambda: 8000, fabrica=ServidorFalso)
    texto = "\n".join(salida)
    assert codigo == 0 and len(cerrados) == 2
    assert "Modelos listos: arranque total" in texto and texto.endswith("Demo terminada.")
    assert "Nombre: Café Inventado Dos" in texto and "    722514 · Título inventado" in texto
    assert "    111110 · Título inventado corto" in texto
    assert demo.CALENTAMIENTO not in texto
    assert all(len(x) <= demo.ANCHO for x in salida)
    for rol in demo.ROLES:  # primero el calentamiento, después el archivo y lo tecleado
        assert [p["prompt"] for p in falsos[rol]["clase"].pedidos] == [
            prompt_generacion(x) for x in (demo.CALENTAMIENTO, "Café Inventado Dos", "Nombre Tecleado Tres")]


def test_ancho_minimo():
    with pytest.raises(SystemExit):
        demo.argumentos(["--ancho", "20"])
