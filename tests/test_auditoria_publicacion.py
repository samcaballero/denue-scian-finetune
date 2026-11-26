"""Pruebas de src/auditoria_publicacion.py con repos de git temporales y solo datos inventados."""

from __future__ import annotations

import json
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from src import auditoria_publicacion as ap
from src.evaluar_corrida_clasificador import indice_de_nombres
from src.evaluar_humo import referencias_de_nombres

# Todo inventado.
NOMBRE = "PANADERIA LUNA AZUL"          # en los 4 conjuntos
NOMBRE_EXTRA = "TALLER COMETA VERDE"     # solo en el "DENUE completo"
ID = "7654321"
SECRETO = "hf_" + "Q" * 34
CORREO = "persona.inventada@correo.test"


def refs_falsas() -> ap.Referencias:
    filas4 = [{"id": ID, "nom_estab": NOMBRE}, {"id": "1234560", "nom_estab": "FARMACIA SOL"}]
    filas_d = filas4 + [{"id": "2345678", "nom_estab": NOMBRE_EXTRA}]
    return ap.Referencias(
        idx_4=indice_de_nombres(referencias_de_nombres(filas4)),
        idx_denue=indice_de_nombres(referencias_de_nombres(filas_d)),
        ids=frozenset({ID, "1234560", "2345678", "461110", "18735"}),
        codigos_scian=frozenset({"461110"}),
        titulos_normalizados=ap.texto_de_busqueda(["Farmacia sol y sombra"]),
        hashes={"denue": "0" * 64},
    )


def _git(raiz: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=Prueba", "-c", "user.email=prueba@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=raiz, check=True, capture_output=True)


def png_con_texto(texto: str) -> bytes:
    def bloque(tipo: bytes, cuerpo: bytes) -> bytes:
        return struct.pack(">I", len(cuerpo)) + tipo + cuerpo + struct.pack(">I", zlib.crc32(tipo + cuerpo) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + bloque(b"IHDR", ihdr) + bloque(b"tEXt", b"Comment\0" + texto.encode("latin-1"))
            + bloque(b"IDAT", zlib.compress(b"\0\0")) + bloque(b"IEND", b""))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    raiz = tmp_path / "repo"
    raiz.mkdir()
    _git(raiz, "init", "-q")
    _git(raiz, "config", "core.autocrlf", "false")
    (raiz / "docs").mkdir()
    (raiz / "src").mkdir()
    (raiz / "results").mkdir()
    (raiz / "tests").mkdir()
    (raiz / "docs" / "notas.md").write_text(f"Ejemplo copiado: {NOMBRE} con id {ID}\notra línea\n", encoding="utf-8")
    (raiz / "docs" / "captura.png").write_bytes(png_con_texto("hecho por alguien"))
    (raiz / "src" / "modulo.py").write_text('TEXTO = "farmacia sol"\n', encoding="utf-8")
    (raiz / "results" / "r.json").write_text(json.dumps({"nota": "farmacia sol", "filas": 18735, "codigo": "461110"}),
                                             encoding="utf-8")
    (raiz / "tests" / "test_x.py").write_text(f'NOMBRE = "{NOMBRE}"\n', encoding="utf-8")
    _git(raiz, "add", "docs", "src", "results", "tests")
    _git(raiz, "commit", "-q", "-m", "primero")
    (raiz / "docs" / "notas.md").write_text("ya sin nada\n", encoding="utf-8")
    _git(raiz, "add", "docs")
    _git(raiz, "commit", "-q", "-m", "segundo")
    (raiz / "docs" / "borrador.md").write_text(f"token {SECRETO}\ncorreo {CORREO}\n| {NOMBRE_EXTRA.lower()} |\n",
                                               encoding="utf-8")
    return raiz


def correr(raiz: Path):
    inv = ap.inventario(raiz)
    refs = refs_falsas()
    res = ap.revisar(inv, refs)
    ap.clasificar_nombres(res, inv, refs, ap.busqueda_de_codigo(inv))
    return inv, res


def test_encuentra_lo_borrado_y_lo_no_versionado(repo):
    inv, res = correr(repo)
    assert len(inv.commits) == 2
    assert inv.no_versionados == ["docs/borrador.md"]
    tipos = {(h.pasada, h.tipo, ap.ubicacion([h], inv)["ruta"]) for h in res.hallazgos if h.categoria is None}
    assert ("4_conjuntos", "llave_varias_palabras", "docs/notas.md") in tipos
    assert ("general", "id", "docs/notas.md") in tipos
    assert ("general", "secreto:token_hf", "docs/borrador.md") in tipos
    assert ("general", "correo", "docs/borrador.md") in tipos
    nota = next(h for h in res.hallazgos if h.tipo == "llave_varias_palabras" and h.categoria is None
                and h.pasada == "4_conjuntos")
    u = ap.ubicacion([nota], inv)
    assert u["en_head"] == []  # solo en el historial
    assert u["commit"] == inv.commits[0][:7]


def test_categorias_automaticas(repo):
    inv, res = correr(repo)
    cat = {(ap.ubicacion([h], inv)["ruta"], h.tipo): h.categoria for h in res.hallazgos}
    assert cat[("tests/test_x.py", "llave_varias_palabras")] == "prueba"
    assert cat[("results/r.json", "llave_varias_palabras")] == "catalogo"  # "farmacia sol" está en un título
    ids = {h.valor: h.categoria for h in res.hallazgos if h.tipo == "id"}
    assert ids["461110"] == "codigo_scian"
    assert ids["18735"] == "explicado_por_el_archivo"
    assert ids[ID] is None


def test_pasada_extra_nombre_completo(repo):
    inv, res = correr(repo)
    extra = [h for h in res.hallazgos if h.pasada == "denue_completo"]
    assert extra and all(h.completo for h in extra)  # es la celda entera de una tabla
    assert all(ap.detiene(h) for h in extra)
    parcial = ap.Hallazgo("llave_varias_palabras", "denue_completo", "COMETA VERDE", "el cometa verde brilla")
    assert not ap.es_nombre_completo(parcial) and not ap.detiene(parcial)


def test_png_con_bloque_de_texto(repo):
    inv, res = correr(repo)
    assert res.imagenes and res.imagenes[0]["bloques_de_texto"] == ["tEXt"]
    assert ap.texto_de_chunks_png(png_con_texto("abc")) == ["Comment: abc"]


def test_la_salida_nunca_trae_valores(repo, capsys):
    inv, res = correr(repo)
    lineas: list[str] = []
    ap.imprimir(res, inv, salida=lineas.append)
    texto = "\n".join(lineas) + json.dumps(ap.resumen(res, inv), ensure_ascii=False)
    for valor in (NOMBRE, NOMBRE.lower(), NOMBRE_EXTRA, NOMBRE_EXTRA.lower(), ID, SECRETO, CORREO, "FARMACIA SOL"):
        assert valor not in texto
    assert "DETIENE" in texto


def test_rutas_prohibidas(tmp_path):
    raiz = tmp_path / "r"
    raiz.mkdir()
    _git(raiz, "init", "-q")
    (raiz / "data").mkdir()
    (raiz / "data" / "manifest.json").write_text("{}", encoding="utf-8")
    (raiz / "data" / "splits.jsonl").write_text("{}\n", encoding="utf-8")
    (raiz / "kaggle.json").write_text("{}", encoding="utf-8")
    _git(raiz, "add", "data", "kaggle.json")
    _git(raiz, "commit", "-q", "-m", "x")
    inv = ap.inventario(raiz)
    res = ap.revisar(inv, refs_falsas())
    assert [p["ruta"] for p in res.rutas_prohibidas] == ["data/splits.jsonl"]
    assert [c["ruta"] for c in res.rutas_credenciales] == ["kaggle.json"]


def test_categoria_id_prefijo_y_unidad():
    refs = refs_falsas()
    h = ap.Hallazgo("id", "general", "1234560", "commit `1234560`")
    assert ap.categoria_id(h, refs, ["1234560abcdef" + "0" * 27], set()) == "prefijo_de_hash"
    h2 = ap.Hallazgo("id", "general", "1234560", "pesa 1234560 bytes")
    assert ap.categoria_id(h2, refs, [], set()) == "explicado_por_el_archivo"
    h3 = ap.Hallazgo("id", "general", "1234560", "el número 1234560 suelto")
    assert ap.categoria_id(h3, refs, [], set()) is None


def test_numeros_explicados_no_acepta_claves_de_id():
    doc = ap.Documento("x", json.dumps({"filas": 1234560, "id": 2345678, "acc": 0.7654321}).encode(), [("c", "a.json")])
    assert ap.numeros_explicados(doc) == {"1234560", "7654321"}


def test_manual_valida_la_categoria():
    h = ap.Hallazgo("id", "general", "1234560", "x 1234560")
    with pytest.raises(ap.ErrorAuditoria):
        ap.aplicar_manual([h], {h.ident(): "frase_comun_de_prosa"})
    ap.aplicar_manual([h], {h.ident(): "posible_dato_real"})
    assert h.categoria == "posible_dato_real" and ap.detiene(h)


def test_correos_aceptados():
    assert ap.otros_en_linea("Co-Authored-By: Claude <noreply@anthropic.com>", frozenset()) == []
    assert ap.otros_en_linea("x <1+u@users.noreply.github.com>", frozenset()) == []
    assert [h.tipo for h in ap.otros_en_linea("ruta C:\\Users\\Otra\\x", frozenset())] == ["ruta_de_usuario"]
    assert ap.otros_en_linea("ruta C:\\Users\\Invitado\\x", frozenset()) == []
