"""Pruebas de src/subir_hf.py: repo temporal, pesos falsos y un cliente de Hugging Face falso (sin red)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import subir_hf as sh

TOKEN = "hf_" + "Z" * 37  # inventado
USUARIO = "usuario-de-prueba"


def _git(raiz: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=Prueba", "-c", "user.email=prueba@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=raiz, check=True, capture_output=True)


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    raiz = tmp_path / "repo"
    raiz.mkdir()
    _git(raiz, "init", "-q")
    _git(raiz, "config", "core.autocrlf", "false")
    (raiz / "data").mkdir()
    (raiz / "data" / "manifest.json").write_text(json.dumps({"x": {"sha256": "0" * 64}}), encoding="utf-8")
    (raiz / "LICENSE").write_text("Licencia de prueba\n", encoding="utf-8")
    for clave in sh.REPOS:
        card = raiz / f"hf/{sh.REPOS[clave]['nombre']}/README.md"
        card.parent.mkdir(parents=True)
        card.write_text(f"# Card de prueba\n\nrepo: {sh.MARCA_USUARIO}/{sh.REPOS[clave]['nombre']}\n", encoding="utf-8")
    _git(raiz, "add", "data/manifest.json", "LICENSE", "hf")
    _git(raiz, "commit", "-q", "-m", "base")
    # Pesos falsos y sus resultados
    resultados: dict[str, dict] = {}
    for clave in sh.REPOS:
        for a in sh.REPOS[clave]["archivos"]:
            if a.esperado[0] == "head":
                continue
            datos = f"pesos falsos de {a.en_repo} ({clave})".encode() * (50 if a.en_repo.endswith(".gguf") else 3)
            ruta = (raiz / a.origen).resolve()
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(datos)
            tipo, res, k = a.esperado
            r = resultados.setdefault(res, {})
            if tipo == "archivos_bajados":
                r.setdefault("corrida_kaggle", {"archivos_bajados": []})["archivos_bajados"].append(
                    {"ruta": k, "bytes": len(datos), "sha256": _sha(datos)})
            else:
                r["herramientas_y_modelos"] = {"gguf": {k: {"bytes": len(datos), "sha256": _sha(datos)}}}
                r["cuantizaciones"] = {"BF16": {"gguf": {"archivo": k, "bytes": len(datos), "sha256": _sha(datos)}}}
                r["regla_demo"] = {"elegida": "BF16"}
    (raiz / "results").mkdir()
    for res, r in resultados.items():
        (raiz / res).write_text(json.dumps(r), encoding="utf-8")
    return raiz


class ClienteFalso:
    def __init__(self, token: str, quien: str = USUARIO, falla: str | None = None, remoto: dict | None = None,
                 existentes: list[str] | None = None):
        self.token, self.quien, self.falla, self.remoto, self.existentes = token, quien, falla, remoto, existentes or []
        self.commits: list[tuple[str, list]] = []
        self.creados: list[tuple[str, bool]] = []

    def usuario(self):
        return self.quien

    def crear_repo(self, repo_id, privado):
        self.creados.append((repo_id, privado))

    def archivos(self, repo_id):
        return [".gitattributes", *self.existentes]

    def commit(self, repo_id, archivos, mensaje):
        if self.falla:
            raise RuntimeError(self.falla)
        self.commits.append((repo_id, archivos))
        return f"https://huggingface.co/{repo_id}/commit/abc"

    def info(self, repo_id, rutas):
        if self.remoto is not None:
            return self.remoto
        res = {}
        for p, f in dict(self.commits[-1][1]).items():
            datos = f if isinstance(f, bytes) else Path(f).read_bytes()
            grande = not isinstance(f, bytes)
            res[p] = {"bytes": len(datos), "sha256": _sha(datos) if grande else None,
                      "blob_id": None if grande else sh.sha1_git(datos)}
        return res


def correr(repo, argv, entorno=None, token=TOKEN, **kw_cliente):
    lineas: list[str] = []
    clientes: list[ClienteFalso] = []
    pedidos: list[str] = []

    def fabrica(t):
        c = ClienteFalso(t, **kw_cliente)
        clientes.append(c)
        return c

    def pedir(msg):
        pedidos.append(msg)
        return token

    codigo = sh.main(argv, entorno=entorno or {}, pedir_token=pedir, cliente=fabrica, raiz=repo, salida=lineas.append)
    return codigo, "\n".join(lineas), clientes, pedidos


def test_dry_run_verifica_y_no_sube(repo):
    codigo, texto, clientes, pedidos = correr(repo, ["--usuario", USUARIO])
    assert codigo == 0
    assert "Dry-run: nada se subió." in texto and "Todos los SHA-256 verificados" in texto
    assert texto.count("SHA-256 ") >= 9
    assert f"Repo {USUARIO}/denue-scian-qwen3-4b-lora (privado al crearse)" in texto
    assert sh.GGUF in texto
    assert clientes == [] and pedidos == []


def test_sha_distinto_detiene(repo):
    a = sh.REPOS["4b"]["archivos"][3]
    ruta = (repo / a.origen).resolve()
    datos = bytearray(ruta.read_bytes())
    datos[0] ^= 1
    ruta.write_bytes(bytes(datos))
    codigo, texto, clientes, _ = correr(repo, ["--usuario", USUARIO, "--subir"])
    assert codigo == 1 and "SHA-256" in texto and "Detenido antes de subir" in texto
    assert clientes == []


def test_bytes_distintos_y_archivo_faltante(repo):
    a = sh.REPOS["06b"]["archivos"][4]
    ruta = (repo / a.origen).resolve()
    ruta.write_bytes(ruta.read_bytes() + b"x")
    codigo, texto, _, _ = correr(repo, ["--usuario", USUARIO, "--repo", "06b"])
    assert codigo == 1 and "bytes y se esperaban" in texto
    ruta.unlink()
    codigo, texto, _, _ = correr(repo, ["--usuario", USUARIO, "--repo", "06b"])
    assert codigo == 1 and "Falta" in texto


def test_card_con_cambios_sin_commit(repo):
    (repo / "hf/denue-scian-qwen3-4b-lora/README.md").write_text("cambiada {{usuario_hf}}\n", encoding="utf-8")
    codigo, texto, _, _ = correr(repo, ["--usuario", USUARIO])
    assert codigo == 1 and "cambios sin commit" in texto


def test_card_sin_marcador(repo):
    card = repo / "hf/denue-scian-qwen3-4b-lora/README.md"
    card.write_text("sin marcador\n", encoding="utf-8")
    _git(repo, "add", "hf")
    _git(repo, "commit", "-q", "-m", "x")
    codigo, texto, _, _ = correr(repo, ["--usuario", USUARIO])
    assert codigo == 1 and "marcador" in texto


def test_hash_del_manifest_detiene(repo):
    a = sh.REPOS["4b"]["archivos"][2]
    sha = _sha((repo / a.origen).read_bytes())
    (repo / "data/manifest.json").write_text(json.dumps({"x": {"sha256": sha}}), encoding="utf-8")
    _git(repo, "add", "data/manifest.json")
    _git(repo, "commit", "-q", "-m", "x")
    codigo, texto, _, _ = correr(repo, ["--usuario", USUARIO])
    assert codigo == 1 and "data/manifest.json" in texto


def test_usuario_invalido(repo):
    codigo, texto, _, _ = correr(repo, ["--usuario", "malo-"])
    assert codigo == 1 and "usuario" in texto


def test_abreviatura_rechazada(repo):
    with pytest.raises(SystemExit) as e:
        correr(repo, ["--usuario", USUARIO, "--sub"])
    assert e.value.code == 2


def test_con_claudecode_se_niega_antes_del_token(repo):
    codigo, texto, clientes, pedidos = correr(repo, ["--usuario", USUARIO, "--subir"], entorno={"CLAUDECODE": "1"})
    assert codigo == 2 and "CLAUDECODE" in texto
    assert clientes == [] and pedidos == []


def test_subida_real_con_cliente_falso(repo, capsys):
    codigo, texto, clientes, pedidos = correr(repo, ["--usuario", USUARIO, "--subir"])
    salida = capsys.readouterr()
    assert codigo == 0, texto
    assert len(pedidos) == 1
    (c,) = clientes
    assert [r for r, _ in c.commits] == [f"{USUARIO}/denue-scian-qwen3-4b-lora", f"{USUARIO}/denue-scian-qwen3-0.6b-lora-gguf"]
    assert all(privado for _r, privado in c.creados)
    card = dict(c.commits[0][1])["README.md"]
    assert f"{USUARIO}/denue-scian-qwen3-4b-lora" in card.decode() and "{{" not in card.decode()
    assert sorted(dict(c.commits[1][1])) == sorted(["README.md", "LICENSE", "adapter_config.json",
                                                    "adapter_model.safetensors", sh.GGUF])
    for t in (texto, salida.out, salida.err):
        assert TOKEN not in t


def test_token_tapado_en_errores(repo, capsys):
    codigo, texto, _, _ = correr(repo, ["--usuario", USUARIO, "--subir"], falla=f"401 Bearer {TOKEN} rechazado")
    salida = capsys.readouterr()
    assert codigo == 1 and "[token oculto]" in texto
    for t in (texto, salida.out, salida.err):
        assert TOKEN not in t


def test_token_de_otro_usuario(repo):
    codigo, texto, clientes, _ = correr(repo, ["--usuario", USUARIO, "--subir"], quien="otra-persona")
    assert codigo == 1 and clientes[0].commits == []


def test_token_desde_hf_token(repo):
    codigo, texto, clientes, pedidos = correr(repo, ["--usuario", USUARIO, "--subir", "--repo", "4b"],
                                              entorno={"HF_TOKEN": TOKEN})
    assert codigo == 0 and pedidos == [] and clientes[0].token == TOKEN and TOKEN not in texto


def test_comprobacion_remota_que_falla(repo):
    remoto = {"README.md": {"bytes": 1, "sha256": None, "blob_id": "x"}}
    codigo, texto, _, _ = correr(repo, ["--usuario", USUARIO, "--subir", "--repo", "4b"], remoto=remoto)
    assert codigo == 1 and "comprobación después de subir" in texto


def test_repo_con_archivos_ajenos(repo):
    codigo, texto, clientes, _ = correr(repo, ["--usuario", USUARIO, "--subir"], existentes=["otro.bin"])
    assert codigo == 1 and "no se mezcla" in texto and clientes[0].commits == []


def test_publico(repo):
    codigo, texto, clientes, _ = correr(repo, ["--usuario", USUARIO, "--subir", "--repo", "4b", "--publico"])
    assert codigo == 0 and clientes[0].creados == [(f"{USUARIO}/denue-scian-qwen3-4b-lora", False)]


def test_importar_no_carga_pandas_ni_huggingface_hub():
    r = subprocess.run([sys.executable, "-c", "import sys, src.subir_hf; "
                        "print(any(m in sys.modules for m in ('pandas', 'numpy', 'huggingface_hub')))"],
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "False"


def test_especificacion_sin_datos():
    for spec in sh.REPOS.values():
        for a in spec["archivos"]:
            assert not a.en_repo.endswith(sh.EXT_DATOS) and not a.origen.endswith(sh.EXT_DATOS)
            assert "splits" not in a.origen and "fase4" not in a.origen
