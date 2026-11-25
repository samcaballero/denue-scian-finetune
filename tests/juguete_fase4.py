"""Datos, config e insumos de juguete para las pruebas del kernel de la Fase 4 (paso 15).

Todo es inventado: nombres, ids, códigos y bytes de los adaptadores. El config parte del
pre-registro real (versionado y sin nombres) con conjuntos chicos y los SHA-256 de los insumos de
juguete. Este archivo no es una prueba (no empieza con test_).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from src import inferencia_fase4_kaggle as fo
from src.denue import REPO_RAIZ
from src.plantilla_fase4 import seleccion_fewshot

CONFIG_REAL = json.loads((REPO_RAIZ / fo.RUTA_CONFIG).read_text(encoding="utf-8"))
CODIGOS_TRAIN = ("461110", "811111", "722511", "311110", "321111", "541110", "431110", "221110", "332110")
FILAS = {"train_final": 60, "validacion": 20, "test": 6, "hard": 3}
MINIMO = 19  # de las 20 filas de validación de juguete
IDS = {"train_final": 31_000_000, "validacion": 41_000_000, "test": 51_000_000, "hard": 61_000_000}
NOMBRE = {"train_final": "Taller {i}", "validacion": "Café {i}", "test": "Fonda {i}", "hard": "Raro {i}"}
RESPUESTA_REFERENCIA = "467111"


def sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def jsonl(regs) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs).encode("utf-8")


def filas(conj: str) -> list[dict]:
    regs = []
    for i in range(FILAS[conj]):
        r = {"id": str(IDS[conj] + 7 * i), "nom_estab": NOMBRE[conj].format(i=i)}
        if conj in ("train_final", "validacion"):
            r["codigo_act"] = CODIGOS_TRAIN[i % len(CODIGOS_TRAIN)]
        regs.append(r)
    return regs


def adapter_config(repo: str) -> dict:
    return {
        "base_model_name_or_path": repo, "bias": "none", "lora_alpha": 32, "lora_dropout": 0.0, "modules_to_save": None,
        "peft_type": "LORA", "r": 32, "target_modules": ["down_proj", "o_proj", "up_proj", "k_proj", "q_proj", "v_proj", "gate_proj"],
        "use_dora": False, "use_rslora": False,
    }


def armar(tmp: Path) -> dict:
    """Entrada (/kaggle/input), raíz de fuentes con el config de juguete, ESPERADOS y SHA256_FUENTES."""
    entrada = tmp / "input"
    datos = {c: jsonl(filas(c)) for c in FILAS}
    archivos = {
        "denue-scian-evaluacion/test.jsonl": datos["test"],
        "denue-scian-evaluacion/hard.jsonl": datos["hard"],
        "denue-scian-clasificador/train_final.jsonl": datos["train_final"],
        "denue-scian-clasificador/val.jsonl": datos["validacion"],
        "denue-scian-06b-barrido/barrido.json": b"{}\n",
        "denue-scian-06b-barrido/06b-lr1e-4-r8/adaptador/adapter_model.safetensors": b"otros pesos inventados",
        "denue-scian-06b-barrido/06b-lr1e-4-r8/predicciones.jsonl": b"",
        "denue-scian-4b/barrido.json": b"{}\n",
    }
    config = copy.deepcopy(CONFIG_REAL)
    referencia = jsonl({"id": r["id"], "respuesta": RESPUESTA_REFERENCIA} for r in filas("validacion"))
    for m in config["modelos"]:
        if m.get("tipo") != "afinado":
            continue
        carpeta = m["origen_kaggle"]["carpeta_en_la_salida"]
        kernel = "denue-scian-06b-barrido" if m["tamano"] == "0.6B" else "denue-scian-4b"
        pesos = f"pesos inventados de {m['id']}".encode()
        cfg = (json.dumps(adapter_config(m["base"]["repo"]), indent=2) + "\n").encode()
        for clave, nombre, contenido in (("adaptador", "adapter_model.safetensors", pesos),
                                         ("adaptador_config", "adapter_config.json", cfg)):
            archivos[f"{kernel}/{carpeta}/{nombre}"] = contenido
            m[clave]["sha256"] = sha(contenido)
        archivos[f"{kernel}/{carpeta.rsplit('/', 1)[0]}/predicciones.jsonl"] = referencia
        m["predicciones_validacion"]["sha256"] = sha(referencia)
    for c, n in FILAS.items():
        config["conjuntos"][c]["filas"] = n
    crit = config["uso_unico"]["reproduccion_validacion_en_el_kernel"]["criterio_de_paro"]
    crit.update(filas=FILAS["validacion"], minimo_identicas=MINIMO)
    sel = seleccion_fewshot(filas("train_final"), config["few_shot"])
    config["few_shot"].update({k: sel["publico"][k] for k in ("sha256_ids", "codigos", "sectores")})
    for ruta, contenido in archivos.items():
        (entrada / ruta).parent.mkdir(parents=True, exist_ok=True)
        (entrada / ruta).write_bytes(contenido)
    raiz = tmp / "fuentes"
    texto = escribir_config(raiz, config)
    esperados = {
        "test.jsonl": sha(datos["test"]), "hard.jsonl": sha(datos["hard"]),
        "train_final.jsonl": sha(datos["train_final"]), "val.jsonl": sha(datos["validacion"]),
    }
    return {"entrada": entrada, "raiz": raiz, "config": config, "esperados": esperados,
            "fuentes": {fo.RUTA_CONFIG: sha(texto)}, "referencia": referencia}


def escribir_config(raiz: Path, config: dict) -> bytes:
    texto = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (raiz / fo.RUTA_CONFIG).parent.mkdir(parents=True, exist_ok=True)
    (raiz / fo.RUTA_CONFIG).write_bytes(texto)
    return texto


def textos_privados() -> list[str]:
    """Ids y nombres de juguete que no deben aparecer en mediciones ni logs."""
    res = []
    for c in FILAS:
        for r in filas(c):
            res += [r["id"], r["nom_estab"]]
    return res
