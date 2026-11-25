"""Plantilla de chat y largos con el tokenizador real (Fase 4, paso 15).

Comprobaciones que pide el pre-registro (`evals/fase4_preregistro.json`) antes de generar la primera
respuesta de cada modelo, escritas una sola vez para la laptop y para el kernel de la Fase 4:

- (a) `tokenizadores_identicos()`: el del 0.6B y el del 4B tienen el mismo vocabulario, los mismos
  tokens agregados, la misma plantilla y dan los mismos ids en todos los prompts;
- (b) 0-shot: `prompt_generacion()` es, byte a byte, lo que da
  `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` con un solo mensaje;
- (c) few-shot: `prompt_fewshot()` con los 5 ejemplos es lo que da la plantilla con los 5 pares
  usuario/asistente y el mensaje de la fila (`mensajes_fewshot()`);
- (d) largos: las fichas reales de cada prompt más las fichas nuevas de su modo caben en su largo
  máximo (256 para los afinados y el 0-shot; `largo_maximo.few_shot` para el few-shot). Nunca se
  trunca.

El tokenizador se recibe ya cargado (las pruebas usan uno falso). Arriba solo se importan la
biblioteca estándar y módulos de `src/` (el preregistro trae numpy), para que el kernel pueda
incrustar este módulo; transformers y huggingface_hub solo se importan en `main()`.

Uso en la laptop, con el entorno aparte del paso 15 (fuera del repo, sin tocar .venv):
  ../venv-tokenizador/Scripts/python -m src.plantilla_fase4 --cache ../venv-tokenizador/hf-cache

Lee las copias de Kaggle de test, hard, train final y validación (con su SHA-256 del manifest),
baja solo los archivos del tokenizador de las dos revisiones fijadas e imprime solo conteos,
máximos y sí/no; nunca nombres, ids, llaves ni prompts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.preregistro_fase4 import (
    ErrorPreregistro,
    elegir_fewshot,
    parsear_jsonl,
    prompt_fewshot,
    sector_de_prefijo,
    sha256_ids,
)
from src.prompt_clasificador import FIN_DE_TURNO, INICIO_DE_TURNO, contenido_usuario, mensajes, prompt_generacion

TOKEN_FIN_DE_TEXTO = "<|endoftext|>"
CAMPOS_RESERVADOS = ("id", "nom_estab")  # test y hard: las copias sin etiquetas
CAMPOS_CON_CODIGO = ("id", "nom_estab", "codigo_act")  # train final y validación

# Modo -> conjuntos que responde, prompt y de qué claves de largo_maximo salen sus límites.
MODOS = {
    "afinado": {"conjuntos": ("validacion", "test", "hard"), "fewshot": False, "largo": "general", "nuevas": "nuevas_afinados"},
    "base-0shot": {"conjuntos": ("test", "hard"), "fewshot": False, "largo": "general", "nuevas": "nuevas_base"},
    "base-fewshot": {"conjuntos": ("test", "hard"), "fewshot": True, "largo": "few_shot", "nuevas": "nuevas_base"},
}

# Archivos del tokenizador en los repos de Qwen3 (sin pesos ni config del modelo).
PATRONES_TOKENIZADOR = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


class ErrorPlantilla(ValueError):
    """Comprobación imposible; el mensaje nunca incluye nombres, ids ni prompts."""


# --- Prompts ---


def mensajes_fewshot(ejemplos: Sequence[tuple[str, str]], nom_estab: str) -> list[dict[str, str]]:
    """Los 5 pares usuario/asistente de los ejemplos, en orden, y el mensaje de usuario de la fila."""
    res: list[dict[str, str]] = []
    for nombre, codigo in ejemplos:
        res.append({"role": "user", "content": contenido_usuario(nombre)})
        res.append({"role": "assistant", "content": codigo})
    return res + mensajes(nom_estab)


def prompt_de(modo: str, nom_estab: str, ejemplos: Sequence[tuple[str, str]] = ()) -> str:
    """El prompt que responde un modelo de ese modo (el texto escrito a mano, no la plantilla)."""
    if MODOS[modo]["fewshot"]:
        return prompt_fewshot(ejemplos, nom_estab)
    return prompt_generacion(nom_estab)


def plantilla_de(tok, modo: str, nom_estab: str, ejemplos: Sequence[tuple[str, str]] = ()) -> str:
    """Lo que da la plantilla del tokenizador para el mismo modo y la misma fila."""
    msgs = mensajes_fewshot(ejemplos, nom_estab) if MODOS[modo]["fewshot"] else mensajes(nom_estab)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def fichas(tok, texto: str) -> list[int]:
    """Ids del texto, sin tokens especiales agregados y sin truncar (la tokenización de la inferencia)."""
    return list(tok(texto, add_special_tokens=False)["input_ids"])


# --- Comprobaciones ---


def tokens_especiales(tok) -> dict:
    """<|im_start|>, <|im_end|> y <|endoftext|>: un solo token cada uno y distintos (comprobación (c) del paso 6b)."""
    ids = {t: tok.convert_tokens_to_ids(t) for t in (INICIO_DE_TURNO, FIN_DE_TURNO, TOKEN_FIN_DE_TEXTO)}
    ok = all(fichas(tok, t) == [i] for t, i in ids.items()) and len(set(ids.values())) == 3
    return {"ok": ok, "im_start": ids[INICIO_DE_TURNO], "im_end": ids[FIN_DE_TURNO], "endoftext": ids[TOKEN_FIN_DE_TEXTO]}


def _agregados(tok) -> dict:
    decoder = getattr(tok, "added_tokens_decoder", None) or {}
    return {int(i): (getattr(t, "content", str(t)), bool(getattr(t, "special", False))) for i, t in decoder.items()}


def tokenizadores_identicos(tok_a, tok_b, textos: Sequence[str]) -> dict:
    """(a): vocabulario, tokens agregados, plantilla e ids de todos los textos."""
    distintos = sum(fichas(tok_a, t) != fichas(tok_b, t) for t in textos)
    res = {
        "vocabulario_igual": tok_a.get_vocab() == tok_b.get_vocab(),
        "tokens_agregados_iguales": _agregados(tok_a) == _agregados(tok_b),
        "plantilla_igual": str(tok_a.chat_template) == str(tok_b.chat_template),
        "textos": len(textos),
        "textos_con_ids_distintos": distintos,
    }
    res["identicos"] = res["vocabulario_igual"] and res["tokens_agregados_iguales"] and res["plantilla_igual"] and not distintos
    return res


def revisar_modo(
    tok,
    modo: str,
    nombres: Mapping[str, Sequence[str]],
    ejemplos: Sequence[tuple[str, str]],
    largo_maximo: Mapping[str, int],
) -> dict:
    """(b) o (c) y (d) de un modo, en todas las filas de sus conjuntos.

    Devuelve, por conjunto, cuántas filas no dan el mismo texto que la plantilla y las fichas
    reales (máximo, media y cuántas no caben con las fichas nuevas del modo). Solo conteos.
    """
    m = MODOS[modo]
    limite, nuevas = largo_maximo[m["largo"]], largo_maximo[m["nuevas"]]
    if m["fewshot"] and not ejemplos:
        raise ErrorPlantilla("El modo few-shot necesita los ejemplos.")
    res: dict = {"limite": limite, "nuevas": nuevas, "conjuntos": {}}
    for conj in m["conjuntos"]:
        if conj not in nombres:
            raise ErrorPlantilla(f"Falta el conjunto {conj} para el modo {modo}.")
        distintos, largos = 0, []
        for n in nombres[conj]:
            prompt = prompt_de(modo, n, ejemplos)
            distintos += plantilla_de(tok, modo, n, ejemplos) != prompt
            largos.append(len(fichas(tok, prompt)))
        if not largos:
            raise ErrorPlantilla(f"El conjunto {conj} está vacío.")
        res["conjuntos"][conj] = {
            "filas": len(largos),
            "plantilla_distinta": distintos,
            "fichas_max": max(largos),
            "fichas_media": round(sum(largos) / len(largos), 2),
            "no_caben": sum(n + nuevas > limite for n in largos),
        }
    c = res["conjuntos"].values()
    res["plantilla_ok"] = all(x["plantilla_distinta"] == 0 for x in c)
    res["largo_ok"] = all(x["no_caben"] == 0 for x in c)
    res["ok"] = res["plantilla_ok"] and res["largo_ok"]
    return res


def comprobar_tokenizadores(
    toks: Mapping[str, object],
    nombres: Mapping[str, Sequence[str]],
    ejemplos: Sequence[tuple[str, str]],
    largo_maximo: Mapping[str, int],
    modos: Sequence[str] = tuple(MODOS),
) -> dict:
    """(a)-(d) con el tokenizador de cada tamaño ({"0.6B": tok, "4B": tok}) en todos sus modos.

    `ok` exige los tokens especiales, (b), (c) y (d) en todos los tamaños y modos; (a) solo se
    registra, porque cada modelo usa su propio tokenizador.
    """
    res: dict = {"tokens_especiales": {}, "modos": {}}
    for tam, tok in toks.items():
        res["tokens_especiales"][tam] = tokens_especiales(tok)
        res["modos"][tam] = {modo: revisar_modo(tok, modo, nombres, ejemplos, largo_maximo) for modo in modos}
    if len(toks) == 2:
        a, b = toks.values()
        textos = [prompt_de(modo, n, ejemplos) for modo in modos for c in MODOS[modo]["conjuntos"] for n in nombres[c]]
        res["a_tokenizadores"] = tokenizadores_identicos(a, b, textos)
    todos = [r for por_modo in res["modos"].values() for r in por_modo.values()]
    res["especiales_ok"] = all(t["ok"] for t in res["tokens_especiales"].values())
    res["b_0shot_ok"] = all(r["plantilla_ok"] for t in res["modos"].values() for m, r in t.items() if not MODOS[m]["fewshot"])
    res["c_fewshot_ok"] = all(r["plantilla_ok"] for t in res["modos"].values() for m, r in t.items() if MODOS[m]["fewshot"])
    res["d_largos_ok"] = all(r["largo_ok"] for r in todos)
    res["ok"] = res["especiales_ok"] and res["b_0shot_ok"] and res["c_fewshot_ok"] and res["d_largos_ok"]
    return res


# --- Selección few-shot sobre la copia de train final ---


def seleccion_fewshot(train: Sequence[Mapping], few_shot: Mapping) -> dict:
    """Rehace la selección del pre-registro sobre la copia de train final y la compara con el config.

    Devuelve los ejemplos (nombre, código), solo para armar los prompts, y lo público: hash de los
    ids, códigos, sectores y si coinciden con `few_shot` del config.
    """
    sector_de = sector_de_prefijo(few_shot["sectores_oficiales"])
    pos = elegir_fewshot([r["codigo_act"] for r in train], sector_de, few_shot["k"], few_shot["semilla"])
    elegidas = [train[p] for p in pos]
    publico = {
        "sha256_ids": sha256_ids([r["id"] for r in elegidas]),
        "codigos": [r["codigo_act"] for r in elegidas],
        "sectores": [sector_de[r["codigo_act"][:2]] for r in elegidas],
    }
    publico["coincide_con_el_config"] = all(publico[k] == few_shot[k] for k in ("sha256_ids", "codigos", "sectores"))
    return {
        "ejemplos": [(r["nom_estab"], r["codigo_act"]) for r in elegidas],
        "ids": [r["id"] for r in elegidas],
        "publico": publico,
    }


# --- Programa principal (laptop) ---


def _leer_copia(raiz: Path, entrada: Mapping, archivo: str, campos: Sequence[str]) -> list[dict]:
    e = entrada["archivos"][archivo]
    datos = (raiz / e["ruta"]).read_bytes()
    if hashlib.sha256(datos).hexdigest() != e["sha256"]:
        raise ErrorPlantilla(f"El SHA-256 de {archivo} no es el del manifest.")
    filas = parsear_jsonl(datos, campos)
    if len(filas) != e["filas"]:
        raise ErrorPlantilla(f"Las filas de {archivo} no son las del manifest.")
    return filas


def _si(v: bool) -> str:
    return "sí" if v else "NO"


def _imprimir(res: Mapping, archivos_iguales: Mapping[str, bool] | None) -> None:
    if archivos_iguales is not None:
        print("(a) Archivos del tokenizador iguales entre las dos revisiones:")
        for f, igual in archivos_iguales.items():
            print(f"  {f}: {_si(igual)}")
    a = res.get("a_tokenizadores")
    if a:
        print(
            f"(a) Tokenizadores idénticos: {_si(a['identicos'])} (vocabulario {_si(a['vocabulario_igual'])}, "
            f"tokens agregados {_si(a['tokens_agregados_iguales'])}, plantilla {_si(a['plantilla_igual'])}, "
            f"{a['textos_con_ids_distintos']} de {a['textos']:,} prompts con ids distintos)"
        )
    for tam, t in res["tokens_especiales"].items():
        print(f"Tokens especiales de una sola ficha y distintos ({tam}): {_si(t['ok'])}")
    for tam, por_modo in res["modos"].items():
        for modo, r in por_modo.items():
            letra = "(c)" if MODOS[modo]["fewshot"] else "(b)"
            for conj, c in r["conjuntos"].items():
                print(
                    f"{tam} {modo} {conj}: {c['filas']:,} filas; {letra} plantilla distinta en {c['plantilla_distinta']}; "
                    f"(d) fichas máx {c['fichas_max']}, media {c['fichas_media']}; máx + {r['nuevas']} = "
                    f"{c['fichas_max'] + r['nuevas']} <= {r['limite']}: {_si(c['no_caben'] == 0)} ({c['no_caben']} no caben)"
                )
    print(f"(b) 0-shot igual a la plantilla en todas las filas: {_si(res['b_0shot_ok'])}")
    print(f"(c) few-shot igual a la plantilla en todas las filas: {_si(res['c_fewshot_ok'])}")
    print(f"(d) todos los prompts caben con sus fichas nuevas: {_si(res['d_largos_ok'])}")
    print(f"Resultado: {'todo pasa' if res['ok'] else 'PARADA 1'}")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import os
    import sys
    import types

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True, help="carpeta de la caché de Hugging Face (fuera del repo)")
    parser.add_argument("--raiz", default=".", help="raíz del repo (por omisión, la carpeta actual)")
    args = parser.parse_args(argv)
    os.environ["HF_HUB_CACHE"] = str(Path(args.cache).resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import huggingface_hub
    import transformers
    from transformers import AutoTokenizer

    from src.humo_kaggle import descargar_verificado, sha256_archivo

    raiz = Path(args.raiz).resolve()
    manifest = json.loads((raiz / "data" / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((raiz / "evals" / "fase4_preregistro.json").read_text(encoding="utf-8"))
    kd = manifest["kaggle_datasets"]
    try:
        conjuntos = {
            "test": _leer_copia(raiz, kd["denue_scian_evaluacion"], "test.jsonl", CAMPOS_RESERVADOS),
            "hard": _leer_copia(raiz, kd["denue_scian_evaluacion"], "hard.jsonl", CAMPOS_RESERVADOS),
            "validacion": _leer_copia(raiz, kd["denue_scian_clasificador"], "val.jsonl", CAMPOS_CON_CODIGO),
        }
        train = _leer_copia(raiz, kd["denue_scian_clasificador"], "train_final.jsonl", CAMPOS_CON_CODIGO)
    except (ErrorPlantilla, ErrorPreregistro) as e:
        sys.exit(f"Me detengo: {e}")
    print(f"transformers {transformers.__version__}, huggingface_hub {huggingface_hub.__version__}")
    print("Copias de Kaggle con el SHA-256 del manifest: " + ", ".join(f"{c} {len(r):,} filas" for c, r in conjuntos.items())
          + f", train final {len(train):,} filas")
    sel = seleccion_fewshot(train, config["few_shot"])
    print(f"Selección few-shot igual a la del config (hash de ids, códigos y sectores): {_si(sel['publico']['coincide_con_el_config'])}")
    if not sel["publico"]["coincide_con_el_config"]:
        sys.exit("PARADA 1: la selección few-shot no es la del config.")

    bases = {}
    for m in config["modelos"]:
        if "base" in m:
            bases.setdefault(m["tamano"], dict(m["base"]))
    mods = types.SimpleNamespace(hf=huggingface_hub)
    toks, rutas = {}, {}
    for tam, base in bases.items():
        ruta, verif = descargar_verificado(mods, base["repo"], base["revision"], list(PATRONES_TOKENIZADOR))
        rutas[tam] = ruta
        print(f"{base['repo']} en {base['revision']}: carpeta de la revisión fijada; {len(verif)} archivos LFS con su SHA-256 verificado")
        toks[tam] = AutoTokenizer.from_pretrained(str(ruta))
    archivos_iguales = None
    if len(rutas) == 2:
        a, b = rutas.values()
        archivos_iguales = {}
        for f in PATRONES_TOKENIZADOR:
            if (a / f).is_file() or (b / f).is_file():
                archivos_iguales[f] = (a / f).is_file() and (b / f).is_file() and sha256_archivo(a / f) == sha256_archivo(b / f)
    nombres = {c: [r["nom_estab"] for r in rs] for c, rs in conjuntos.items()}
    res = comprobar_tokenizadores(toks, nombres, sel["ejemplos"], config["largo_maximo"])
    _imprimir(res, archivos_iguales)
    if not res["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
