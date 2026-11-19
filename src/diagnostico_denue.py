"""Diagnóstico del DENUE (Fase 1, paso 2a): codificación, clee y nom_estab.

Imprime y guarda solo conteos y códigos SCIAN; nunca filas de datos.
Uso: .venv/Scripts/python -m src.diagnostico_denue
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from collections import Counter

import numpy as np

from src.denue import REPO_RAIZ, HashNoCoincideError, cargar_denue, entrada_manifest

COLUMNAS = ["id", "clee", "nom_estab", "codigo_act", "nombre_act"]
TEXTOS_TIPO_NA = ["", "NA", "NULL", "NONE", "N/A"]
SEIS_DIGITOS = r"[0-9]{6}"  # dígitos ASCII, como la regla 5 de CLAUDE.md
NO_SON_6_DIGITOS = "(no son 6 dígitos)"
SEMILLA = 42  # este diagnóstico no usa aleatoriedad; se registra por la regla 1


def _git(*args: str) -> str:
    resultado = subprocess.run(
        ["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True
    )
    return resultado.stdout.strip()


def _pct(parte: int, total: int) -> float:
    return round(100 * parte / total, 4) if total else 0.0


def perfil_de_bytes(entrada: dict) -> dict:
    """Conteos de bytes del CSV completo; verifica el hash sobre los mismos bytes."""
    datos = entrada["ruta"].read_bytes()
    sha = hashlib.sha256(datos).hexdigest()
    if sha != entrada["sha256"]:
        raise HashNoCoincideError(f"SHA-256 calculado {sha}, registrado {entrada['sha256']}.")
    conteo = np.bincount(np.frombuffer(datos, dtype=np.uint8), minlength=256)
    try:
        datos.decode("utf-8")
        utf8_valido, primer_invalido = True, None
    except UnicodeDecodeError as e:
        utf8_valido, primer_invalido = False, e.start
    control = int(conteo[0x00:0x09].sum() + conteo[0x0B] + conteo[0x0C] + conteo[0x0E:0x20].sum())
    return {
        "tamano_bytes": len(datos),
        "bytes_0x80_0x9F": int(conteo[0x80:0xA0].sum()),
        "bytes_0xA0_0xFF": int(conteo[0xA0:].sum()),
        "bytes_de_control_sin_tab_lf_cr": control,
        "bytes_CR": int(conteo[0x0D]),
        "bytes_LF": int(conteo[0x0A]),
        "bom_utf8": datos[:3] == b"\xef\xbb\xbf",
        "utf8_valido": utf8_valido,
        "utf8_primer_byte_invalido": primer_invalido,
        "conclusion": "latin-1 (sin bytes 0x80-0x9F, cp1252 lee igual)"
        if conteo[0x80:0xA0].sum() == 0
        else "hay bytes 0x80-0x9F: revisar cp1252",
    }


def diagnostico_clee(df) -> dict:
    total = len(df)
    clee, codigo = df["clee"], df["codigo_act"]
    longitudes = clee.str.len().value_counts().sort_index()
    largo_max = int(longitudes.index.max())

    por_posicion = {
        p: int((clee.str.slice(p, p + 6) == codigo).sum()) for p in range(largo_max - 5)
    }
    mejor = max(por_posicion, key=por_posicion.get)
    fragmento = clee.str.slice(mejor, mejor + 6)
    coincide = fragmento == codigo
    discrepa = ~coincide

    # Etiquetas antes de contar: un fragmento que no es de 6 dígitos no se muestra.
    frag_mostrable = fragmento.where(fragmento.str.fullmatch(SEIS_DIGITOS), NO_SON_6_DIGITOS)
    cod_mostrable = codigo.where(codigo.str.fullmatch(SEIS_DIGITOS), NO_SON_6_DIGITOS)
    pares = Counter(zip(frag_mostrable[discrepa], cod_mostrable[discrepa])).most_common(10)

    return {
        "longitudes_clee": {int(k): int(v) for k, v in longitudes.items()},
        "valores_unicos_id": int(df["id"].nunique()),
        "valores_unicos_clee": int(clee.nunique()),
        "codigo_act_con_6_digitos": int(codigo.str.fullmatch(SEIS_DIGITOS).sum()),
        "coincidencias_por_posicion_inicial_0": por_posicion,
        "mejor_posicion_inicial_0": mejor,
        "caracteres_1_indexados": f"{mejor + 1} a {mejor + 6}",
        "coincidencias": int(coincide.sum()),
        "discrepancias": int(discrepa.sum()),
        "pct_coincidencias": _pct(int(coincide.sum()), total),
        "discrepancias_con_fragmento_de_6_digitos": int(
            (discrepa & fragmento.str.fullmatch(SEIS_DIGITOS)).sum()
        ),
        "top10_pares_discrepantes": [
            {"fragmento_clee": f, "codigo_act": c, "filas": n} for (f, c), n in pares
        ],
    }


def diagnostico_nom_estab(df) -> dict:
    total = len(df)
    nombre = df["nom_estab"]
    normalizado = nombre.str.strip().str.upper()
    nombre_act_norm = df["nombre_act"].str.strip().str.upper()
    igual_exacto = int((nombre == df["nombre_act"]).sum())
    igual_norm = int((normalizado == nombre_act_norm).sum())
    return {
        "exactos": {v: int((nombre == v).sum()) for v in TEXTOS_TIPO_NA},
        "solo_espacios_no_vacio": int(((normalizado == "") & (nombre != "")).sum()),
        "tras_strip_y_mayusculas": {v: int((normalizado == v).sum()) for v in TEXTOS_TIPO_NA},
        "nombre_act_vacio": int((df["nombre_act"] == "").sum()),
        "igual_a_nombre_act": {
            "exacto": igual_exacto,
            "pct_exacto": _pct(igual_exacto, total),
            "tras_strip_y_mayusculas": igual_norm,
            "pct_tras_strip_y_mayusculas": _pct(igual_norm, total),
            "nota": "Solo conteo; qué hacer con esas filas se decide en el análisis exploratorio.",
        },
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos bien aunque la salida vaya a un archivo o tubería
    fecha = datetime.date.today().isoformat()
    entrada = entrada_manifest()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""

    codificacion = perfil_de_bytes(entrada)
    df = cargar_denue(COLUMNAS)
    clee = diagnostico_clee(df)
    nom_estab = diagnostico_nom_estab(df)

    resultado = {
        "fecha": fecha,
        "experimento": "fase1_paso2a_diagnostico_denue",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": "src/diagnostico_denue.py",
            "cargador": "src/denue.py:cargar_denue",
            "columnas": COLUMNAS,
            "codificacion": entrada["codificacion"],
        },
        "semilla": SEMILLA,
        "split": "ninguno: CSV completo (todavía no hay splits)",
        "hash_split": entrada["sha256"],
        "hash_split_origen": "data/manifest.json, fuentes.denue_cdmx.sha256",
        "tamano_muestra": len(df),
        "horas_gpu": None,
        "codificacion_csv": codificacion,
        "clee_vs_codigo_act": clee,
        "nom_estab": nom_estab,
    }
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_diagnostico-denue.json"
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Resumen en la terminal: solo conteos y códigos SCIAN.
    print(f"Filas: {len(df):,}  |  commit {commit[:7]}  |  árbol limpio: {arbol_limpio}")
    print(f"Codificación: {json.dumps(codificacion, ensure_ascii=False)}")
    print(
        f"clee: longitudes {clee['longitudes_clee']}; únicos id {clee['valores_unicos_id']:,}, "
        f"clee {clee['valores_unicos_clee']:,}; codigo_act de 6 dígitos {clee['codigo_act_con_6_digitos']:,}"
    )
    print(
        f"clee caracteres {clee['caracteres_1_indexados']} vs codigo_act: "
        f"{clee['coincidencias']:,} coinciden ({clee['pct_coincidencias']} %), "
        f"{clee['discrepancias']:,} discrepan"
    )
    for par in clee["top10_pares_discrepantes"]:
        print(f"  {par['fragmento_clee']} -> {par['codigo_act']}: {par['filas']:,}")
    print(f"nom_estab exactos: {nom_estab['exactos']}")
    print(f"nom_estab solo espacios: {nom_estab['solo_espacios_no_vacio']}")
    print(f"nom_estab tras strip y mayúsculas: {nom_estab['tras_strip_y_mayusculas']}")
    print(f"nom_estab == nombre_act: {nom_estab['igual_a_nombre_act']}")
    print(f"Guardado en {salida.relative_to(REPO_RAIZ).as_posix()}")


if __name__ == "__main__":
    main()
