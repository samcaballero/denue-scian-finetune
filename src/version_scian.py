"""Versión del SCIAN de `codigo_act` (Fase 1, paso 2b).

La versión se decide por los códigos presentes, no por conteos: si todos los
`codigo_act` existen en una versión y alguno no existe en la otra, es la primera.
Si todos existen en ambas, o hay ausentes en las dos, no se decide y el script
termina con error.

La evidencia secundaria (no decide): `nombre_act` contra los títulos de cada
versión, las tablas comparativas 2023-2018 y los códigos que guarda `clee`
(caracteres 6 a 11).

Imprime y guarda solo conteos, códigos y títulos del catálogo; nunca textos de
`nombre_act` ni filas del DENUE.
Uso: .venv/Scripts/python -m src.version_scian
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict

from src.denue import REPO_RAIZ, cargar_denue, entrada_manifest
from src.scian import (
    CATALOGO,
    FUENTE_TABLAS,
    VERSIONES,
    archivo_verificado,
    codigos_en_tablas,
    leer_catalogo,
    leer_manifest,
    texto_de_paginas,
)

COLUMNAS = ["clee", "codigo_act", "nombre_act"]
SEIS_DIGITOS = r"[0-9]{6}"  # dígitos ASCII, como la regla 5 de CLAUDE.md
CATEGORIAS = ("solo_2018", "solo_2023", "ambas", "ninguna")
SEMILLA = 42  # este análisis no usa aleatoriedad; se registra por la regla 1
REGLA = (
    "Si todos los codigo_act existen en una versión y alguno no existe en la otra, "
    "la versión es la primera; si todos existen en ambas o hay ausentes en las dos, no se decide."
)


def decidir_version(codigos: set[str], cat_2023: set[str], cat_2018: set[str]) -> str | None:
    """'2023', '2018' o None (sin decisión), según la regla del paso 2b."""
    faltan_2023 = codigos - cat_2023
    faltan_2018 = codigos - cat_2018
    if not faltan_2023 and faltan_2018:
        return "2023"
    if not faltan_2018 and faltan_2023:
        return "2018"
    return None


def clasificar_codigo(codigo: str, cat_2023: set[str], cat_2018: set[str]) -> str:
    en_2023, en_2018 = codigo in cat_2023, codigo in cat_2018
    if en_2023 and en_2018:
        return "ambas"
    if en_2023:
        return "solo_2023"
    if en_2018:
        return "solo_2018"
    return "ninguna"


def resumen_por_categoria(
    filas_por_codigo: dict[str, int], cat_2023: set[str], cat_2018: set[str]
) -> dict[str, dict[str, int]]:
    """Códigos distintos y filas en cada categoría: solo 2018, solo 2023, ambas, ninguna."""
    resumen = {k: {"codigos": 0, "filas": 0} for k in CATEGORIAS}
    for codigo, filas in filas_por_codigo.items():
        k = clasificar_codigo(codigo, cat_2023, cat_2018)
        resumen[k]["codigos"] += 1
        resumen[k]["filas"] += filas
    return resumen


def clave_titulo(titulo: str) -> str:
    """Comparación laxa de títulos: NFC, espacios colapsados, sin distinguir mayúsculas.

    Solo sirve para comparar títulos; no es la normalización de la regla 2.
    """
    return " ".join(unicodedata.normalize("NFC", titulo).split()).casefold()


def _a_dict(serie) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(serie.items())}


def _top(filas_por_codigo: dict[str, int], n: int = 10) -> list[dict]:
    orden = sorted(filas_por_codigo.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [{"codigo": c, "filas": f} for c, f in orden]


def _git(*args: str) -> str:
    resultado = subprocess.run(
        ["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True
    )
    return resultado.stdout.strip()


def nombre_act_contra_titulos(pares: dict[tuple[str, str], int], catalogos: dict) -> dict:
    """Coincidencia de cada par (codigo_act, nombre_act) con el título de cada versión."""
    por_version = {}
    for v in VERSIONES:
        cat = catalogos[v]
        res = {k: {"pares": 0, "filas": 0} for k in ("exacto", "solo_laxo", "distinto", "codigo_inexistente")}
        for (codigo, nombre), filas in pares.items():
            if codigo not in cat:
                k = "codigo_inexistente"
            elif nombre == cat[codigo]:
                k = "exacto"
            elif clave_titulo(nombre) == clave_titulo(cat[codigo]):
                k = "solo_laxo"
            else:
                k = "distinto"
            res[k]["pares"] += 1
            res[k]["filas"] += filas
        por_version[v] = res

    # Con qué versión coincide cada par (comparación laxa).
    cruce = {k: {"pares": 0, "filas": 0} for k in CATEGORIAS}
    for (codigo, nombre), filas in pares.items():
        clave = clave_titulo(nombre)
        en = {
            v: codigo in catalogos[v] and clave == clave_titulo(catalogos[v][codigo])
            for v in VERSIONES
        }
        k = clasificar_codigo(
            codigo,
            {codigo} if en["2023"] else set(),
            {codigo} if en["2018"] else set(),
        )
        cruce[k]["pares"] += 1
        cruce[k]["filas"] += filas

    nombres_por_codigo = Counter(codigo for codigo, _ in pares)
    varios = sorted(c for c, n in nombres_por_codigo.items() if n > 1)
    return {
        "pares_codigo_nombre_act": len(pares),
        "codigos_con_mas_de_un_nombre_act": {"total": len(varios), "codigos": varios},
        "por_version": por_version,
        "coincidencia_laxa_por_par": cruce,
        "nota": "Comparación laxa: NFC, espacios colapsados, sin distinguir mayúsculas.",
    }


def titulos_distintos(pares, filas_por_codigo, catalogos, tablas) -> dict:
    """Clases con el mismo código y título distinto (laxo) entre versiones."""
    cat_2023, cat_2018 = catalogos["2023"], catalogos["2018"]
    comunes = sorted(cat_2023.keys() & cat_2018.keys())
    cambiados = [c for c in comunes if clave_titulo(cat_2023[c]) != clave_titulo(cat_2018[c])]
    solo_formato = [
        c for c in comunes
        if cat_2023[c] != cat_2018[c] and clave_titulo(cat_2023[c]) == clave_titulo(cat_2018[c])
    ]

    coincidencia: dict[str, Counter] = defaultdict(Counter)
    for (codigo, nombre), filas in pares.items():
        if codigo in cambiados:
            clave = clave_titulo(nombre)
            if clave == clave_titulo(cat_2023[codigo]):
                coincidencia[codigo]["2023"] += filas
            elif clave == clave_titulo(cat_2018[codigo]):
                coincidencia[codigo]["2018"] += filas
            else:
                coincidencia[codigo]["ninguna"] += filas

    detalle = []
    total = Counter()
    for c in cambiados:
        if c not in filas_por_codigo:
            continue
        por = {k: coincidencia[c].get(k, 0) for k in ("2023", "2018", "ninguna")}
        total.update(por)
        detalle.append(
            {
                "codigo": c,
                "titulo_2023": cat_2023[c],
                "titulo_2018": cat_2018[c],
                "en_tablas_comparativas": c in tablas,
                "filas": filas_por_codigo[c],
                "filas_por_coincidencia_de_nombre_act": por,
            }
        )
    return {
        "clases_en_ambas_versiones": len(comunes),
        "con_titulo_distinto": len(cambiados),
        "con_titulo_distinto_en_tablas_comparativas": sum(c in tablas for c in cambiados),
        "solo_difieren_en_mayusculas_o_espacios": len(solo_formato),
        "presentes_en_el_denue": len(detalle),
        "filas_por_coincidencia_de_nombre_act": {k: total.get(k, 0) for k in ("2023", "2018", "ninguna")},
        "detalle": detalle,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    manifest = leer_manifest()
    entrada_denue = entrada_manifest()

    catalogos = {
        v: leer_catalogo(archivo_verificado(manifest["catalogos"][CATALOGO[v]])) for v in VERSIONES
    }
    set_2023, set_2018 = set(catalogos["2023"]), set(catalogos["2018"])
    tablas = codigos_en_tablas(texto_de_paginas(archivo_verificado(manifest["fuentes"][FUENTE_TABLAS])))

    df = cargar_denue(COLUMNAS)
    codigo = df["codigo_act"]

    # 1. Decisión por los códigos presentes.
    filas_por_codigo = _a_dict(codigo.value_counts())
    no_6 = ~codigo.str.fullmatch(SEIS_DIGITOS)
    ausentes = {
        v: {c: f for c, f in filas_por_codigo.items() if c not in catalogos[v]} for v in VERSIONES
    }
    version = decidir_version(set(filas_por_codigo), set_2023, set_2018)

    # 2. Evidencia secundaria.
    pares = {
        (str(c), str(n)): int(f) for (c, n), f in df.groupby(["codigo_act", "nombre_act"]).size().items()
    }
    nombre_act = nombre_act_contra_titulos(pares, catalogos)
    cambios = titulos_distintos(pares, filas_por_codigo, catalogos, tablas)

    solo = {"solo_2023": set_2023 - set_2018, "solo_2018": set_2018 - set_2023}
    tablas_res = {
        "codigos_en_tablas": len(tablas),
        "codigos_en_tablas_que_no_estan_en_ningun_catalogo": sorted(tablas - set_2023 - set_2018),
        **{
            k: {"total": len(s), "en_tablas": len(s & tablas), "no_en_tablas": sorted(s - tablas)}
            for k, s in solo.items()
        },
        "codigos_del_denue_ausentes_en_alguna_version_que_estan_en_tablas": {
            v: {"total": len(a), "en_tablas": sum(c in tablas for c in a)} for v, a in ausentes.items()
        },
    }

    fragmento = df["clee"].str.slice(5, 11)
    discrepa = fragmento != codigo
    filas_frag = _a_dict(fragmento.value_counts())
    filas_frag_discrepa = _a_dict(fragmento[discrepa].value_counts())
    ninguna = {c: f for c, f in filas_frag.items() if clasificar_codigo(c, set_2023, set_2018) == "ninguna"}
    solo_2018_discrepa = {
        c: f for c, f in filas_frag_discrepa.items() if clasificar_codigo(c, set_2023, set_2018) == "solo_2018"
    }
    clee = {
        "caracteres_1_indexados": "6 a 11",
        "fragmentos_que_no_son_6_digitos": int((~fragmento.str.fullmatch(SEIS_DIGITOS)).sum()),
        "todas_las_filas": {"filas": len(df), **resumen_por_categoria(filas_frag, set_2023, set_2018)},
        "filas_que_discrepan_de_codigo_act": {
            "filas": int(discrepa.sum()),
            **resumen_por_categoria(filas_frag_discrepa, set_2023, set_2018),
        },
        "top10_en_ninguna_version_todas_las_filas": _top(ninguna),
        "top10_solo_2018_en_filas_que_discrepan": _top(solo_2018_discrepa),
        "nota": "Sin el catálogo del SCIAN 2013 no se afirma de qué versión vienen los códigos de 'ninguna'.",
    }

    resultado = {
        "fecha": fecha,
        "experimento": "fase1_paso2b_version_scian",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": "src/version_scian.py",
            "cargador": "src/denue.py:cargar_denue",
            "columnas": COLUMNAS,
            "codificacion": entrada_denue["codificacion"],
            "catalogos": {
                v: {"ruta": manifest["catalogos"][CATALOGO[v]]["ruta"], "sha256": manifest["catalogos"][CATALOGO[v]]["sha256"]}
                for v in VERSIONES
            },
            "tablas_comparativas_sha256": manifest["fuentes"][FUENTE_TABLAS]["sha256"],
            "regla_de_decision": REGLA,
        },
        "semilla": SEMILLA,
        "split": "ninguno: CSV completo (todavía no hay splits)",
        "hash_split": entrada_denue["sha256"],
        "hash_split_origen": "data/manifest.json, fuentes.denue_cdmx.sha256",
        "tamano_muestra": len(df),
        "horas_gpu": None,
        "codigo_act": {
            "codigos_distintos": len(filas_por_codigo),
            "filas_que_no_son_6_digitos_ascii": int(no_6.sum()),
            "codigos_distintos_que_no_son_6_digitos_ascii": int(codigo[no_6].nunique()),
            "filas_por_codigo": filas_por_codigo,
        },
        **{
            f"ausentes_en_{v}": {
                "codigos": len(a),
                "filas": sum(a.values()),
                "detalle": dict(sorted(a.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
            for v, a in ausentes.items()
        },
        "decision": {"version": version, "regla": REGLA},
        "evidencia_secundaria": {
            "nombre_act_contra_titulos": nombre_act,
            "titulos_distintos_entre_versiones": cambios,
            "tablas_comparativas": tablas_res,
            "clee": clee,
        },
    }
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_version-scian.json"
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Resumen en la terminal: solo conteos y códigos.
    print(f"Filas: {len(df):,}  |  commit {commit[:7]}  |  árbol limpio: {arbol_limpio}")
    print(
        f"codigo_act: {len(filas_por_codigo)} códigos distintos; "
        f"{int(no_6.sum())} filas no son 6 dígitos ASCII"
    )
    for v, a in ausentes.items():
        lista = ", ".join(f"{c} ({f:,})" for c, f in sorted(a.items(), key=lambda kv: (-kv[1], kv[0])))
        print(f"Ausentes en {v}: {len(a)} códigos, {sum(a.values()):,} filas: {lista or '-'}")
    print(f"Decisión: {version}")
    print(f"nombre_act contra títulos ({nombre_act['pares_codigo_nombre_act']} pares):")
    for v, res in nombre_act["por_version"].items():
        print(f"  {v}: {res}")
    print(f"  coincidencia laxa por par: {nombre_act['coincidencia_laxa_por_par']}")
    print(f"  códigos con más de un nombre_act: {nombre_act['codigos_con_mas_de_un_nombre_act']['total']}")
    print(
        f"Títulos distintos entre versiones: {cambios['con_titulo_distinto']} de "
        f"{cambios['clases_en_ambas_versiones']} clases comunes "
        f"({cambios['con_titulo_distinto_en_tablas_comparativas']} en las tablas comparativas; "
        f"{cambios['solo_difieren_en_mayusculas_o_espacios']} solo difieren en mayúsculas o espacios); "
        f"{cambios['presentes_en_el_denue']} en el DENUE, filas por coincidencia de nombre_act: "
        f"{cambios['filas_por_coincidencia_de_nombre_act']}"
    )
    for d in cambios["detalle"]:
        print(f"  {d['codigo']}: {d['filas']:,} filas; {d['filas_por_coincidencia_de_nombre_act']}")
    print(
        f"Tablas comparativas: {len(tablas)} códigos; solo 2023 "
        f"{tablas_res['solo_2023']['en_tablas']}/{tablas_res['solo_2023']['total']} en tablas; solo 2018 "
        f"{tablas_res['solo_2018']['en_tablas']}/{tablas_res['solo_2018']['total']} en tablas; "
        f"en ningún catálogo: {tablas_res['codigos_en_tablas_que_no_estan_en_ningun_catalogo']}"
    )
    print(f"clee, todas las filas: {clee['todas_las_filas']}")
    print(f"clee, filas que discrepan: {clee['filas_que_discrepan_de_codigo_act']}")
    print(f"clee, top 10 en ninguna versión: {clee['top10_en_ninguna_version_todas_las_filas']}")
    print(f"clee, top 10 solo 2018 en discrepancias: {clee['top10_solo_2018_en_filas_que_discrepan']}")
    print(f"Guardado en {salida.relative_to(REPO_RAIZ).as_posix()}")

    if version is None:
        sys.exit("Sin decisión: todos los códigos existen en ambas versiones o hay ausentes en las dos.")


if __name__ == "__main__":
    main()
