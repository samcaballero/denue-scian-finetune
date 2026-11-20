"""Llaves normalizadas de `nom_estab` y tamaños de grupo (Fase 1, paso 3).

Compara las tres variantes de puntuación de `src/normalizacion.py` (A: todo signo
a espacio; B: todo signo se borra; C: punto y apóstrofo se borran y el resto a
espacio) y describe los grupos de la variante elegida.

Imprime y guarda solo conteos y puntos de código; nunca nombres ni filas del DENUE.
Uso: .venv/Scripts/python -m src.grupos_nombres
"""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping

from src.denue import REPO_RAIZ, cargar_denue, entrada_manifest
from src.normalizacion import VARIANTE, VARIANTES, _normalizar

COLUMNAS = ["nom_estab"]
SEMILLA = 42  # este análisis no usa aleatoriedad; se registra por la regla 1
CUBETAS = (("1", 1, 1), ("2-9", 2, 9), ("10-99", 10, 99), ("100+", 100, None))
# Marca de la Ñ para medir el efecto de Ñ→N: Ŋ (U+014A) es una letra sin
# descomposición que no puede aparecer en un CSV latin-1.
MARCA_ENIE = "Ŋ"
DEFINICIONES = {
    "A": "todo signo se cambia por espacio",
    "B": "todo signo se borra",
    "C": "el punto y el apóstrofo (' y ’) se borran; el resto de signos se cambia por espacio",
}
NOTA_FUSIONES = (
    "X_en_Y: llaves_de_origen = llaves de X que comparten llave de Y con otra llave de X; "
    "llaves_resultantes = llaves de Y que juntan más de una llave de X; filas = filas en esas llaves de Y."
)


def _git(*args: str) -> str:
    resultado = subprocess.run(
        ["git", *args], cwd=REPO_RAIZ, capture_output=True, text=True, check=True
    )
    return resultado.stdout.strip()


def es_signo(c: str) -> bool:
    """Carácter que la normalización trata como puntuación: no es letra, número ni espacio."""
    return not (c.isspace() or unicodedata.category(c)[0] in "LN")


def inventario_de_caracteres(filas: Mapping[str, int]) -> dict[str, dict]:
    """Caracteres que no son letra ASCII, dígito ASCII ni espacio, con sus conteos.

    `nombres` cuenta nombres distintos que contienen el carácter; `filas`, sus filas.
    """
    nombres_c: Counter = Counter()
    filas_c: Counter = Counter()
    for nombre, f in filas.items():
        for c in set(nombre):
            if not (c.isascii() and (c.isalnum() or c == " ")):
                nombres_c[c] += 1
                filas_c[c] += f
    orden = sorted(filas_c, key=lambda c: (-filas_c[c], ord(c)))
    return {
        f"U+{ord(c):04X}": {
            "caracter": c,
            "nombre_unicode": unicodedata.name(c, "?"),
            "categoria": unicodedata.category(c),
            "nombres": nombres_c[c],
            "filas": filas_c[c],
        }
        for c in orden
    }


def llaves(nombres, se_borran) -> dict[str, str]:
    return {n: _normalizar(n, se_borran) for n in nombres}


def resumen_llaves(llave: Mapping[str, str], filas: Mapping[str, int]) -> dict[str, int]:
    distintas = set(llave.values())
    vacios = [n for n, k in llave.items() if k == ""]
    return {
        "llaves_distintas": len(distintas),
        "filas_llave_vacia": sum(filas[n] for n in vacios),
        "nombres_distintos_llave_vacia": len(vacios),
    }


def fusiones(llave_x: Mapping[str, str], llave_y: Mapping[str, str], filas: Mapping[str, int]) -> dict:
    """Llaves de X que se juntan en Y (ver NOTA_FUSIONES)."""
    origen_por_destino: dict[str, set] = defaultdict(set)
    for nombre, ky in llave_y.items():
        origen_por_destino[ky].add(llave_x[nombre])
    juntas = {ky for ky, origen in origen_por_destino.items() if len(origen) > 1}
    origen = set().union(*(origen_por_destino[ky] for ky in juntas))
    return {
        "llaves_de_origen": len(origen),
        "llaves_resultantes": len(juntas),
        "filas": sum(filas[n] for n, ky in llave_y.items() if ky in juntas),
    }


def grupos_por_tamano(filas_por_llave: Mapping[str, int]) -> dict:
    """Grupos por número de filas, sin contar la llave vacía (que se reporta aparte)."""
    sin_vacia = {k: f for k, f in filas_por_llave.items() if k != ""}
    cubetas = {}
    for nombre, minimo, maximo in CUBETAS:
        tamanos = [f for f in sin_vacia.values() if f >= minimo and (maximo is None or f <= maximo)]
        cubetas[nombre] = {"grupos": len(tamanos), "filas": sum(tamanos)}
    return {
        "grupos": cubetas,
        "grupo_mas_grande_filas": max(sin_vacia.values(), default=0),
        "filas_llave_vacia": filas_por_llave.get("", 0),
    }


def perfil_nom_estab(filas: Mapping[str, int]) -> dict:
    def contar(pred):
        nombres = [n for n in filas if pred(n)]
        return {"nombres": len(nombres), "filas": sum(filas[n] for n in nombres)}

    return {
        "con_minusculas_ascii": contar(lambda n: any("a" <= c <= "z" for c in n)),
        "con_espacios_dobles": contar(lambda n: "  " in n),
        "con_espacios_al_inicio_o_al_final": contar(lambda n: n != n.strip(" ")),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fecha = datetime.date.today().isoformat()
    commit = _git("rev-parse", "HEAD")
    arbol_limpio = _git("status", "--porcelain", "--untracked-files=no") == ""
    entrada = entrada_manifest()

    df = cargar_denue(COLUMNAS)
    conteo = df["nom_estab"].value_counts()
    filas = {str(n): int(f) for n, f in conteo.items()}

    # 1. Las tres variantes.
    por_variante = {v: llaves(filas, VARIANTES[v]) for v in DEFINICIONES}
    pares = [("A", "B"), ("A", "C"), ("B", "C")]
    comparacion = {
        "definiciones": DEFINICIONES,
        "por_variante": {v: resumen_llaves(k, filas) for v, k in por_variante.items()},
        "fusiones": {
            f"{x}_en_{y}": fusiones(por_variante[x], por_variante[y], filas)
            for a, b in pares
            for x, y in ((a, b), (b, a))
        },
        "nota_fusiones": NOTA_FUSIONES,
        "nombres_con_llave_distinta": {
            f"{x}_{y}": {
                "nombres": sum(por_variante[x][n] != por_variante[y][n] for n in filas),
                "filas": sum(f for n, f in filas.items() if por_variante[x][n] != por_variante[y][n]),
            }
            for x, y in pares
        },
    }

    # 2. Efecto de cada signo: borrar solo ese signo (el resto a espacio) contra A.
    caracteres = inventario_de_caracteres(filas)
    llave_a = por_variante["A"]
    por_signo = {}
    for cp, info in caracteres.items():
        c = info["caracter"]
        if not es_signo(c):
            continue
        borrado = dict(llave_a)
        for n in filas:
            if c in n:
                borrado[n] = _normalizar(n, frozenset({c}))
        por_signo[cp] = {
            "caracter": c,
            "borrarlo_junta": fusiones(llave_a, borrado, filas),
            "espacio_junta": fusiones(borrado, llave_a, filas),
        }
    comparacion["por_signo_frente_a_A"] = por_signo

    # 3. Ñ→N contra conservar la Ñ (con la variante elegida).
    elegida = por_variante[VARIANTE]
    con_enie = {
        n: _normalizar(n.replace("Ñ", MARCA_ENIE).replace("ñ", MARCA_ENIE), VARIANTES[VARIANTE])
        for n in filas
    }
    efecto_enie = {
        "llaves_conservando_la_enie": len(set(con_enie.values())),
        "llaves_con_enie_a_n": len(set(elegida.values())),
        "fusiones": fusiones(con_enie, elegida, filas),
        "nota": f"Para conservar la Ñ se cambia por {MARCA_ENIE} (U+014A) antes de normalizar.",
    }

    # 4. Grupos de la variante elegida.
    filas_por_llave: Counter = Counter()
    nombres_por_llave: Counter = Counter()
    for n, f in filas.items():
        filas_por_llave[elegida[n]] += f
        nombres_por_llave[elegida[n]] += 1
    tamanos = grupos_por_tamano(filas_por_llave)
    resultado_elegida = {
        "variante": VARIANTE,
        "nombres_distintos": len(filas),
        "llaves_distintas": len(filas_por_llave),
        "filas_llave_vacia": tamanos["filas_llave_vacia"],
        "nombres_distintos_llave_vacia": nombres_por_llave.get("", 0),
        "llaves_con_mas_de_un_nom_estab_distinto": sum(
            1 for k, v in nombres_por_llave.items() if k != "" and v > 1
        ),
        "grupos_por_tamano_en_filas": tamanos["grupos"],
        "grupo_mas_grande_filas": tamanos["grupo_mas_grande_filas"],
        "nota": "Los grupos por tamaño y el grupo más grande no cuentan la llave vacía.",
    }

    resultado = {
        "fecha": fecha,
        "experimento": "fase1_paso3_normalizacion_nombres",
        "commit": commit,
        "arbol_limpio": arbol_limpio,
        "modelo": None,
        "configuracion": {
            "script": "src/grupos_nombres.py",
            "cargador": "src/denue.py:cargar_denue",
            "funcion": "src/normalizacion.py:normalizar_nombre",
            "columnas": COLUMNAS,
            "codificacion": entrada["codificacion"],
            "variante_elegida": VARIANTE,
            "python": platform.python_version(),
            "unicodedata_version": unicodedata.unidata_version,
        },
        "semilla": SEMILLA,
        "split": "ninguno: CSV completo (todavía no hay splits)",
        "hash_split": entrada["sha256"],
        "hash_split_origen": "data/manifest.json, fuentes.denue_cdmx.sha256",
        "tamano_muestra": len(df),
        "horas_gpu": None,
        "nom_estab": {"nombres_distintos": len(filas), **perfil_nom_estab(filas)},
        "caracteres": caracteres,
        "comparacion_variantes": comparacion,
        "efecto_enie": efecto_enie,
        "variante_elegida": resultado_elegida,
    }
    salida = REPO_RAIZ / "results" / f"{fecha}_fase1_normalizacion-nombres.json"
    with open(salida, "w", encoding="utf-8", newline="\n") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Resumen en la terminal: solo conteos y puntos de código.
    print(f"Filas: {len(df):,}  |  commit {commit[:7]}  |  árbol limpio: {arbol_limpio}")
    print(f"nom_estab: {resultado['nom_estab']}")
    for cp, info in caracteres.items():
        print(f"  {cp} {info['categoria']} {info['nombre_unicode']}: {info['nombres']:,} nombres, {info['filas']:,} filas")
    for v, r in comparacion["por_variante"].items():
        print(f"Variante {v}: {r}")
    for k, r in comparacion["fusiones"].items():
        print(f"  {k}: {r}")
    print(f"  nombres con llave distinta: {comparacion['nombres_con_llave_distinta']}")
    for cp, r in por_signo.items():
        print(f"  {cp}: borrarlo junta {r['borrarlo_junta']}; espacio junta {r['espacio_junta']}")
    print(f"Ñ→N: {efecto_enie['llaves_conservando_la_enie']:,} → {efecto_enie['llaves_con_enie_a_n']:,} llaves; {efecto_enie['fusiones']}")
    print(f"Variante elegida: {resultado_elegida}")
    print(f"Guardado en {salida.relative_to(REPO_RAIZ).as_posix()}")


if __name__ == "__main__":
    main()
