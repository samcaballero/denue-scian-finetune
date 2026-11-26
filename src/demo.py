"""Demo de la charla (Fase 4, paso 19, Tema B): el Qwen3-0.6B base y el afinado, uno tras otro, en la laptop sin GPU.

Se lanza con scripts/demo.ps1, que pasa las rutas absolutas:

  .venv/Scripts/python -m src.demo --llama-server RUTA --dir-gguf RUTA [--nombres RUTA] [--ancho N] [--pausa]

- Antes de cargar nada verifica:
  - el SHA-256 de los dos GGUF de la cuantización que eligió la regla de la demo y de cada binario de llama.cpp, contra
    results/*_fase4_gguf-concordancia.json;
  - el del catálogo SCIAN, contra data/manifest.json.
- Levanta los dos llama-server en 127.0.0.1 y sin red (--offline). Antes de cada uno exige 3 GB libres; si no alcanza,
  dice cuánta memoria hay y qué procesos ocupan más, y no cierra nada. Calienta cada modelo con un nombre inventado y
  mide el arranque.
- Toma los nombres de data/demo/nombres.txt (ignorado por git; salta líneas vacías y las que empiezan con #) y después
  los que se teclean, hasta una línea vacía o "salir".
- A cada modelo le hace la misma pregunta que en el Tema A: prompt_generacion, greedy, corte de Kaggle y /detokenize.
  Muestra la categoría de la regla 5 con el título del catálogo oficial, nunca del modelo, y los ms de punta a punta.

Nada sale de la laptop: el cliente solo habla con 127.0.0.1. La demo no guarda nombres ni respuestas en ningún archivo
(los logs de llama-server, en data/demo/logs/, no llevan los prompts).
Solo usa la biblioteca estándar y módulos de src/ que tampoco traen nada más.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys
import textwrap
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from src import llama_servidor as ls
from src.gguf_fase4 import (DIR_GGUF, LLAMA_SERVER, NUEVAS, PATRON_CONCORDANCIA, PUERTOS, ROLES, nombre_gguf,
                            programas_abiertos, sha256_archivo)
from src.metricas_clasificador import CODIGO_EXISTENTE, CODIGO_INEXISTENTE, categoria, limpiar
from src.prompt_clasificador import RE_CODIGO, prompt_generacion, tiene_marca_especial
from src.scian import CATALOGO, leer_catalogo

REPO_RAIZ = Path(__file__).resolve().parents[1]
NOMBRES = REPO_RAIZ / "data" / "demo" / "nombres.txt"
LOGS = REPO_RAIZ / "data" / "demo" / "logs"
ANCHO = 56  # columnas: se lee a 20 pt en una ventana angosta
MINIMO_ANCHO = 32
SANGRIA = "    "
LINEAS_FORMATO_INVALIDO = 2
MAXIMO_CARACTERES = 200  # el prompt cabe de sobra en el contexto de 512 fichas
CALENTAMIENTO = "Negocio Inventado de Prueba"  # nombre inventado para calentar cada modelo (no se muestra)
ETIQUETAS = {"base": "Base (sin afinar)", "afinado": "Afinado"}
PROCESOS_A_MOSTRAR = 5


class ErrorDemo(RuntimeError):
    """La demo no puede arrancar; el mensaje nunca incluye nombres ni respuestas. `comando`: el arreglo, si lo hay."""

    def __init__(self, mensaje: str, comando: str | None = None):
        super().__init__(mensaje)
        self.comando = comando


# --- Verificación de archivos ---


def resultado_concordancia(raiz: Path = REPO_RAIZ) -> dict:
    hay = sorted((Path(raiz) / "results").glob(PATRON_CONCORDANCIA))
    if len(hay) != 1:
        raise ErrorDemo("Se necesita exactamente un resultado de concordancia en results/ (paso 19, Tema A).")
    return json.loads(hay[0].read_text(encoding="utf-8"))


def esperados(res: Mapping) -> dict:
    """La cuantización elegida, los GGUF de los dos modelos (archivo, bytes y SHA-256) y los binarios de llama.cpp."""
    cuant = res["regla_demo"]["elegida"]
    hm = res["herramientas_y_modelos"]
    gguf = {}
    for rol in ROLES:
        archivo = nombre_gguf(rol, cuant)
        e = hm["gguf"].get(archivo)
        if not e:
            raise ErrorDemo(f"{archivo} no está en el resultado de concordancia.")
        gguf[rol] = {"archivo": archivo, "bytes": e["bytes"], "sha256": e["sha256"]}
    return {"cuantizacion": cuant, "gguf": gguf, "binarios": dict(hm["llama_cpp"]["binarios_extraidos"])}


def verificar_archivos(llama_server: Path, dir_gguf: Path, esperado: Mapping) -> dict[str, Path]:
    """Exige cada binario de llama.cpp y los dos GGUF con su SHA-256 registrado; devuelve la ruta de cada GGUF."""
    llama_server = Path(llama_server)
    if llama_server.name not in esperado["binarios"]:
        raise ErrorDemo(f"{llama_server.name} no es un binario registrado de llama.cpp.")
    for archivo, sha in sorted(esperado["binarios"].items()):
        ruta = llama_server.parent / archivo
        if not ruta.is_file() or sha256_archivo(ruta) != sha:
            raise ErrorDemo(f"{archivo} falta o no tiene el SHA-256 registrado en el resultado de concordancia.")
    rutas = {}
    for rol, g in esperado["gguf"].items():
        ruta = Path(dir_gguf) / g["archivo"]
        if not ruta.is_file() or ruta.stat().st_size != g["bytes"] or sha256_archivo(ruta) != g["sha256"]:
            raise ErrorDemo(f"{g['archivo']} falta o no tiene los bytes o el SHA-256 del resultado de concordancia.")
        rutas[rol] = ruta
    return rutas


def cargar_titulos(raiz: Path = REPO_RAIZ) -> dict[str, str]:
    """{código: título} del catálogo SCIAN de la versión fijada, con su SHA-256 verificado contra el manifest."""
    manifest = json.loads((Path(raiz) / "data" / "manifest.json").read_text(encoding="utf-8"))
    e = manifest["catalogos"][CATALOGO[manifest["scian_version"]]]
    ruta = Path(raiz) / e["ruta"]
    if not ruta.is_file() or sha256_archivo(ruta) != e["sha256"]:
        raise ErrorDemo(f"{e['ruta']} falta o no tiene el SHA-256 de data/manifest.json.")
    return leer_catalogo(ruta)


def leer_nombres(ruta: Path) -> list[str]:
    """Los nombres del archivo, uno por línea; salta líneas vacías y las que empiezan con #. Acepta BOM y CRLF."""
    lineas = Path(ruta).read_text(encoding="utf-8-sig").splitlines()
    return [x.strip() for x in lineas if x.strip() and not x.strip().startswith("#")]


# --- Pregunta y salida ---


def preguntar(cliente: ls.Cliente, nombre: str, n_predict: int, fin: Sequence[int]) -> dict:
    """La pregunta del Tema A (/completion + /detokenize), medida de punta a punta en el cliente."""
    t0 = time.perf_counter()
    c = cliente.completar(prompt_generacion(nombre), n_predict)
    r = ls.respuesta_de(c, fin)
    texto = cliente.detokenizar(r["fichas"])
    return {"respuesta": texto, "ms": (time.perf_counter() - t0) * 1000, "al_tope": r["al_tope"]}


def envolver(texto: str, ancho: int, sangria: str = "", max_lineas: int | None = None,
             marcador: str = " …") -> list[str]:
    return textwrap.wrap(texto, width=ancho, initial_indent=sangria, subsequent_indent=sangria,
                         break_long_words=True, break_on_hyphens=False, max_lines=max_lineas,
                         placeholder=marcador) or [sangria.rstrip()]


def texto_respuesta(respuesta: str, titulos: Mapping[str, str]) -> tuple[str, str]:
    """(categoría de la regla 5, texto que se muestra)."""
    cat = categoria(respuesta, titulos)
    if cat == CODIGO_EXISTENTE:
        codigo = limpiar(respuesta)
        return cat, f"{codigo} · {titulos[codigo]}"
    if cat == CODIGO_INEXISTENTE:
        return cat, f"{limpiar(respuesta)} · código inexistente"
    visible = " ".join(respuesta.split())
    if not visible:
        return cat, "formato inválido: respuesta vacía"
    if RE_CODIGO.fullmatch(visible):  # inválida solo por espacios raros en las orillas: se muestran
        return cat, f"formato inválido: {ascii(respuesta)}"
    return cat, f"formato inválido: «{visible}»"


def lineas_modelo(rol: str, res: Mapping, titulos: Mapping[str, str], ancho: int) -> list[str]:
    cat, texto = texto_respuesta(res["respuesta"], titulos)
    maximo = LINEAS_FORMATO_INVALIDO if cat not in (CODIGO_EXISTENTE, CODIGO_INEXISTENTE) else None
    marcador = " …»" if texto.endswith("»") else " …"
    return [*envolver(f"{ETIQUETAS[rol]} · {res['ms']:,.0f} ms", ancho, "  "),
            *envolver(texto, ancho, SANGRIA, max_lineas=maximo, marcador=marcador)]


def problema_del_nombre(nombre: str) -> str | None:
    if tiene_marca_especial(nombre):
        return "lleva «<|», que podría leerse como token especial: no se manda."
    if len(nombre) > MAXIMO_CARACTERES:
        return f"tiene más de {MAXIMO_CARACTERES} caracteres: no se manda."
    return None


def mostrar(nombre: str, clientes: Mapping[str, ls.Cliente], fin: Mapping[str, Sequence[int]],
            titulos: Mapping[str, str], ancho: int, escribir: Callable[[str], None], encabezado: bool = True) -> None:
    """Las respuestas de los dos modelos; con `encabezado`, antes la raya y el nombre (los tecleados ya se ven)."""
    if encabezado:
        escribir("─" * ancho)
        for x in envolver(f"Nombre: {nombre}", ancho):
            escribir(x)
    problema = problema_del_nombre(nombre)
    if problema:
        for x in envolver(f"Este nombre {problema}", ancho, "  "):
            escribir(x)
        return
    for rol in ROLES:
        try:
            res = preguntar(clientes[rol], nombre, NUEVAS[rol], fin[rol])
        except ls.ErrorServidor as e:
            for x in envolver(f"{ETIQUETAS[rol]}: sin respuesta del servidor ({e})", ancho, "  "):
                escribir(x)
            continue
        for x in lineas_modelo(rol, res, titulos, ancho):
            escribir(x)


def _leer(entrada: Callable[[str], str], aviso: str) -> str | None:
    try:
        return entrada(aviso)
    except EOFError:
        return None


def sesion(clientes: Mapping[str, ls.Cliente], fin: Mapping[str, Sequence[int]], titulos: Mapping[str, str],
           nombres: Iterable[str], *, ancho: int = ANCHO, pausa: bool = False,
           entrada: Callable[[str], str] = input, escribir: Callable[[str], None] = print) -> int:
    """Los nombres del archivo y después los tecleados; devuelve cuántos nombres se mostraron."""
    n = 0
    for nombre in nombres:
        if pausa:
            r = _leer(entrada, "[Enter] siguiente · salir: ")
            if r is None:
                escribir("")
            if r is None or r.strip().lower() == "salir":
                return n
        mostrar(nombre, clientes, fin, titulos, ancho, escribir)
        n += 1
    while True:
        escribir("─" * ancho)
        r = _leer(entrada, "Nombre (Enter para salir): ")
        if r is None:  # fin de la entrada (por ejemplo, sin teclado): el aviso quedó sin salto de línea
            escribir("")
        if r is None or r.strip().lower() in ("", "salir"):
            return n
        mostrar(r.strip(), clientes, fin, titulos, ancho, escribir, encabezado=False)
        n += 1


# --- Arranque ---


def mensaje_memoria(libre_mb: int, rol: str, procesos: Sequence[Mapping], ancho: int) -> list[str]:
    texto = (f"Memoria insuficiente al cargar el modelo {rol}: había {libre_mb:,} MB libres y cada modelo necesita "
             f"{ls.MINIMO_MEMORIA_MB:,} MB libres antes de cargarse. No se cerró nada.")
    lineas = envolver(texto, ancho)
    if procesos:
        lineas.append("Procesos que más ocupan:")
        lineas += [f"  {p['proceso']}: {p['mb']:,} MB" for p in procesos[:PROCESOS_A_MOSTRAR]]
    return lineas


def argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--llama-server", default=str(LLAMA_SERVER), help="ruta absoluta de llama-server.exe")
    p.add_argument("--dir-gguf", default=str(DIR_GGUF), help="carpeta con los GGUF")
    p.add_argument("--nombres", default=None, help=f"archivo de nombres (por omisión, {NOMBRES.relative_to(REPO_RAIZ).as_posix()})")
    p.add_argument("--ancho", type=int, default=ANCHO, help=f"columnas de la salida (por omisión, {ANCHO})")
    p.add_argument("--pausa", action="store_true", help="espera Enter antes de cada nombre del archivo")
    a = p.parse_args(argv)
    if a.ancho < MINIMO_ANCHO:
        p.error(f"--ancho debe ser al menos {MINIMO_ANCHO}.")
    return a


def main(argv: Sequence[str] | None = None, *, raiz: Path = REPO_RAIZ, entrada: Callable[[str], str] = input,
         escribir: Callable[[str], None] | None = None, libre: Callable[[], int] = ls.memoria_libre_mb,
         procesos: Callable[[], list[dict]] = programas_abiertos, fabrica: Callable = ls.Servidor) -> int:
    t0 = time.perf_counter()
    if escribir is None:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        escribir = functools.partial(print, flush=True)
    a = argumentos(argv)
    w = a.ancho
    medida = {"libre_mb": None, "rol": None}

    def libre_medida() -> int:  # la memoria libre en el momento de la regla, para el mensaje
        medida["libre_mb"] = libre()
        return medida["libre_mb"]

    try:
        esperado = esperados(resultado_concordancia(raiz))
        for x in envolver(f"Clasificador SCIAN · Qwen3-0.6B en esta laptop: base sin afinar y afinado (LoRA), "
                          f"{esperado['cuantizacion']}, sin red", w):
            escribir(x)
        rutas = verificar_archivos(Path(a.llama_server), Path(a.dir_gguf), esperado)
        titulos = cargar_titulos(raiz)
        ruta_nombres = Path(a.nombres) if a.nombres else NOMBRES
        if a.nombres and not ruta_nombres.is_file():
            raise ErrorDemo("No existe el archivo de nombres que se pasó con --nombres.")
        nombres = leer_nombres(ruta_nombres) if ruta_nombres.is_file() else []
        t_verificar = time.perf_counter() - t0
        escribir(f"Archivos verificados (SHA-256) en {t_verificar:.1f} s")
        for rol in ROLES:
            if ls.puerto_ocupado(PUERTOS[rol]):
                raise ErrorDemo(f"El puerto {PUERTOS[rol]} está ocupado: quedó otro llama-server abierto. "
                                "Ciérralo con:", comando="Stop-Process -Name llama-server")
        with contextlib.ExitStack() as pila:
            servidores = {}
            for rol in ROLES:
                medida["rol"] = rol
                servidores[rol] = pila.enter_context(fabrica(a.llama_server, rutas[rol], PUERTOS[rol],
                                                             LOGS / f"{rol}.log", libre=libre_medida))
            clientes = {rol: s.cliente for rol, s in servidores.items()}
            fin = {rol: ls.ids_fin(c) for rol, c in clientes.items()}
            for rol in ROLES:
                preguntar(clientes[rol], CALENTAMIENTO, NUEVAS[rol], fin[rol])
            escribir(f"Modelos listos: arranque total {time.perf_counter() - t0:.1f} s")
            if not nombres:
                escribir("Sin nombres en archivo: escríbelos aquí.")
            sesion(clientes, fin, titulos, nombres, ancho=w, pausa=a.pausa, entrada=entrada, escribir=escribir)
        escribir("Demo terminada.")
        return 0
    except ls.MemoriaInsuficiente:
        for x in mensaje_memoria(medida["libre_mb"], medida["rol"], procesos(), w):
            escribir(x)
        return 1
    except (ErrorDemo, ls.ErrorServidor) as e:
        for x in envolver(f"La demo no arranca: {e}", w):
            escribir(x)
        if getattr(e, "comando", None):
            escribir(f"  {e.comando}")
        return 1
    except KeyboardInterrupt:
        escribir("")
        escribir("Demo terminada.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
