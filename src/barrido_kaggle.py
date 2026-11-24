"""Kernel de Kaggle del barrido del clasificador (Fase 2, paso 11).

Corre varias configuraciones de QLoRA (las `CORRIDAS` que incrusta `src/construir_kernel.py`),
una tras otra, en la misma sesión de Kaggle y con una sola T4. Cada configuración corre en su
propio proceso de Python, que llama a `src.clasificador_kaggle.main()` sin cambios: es el mismo
camino de código de la corrida 06b-inicial (datos con su SHA-256, instalación, imports,
descarga verificada, carga, tokens, LoRA, entrenamiento con curvas, adaptador, inferencia y
cierre). Al terminar cada proceso, la memoria de la GPU se libera; se comprueba con nvidia-smi.

Etapas del orquestador (lo que puede fallar barato va antes de instalar):
 1. nvidia-smi: toda GPU debe ser T4.
 2. Datos: en /kaggle/input solo los dos archivos, con el SHA-256 del manifest (sin leer filas).
 3. Instalación común con versiones fijadas (`PAQUETES_BARRIDO`): las 8 de 06b-inicial, las que
    pip agregó o cambió en 06b-inicial y las de la imagen, todas a la versión de 06b-inicial. Si
    una versión fijada con pip no quedó instalada, aborta antes de entrenar (la instalación
    falló). Si difiere una que no se puede fijar con pip (torch, torchvision, triton o el
    controlador de la GPU, que van con la imagen y su CUDA), la registra como advertencia y sigue.
 4. Cada configuración, en orden:
    - candado de tiempo: se omite si lo transcurrido más FACTOR_CANDADO veces su estimación pasa
      del presupuesto (el tiempo máximo de Kaggle menos un margen para cerrar);
    - proceso propio, con su salida en /kaggle/working/<nombre>/ y su log completo en
      /kaggle/working/logs/<nombre>.log (ahí queda la pérdida de validación del paso 0, que
      `trainer.train()` borra del historial de mediciones.json);
    - tiempo máximo de FACTOR_TIEMPO_MAXIMO veces su estimación, sin pasar del presupuesto: si se
      pasa, se mata el proceso y su grupo, se registra como falla y sigue la siguiente;
    - al salir: código de salida, duración, memoria usada en la GPU 0 y un resumen de su
      mediciones.json (estado, fallas, pasos y SHA-256 del orden del train, que debe ser el de
      06b-inicial).
 5. Cierre: barrido.json con todo lo anterior y el SHA-256 de cada archivo de la salida.

El orquestador nunca importa torch ni lee nom_estab. Sale con 0 aunque una configuración falle,
para que Kaggle guarde la salida de las demás; solo sale con 1 si aborta antes de entrenar.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import src.clasificador_kaggle as kc
import src.humo_kaggle as hk

ARCHIVO_BARRIDO = "barrido.json"
CARPETA_LOGS = "logs"
CARPETA_ENTRADAS = "barrido_entradas"  # JSON de entrada de cada proceso, en DIR_TRABAJO
RAIZ_FUENTES = Path(__file__).resolve().parents[1]  # en Kaggle, /tmp/denue_src
RE_NOMBRE = re.compile(r"[a-z0-9][a-z0-9.-]*")

# --- Entorno de 06b-inicial (results/2026-11-23_fase2_06b-inicial.json, sección para_el_barrido) ---

VERSIONES_06B_INICIAL = {
    "unsloth": "2026.11.9",
    "unsloth_zoo": "2026.11.8",
    "transformers": "5.17.0",
    "trl": "1.15.0",
    "datasets": "4.8.5",
    "peft": "0.21.3",
    "bitsandbytes": "0.50.2",
    "accelerate": "1.15.0",
    "torch": "2.11.0+cu130",
    "torchvision": "0.26.0+cu130",
    "xformers": "0.0.36",
    "triton": "3.6.0",
    "huggingface_hub": "1.33.0",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
    "numpy": "2.1.3",
}
# Además de las 8 de hk.PAQUETES: lo que pip agregó (xformers, cut-cross-entropy, hf-transfer,
# msgspec, mslk, structlog y tyro) o cambió (torchao) al instalar en 06b-inicial, y las librerías
# clave que ya traía la imagen, todo a la versión que quedó en 06b-inicial.
PAQUETES_EXTRA = (
    "xformers==0.0.36",
    "torchao==0.18.0",
    "cut-cross-entropy==25.1.1",
    "hf-transfer==0.1.9",
    "msgspec==0.22.0",
    "mslk==0.0.0",
    "structlog==26.1.0",
    "tyro==1.0.17",
    "huggingface_hub==1.33.0",
    "tokenizers==0.23.3",
    "safetensors==0.8.0",
    "numpy==2.1.3",
)
PAQUETES_BARRIDO = tuple(hk.PAQUETES) + PAQUETES_EXTRA
NO_FIJABLES = ("torch", "torchvision", "triton")  # compilaciones CUDA de la imagen
CONTROLADOR_06B_INICIAL = "580.178.04"  # nvidia-smi en mediciones.json de 06b-inicial
SHA256_ORDEN_06B_INICIAL = "b6b53a8508b9aef2cc0323c60397dddb023869d50a4accacf35887a3ab897980"

# --- Tiempos de 06b-inicial (s; duraciones por etapa en results/2026-11-23_fase2_06b-inicial.json) ---

S_ANTES_DE_ENTRENAR_06B = 238.455  # datos, instalación, imports, descarga, carga, tokens y LoRA
S_ENTRENAMIENTO_06B = 3883.48  # 5,855 pasos con las 13 evaluaciones
S_DESPUES_DE_ENTRENAR_06B = 228.069  # adaptador, inferencia y cierre
PASOS_06B = 5855
FILAS_TRAIN_06B = 187_350  # train_final.jsonl (results: tamano_muestra.train)
# Rank 8 se estima igual que 16 (no más rápido) y rank 32 con un 5 % más de entrenamiento: en el
# 0.6B, LoRA es del 1 al 5 % de los FLOPs y el tiempo lo dominan el modelo base y la cabeza.
FACTOR_ENTRENAMIENTO_POR_RANK = {8: 1.0, 16: 1.0, 32: 1.05}
S_ORQUESTADOR = 150  # GPU, hashes de los datos e instalación común (06b-inicial: 48 s de pip)

TIEMPO_KAGGLE_S = 27_000  # el -t de `kaggle kernels push`: unas 1.5 veces el total estimado
MARGEN_CIERRE_S = 900
PRESUPUESTO_S = TIEMPO_KAGGLE_S - MARGEN_CIERRE_S
FACTOR_CANDADO = 1.3
FACTOR_TIEMPO_MAXIMO = 2.0
MEMORIA_LIBRE_MAXIMA_MIB = 500  # memoria usada en la GPU 0 tras un proceso; más = no se liberó


class AbortoBarrido(hk.AbortoHumo):
    """Condición que detiene el barrido antes de entrenar; el mensaje nunca incluye un nom_estab."""


def informar(mensaje: str) -> None:
    print(f"[barrido {time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


# --- Funciones puras (se prueban en la laptop) ---


def estimacion_s(corrida: Mapping) -> float:
    """Segundos estimados de una configuración, a partir de los tiempos reales de 06b-inicial."""
    pasos = kc.pasos_por_epoca(FILAS_TRAIN_06B, corrida["lote"], corrida["gradient_accumulation_steps"]) * corrida["epocas"]
    factor = FACTOR_ENTRENAMIENTO_POR_RANK[corrida["lora"]["r"]]
    return round(S_ANTES_DE_ENTRENAR_06B + S_ENTRENAMIENTO_06B * factor * pasos / PASOS_06B + S_DESPUES_DE_ENTRENAR_06B, 1)


def estimacion_total_s(corridas: Sequence[Mapping]) -> float:
    return round(S_ORQUESTADOR + sum(estimacion_s(c) for c in corridas), 1)


def decidir(estimado_s: float, transcurrido_s: float, presupuesto_s: float = PRESUPUESTO_S) -> dict:
    """¿Se lanza la configuración? Y su tiempo máximo: FACTOR_TIEMPO_MAXIMO × estimación, sin pasar del presupuesto."""
    if transcurrido_s + FACTOR_CANDADO * estimado_s > presupuesto_s:
        return {
            "lanzar": False,
            "motivo": (
                f"lo transcurrido ({transcurrido_s:.0f} s) más {FACTOR_CANDADO} veces la estimación "
                f"({estimado_s:.0f} s) pasa del presupuesto ({presupuesto_s:.0f} s)"
            ),
        }
    return {"lanzar": True, "tiempo_maximo_s": round(min(FACTOR_TIEMPO_MAXIMO * estimado_s, presupuesto_s - transcurrido_s), 1)}


def validar_corridas(corridas: Sequence[Mapping]) -> list[dict]:
    """Cada corrida validada como en 06b-inicial, con nombres distintos y aptos para carpeta."""
    res = [kc.validar_corrida(c) for c in corridas]
    nombres = [c["nombre"] for c in res]
    if not res or len(set(nombres)) != len(nombres):
        raise AbortoBarrido("Las corridas deben ser al menos una y con nombres distintos.")
    malos = [n for n in nombres if not RE_NOMBRE.fullmatch(n)]
    if malos:
        raise AbortoBarrido(f"Nombres de corrida no aptos para carpeta: {malos}.")
    for c in res:
        if c["lora"]["r"] not in FACTOR_ENTRENAMIENTO_POR_RANK:
            raise AbortoBarrido(f"Sin factor de tiempo para el rank {c['lora']['r']}.")
    return res


def revisar_versiones(instaladas: Mapping[str, str | None], controlador: str | None) -> dict:
    """Versiones fijadas con pip (si difieren, aborta) y no fijables (si difieren, advertencia)."""
    fijadas = dict(p.split("==") for p in PAQUETES_BARRIDO)
    distintas = {k: {"esperada": v, "instalada": instaladas.get(k)} for k, v in fijadas.items() if instaladas.get(k) != v}
    advertencias = {
        k: {"06b_inicial": VERSIONES_06B_INICIAL[k], "instalada": instaladas.get(k)}
        for k in NO_FIJABLES
        if instaladas.get(k) != VERSIONES_06B_INICIAL[k]
    }
    if controlador != CONTROLADOR_06B_INICIAL:
        advertencias["controlador_nvidia"] = {"06b_inicial": CONTROLADOR_06B_INICIAL, "instalada": controlador}
    return {
        "fijadas_con_pip": fijadas,
        "fijadas_distintas": distintas,
        "no_fijables": {k: instaladas.get(k) for k in NO_FIJABLES},
        "controlador_nvidia": controlador,
        "advertencias": advertencias,
    }


def resumen_de_mediciones(ruta: Path) -> dict:
    """Lo que el barrido guarda de la mediciones.json de una configuración (sin textos de error)."""
    if not ruta.is_file():
        return {"existe": False}
    try:
        m = json.loads(ruta.read_bytes())
    except (OSError, ValueError):
        return {"existe": True, "legible": False}
    ent = m.get("entrenamiento", {}) if isinstance(m.get("entrenamiento"), dict) else {}
    orden = ent.get("orden_train", {}) if isinstance(ent.get("orden_train"), dict) else {}
    return {
        "existe": True,
        "legible": True,
        "estado": m.get("estado"),
        "fallas": len(m.get("fallas", [])),
        "etapas_con_falla": [f.get("etapa") for f in m.get("fallas", []) if isinstance(f, dict)],
        "aborto_o_error": "aborto" in m or "error" in m,
        "pasos_hechos": ent.get("pasos_hechos"),
        "sha256_orden_train": orden.get("sha256_orden"),
        "orden_igual_al_de_06b_inicial": orden.get("sha256_orden") == SHA256_ORDEN_06B_INICIAL,
        "filas_inferidas": (m.get("inferencia") or {}).get("filas"),
        "segundos_desde_arranque": m.get("segundos_desde_arranque"),
    }


def parsear_memoria(salida: str) -> dict[int, int]:
    """Memoria usada (MiB) por GPU, de `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`."""
    res = {}
    for linea in salida.strip().splitlines():
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) == 2:
            with contextlib.suppress(ValueError):
                res[int(partes[0])] = int(float(partes[1]))
    return res


# --- Procesos ---


def matar(proc: subprocess.Popen) -> None:
    """Termina el proceso y su grupo (pip, compiladores...); a los 30 s, a la fuerza."""
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, 15)
        try:
            proc.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, 9)
    else:
        proc.kill()
    proc.wait()


def ejecutar(comando: Sequence[str], *, env: Mapping[str, str], cwd: Path, log: Path, tiempo_maximo_s: float,
             eco: Callable[[str], None], espera_s: float = 2.0) -> dict:
    """Corre un proceso, copia su salida al log (tal cual) y al eco; lo mata si pasa del tiempo máximo."""
    t0 = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", newline="\n") as archivo:
        proc = subprocess.Popen(
            list(comando), env=dict(env), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=True,
        )

        def copiar():
            with contextlib.suppress(ValueError, OSError):  # el archivo pudo cerrarse tras matar el proceso
                for linea in proc.stdout:
                    archivo.write(linea)
                    archivo.flush()
                    eco(linea.rstrip("\n"))

        hilo = threading.Thread(target=copiar, daemon=True)
        hilo.start()
        matado = False
        while True:
            try:
                proc.wait(timeout=espera_s)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() - t0 > tiempo_maximo_s:
                    matar(proc)
                    matado = True
                    break
        hilo.join(timeout=60)
    return {"codigo_de_salida": proc.returncode, "segundos": round(time.monotonic() - t0, 1), "matado_por_tiempo": matado}


@dataclass
class Dependencias:
    """Lo que el orquestador usa del mundo exterior; las pruebas lo reemplazan."""

    dir_salida: Path
    dir_trabajo: Path
    dir_entrada: Path
    consultar_gpus: Callable[[], list[dict]]
    instalar: Callable[[hk.Registro, Path], None]
    versiones: Callable[[Sequence[str]], dict[str, str | None]]
    memoria_gpu: Callable[[], dict[int, int]]
    estimar: Callable[[Mapping], float] = estimacion_s
    presupuesto_s: float = PRESUPUESTO_S
    modulo_una: str = "src.barrido_kaggle"
    raiz_fuentes: Path = RAIZ_FUENTES
    python: str = sys.executable
    reloj: Callable[[], float] = time.time
    espera_s: float = 2.0


def verificar_datos(dir_entrada: Path, esperados: Mapping[str, str]) -> dict:
    """Solo los dos archivos en la entrada y su SHA-256; no se leen filas."""
    listado = kc.listar_entrada(dir_entrada)
    rutas = kc.revisar_entrada(listado)
    calculados = {a: hk.sha256_archivo(Path(dir_entrada) / rutas[a]) for a in kc.ARCHIVOS_ENTRADA}
    distintos = sorted(a for a in kc.ARCHIVOS_ENTRADA if calculados[a] != esperados[a])
    if distintos:
        raise AbortoBarrido(f"El SHA-256 de {distintos} no es el del manifest.")
    return {"entrada": listado, "sha256_verificados": calculados}


def correr_configuracion(reg: hk.Registro, corrida: Mapping, esperados: Mapping[str, str], fuentes: Mapping[str, str],
                         dep: Dependencias, t_arranque: float) -> dict:
    """Una configuración en su propio proceso, con candado de tiempo y tiempo máximo."""
    nombre = corrida["nombre"]
    estimado = dep.estimar(corrida)
    transcurrido = dep.reloj() - t_arranque
    decision = decidir(estimado, transcurrido, dep.presupuesto_s)
    res: dict = {"nombre": nombre, "estimado_s": estimado, "transcurrido_al_empezar_s": round(transcurrido, 1), **decision}
    if not decision["lanzar"]:
        reg.falla(f"configuracion_{nombre}", f"Omitida por tiempo: {decision['motivo']}.")
        informar(f"{nombre}: omitida por tiempo")
        return {**res, "estado": "omitida"}

    salida = dep.dir_salida / nombre
    trabajo = dep.dir_trabajo / nombre
    if salida.exists():
        raise AbortoBarrido(f"La carpeta de salida de {nombre} ya existe.")
    entrada = dep.dir_trabajo / CARPETA_ENTRADAS / f"{nombre}.json"
    entrada.parent.mkdir(parents=True, exist_ok=True)
    entrada.write_text(
        json.dumps({"corrida": dict(corrida), "esperados": dict(esperados), "fuentes": dict(fuentes),
                    "salida": str(salida), "trabajo": str(trabajo)}, ensure_ascii=False),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(dep.raiz_fuentes),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",  # el log del proceso se lee en UTF-8, sin depender del locale
    }
    log = dep.dir_salida / CARPETA_LOGS / f"{nombre}.log"
    informar(f"{nombre}: empieza (estimado {estimado / 60:.0f} min; tiempo máximo {decision['tiempo_maximo_s'] / 60:.0f} min)")
    ej = ejecutar(
        [dep.python, "-u", "-m", dep.modulo_una, "--una", str(entrada)], env=env, cwd=dep.dir_trabajo, log=log,
        tiempo_maximo_s=decision["tiempo_maximo_s"], eco=lambda linea: print(f"[{nombre}] {linea}", flush=True),
        espera_s=dep.espera_s,
    )
    memoria = dep.memoria_gpu()
    mediciones = resumen_de_mediciones(salida / kc.ARCHIVO_MEDICIONES)
    res.update({
        **ej,
        "log": f"{CARPETA_LOGS}/{nombre}.log",
        "carpeta": nombre,
        "memoria_gpu0_al_salir_mib": memoria.get(0),
        "memoria_liberada": memoria.get(0) is not None and memoria[0] <= MEMORIA_LIBRE_MAXIMA_MIB,
        "mediciones": mediciones,
    })
    if ej["matado_por_tiempo"]:
        reg.falla(f"configuracion_{nombre}", f"Pasó de su tiempo máximo ({decision['tiempo_maximo_s']:.0f} s); se mató el proceso.")
    elif ej["codigo_de_salida"] != 0:
        reg.falla(f"configuracion_{nombre}", f"El proceso salió con código {ej['codigo_de_salida']}.")
    elif mediciones.get("estado") != "completo":
        reg.falla(f"configuracion_{nombre}", f"Su mediciones.json termina en estado {mediciones.get('estado')}.")
    if not res["memoria_liberada"]:
        reg.falla(f"memoria_{nombre}", f"La GPU 0 tiene {memoria.get(0)} MiB en uso al salir el proceso.")
    if mediciones.get("legible") and mediciones.get("sha256_orden_train") and not mediciones["orden_igual_al_de_06b_inicial"]:
        reg.falla(f"orden_{nombre}", "El orden del train no es el de 06b-inicial.")
    ok = not ej["matado_por_tiempo"] and ej["codigo_de_salida"] == 0 and mediciones.get("estado") == "completo"
    res["estado"] = "completa" if ok else "fallida"
    informar(f"{nombre}: {res['estado']} en {ej['segundos'] / 60:.1f} min; GPU 0 con {memoria.get(0)} MiB al salir")
    return res


def orquestar(reg: hk.Registro, esperados: Mapping[str, str], fuentes: Mapping[str, str], corridas: Sequence[Mapping],
              dep: Dependencias, t_arranque: float) -> None:
    with reg.etapa("gpu"):
        gpus = dep.consultar_gpus()
        hk.verificar_t4(gpus)
        reg.datos["gpu"] = {"nvidia_smi": gpus, "machine_shape_pedido": hk.MACHINE_SHAPE}

    with reg.etapa("datos"):
        reg.datos["datos"] = verificar_datos(dep.dir_entrada, esperados)

    with reg.etapa("instalacion"):
        dep.instalar(reg, dep.dir_salida)

    with reg.etapa("versiones"):
        nombres = sorted({p.split("==")[0] for p in PAQUETES_BARRIDO} | set(NO_FIJABLES))
        controlador = gpus[0].get("driver") if gpus else None
        rev = revisar_versiones(dep.versiones(nombres), controlador)
        reg.datos["versiones"] = rev
        reg.datos["advertencias"] = rev["advertencias"]
        if rev["fijadas_distintas"]:
            raise AbortoBarrido(f"Versiones fijadas con pip que no quedaron instaladas: {sorted(rev['fijadas_distintas'])}.")
        for k, v in rev["advertencias"].items():
            informar(f"ADVERTENCIA: {k} es {v['instalada']}; en 06b-inicial era {v['06b_inicial']}")
        informar("versiones fijadas con pip: todas iguales a las de 06b-inicial")

    reg.datos["configuraciones"] = []
    for corrida in corridas:
        with reg.etapa(f"configuracion_{corrida['nombre']}"):
            r = correr_configuracion(reg, corrida, esperados, fuentes, dep, t_arranque)
            reg.datos["configuraciones"].append(r)

    with reg.etapa("cierre"):
        reg.datos["archivos_salida"] = kc.archivos_salida(dep.dir_salida, (ARCHIVO_BARRIDO, ARCHIVO_BARRIDO + ".tmp"))
        todas = all(c.get("estado") == "completa" for c in reg.datos["configuraciones"])
        reg.datos["estado"] = "completo" if todas and not reg.datos["fallas"] else "completo_con_fallas"


# --- Dependencias reales (Kaggle) ---


def _instalar(reg: hk.Registro, dir_salida: Path) -> None:
    antes = hk._pip("freeze").stdout
    (dir_salida / "pip_freeze_antes.txt").write_text(antes, encoding="utf-8")
    t = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", *PAQUETES_BARRIDO], text=True
    )
    segundos = time.perf_counter() - t
    if r.returncode != 0:
        raise AbortoBarrido(f"pip install falló con código {r.returncode}.")
    despues = hk._pip("freeze").stdout
    (dir_salida / "pip_freeze_despues.txt").write_text(despues, encoding="utf-8")
    reg.datos["instalacion"] = {
        "paquetes": list(PAQUETES_BARRIDO),
        "segundos_pip_install": round(segundos, 1),
        "cambios_freeze": hk.diferencias_freeze(antes, despues),
    }


def _versiones(nombres: Sequence[str]) -> dict[str, str | None]:
    from importlib import metadata

    res = {}
    for n in nombres:
        try:
            res[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            res[n] = None
    return res


def _memoria_gpu() -> dict[int, int]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    return parsear_memoria(r.stdout) if r.returncode == 0 else {}


def dependencias_kaggle() -> Dependencias:
    return Dependencias(
        dir_salida=hk.DIR_SALIDA,
        dir_trabajo=hk.DIR_TRABAJO,
        dir_entrada=hk.DIR_ENTRADA,
        consultar_gpus=hk.consultar_gpus,
        instalar=_instalar,
        versiones=_versiones,
        memoria_gpu=_memoria_gpu,
    )


# --- Puntos de entrada ---


def correr_una(ruta: str | Path) -> None:
    """Proceso de una configuración: la salida y el trabajo en su carpeta, y clasificador_kaggle.main() sin cambios."""
    d = json.loads(Path(ruta).read_text(encoding="utf-8"))
    hk.DIR_SALIDA = Path(d["salida"])
    hk.DIR_TRABAJO = Path(d["trabajo"])
    kc.main(d["esperados"], time.time(), d["fuentes"], d["corrida"])


def main(
    esperados: Mapping[str, str],
    t_arranque: float | None = None,
    fuentes: Mapping[str, str] | None = None,
    corridas: Sequence[Mapping] | None = None,
    dep: Dependencias | None = None,
) -> None:
    """Punto de entrada en Kaggle. `esperados` y `corridas` los incrusta el generador."""
    t_arranque = time.time() if t_arranque is None else t_arranque
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    hk.informar = informar
    corridas = validar_corridas(corridas or [])
    if set(esperados) != set(kc.ARCHIVOS_ENTRADA):
        raise AbortoBarrido(f"Los SHA-256 esperados deben ser exactamente los de {list(kc.ARCHIVOS_ENTRADA)}.")
    dep = dep or dependencias_kaggle()
    dep.dir_salida.mkdir(parents=True, exist_ok=True)
    dep.dir_trabajo.mkdir(parents=True, exist_ok=True)
    reg = hk.Registro(dep.dir_salida / ARCHIVO_BARRIDO, t_arranque)
    reg.datos["descripcion"] = "Barrido del clasificador (Fase 2, paso 11): una configuración por proceso."
    reg.datos["configuracion"] = {
        "corridas": [dict(c) for c in corridas],
        "paquetes_fijados": list(PAQUETES_BARRIDO),
        "no_fijables": list(NO_FIJABLES),
        "versiones_06b_inicial": VERSIONES_06B_INICIAL,
        "controlador_06b_inicial": CONTROLADOR_06B_INICIAL,
        "sha256_orden_06b_inicial": SHA256_ORDEN_06B_INICIAL,
        "estimado_s": {c["nombre"]: dep.estimar(c) for c in corridas},
        "presupuesto_s": dep.presupuesto_s,
        "tiempo_kaggle_s": TIEMPO_KAGGLE_S,
        "factor_candado": FACTOR_CANDADO,
        "factor_tiempo_maximo": FACTOR_TIEMPO_MAXIMO,
        "memoria_libre_maxima_mib": MEMORIA_LIBRE_MAXIMA_MIB,
    }
    reg.datos["fuentes_incrustadas_sha256"] = dict(fuentes or {})
    reg.datos["datos_sha256_esperados"] = dict(esperados)
    reg.escribir()
    try:
        orquestar(reg, esperados, fuentes or {}, corridas, dep, t_arranque)
    except hk.AbortoHumo as e:
        reg.datos["estado"] = "abortado"
        reg.datos["aborto"] = str(e)
        reg.escribir()
        informar(f"ABORTO: {e}")
        sys.exit(1)
    except Exception:
        reg.datos["estado"] = "error"
        reg.datos["error"] = traceback.format_exc()[-4000:]
        reg.escribir()
        print(reg.datos["error"], flush=True)
        sys.exit(1)
    reg.escribir()
    informar(f"estado final: {reg.datos['estado']}; fallas: {len(reg.datos['fallas'])}")


def _cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Proceso de una configuración del barrido.")
    parser.add_argument("--una", required=True, help="JSON de entrada que escribe el orquestador")
    args = parser.parse_args(argv)
    correr_una(args.una)


if __name__ == "__main__":
    _cli()
