"""Lanzador y cliente de llama-server en la laptop (Fase 4, paso 19): concordancia, latencia y demo.

Solo usa la biblioteca estándar y `src/prompt_clasificador.py` / `src/humo_kaggle.py` (que tampoco traen nada más), para
que lo importen el entorno del proyecto (.venv) y el del tokenizador (venv-tokenizador, fuera del repo).

- Memoria: `memoria_libre_mb()` y `memoria_proceso(pid)` con ctypes (GlobalMemoryStatusEx y GetProcessMemoryInfo).
  `exigir_memoria()` detiene todo antes de cargar pesos si quedan menos de 3 GB libres (regla del paso 19).
- `Servidor`: arranca un llama-server por modelo en 127.0.0.1, con su log en un archivo ignorado por git, espera a
  `/health`, mide el arranque y lo cierra al salir.
- `Cliente`: `/completion`, `/tokenize` y `/detokenize` sin pasar por ningún proxy del sistema.
- Decodificación (decisión D6 del plan): prompt crudo de `prompt_generacion` (sin la plantilla de chat de llama.cpp),
  greedy, `cache_prompt` apagado y las fichas generadas de vuelta; `respuesta_de()` las corta con `cortar_respuesta`
  en `<|im_end|>` y `<|endoftext|>`, como en Kaggle.

Los mensajes de error nunca incluyen prompts, nombres ni respuestas.
"""

from __future__ import annotations

import ctypes
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from src.humo_kaggle import cortar_respuesta

HOST = "127.0.0.1"
MINIMO_MEMORIA_MB = 3072  # regla del paso 19: con menos de 3 GB libres no se cargan pesos
HILOS = 4  # núcleos físicos del i7-1185G7
CONTEXTO = 512  # el prompt más largo del afinado es de 95 fichas (paso 15) + 32 nuevas
NUEVAS_AFINADO = 16  # pre-registro, decodificacion.afinados.max_new_tokens
NUEVAS_BASE = 32  # pre-registro, decodificacion.base.max_new_tokens
TOKENS_FIN = ("<|im_end|>", "<|endoftext|>")  # pre-registro, tokens_de_fin (los mismos para afinados y base)
ESPERA_ARRANQUE_S = 180
TIEMPO_MAXIMO_PETICION_S = 300

# Banderas fijas del servidor (D6): un solo slot, contexto chico, sin interfaz web, sin caché de prompts en RAM y sin red.
BANDERAS_SERVIDOR = (
    "-c", str(CONTEXTO), "-np", "1", "-t", str(HILOS), "--host", HOST,
    "--no-webui", "--cache-ram", "0", "--offline",
)

# Greedy explícito (D6). Con temperature 0 llama.cpp ya elige el más probable; top_k 1 lo asegura aunque cambie el
# orden de los muestreadores. cache_prompt apagado: el README de llama-server advierte que reusar la caché puede dar
# resultados no deterministas. return_tokens: las fichas generadas, para cortarlas igual que en Kaggle.
PARAMETROS_GREEDY: dict = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "xtc_probability": 0.0,
    "mirostat": 0,
    "seed": 42,
    "n_probs": 0,
    "cache_prompt": False,
    "return_tokens": True,
    "stream": False,
}


class ErrorServidor(RuntimeError):
    """Problema con llama-server o con su respuesta; el mensaje nunca incluye prompts ni respuestas."""


class MemoriaInsuficiente(ErrorServidor):
    """Quedan menos de MINIMO_MEMORIA_MB libres: hay que cerrar programas antes de cargar pesos."""


# --- Memoria (Windows, con ctypes) ---


class _EstadoMemoria(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ContadoresProceso(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS_EX
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _mb(n: int) -> int:
    return round(n / (1024 * 1024))


def estado_memoria() -> dict:
    """RAM total y libre del sistema, en MB."""
    if sys.platform != "win32":
        raise ErrorServidor("La lectura de memoria solo está implementada para Windows.")
    e = _EstadoMemoria()
    e.dwLength = ctypes.sizeof(_EstadoMemoria)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(e)):
        raise ErrorServidor("GlobalMemoryStatusEx falló.")
    return {"total_mb": _mb(e.ullTotalPhys), "libre_mb": _mb(e.ullAvailPhys)}


def memoria_libre_mb() -> int:
    return estado_memoria()["libre_mb"]


def exigir_memoria(minimo_mb: int = MINIMO_MEMORIA_MB, libre: Callable[[], int] = memoria_libre_mb) -> int:
    """La memoria libre en MB; MemoriaInsuficiente si es menor que `minimo_mb` (antes de cargar pesos)."""
    mb = libre()
    if mb < minimo_mb:
        raise MemoriaInsuficiente(
            f"Hay {mb:,} MB libres y se necesitan al menos {minimo_mb:,} MB antes de cargar pesos: me detengo. "
            "Pídele a Samuel que cierre programas (por ejemplo, pestañas de Chrome) y vuelve a correr."
        )
    return mb


def memoria_proceso(pid: int) -> dict:
    """Memoria de un proceso en MB: en uso (working set), su pico y la privada."""
    if sys.platform != "win32":
        raise ErrorServidor("La lectura de memoria solo está implementada para Windows.")
    k32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    k32.CloseHandle.argtypes = (ctypes.c_void_p,)
    psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)
    h = k32.OpenProcess(0x1000 | 0x0010, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
    if not h:
        raise ErrorServidor(f"No se pudo abrir el proceso {pid} para leer su memoria.")
    try:
        c = _ContadoresProceso()
        c.cb = ctypes.sizeof(_ContadoresProceso)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            raise ErrorServidor(f"GetProcessMemoryInfo falló para el proceso {pid}.")
    finally:
        k32.CloseHandle(h)
    return {"en_uso_mb": _mb(c.WorkingSetSize), "pico_mb": _mb(c.PeakWorkingSetSize), "privada_mb": _mb(c.PrivateUsage)}


# --- Cliente HTTP ---


class Cliente:
    """Peticiones a un llama-server en 127.0.0.1, sin proxies del sistema."""

    def __init__(self, puerto: int, host: str = HOST, tiempo_maximo_s: float = TIEMPO_MAXIMO_PETICION_S):
        if host != HOST:
            raise ErrorServidor("El cliente solo habla con 127.0.0.1.")
        self.base = f"http://{host}:{int(puerto)}"
        self.tiempo_maximo_s = tiempo_maximo_s
        self._abridor = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _pedir(self, ruta: str, cuerpo: Mapping | None = None) -> tuple[int, dict]:
        datos = None if cuerpo is None else json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base + ruta, data=datos, method="GET" if cuerpo is None else "POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with self._abridor.open(req, timeout=self.tiempo_maximo_s) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, {}
        except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ErrorServidor(f"Sin respuesta válida de {self.base}{ruta}: {type(e).__name__}.") from e

    def salud(self) -> bool:
        try:
            estado, cuerpo = self._pedir("/health")
        except ErrorServidor:
            return False
        return estado == 200 and cuerpo.get("status") == "ok"

    def _post(self, ruta: str, cuerpo: Mapping) -> dict:
        estado, res = self._pedir(ruta, cuerpo)
        if estado != 200:
            raise ErrorServidor(f"{ruta} respondió con el código HTTP {estado}.")
        return res

    def tokenizar(self, texto: str, add_special: bool = True) -> list[int]:
        """Ids de `texto`. add_special=True es lo que hace /completion con un prompt de texto."""
        res = self._post("/tokenize", {"content": texto, "add_special": add_special, "parse_special": True,
                                       "with_pieces": False})
        ids = res.get("tokens")
        if not isinstance(ids, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in ids):
            raise ErrorServidor("/tokenize no devolvió una lista de ids.")
        return ids

    def detokenizar(self, ids: Sequence[int]) -> str:
        res = self._post("/detokenize", {"tokens": list(ids)})
        if not isinstance(res.get("content"), str):
            raise ErrorServidor("/detokenize no devolvió texto.")
        return res["content"]

    def completar(self, prompt: str, n_predict: int) -> dict:
        """La respuesta de /completion con los PARAMETROS_GREEDY, más `ms`: el tiempo de la petición en el cliente."""
        cuerpo = {"prompt": prompt, "n_predict": int(n_predict), **PARAMETROS_GREEDY}
        t0 = time.perf_counter()
        res = self._post("/completion", cuerpo)
        ms = (time.perf_counter() - t0) * 1000
        toks = res.get("tokens")
        if not isinstance(toks, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in toks):
            raise ErrorServidor("/completion no devolvió las fichas generadas (return_tokens).")
        return {**res, "ms": ms}


def ids_fin(cliente: Cliente, tokens: Sequence[str] = TOKENS_FIN) -> list[int]:
    """Los ids de los tokens de fin según el servidor; cada uno debe ser una sola ficha."""
    res = []
    for t in tokens:
        ids = cliente.tokenizar(t, add_special=False)
        if len(ids) != 1:
            raise ErrorServidor(f"{t} no es una sola ficha en el servidor.")
        res.append(ids[0])
    return res


def respuesta_de(completado: Mapping, fin: Sequence[int]) -> dict:
    """Corte de Kaggle sobre las fichas generadas por el servidor.

    `cortar_respuesta` corta antes del primer token de fin de Kaggle. Si el servidor paró por un token de fin propio
    (stop_type "eos") sin que aparezca ninguno de Kaggle en las fichas devueltas, la parada queda marcada: puede ser
    que el servidor no devuelva el token de fin, o que haya parado en otro token de fin de generación de llama.cpp.
    """
    fichas = list(completado["tokens"])
    corte, motivo = cortar_respuesta(fichas, fin)
    stop = completado.get("stop_type")
    if stop not in ("eos", "limit"):
        raise ErrorServidor(f"Parada inesperada del servidor: {stop!r}.")
    fin_visible = motivo == "token_de_fin"
    if not fin_visible and stop == "eos":
        motivo = "token_de_fin"
    return {
        "fichas": corte,
        "motivo": motivo,
        "al_tope": motivo == "limite_de_tokens",
        "stop_type": stop,
        "fin_visible": fin_visible,
        "fichas_devueltas": len(fichas),
    }


# --- Servidor ---


def comando_servidor(exe: str | Path, gguf: str | Path, puerto: int, hilos: int = HILOS,
                     contexto: int = CONTEXTO) -> list[str]:
    banderas = list(BANDERAS_SERVIDOR)
    banderas[banderas.index("-t") + 1] = str(hilos)
    banderas[banderas.index("-c") + 1] = str(contexto)
    return [str(exe), "-m", str(gguf), "--port", str(int(puerto)), *banderas]


def puerto_ocupado(puerto: int, host: str = HOST) -> bool:
    try:
        with socket.create_connection((host, int(puerto)), timeout=0.5):
            return True
    except OSError:
        return False


class Servidor:
    """Un llama-server en 127.0.0.1 mientras dura el bloque `with`.

    Antes de arrancar exige memoria libre y el puerto desocupado; espera a /health y guarda el tiempo de arranque.
    La salida del proceso va a `log` (un archivo ignorado por git) y nunca a la terminal.
    """

    def __init__(self, exe: str | Path, gguf: str | Path, puerto: int, log: str | Path, *, hilos: int = HILOS,
                 contexto: int = CONTEXTO, espera_s: float = ESPERA_ARRANQUE_S, minimo_mb: int = MINIMO_MEMORIA_MB,
                 libre: Callable[[], int] = memoria_libre_mb, popen: Callable = subprocess.Popen):
        self.comando = comando_servidor(exe, gguf, puerto, hilos, contexto)
        self.puerto = int(puerto)
        self.log = Path(log)
        self.espera_s = espera_s
        self.minimo_mb = minimo_mb
        self.libre = libre
        self.popen = popen
        self.proceso = None
        self.cliente = Cliente(self.puerto)
        self.arranque_s: float | None = None
        self.memoria_libre_antes_mb: int | None = None

    def __enter__(self) -> "Servidor":
        for ruta in (self.comando[0], self.comando[2]):
            if not Path(ruta).is_file():
                raise ErrorServidor(f"No existe {ruta}.")
        if puerto_ocupado(self.puerto):
            raise ErrorServidor(f"El puerto {self.puerto} ya está ocupado: ¿quedó otro llama-server abierto?")
        self.memoria_libre_antes_mb = exigir_memoria(self.minimo_mb, self.libre)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._archivo_log = open(self.log, "wb")
        try:
            self.proceso = self.popen(self.comando, stdout=self._archivo_log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, creationflags=flags)
            while not self.cliente.salud():
                if self.proceso.poll() is not None:
                    raise ErrorServidor(f"llama-server terminó al arrancar (código {self.proceso.returncode}); "
                                        f"ver su log en {self.log}.")
                if time.perf_counter() - t0 > self.espera_s:
                    raise ErrorServidor(f"llama-server no respondió /health en {self.espera_s:.0f} s.")
                time.sleep(0.2)
        except BaseException:
            self.cerrar()
            raise
        self.arranque_s = time.perf_counter() - t0
        return self

    @property
    def pid(self) -> int:
        return self.proceso.pid

    def cerrar(self) -> None:
        if self.proceso is not None and self.proceso.poll() is None:
            self.proceso.terminate()
            try:
                self.proceso.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proceso.kill()
                self.proceso.wait(timeout=15)
        if getattr(self, "_archivo_log", None) is not None and not self._archivo_log.closed:
            self._archivo_log.close()

    def __exit__(self, *exc) -> None:
        self.cerrar()
