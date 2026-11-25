"""Revisión local de la salida del kernel de la Fase 4 (paso 17, punto 1). Nunca lee etiquetas de test ni de hard.

Lee lo que bajó `kaggle kernels output samcaballero/denue-scian-fase4-inferencia -p kaggle-output/fase4`
(carpeta ignorada por git): el log de la CLI y la carpeta `fase4/` del kernel. En este orden:

a. Descarga: cada archivo con los bytes y el SHA-256 de `archivos_salida` de `mediciones.json`; de más
   solo pueden estar `mediciones.json` y el log de la CLI; el JSONL de cada etapa completa con el SHA-256
   de `mediciones.etapas[].predicciones` y del JSON de su etapa; ningún archivo con un SHA-256 del
   manifest ni con el de test o hard con etiquetas.
b. Revisión de nombres con la regla acotada (método del paso 6c indexado, con la precisión del paso 11):
   el log de la CLI, `logs/*.log`, `pip_freeze_*.txt` y los JSON (texto crudo y valores), contra los 4
   archivos que vio el kernel (copias de test y hard sin etiquetas; train final y validación del paquete
   del clasificador), cada uno con su SHA-256 del manifest. Los JSONL de predicciones quedan fuera: pueden
   traer nombres y solo viven en la carpeta ignorada.
c. Ids: los de cada JSONL de test y hard son exactamente los de su copia, en orden de int(id); los de
   validación, los de val.jsonl.
d. Resumen: etapas, compuertas (con un recálculo local contra las predicciones de validación de
   referencia del config), tiempos, horas de GPU, s/fila, % al tope, VRAM pico, fichas máximas, versiones
   y la imagen de Docker (pip freeze y entorno contra los del kernel del 4B, que corrió con la imagen fijada).

`revisar()` devuelve todo (con las respuestas crudas solo en memoria, para la evaluación final); la
terminal solo recibe estados, conteos, tiempos, versiones y hashes.

Uso (solo imprime; no escribe nada):
  .venv/Scripts/python -m src.salida_fase4

El chat de Claude del proyecto propuso el paso; Samuel lo aprobó; Claude Code lo redactó.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import src.paquete_evaluacion as pe
from src import construir_kernel as ck
from src import inferencia_fase4_kaggle as fo
from src.clasificador_kaggle import ARCHIVO_TRAIN, ARCHIVO_VAL
from src.denue import REPO_RAIZ
from src.evaluar_corrida_clasificador import (
    buscar_nombres_indexada,
    clasificar,
    indice_de_nombres,
    palabras_del_texto,
    posible_fuga,
)
from src.evaluar_humo import inventario, leer_log, referencias_de_nombres, texto_log, textos_de_json
from src.humo_kaggle import diferencias_freeze, leer_jsonl_verificado, sha256_bytes
from src.muestra_humo import hashes_del_manifest, leer_verificado
from src.preregistro_fase4 import parsear_jsonl

SCRIPT = "src/salida_fase4.py"
RUTA_CONFIG = fo.RUTA_CONFIG
CARPETA = "kaggle-output/fase4"
SUB = fo.DIR_SALIDA.name  # el kernel escribe en /kaggle/working/fase4
LOG_CLI = "denue-scian-fase4-inferencia.log"  # lo escribe la CLI de Kaggle junto a la salida
ESPEC = ck.KERNELS["fase4-inferencia"]
ENTRADA_PAQUETE = "denue_scian_clasificador"
CONJUNTOS = ("validacion", "test", "hard")
ARCHIVOS_TEXTO = (".log", ".txt", ".json")
FREEZE_ANTES = "pip_freeze_antes.txt"
FREEZE_DESPUES = "pip_freeze_despues.txt"
# Kernel de referencia para la imagen: el del 4B corrió con la imagen fijada (bitácora del paso 13).
REFERENCIA_IMAGEN = {
    "kernel": "samcaballero/denue-scian-4b",
    "freeze_antes": "kaggle-output/4b/pip_freeze_antes.txt",
    "mediciones": "kaggle-output/4b/4b-lr2e-4-r32/mediciones.json",
}
# Observaciones que no están en la salida.
OBSERVACIONES = {
    "estado_cli": "COMPLETE",  # kaggle kernels status, 2026-11-25 17:52 UTC
    "version_kaggle": 1,
    "docker_image_metadata_pull_m": ck.DOCKER_06B_INICIAL,  # kaggle kernels pull -m, 2026-11-25 18:24 UTC
    "pagina_kaggle": "Latest Container Image (texto de la página, visto por Samuel)",
}
CLAVES_INFERENCIA = (
    "filas", "segundos", "s_por_fila", "s_por_fila_mediana", "s_por_fila_p90", "al_tope", "pct_al_tope",
    "fichas_prompt_max", "fichas_prompt_media", "fichas_generadas_max", "fichas_generadas_media", "terminacion",
    "respaldo_parametros_de_muestreo",
)


class ErrorSalida(RuntimeError):
    """PARADA del punto 1; el mensaje nunca incluye un nom_estab, una llave, un id ni una respuesta."""


# --- Carga ---


def referencias(manifest: Mapping, raiz: Path = REPO_RAIZ) -> dict[str, list[dict]]:
    """Los 4 archivos que vio el kernel, con su SHA-256 del manifest: test y hard (solo id y nom_estab) y
    train final y validación del paquete del clasificador (id, nom_estab y codigo_act)."""
    res: dict[str, list[dict]] = {}
    ev = manifest["kaggle_datasets"][pe.ENTRADA]["archivos"]
    for conj, archivo in (("test", pe.ARCHIVO_TEST), ("hard", pe.ARCHIVO_HARD)):
        res[conj] = parsear_jsonl(leer_verificado(Path(raiz) / ev[archivo]["ruta"], ev[archivo]["sha256"]), pe.CAMPOS)
    paq = manifest["kaggle_datasets"][ENTRADA_PAQUETE]["archivos"]
    for conj, archivo in (("train_final", ARCHIVO_TRAIN), ("validacion", ARCHIVO_VAL)):
        res[conj] = leer_jsonl_verificado(Path(raiz) / paq[archivo]["ruta"], paq[archivo]["sha256"])
    return res


def leer_predicciones(datos: bytes) -> list[dict]:
    """Registros del JSONL de una etapa: claves de fo.CAMPOS_PREDICCION en orden, ids de dígitos sin repetir y
    tipos correctos. Nunca se imprime una respuesta."""
    try:
        regs = fo.leer_jsonl(datos, fo.CAMPOS_PREDICCION)
    except (ValueError, UnicodeDecodeError) as e:
        raise ErrorSalida(f"Un JSONL de predicciones no es válido: {type(e).__name__}.") from e
    for r in regs:
        ok = (isinstance(r["respuesta"], str) and isinstance(r["al_tope"], bool)
              and all(isinstance(r[k], int) and not isinstance(r[k], bool) for k in ("fichas_prompt", "fichas_generadas"))
              and isinstance(r["segundos"], (int, float)) and not isinstance(r["segundos"], bool))
        if not ok:
            raise ErrorSalida("Una predicción tiene un campo con un tipo inesperado.")
    return regs


def cargar(raiz: Path = REPO_RAIZ, carpeta: str = CARPETA) -> SimpleNamespace:
    """Inventario, textos, JSON y predicciones de la descarga (sin imprimir nada)."""
    base = Path(raiz) / carpeta
    salida = base / SUB
    if not (salida / fo.ARCHIVO_MEDICIONES).is_file() or not (base / LOG_CLI).is_file():
        raise ErrorSalida(f"Falta {SUB}/{fo.ARCHIVO_MEDICIONES} o el log de la CLI en {carpeta}.")
    inv = inventario(base)
    textos: dict[str, str] = {}
    jsons: dict[str, object] = {}
    for a in inv:
        r = a["ruta"]
        if r.startswith(f"{SUB}/{fo.CARPETA_PREDICCIONES}/"):
            continue
        if not r.endswith(ARCHIVOS_TEXTO):
            raise ErrorSalida(f"Archivo inesperado en la descarga: {r}.")
        datos = (base / r).read_bytes()
        if r == LOG_CLI:
            textos[r] = texto_log(leer_log(datos))
            continue
        textos[r] = datos.decode("utf-8")
        if r.endswith(".json"):
            jsons[r] = json.loads(textos[r])
    preds: dict[tuple[str, str], dict] = {}
    for a in inv:
        r = a["ruta"]
        if r.startswith(f"{SUB}/{fo.CARPETA_PREDICCIONES}/"):
            partes = r.split("/")
            if len(partes) != 4 or not partes[3].endswith(".jsonl"):
                raise ErrorSalida(f"Ruta de predicciones inesperada: {r}.")
            datos = (base / r).read_bytes()
            preds[(partes[2], partes[3][: -len(".jsonl")])] = {"ruta": r, "sha256": a["sha256"], "bytes": a["bytes"],
                                                               "registros": leer_predicciones(datos)}
    return SimpleNamespace(base=base, carpeta=carpeta, inventario=inv, textos=textos, jsons=jsons,
                           mediciones=jsons[f"{SUB}/{fo.ARCHIVO_MEDICIONES}"], predicciones=preds)


# --- a. Descarga ---


def verificar_descarga(inv: Sequence[Mapping], mediciones: Mapping, etapas_json: Mapping[str, Mapping],
                       hashes_prohibidos: set[str]) -> dict:
    """Lo bajado contra `archivos_salida`; el JSONL de cada etapa completa contra mediciones y el JSON de su etapa."""
    locales = {a["ruta"]: a for a in inv}
    kaggle = {f"{SUB}/{a['ruta']}": a for a in mediciones.get("archivos_salida", [])}
    sin_hash = {f"{SUB}/{fo.ARCHIVO_MEDICIONES}", LOG_CLI}
    faltan = sorted(set(kaggle) - set(locales))
    distintos = sorted(r for r in set(kaggle) & set(locales)
                       if (locales[r]["bytes"], locales[r]["sha256"]) != (kaggle[r]["bytes"], kaggle[r]["sha256"]))
    extra = sorted(set(locales) - set(kaggle) - sin_hash)
    prohibidos = sorted(r for r, a in locales.items() if a["sha256"] in hashes_prohibidos)
    etapas_mal: list[str] = []
    completas = 0
    for e in mediciones.get("etapas", []):
        if e.get("estado") != "completa":
            continue
        completas += 1
        p = e.get("predicciones") or {}
        ruta = f"{SUB}/{p.get('ruta')}"
        arch = (etapas_json.get(e["id"]) or {}).get("archivo") or {}
        loc = locales.get(ruta)
        if (loc is None or loc["sha256"] != p.get("sha256") or loc["sha256"] != arch.get("sha256")
                or loc["bytes"] != p.get("bytes") or loc["bytes"] != arch.get("bytes") or p.get("filas") != arch.get("filas")):
            etapas_mal.append(e["id"])
    jsonl = sorted(r for r in locales if r.startswith(f"{SUB}/{fo.CARPETA_PREDICCIONES}/"))
    res = {
        "archivos_bajados": len(locales),
        "con_sha256_de_kaggle": len(kaggle),
        "coinciden_con_kaggle": len(set(kaggle) & set(locales)) - len(distintos),
        "sin_sha256_de_kaggle": sorted(sin_hash & set(locales)),
        "faltan": faltan,
        "distintos": distintos,
        "extra": extra,
        "con_sha256_del_manifest_o_de_etiquetas": prohibidos,
        "etapas_completas": completas,
        "jsonl_de_predicciones": len(jsonl),
        "etapas_con_predicciones_distintas": etapas_mal,
    }
    if faltan or distintos or extra or prohibidos or etapas_mal or len(jsonl) != completas or not sin_hash <= set(locales):
        raise ErrorSalida(
            f"La descarga no coincide con lo que calculó Kaggle: faltan {len(faltan)}, distintos {len(distintos)}, "
            f"extra {len(extra)}, con hash prohibido {len(prohibidos)}, etapas con predicciones distintas "
            f"{len(etapas_mal)}, JSONL {len(jsonl)} de {completas} etapas completas."
        )
    return res


def hashes_prohibidos(manifest: Mapping, config: Mapping) -> set[str]:
    return set(hashes_del_manifest(manifest)) | {config["conjuntos"]["test"]["sha256"], config["conjuntos"]["hard"]["sha256"]}


# --- b. Revisión de nombres ---


def revisar_nombres(textos: Mapping[str, str], jsons: Mapping[str, object], idx: Mapping, texto_kernel: str) -> dict:
    """Regla acotada sobre cada texto y sobre los valores de cada JSON; solo conteos."""
    palabras_kernel = palabras_del_texto(texto_kernel)
    entradas = dict(sorted(textos.items()))
    for nombre, obj in sorted(jsons.items()):
        entradas[f"{nombre} (valores)"] = textos_de_json(obj)
    archivos: dict[str, dict] = {}
    for nombre, texto in entradas.items():
        resumen, encontrados = buscar_nombres_indexada(texto, idx)
        archivos[nombre] = {
            "nombres_crudos": resumen["nombres_crudos"],
            "llaves_varias_palabras": resumen["llaves_varias_palabras"],
            "llaves_una_palabra": resumen["llaves_una_palabra"],
            "clasificacion": clasificar(encontrados, texto_kernel, palabras_kernel),
        }
    fugas = sorted(n for n, r in archivos.items() if posible_fuga(r["clasificacion"]))
    tapados = {n: len(o.get("tapados", [])) for n, o in jsons.items() if isinstance(o, Mapping) and o.get("tapados")}
    return {
        "nombres_buscados": len(idx["nombres"]),
        "llaves_varias_palabras_buscadas": len(idx["varias"]),
        "llaves_una_palabra_buscadas": len(idx["una"]),
        "archivos_revisados": len(archivos),
        "tapados_por_el_kernel": tapados,
        "archivos": archivos,
        "archivos_con_posible_fuga": fugas,
        "posible_fuga": bool(fugas),
    }


# --- c. Ids ---


def verificar_ids(preds: Mapping[tuple[str, str], Mapping], ids_conjunto: Mapping[str, Sequence[str]]) -> dict:
    """Cada JSONL con exactamente los ids de su conjunto, en orden de int(id)."""
    mal = []
    for (modelo, conj), p in sorted(preds.items()):
        ids = [r["id"] for r in p["registros"]]
        if conj not in ids_conjunto or ids != sorted(ids_conjunto[conj], key=int):
            mal.append(f"{modelo}/{conj}")
    if mal:
        raise ErrorSalida(f"Ids que no son los de su conjunto (o fuera de orden): {mal}.")
    return {f"{m}/{c}": len(p["registros"]) for (m, c), p in sorted(preds.items())}


# --- d. Resumen ---


def leer_referencia_validacion(config: Mapping, modelo: str, raiz: Path = REPO_RAIZ) -> dict[str, str]:
    """{id: respuesta} de las predicciones de validación de referencia del config, con su SHA-256."""
    m = next(x for x in config["modelos"] if x["id"] == modelo)
    pv = m["predicciones_validacion"]
    datos = leer_verificado(Path(raiz) / pv["ruta"], pv["sha256"])
    return {r["id"]: r["respuesta"] for r in fo.leer_jsonl(datos, fo.CAMPOS_REFERENCIA)}


def compuertas(mediciones: Mapping, preds: Mapping[tuple[str, str], Mapping], referencias_val: Mapping[str, Mapping[str, str]],
               minimo: int) -> dict:
    """Idénticas en validación por afinado: las del kernel y el recálculo local contra la referencia."""
    res = {}
    for modelo, ref in referencias_val.items():
        kern = (mediciones.get("reproduccion_validacion") or {}).get(modelo) or {}
        nue = {r["id"]: r["respuesta"] for r in preds.get((modelo, "validacion"), {}).get("registros", [])}
        identicas = sum(nue.get(i) == v for i, v in ref.items())
        res[modelo] = {
            "kernel_identicas": kern.get("identicas"),
            "kernel_cumple": kern.get("cumple"),
            "metodo_adaptador": kern.get("metodo_adaptador"),
            "local_identicas": identicas,
            "local_mismos_ids": set(nue) == set(ref),
            "filas": len(ref),
            "minimo": minimo,
            "cumple": kern.get("cumple") is True and identicas >= minimo and identicas == kern.get("identicas"),
        }
    return res


def resumen_etapas(mediciones: Mapping, etapas_json: Mapping[str, Mapping], preds: Mapping[tuple[str, str], Mapping]) -> list[dict]:
    est = (mediciones.get("estimaciones") or {}).get("etapas") or {}
    res = []
    for e in mediciones.get("etapas", []):
        ej = etapas_json.get(e["id"]) or {}
        inf = ej.get("inferencia") or {}
        p = preds.get((e["modelo"], e["conjunto"]))
        al_tope_jsonl = sum(r["al_tope"] for r in p["registros"]) if p else None
        rev = (ej.get("antes_de_generar") or {}).get("plantilla_y_largos") or {}
        res.append({
            "id": e["id"],
            "modelo": e["modelo"],
            "conjunto": e["conjunto"],
            "estado": e.get("estado"),
            "codigo_de_salida": e.get("codigo_de_salida"),
            "matado_por_tiempo": e.get("matado_por_tiempo"),
            "memoria_liberada": e.get("memoria_liberada"),
            "segundos_proceso": e.get("segundos"),
            "estimado_mejor_s": (est.get(e["id"]) or {}).get("mejor_s"),
            "estimado_peor_s": (est.get(e["id"]) or {}).get("peor_s"),
            "tiempo_maximo_s": e.get("tiempo_maximo_s"),
            "fallas_en_la_etapa": len(ej.get("fallas", [])),
            "inferencia": {k: inf.get(k) for k in CLAVES_INFERENCIA},
            "vram_pico_mib": inf.get("vram_pico_mib"),
            "al_tope_jsonl": al_tope_jsonl,
            "al_tope_coincide": p is not None and al_tope_jsonl == inf.get("al_tope"),
            "fichas_prompt_max_revision": {c: v.get("fichas_max") for c, v in (rev.get("conjuntos") or {}).items()},
            "metodo_adaptador": (ej.get("adaptador") or {}).get("metodo"),
            "entorno": {k: (ej.get("entorno") or {}).get(k) for k in ("python", "torch_cuda", "cudnn")},
            "versiones_entorno": (ej.get("entorno") or {}).get("versiones"),
        })
    return res


def imagen(textos: Mapping[str, str], mediciones: Mapping, etapas: Sequence[Mapping], raiz: Path = REPO_RAIZ) -> dict:
    """¿Se usó la imagen fijada? pip freeze de antes de instalar, entorno y controlador contra los del 4B."""
    ref_freeze = (Path(raiz) / REFERENCIA_IMAGEN["freeze_antes"]).read_text(encoding="utf-8")
    ref_med = json.loads((Path(raiz) / REFERENCIA_IMAGEN["mediciones"]).read_bytes())
    propio = textos[f"{SUB}/{FREEZE_ANTES}"]
    dif = diferencias_freeze(ref_freeze, propio)
    ref_ent = ref_med.get("entorno") or {}
    entornos = {json.dumps(e["entorno"], sort_keys=True) for e in etapas if e["estado"] == "completa"}
    versiones = {json.dumps(e["versiones_entorno"], sort_keys=True) for e in etapas if e["estado"] == "completa"}
    ent = json.loads(next(iter(entornos))) if len(entornos) == 1 else None
    ver = json.loads(next(iter(versiones))) if len(versiones) == 1 else None
    gpus = (mediciones.get("gpu") or {}).get("nvidia_smi") or [{}]
    ref_gpus = (ref_med.get("gpu") or {}).get("nvidia_smi") or [{}]
    distintos_entorno = sorted(k for k in ("python", "torch_cuda", "cudnn") if ent is None or ent.get(k) != ref_ent.get(k))
    distintas_versiones = sorted(k for k in (ver or {}) if (ver or {}).get(k) != (ref_ent.get("versiones") or {}).get(k))
    res = {
        "referencia": REFERENCIA_IMAGEN,
        "pip_freeze_antes_identico_al_del_4b": propio == ref_freeze,
        "pip_freeze_antes_diferencias": dif,
        "entorno_igual_en_todas_las_etapas": len(entornos) == 1,
        "versiones_iguales_en_todas_las_etapas": len(versiones) == 1,
        "entorno": ent,
        "entorno_4b": {k: ref_ent.get(k) for k in ("python", "torch_cuda", "cudnn")},
        "entorno_distinto_del_4b": distintos_entorno,
        "versiones_distintas_del_4b": {k: [(ref_ent.get("versiones") or {}).get(k), ver.get(k)] for k in distintas_versiones},
        "controlador": gpus[0].get("driver"),
        "controlador_4b": ref_gpus[0].get("driver"),
        "docker_image_metadata": OBSERVACIONES["docker_image_metadata_pull_m"],
        "docker_image_fijada": ck.DOCKER_06B_INICIAL,
    }
    res["misma_imagen_que_el_4b"] = (res["pip_freeze_antes_identico_al_del_4b"] and not distintos_entorno
                                     and not distintas_versiones and res["controlador"] == res["controlador_4b"]
                                     and res["docker_image_metadata"] == res["docker_image_fijada"])
    return res


def costos(mediciones: Mapping, etapas: Sequence[Mapping]) -> dict:
    """Horas de GPU (total del kernel y por modelo), s/fila y % al tope por modelo y conjunto."""
    por_modelo: dict[str, dict] = {}
    for e in etapas:
        m = por_modelo.setdefault(e["modelo"], {"segundos_proceso": 0.0, "conjuntos": {}})
        m["segundos_proceso"] = round(m["segundos_proceso"] + (e["segundos_proceso"] or 0), 1)
        inf = e["inferencia"]
        m["conjuntos"][e["conjunto"]] = {
            "segundos_proceso": e["segundos_proceso"], "segundos_inferencia": inf.get("segundos"),
            "filas": inf.get("filas"), "s_por_fila": inf.get("s_por_fila"), "al_tope": inf.get("al_tope"),
            "pct_al_tope": inf.get("pct_al_tope"), "vram_pico_mib": e["vram_pico_mib"],
        }
    for m in por_modelo.values():
        m["horas_gpu_procesos"] = round(m["segundos_proceso"] / 3600, 3)
    total_s = mediciones.get("segundos_desde_arranque")
    etapas_s = sum(e["segundos_proceso"] or 0 for e in etapas)
    return {
        "segundos_kernel": total_s,
        "horas_gpu_kernel": round(total_s / 3600, 3) if isinstance(total_s, (int, float)) else None,
        "segundos_etapas": round(etapas_s, 1),
        "segundos_fuera_de_etapas": round(total_s - etapas_s, 1) if isinstance(total_s, (int, float)) else None,
        "duraciones_orquestador_s": {k: v for k, v in (mediciones.get("duraciones_s") or {}).items()
                                     if not k.startswith("etapa_")},
        "por_modelo": por_modelo,
        "nota": ("Horas de GPU del kernel: segundos_desde_arranque de mediciones.json (la sesión con la T4 desde el "
                 "arranque del orquestador hasta el cierre). Por modelo: la suma de los segundos de los procesos de sus "
                 "etapas (carga, revisión de plantilla y largos e inferencia); el resto es instalación, comprobaciones "
                 "y orquestador, compartido."),
    }


# --- Todo junto ---


def revisar(raiz: Path = REPO_RAIZ, manifest: Mapping | None = None, carpeta: str = CARPETA,
            con_nombres: bool = True) -> SimpleNamespace:
    """a a d; PARADA (ErrorSalida) ante cualquier problema. Devuelve el resumen y las predicciones en memoria."""
    from src.scian import leer_manifest

    manifest = manifest if manifest is not None else leer_manifest()
    config_bytes = (Path(raiz) / RUTA_CONFIG).read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    ent = cargar(raiz, carpeta)
    med = ent.mediciones
    etapas_json = {Path(r).stem: o for r, o in ent.jsons.items() if r.startswith(f"{SUB}/{fo.CARPETA_ETAPAS}/")}

    descarga = verificar_descarga(ent.inventario, med, etapas_json, hashes_prohibidos(manifest, config))

    refs = referencias(manifest, raiz)
    nombres = None
    if con_nombres:
        idx = indice_de_nombres(referencias_de_nombres([r for c in refs.values() for r in c]))
        texto_kernel = (Path(raiz) / ESPEC.carpeta_relativa / ESPEC.archivo_kernel).read_text(encoding="utf-8")
        nombres = revisar_nombres(ent.textos, ent.jsons, idx, texto_kernel)
        nombres["filas_de_referencia"] = {c: len(v) for c, v in refs.items()}
        if nombres["posible_fuga"]:
            raise ErrorSalida(f"Posible fuga de nombres en {len(nombres['archivos_con_posible_fuga'])} archivos: "
                              f"{nombres['archivos_con_posible_fuga']}. Solo conteos; decide Samuel.")

    ids = verificar_ids(ent.predicciones, {c: [r["id"] for r in refs[c]] for c in CONJUNTOS})

    crit = fo.criterio_de_paro(config)
    refs_val = {m: leer_referencia_validacion(config, m, raiz) for m in crit["modelos"]}
    comp = compuertas(med, ent.predicciones, refs_val, crit["minimo_identicas"])
    etapas = resumen_etapas(med, etapas_json, ent.predicciones)
    img = imagen(ent.textos, med, etapas, raiz)
    cst = costos(med, etapas)
    plan = fo.validar_config_kernel(config)
    problemas = []
    if len(etapas) != len(plan["etapas"]) or [e["id"] for e in etapas] != [e["id"] for e in plan["etapas"]]:
        problemas.append("las etapas de mediciones no son las del plan")
    problemas += [f"{e['id']} {e['estado']}" for e in etapas if e["estado"] != "completa"]
    problemas += [f"{e['id']} al tope distinto" for e in etapas if e["estado"] == "completa" and not e["al_tope_coincide"]]
    problemas += [f"compuerta {m}" for m, g in comp.items() if not g["cumple"]]
    if med.get("estado") != "completo" or med.get("fallas"):
        problemas.append(f"estado del kernel {med.get('estado')} con {len(med.get('fallas', []))} fallas")
    return SimpleNamespace(
        config=config, sha_config=sha256_bytes(config_bytes), manifest=manifest, descarga=descarga, nombres=nombres,
        ids=ids, compuertas=comp, etapas=etapas, imagen=img, costos=cst, mediciones=med, predicciones=ent.predicciones,
        problemas=problemas, versiones=med.get("versiones"), advertencias=med.get("advertencias"),
    )


def respuestas(r: SimpleNamespace, modelo: str, conjunto: str) -> dict[str, str]:
    """{id: respuesta cruda} de una etapa (solo en memoria; nunca se imprime)."""
    return {x["id"]: x["respuesta"] for x in r.predicciones[(modelo, conjunto)]["registros"]}


def al_tope(r: SimpleNamespace, modelo: str, conjunto: str) -> dict[str, bool]:
    return {x["id"]: x["al_tope"] for x in r.predicciones[(modelo, conjunto)]["registros"]}


# --- Terminal ---


def _h(s) -> str:
    return "—" if s is None else f"{s / 3600:.2f} h"


def imprimir(r: SimpleNamespace) -> None:
    d = r.descarga
    print(f"Config {RUTA_CONFIG}: SHA-256 {r.sha_config}")
    print(f"Descarga: {d['archivos_bajados']} archivos; {d['coinciden_con_kaggle']} de {d['con_sha256_de_kaggle']} con "
          f"bytes y SHA-256 de Kaggle; sin hash de Kaggle: {d['sin_sha256_de_kaggle']}; faltan {len(d['faltan'])}, "
          f"distintos {len(d['distintos'])}, extra {len(d['extra'])}; con hash del manifest o de etiquetas: "
          f"{len(d['con_sha256_del_manifest_o_de_etiquetas'])}; {d['jsonl_de_predicciones']} JSONL = "
          f"{d['etapas_completas']} etapas completas, cada uno con el SHA-256 de mediciones y de su etapa.")
    if r.nombres:
        n = r.nombres
        print(f"Nombres (regla acotada): {n['archivos_revisados']} textos revisados contra {n['nombres_buscados']:,} "
              f"nombres crudos y {n['llaves_varias_palabras_buscadas']:,} llaves de varias palabras "
              f"({n['filas_de_referencia']}); posible fuga en {len(n['archivos_con_posible_fuga'])}.")
        tot = {k: sum(a["clasificacion"][k] for a in n["archivos"].values()) for k in (
            "nombres_crudos_con_letras", "nombres_crudos_cortos_solo_dentro_de_un_token", "nombres_crudos_posible_fuga",
            "llaves_varias_palabras_con_letras", "llaves_una_palabra_con_letras", "nombres_crudos_solo_digitos")}
        print(f"  sumas por archivo: {tot}; tapados por el kernel: {n['tapados_por_el_kernel']}")
    print(f"Ids: {len(r.ids)} JSONL con exactamente los ids de su conjunto, en orden de int(id).")
    print(f"Kernel: estado {r.mediciones.get('estado')}, fallas {len(r.mediciones.get('fallas', []))}; "
          f"{sum(e['estado'] == 'completa' for e in r.etapas)} de {len(r.etapas)} etapas completas.")
    for m, g in r.compuertas.items():
        print(f"  compuerta {m}: kernel {g['kernel_identicas']} idénticas (cumple {g['kernel_cumple']}), recálculo local "
              f"{g['local_identicas']} de {g['filas']} (mínimo {g['minimo']}); adaptador por {g['metodo_adaptador']}")
    c = r.costos
    print(f"Tiempo del kernel: {_h(c['segundos_kernel'])} ({c['segundos_kernel']} s); etapas {_h(c['segundos_etapas'])}; "
          f"fuera de etapas {c['segundos_fuera_de_etapas']} s; orquestador {c['duraciones_orquestador_s']}")
    print("Etapas: id | estado | proceso s (estimado mejor/peor) | s/fila (mediana, p90) | al tope (JSONL) | "
          "fichas prompt máx / generadas máx | VRAM pico asignada/reservada MiB")
    for e in r.etapas:
        i = e["inferencia"]
        v = e["vram_pico_mib"] or {}
        print(f"  {e['id']} | {e['estado']} | {e['segundos_proceso']} ({e['estimado_mejor_s']}/{e['estimado_peor_s']}) | "
              f"{i['s_por_fila']} ({i['s_por_fila_mediana']}, {i['s_por_fila_p90']}) | {i['al_tope']} = {i['pct_al_tope']} % "
              f"({e['al_tope_jsonl']}) | {i['fichas_prompt_max']} / {i['fichas_generadas_max']} | "
              f"{v.get('asignada')}/{v.get('reservada')} | salida {e['codigo_de_salida']}, matado {e['matado_por_tiempo']}, "
              f"memoria liberada {e['memoria_liberada']}, fallas {e['fallas_en_la_etapa']}")
    ver = r.versiones or {}
    print(f"Versiones: fijadas distintas {ver.get('fijadas_distintas')}; no fijables {ver.get('no_fijables')}; "
          f"controlador {ver.get('controlador_nvidia')}; advertencias {r.advertencias}")
    im = r.imagen
    print(f"Imagen: metadata {im['docker_image_metadata'].rsplit(':', 1)[-1][:8]}… (fijada "
          f"{im['docker_image_fijada'].rsplit(':', 1)[-1][:8]}…); pip freeze de antes idéntico al del 4B: "
          f"{im['pip_freeze_antes_identico_al_del_4b']} (diferencias {im['pip_freeze_antes_diferencias']}); entorno "
          f"{im['entorno']} (4B {im['entorno_4b']}); versiones distintas del 4B {im['versiones_distintas_del_4b']}; "
          f"controlador {im['controlador']} (4B {im['controlador_4b']}); misma imagen que el 4B: {im['misma_imagen_que_el_4b']}")
    if r.problemas:
        print(f"PROBLEMAS: {r.problemas}")
    else:
        print("Todo pasa: 14 etapas completas y las dos compuertas cumplen.")


def main(argv: Sequence[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        r = revisar()
    except ErrorSalida as e:
        sys.exit(f"PARADA: {e}")
    imprimir(r)
    if r.problemas:
        sys.exit(1)


if __name__ == "__main__":
    main()
