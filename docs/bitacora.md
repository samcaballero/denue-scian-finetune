# Bitácora del proyecto

> Las secciones del 2026-11-17 son una copia con formato de `../evidencia/bitacora.txt` (fuera del repo), hecha el 2026-11-18. Su texto es el original, salvo dos cambios pedidos por Samuel:
> (1) la fecha, el título y la evidencia de la parte 1 los indicó Samuel, porque esas líneas no tenían fecha en el archivo;
> (2) en la parte 2 se restauraron los acentos de: último, falló, código, desinstaló, también, dejó, resolvió y técnicas.
> Lo demás solo es formato Markdown: títulos, viñetas, rutas y comandos entre comillas invertidas y enlaces a las capturas.
> La sección "Fase 0, cierre" del 2026-11-18 la dictó Samuel durante la sesión de Claude Code de ese día, salvo su última viñeta, que Claude Code agregó el 2026-11-19 con los datos de la instrucción del paso 2a. Las de "Fase 1, paso 1" y "Fase 1, paso 2a" las redactó Claude Code a partir de los planes aprobados por Samuel; la instrucción del paso 2a la redactó el chat de Claude del proyecto con la estructura que pidió Samuel, y Samuel la aprobó sin cambios.

## 2026-11-17 - Fase 0, parte 1

- Horas de GPU disponibles en Kaggle por semana: 30 horas.
- Usuarios — GitHub: samcaballero | Kaggle: samcaballero | Hugging Face: samcaballero
- https://github.com/samcaballero/denue-scian-finetune
- Edición del DENUE: 2026/05 (Periodo 05/2026)
- Archivo CSV: denue_inegi_09_.csv
- Tamaño: 253,929 KB (~248 MB)
- Evidencia: [2026-11-17_fase0_gpu-kaggle.png](evidencia/2026-11-17_fase0_gpu-kaggle.png)

## 2026-11-17 - Fase 0, parte 2: entorno local

- Versiones: Git 2.56.0.windows.4 | Python 3.12.10 | Claude Code 2.1.327 (npm, canal latest) | Kaggle CLI 2.2.5 | Node v24.13.0 | pip 26.3
- Python se mantiene en 3.12.10: es el último instalador de 3.12 para Windows; el entrenamiento corre en Kaggle con su propio Python.
- Incidente Git: `winget upgrade` falló (código 4, "Acceso denegado" al crear `C:\Program Files\Git\dev`) y el rollback desinstaló también la 2.53. Tras reiniciar, `winget install` dejó la 2.56 en `AppData\Local\Programs\Git`.
- Incidente Kaggle: la CLI 2.2.5 no arrancaba (`ModuleNotFoundError: six`). Se resolvió con `pip install six`.
- Llave de Kaggle en `C:\Users\Invitado\.kaggle\access_token`, fuera del repo. Verificada con: `kaggle kernels list --mine`
- Git: user.name "Sam Caballero", correo privado noreply de GitHub, bloqueo de pushes que expongan el correo, rama por defecto main.
- Repo clonado en `C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune\`
- Evidencia: [2026-11-17_fase0_kaggle-cli.png](evidencia/2026-11-17_fase0_kaggle-cli.png) (la laptop controla Kaggle desde la terminal; diapositiva "Decisiones técnicas", flujo Claude Code -> Kaggle)
- Evidencia: [2026-11-17_fase0_herramientas.png](evidencia/2026-11-17_fase0_herramientas.png) (versiones del entorno; respaldo para preguntas sobre el setup)

## 2026-11-18 - Fase 0, cierre: estructura del repo

- Modelo de Claude Code: de Fable 5.2 (guardado por error como predeterminado) a Opus 5.5 con esfuerzo xhigh (subido a propósito para obtener más calidad).
- Primera sesión de Claude Code en el repo: CLAUDE.md, carpetas, .gitignore, evidencia, bitácora y README.
- Hallazgos de Claude Code: ruta real del CSV, tamaño real de unos 248 MB, codificación latin-1, columnas de contacto excluidas y correo personal en el commit inicial.
- Correo del commit inicial corregido: se reescribió el autor con el noreply de GitHub (commit 012e43a, mismo contenido). Pendiente: borrar y recrear el repo en GitHub para eliminar el commit viejo, y subir el historial limpio.
- Evidencia: [2026-11-18_fase0_plan-claude-code.png](evidencia/2026-11-18_fase0_plan-claude-code.png) y [2026-11-18_fase0_hallazgos-claude-code.png](evidencia/2026-11-18_fase0_hallazgos-claude-code.png), para la diapositiva "Decisiones técnicas: qué propuso Claude Code y qué revisé yo".
- Cierre del pendiente (agregado el 2026-11-19): el repo se borró y se recreó en GitHub el 2026-11-18. El commit viejo `1f60028` da 404 (página no encontrada), y los commits `012e43a` y `648a6ba` aparecen con el usuario samcaballero.

## 2026-11-18 - Fase 1, paso 1: entorno local de Python

- Entorno virtual `.venv/` creado con `py -3.12` (Python 3.12.10). Solo para trabajo en CPU (pandas, scikit-learn); sin torch, transformers ni unsloth, porque el entrenamiento corre en Kaggle.
- Paquetes principales: pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2, pyarrow 26.0.1, pytest 9.2.0. Versiones exactas de todo el entorno en `requirements.txt`.
- CSV original: 462,732 filas y 42 columnas, leído con latin-1. SHA-256, tamaño y número de filas registrados en `data/manifest.json`.
- Diccionario de datos: codificación latin-1; diferencias con el encabezado del CSV: los mismos 42 campos con los mismos nombres; solo cambia el orden de `clee` (columna 2 en el CSV, última en el diccionario).
- Reglas nuevas en `CLAUDE.md`: Python siempre con `.venv/Scripts/python`; etiqueta "CAMBIO DE DISEÑO" para cambios en modelos, tareas, splits, métricas o reglas de evaluación; el clasificador responde solo el código SCIAN de 6 dígitos y cada respuesta se clasifica como formato inválido, código inexistente o código existente ("% formato válido" y "% códigos inexistentes" sobre el total de respuestas); `docs/sesiones/` se versiona solo tras una revisión de datos sensibles aprobada por Samuel.

## 2026-11-19 - Fase 1, paso 2a: cargador único del DENUE

- Claude Code 2.1.328 (el 2026-11-17 era la 2.1.327; se actualizó solo).
- Finales de línea: `core.autocrlf=true` en la configuración de sistema de Git for Windows. El repo guarda todo en LF; en disco, `.gitignore`, `LICENSE` y `requirements.txt` tienen CRLF y el resto LF. El aviso de `git add` del paso 1 ("LF will be replaced by CRLF the next time Git touches it") venía de los archivos con LF; no afecta lo que se sube ni al CSV. Riesgo futuro: hashes de conjuntos congelados guardados como texto en git.
- Diccionario de datos: según el diccionario, `codigo_act` y `nombre_act` usan el SCIAN 2018, pero la edición se basa en los Censos Económicos 2024 (SCIAN 2023): la versión queda por confirmar en el paso 2b. `clee` es la llave estadística única del RENEM y vincula los establecimientos de una misma empresa; el diccionario no dice que contenga el código de actividad. Si un negocio no tiene nombre comercial, `nom_estab` lleva el nombre genérico o la descripción de la actividad.
- Codificación confirmada en el archivo completo: no es UTF-8 válido y tiene 0 bytes entre 0x80 y 0x9F, así que latin-1 y cp1252 lo leen igual; se queda latin-1. Fin de línea LF, sin CR.
- Cargador único `cargar_denue()` en `src/denue.py`: verifica el SHA-256 contra `data/manifest.json`, rechaza `telefono`, `correoelec`, `www` y `raz_social`, y conserva textos como "NA" o "NULL". 19 pruebas de pytest pasan sobre CSV sintéticos.
- `clee` contra `codigo_act`: los caracteres 6 a 11 de `clee` coinciden con `codigo_act` en 400,328 de 462,732 filas (86.51 %) y discrepan en 62,404. Todas las `clee` miden 28 caracteres; `id` y `clee` no se repiten en ninguna de las 462,732 filas.
- `nom_estab`: 212 vacíos; 0 "NA", 0 "NULL", 0 "NONE" y 0 "N/A" (tampoco tras quitar espacios y pasar a mayúsculas).
- `nom_estab` igual a `nombre_act`: 0 exactos y 32 (0.0069 %) tras quitar espacios al inicio y al final y pasar a mayúsculas. No se decidió nada sobre esas filas: se decide en el análisis exploratorio.
- Números en `results/2026-11-19_fase1_diagnostico-denue.json`.
- Reglas nuevas en `CLAUDE.md`: cargador único; `clee` nunca es entrada ni variable de un modelo; Claude Code no imprime filas del DENUE; quién propuso, redactó y aprobó en los registros. La versión del SCIAN queda pendiente para el paso 2b.
