# Bitácora del proyecto

> Las secciones del 2026-11-17 son una copia con formato de `../evidencia/bitacora.txt` (fuera del repo), hecha el 2026-11-18. Su texto es el original, salvo dos cambios pedidos por Samuel:
> (1) la fecha, el título y la evidencia de la parte 1 los indicó Samuel, porque esas líneas no tenían fecha en el archivo;
> (2) en la parte 2 se restauraron los acentos de: último, falló, código, desinstaló, también, dejó, resolvió y técnicas.
> Lo demás solo es formato Markdown: títulos, viñetas, rutas y comandos entre comillas invertidas y enlaces a las capturas.
> La sección "Fase 0, cierre" del 2026-11-18 la dictó Samuel durante la sesión de Claude Code de ese día, salvo su última viñeta, que Claude Code agregó el 2026-11-19 con los datos de la instrucción del paso 2a. Las de "Fase 1, paso 1" y "Fase 1, paso 2a" las redactó Claude Code a partir de los planes aprobados por Samuel; la instrucción del paso 2a la redactó el chat de Claude del proyecto con la estructura que pidió Samuel, y Samuel la aprobó sin cambios. La sección "Fase 1, paso 2b" la redactó Claude Code a partir del plan aprobado por Samuel; la instrucción del paso 2b la redactó el chat de Claude del proyecto y Samuel la aprobó. El enlace a la captura de `clee` en la sección del paso 2a se agregó durante el paso 2b. La sección "Fase 1, paso 3" la redactó Claude Code a partir del plan aprobado por Samuel; la instrucción del paso 3 la redactó el chat de Claude del proyecto y Samuel la aprobó. El enlace a la captura de la versión del SCIAN en la sección del paso 2b se agregó durante el paso 3.

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
- Evidencia: [2026-11-19_fase1_clee-etiqueta.png](evidencia/2026-11-19_fase1_clee-etiqueta.png) (`clee` contra `codigo_act`), para la diapositiva "Decisiones técnicas".
- `nom_estab`: 212 vacíos; 0 "NA", 0 "NULL", 0 "NONE" y 0 "N/A" (tampoco tras quitar espacios y pasar a mayúsculas).
- `nom_estab` igual a `nombre_act`: 0 exactos y 32 (0.0069 %) tras quitar espacios al inicio y al final y pasar a mayúsculas. No se decidió nada sobre esas filas: se decide en el análisis exploratorio.
- Números en `results/2026-11-19_fase1_diagnostico-denue.json`.
- Reglas nuevas en `CLAUDE.md`: cargador único; `clee` nunca es entrada ni variable de un modelo; Claude Code no imprime filas del DENUE; quién propuso, redactó y aprobó en los registros. La versión del SCIAN queda pendiente para el paso 2b.

## 2026-11-20 - Fase 1, paso 2b: catálogo SCIAN y versión de codigo_act

- Empezó el 2026-11-19 y terminó el 2026-11-20.
- `../evidencia/` tiene 8 archivos (7 capturas PNG y `bitacora.txt`); ninguno es del SCIAN, así que no hubo copias que comparar con las descargas.
- Fuentes oficiales descargadas del INEGI el 2026-11-19 a `../datos-originales/scian/` (fuera del repo, solo lectura): `estructura2023.pdf`, `estructura2018.pdf` y `tablaxv.pdf` (tablas comparativas 2023-2018). URL, SHA-256 y tamaño en `data/manifest.json`. Los tres PDF están cifrados sin contraseña de apertura y permiten extraer texto.
- Extractor `src/scian.py` con pypdfium2 5.15.2 (un solo paquete, sin dependencias; abre PDF cifrados con AES). Toma los títulos solo de las páginas "Versión en Español": cada estructura trae también una versión en inglés y otra español-inglés, y en la de 2018 la de inglés va primero. Quita todas las marcas "T" finales: en el PDF oficial de 2023 la subrama 23611 trae una marca doble ("Edificación residencialT T"). 27 pruebas de pytest del extractor y 12 del análisis de la versión pasan sobre datos sintéticos (58 en total en el repo).
- Candado cumplido: SCIAN 2023 con 1,086 clases, 610 subramas y 305 ramas; SCIAN 2018 con 1,084 clases, 615 subramas y 306 ramas. Los códigos únicos de las tres secciones de cada PDF coinciden con los de la sección en español.
- Versión de `codigo_act`: SCIAN 2023. Los 931 códigos distintos del DENUE (0 que no son 6 dígitos ASCII) existen en el SCIAN 2023; 31 de ellos (1,731 filas) no existen en el SCIAN 2018, por ejemplo 931412 (698 filas), 519212 (290), 931411 (150), 516110 (71) y 315123 (46). Ningún código falta en el 2023. Registrada en `data/manifest.json` (`scian_version`).
- Evidencia: [2026-11-19_fase1_version-scian.png](evidencia/2026-11-19_fase1_version-scian.png) (versión decidida y los 31 códigos que no existen en el SCIAN 2018), para la diapositiva "Decisiones técnicas".
- Evidencia secundaria (no decide): `nombre_act` coincide de forma exacta con el título 2023 en 889 de 931 pares código-título (451,696 filas) y con el 2018 en 885 (459,344 filas). En las 30 clases del DENUE cuyo título cambió entre versiones (mismo código), coincide con el título 2018 en 9,469 filas, con el 2023 en ninguna y con ninguno de los dos en 74: los códigos son del SCIAN 2023, pero `nombre_act` conserva el título 2018 de esas clases. En 14 códigos (1,567 filas), `nombre_act` no coincide con ninguna versión. Las tablas comparativas incluyen las 32 clases con título distinto, los 34 códigos que solo existen en 2023 y los 32 que solo existen en 2018.
- `clee` (caracteres 6 a 11): 31 de sus códigos distintos existen solo en 2018 (1,314 filas), 29 solo en 2023 (496), 920 en ambas (447,750) y 33 en ninguna (13,172). En las 62,404 filas que discrepan de `codigo_act`, el código de `clee` no existe en ninguna de las dos versiones en el 21.11 % (los más frecuentes son 722219 y 722212), solo existe en 2018 en el 2.11 %, existe en ambas en el 76.72 % y solo existe en 2023 en el 0.07 %. La hipótesis del paso 2a (`clee` conserva el código de una versión anterior) explica solo una parte: en la mayoría de las discrepancias, el código de `clee` es válido en las dos versiones. Sin el catálogo 2013 no se puede confirmar de qué versión vienen los códigos que no existen en ninguna.
- Catálogos en `data/scian_2023_clases.csv` y `data/scian_2018_clases.csv` (`codigo,titulo`; UTF-8; LF; ordenados por código), con su SHA-256 en el manifest. Solo se versiona el de la versión decidida, con crédito al INEGI en `README.md`.
- `.gitattributes` nuevo, con LF para `*.csv`, `*.jsonl` y `*.json`. No cambió ningún archivo ya versionado: los dos JSON del repo ya tenían LF en git y en disco.
- Captura del paso 2a copiada a `docs/evidencia/` (SHA-256 idéntico al original): solo muestra conteos y códigos SCIAN; trae un bloque de metadatos XMP con solo la orientación de la imagen.
- Números en `results/2026-11-19_fase1_catalogo-scian.json` y `results/2026-11-20_fase1_version-scian.json`.

## 2026-11-20 - Fase 1, paso 3: normalización de nombres para agrupar

- Función única `normalizar_nombre(nombre: str) -> str` en `src/normalizacion.py`, solo con `unicodedata` de la biblioteca estándar y sin depender del locale. Pasos, en orden: forma NFKD; sin marcas combinantes (Ñ→N, Ü→U); mayúsculas; el punto y el apóstrofo se borran y cualquier otro carácter que no sea letra, número ni espacio se cambia por espacio; espacios colapsados y recortados. La llave solo sirve para agrupar (splits, baseline de diccionario y hard set); los modelos Qwen reciben el `nom_estab` original.
- `nom_estab` ya viene en mayúsculas y casi sin signos: 0 nombres con minúsculas ASCII y ninguno con apóstrofo, º, ª o espacio no separable. Fuera de letras y dígitos ASCII solo aparecen la Ñ (10,158 nombres distintos), vocales con acento o tilde, la Ü (1 nombre) y 12 signos: guion (1,978), & (1,696), punto (1,449), coma (130), paréntesis (121 cada uno), @ (99), # (49), / (28), _ (28), + (8) y ? (6).
- Puntuación: se compararon tres variantes sobre los 320,440 `nom_estab` distintos: (A) todo signo se cambia por espacio, 319,893 llaves; (B) todo signo se borra, 319,900; (C) el punto y el apóstrofo se borran y el resto se cambia por espacio, 319,876. Las tres dan llave vacía solo a los 212 `nom_estab` vacíos; ningún nombre es de pura puntuación. Frente a A, B junta 194 llaves en 97 y separa en 208 llaves lo que A junta en 104; C junta 52 en 26 y separa en 18 lo que A junta en 9. Borrar el guion separa más de lo que junta (166 llaves frente a 110); borrar el punto junta más de lo que separa (52 frente a 18). Claude Code recomendó C y Samuel la aprobó.
- Ñ→N junta 74 llaves en 37 (286 filas) frente a conservar la Ñ.
- Con C: 319,876 llaves, de las cuales 543 juntan más de un `nom_estab` distinto. Grupos, sin contar la llave vacía: 286,829 de 1 fila, 30,741 de 2 a 9, 2,151 de 10 a 99 y 154 de 100 o más (37,870 filas); el más grande tiene 1,917 filas.
- Pendiente para el paso de los splits: qué hacer con las 212 filas sin nombre (llave vacía).
- 104 pruebas nuevas de pytest pasan sobre texto sintético (162 en total en el repo).
- Regla 2 de `CLAUDE.md` precisada con la normalización exacta (no es un CAMBIO DE DISEÑO).
- Dos reglas nuevas en `CLAUDE.md`, marcadas como "CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-18; registrado el 2026-11-20": el baseline de diccionario como segundo baseline obligatorio (sección a) y la regla 7 de la sección c (antes de congelar validación y test, se reportan las filas de train excluidas por compartir llave con ellos; si pasan de 10 %, se detiene y decide Samuel). Faltaban en `CLAUDE.md`; los textos los redactó el chat de Claude del proyecto y Samuel los aprobó.
- Captura del paso 2b copiada a `docs/evidencia/` (SHA-256 idéntico al original) y enlazada en la sección del paso 2b: solo muestra conteos, códigos SCIAN y la ruta del JSON, y no tiene metadatos de texto. No existe `../evidencia/2026-11-19_fase1_errata-scian.png`.
- Números en `results/2026-11-20_fase1_normalizacion-nombres.json`.
