# Bitácora del proyecto

> Las secciones del 2026-11-17 son una copia con formato de `../evidencia/bitacora.txt` (fuera del repo), hecha el 2026-11-18. Su texto es el original, salvo dos cambios pedidos por Samuel:
> (1) la fecha, el título y la evidencia de la parte 1 los indicó Samuel, porque esas líneas no tenían fecha en el archivo;
> (2) en la parte 2 se restauraron los acentos de: último, falló, código, desinstaló, también, dejó, resolvió y técnicas.
> Lo demás solo es formato Markdown: títulos, viñetas, rutas y comandos entre comillas invertidas y enlaces a las capturas.
> La sección "Fase 0, cierre" del 2026-11-18 la dictó Samuel durante la sesión de Claude Code de ese día, salvo su última viñeta, que Claude Code agregó el 2026-11-19 con los datos de la instrucción del paso 2a. Las de "Fase 1, paso 1" y "Fase 1, paso 2a" las redactó Claude Code a partir de los planes aprobados por Samuel; la instrucción del paso 2a la redactó el chat de Claude del proyecto con la estructura que pidió Samuel, y Samuel la aprobó sin cambios. La sección "Fase 1, paso 2b" la redactó Claude Code a partir del plan aprobado por Samuel; la instrucción del paso 2b la redactó el chat de Claude del proyecto y Samuel la aprobó. El enlace a la captura de `clee` en la sección del paso 2a se agregó durante el paso 2b. La sección "Fase 1, paso 3" la redactó Claude Code a partir del plan aprobado por Samuel; la instrucción del paso 3 la redactó el chat de Claude del proyecto y Samuel la aprobó. El enlace a la captura de la versión del SCIAN en la sección del paso 2b se agregó durante el paso 3. La sección "Fase 1, paso 3b" la redactó Claude Code a partir del plan aprobado por Samuel; la instrucción del paso 3b la redactó el chat de Claude del proyecto y Samuel la aprobó. La sección "Fase 1, paso 4" la redactó Claude Code a partir del plan aprobado por Samuel; la instrucción del paso 4 la redactó el chat de Claude del proyecto y Samuel la aprobó. La sección "Fase 1, paso 5" la redactó Claude Code a partir del plan aprobado por Samuel; la instrucción del paso 5 la redactó el chat de Claude del proyecto y Samuel la aprobó.

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

## 2026-11-20 - Fase 1, paso 3b: candados de configuración y ajustes de reglas

- Candados en `.claude/settings.json` (versionado): 38 reglas `deny` de permisos de Claude Code, 19 para Bash y 19 para PowerShell. Rechazan `git push` en sus formas directas, también con opciones globales de git antes del subcomando (`git -C <dir> push`), y `git add .`, `git add -A`, `git add --all` y `git add -f` (o `--force`) con el signo en cualquier posición. Según la documentación de Claude Code, las reglas `deny` aplican en todos los modos, incluido el automático, y ninguna regla `allow` de otro archivo las anula. No cubren `bash -c "…"`, programas que llamen a git por su cuenta ni banderas combinadas como `-fA`.
- `.claude/settings.local.json` ya existía, ignorado por la configuración global de git, con una sola regla `allow` para pytest; no se tocó.
- Prueba de bloqueo: Claude Code intentó `git push --dry-run` y `git add -A --dry-run` en Bash y en PowerShell, y `git -C . push --dry-run` en Bash. Las reglas rechazaron los cinco intentos ("Permission to use … has been denied"). El control `git add --dry-run CLAUDE.md` sí corrió, y el índice no cambió.
- Candado local en esta laptop (aprobado por Samuel): el hook `.git/hooks/pre-push` rechaza el push cuando existe la variable `CLAUDECODE`, que definen las shells de Claude Code (Bash y PowerShell) y no la terminal de Samuel. No se versiona ni viaja con el repo. Prueba: `bash -c 'git push --dry-run …'`, una vía que las reglas `deny` no cubren, contra un repo vacío local creado para la prueba, sin tocar GitHub. Las reglas lo dejaron pasar, el hook lo rechazó y el repo de prueba siguió sin referencias. El mismo script, ejecutado sin `CLAUDECODE`, termina con código 0, así que no bloquea el push de Samuel.
- Reglas nuevas en `CLAUDE.md` (sección e): el push lo hace siempre Samuel desde PowerShell, después de que el chat de Claude del proyecto revisa el reporte; los candados (`.claude/settings.json` y el hook local) no se modifican sin aprobación de Samuel; el modo automático solo se usa con un plan ya aprobado.
- Regla 2 precisada (aprobada por Samuel): la llave sirve para agrupar y es la entrada del baseline de diccionario; los modelos Qwen y el baseline TF-IDF reciben el `nom_estab` original. Regla 6: "incluidos los baselines".
- Regla 7 (CAMBIO DE DISEÑO aprobado por Samuel, privacidad): las 20 llaves que más filas excluyen se muestran solo a Samuel; en `results/` y en git solo van sus conteos y tamaños.
- Regla nueva en la sección d: ningún `nom_estab` ni llave normalizada se escribe en `results/` ni en archivos versionados, y los ejemplos para la charla los elige Samuel. Una revisión de los archivos versionados encontró coincidencias con llaves solo en títulos del catálogo SCIAN, en nombres inventados de las pruebas y en dos frases propias de la bitácora y del registro; ninguna se copió del DENUE, y la regla aclara que esas coincidencias no cuentan como copia.

## 2026-11-20 - Fase 1, paso 4: análisis exploratorio para el clasificador

- Script `src/exploracion_clasificador.py`: usa `cargar_denue()`, `normalizar_nombre()` y `data/scian_2023_clases.csv`. Solo mide y propone: no filtra, no congela y no cambia `CLAUDE.md`. Los títulos de sector, subsector, rama y subrama salen de la estructura oficial 2023 con `src/scian.py` (hash y candado verificados), porque el catálogo versionado solo trae clases. En el nivel de 2 dígitos se usan los sectores oficiales: 31-33 y 48-49 cuentan como un sector cada uno. 24 pruebas nuevas de pytest pasan sobre datos sintéticos (186 en total en el repo).
- Clases: 931 de las 1,086 del catálogo tienen filas y 155 no tienen ninguna. Filas por clase presente: mínimo 1, cuartiles 11, 37 y 211, y máximo 38,996. Con menos de 10, 50 y 100 filas hay 208, 509 y 600 clases. Las 10, 50 y 100 clases más grandes cubren el 31.22 %, el 65.43 % y el 80.5 % de las filas. Tienen filas 20 de los 20 sectores, 89 de los 94 subsectores, 282 de las 305 ramas y 551 de las 610 subramas.
- Nombres problemáticos: las 212 filas sin nombre (llave vacía) están en 105 clases. El DENUE marca los negocios sin nombre comercial con la frase "SIN NOMBRE" después de una descripción de la actividad; ninguna llave es solo la frase. Tienen la frase 43,979 filas (9.51 % de las filas con nombre), en 9,523 llaves y 501 clases. Otras 367 filas tienen una errata de la frase y no se cuentan como marcador. Llaves iguales al título normalizado de su propia clase: 141 filas; al de otra clase: 305; a `nombre_act` normalizado: 144. Sin la frase "SIN NOMBRE", la descripción es igual al título de su clase en 2,191 filas.
- Techo de ambigüedad (llaves de 2 o más filas, sin la vacía): 88.9 % en clase, 91.21 % en subrama, 92.28 % en rama, 93.41 % en subsector y 95.45 % en sector. Esas llaves suman 175,691 filas, el 37.99 % de las filas con nombre, y 23,128 de las 33,046 llaves repetidas tienen una sola clase. De los 154 grupos de 100 o más filas, 12 tienen una sola clase y en 7 la clase mayoritaria no llega a la mitad. De esos 154 grupos, 69 (17,616 filas) tienen la frase "SIN NOMBRE", 68 (18,252 filas) son genéricos según la definición del script y 37 (9,710 filas) son las dos cosas.
- Regla 7, estimación (sin la semilla 42; simulaciones con las semillas 1000 a 1099): con 3,000 filas de validación y test sacadas al azar entre las filas con nombre, se excluiría de train el 8.97 % de las filas candidatas (cálculo analítico); en las simulaciones, media 8.97 %, percentil 5 8.63 % y percentil 95 9.38 %. Los 154 grupos grandes aportan el 73.85 % de lo excluido; los grandes con la frase "SIN NOMBRE", el 34.95 %, y los grandes genéricos, el 36.53 % (las dos partes se traslapan). Con 3,000 filas en cada conjunto (6,000 en total), el 11.84 %.
- Topes por clase en train (filas con nombre, sin separar validación y test): con 100, 200, 500 y 1,000 filas por clase quedarían 47,349, 74,850, 130,289 y 190,188 de las 462,520 filas, y se recortarían 330, 239, 149 y 100 clases.
- Figuras en `docs/figuras/` (PNG de 1920×1080, sin nombres de negocio): `2026-11-20_fase1_cola-larga-clases.png`, para la diapositiva de los datos (931 clases y su cola larga); `2026-11-20_fase1_filas-por-sector.png`, para la diapositiva del DENUE de la CDMX por sector; `2026-11-20_fase1_techo-ambiguedad.png`, para la diapositiva "¿Cuánto se puede acertar solo con el nombre?"; `2026-11-20_fase1_tamanos-grupo.png`, para la diapositiva del split por nombre.
- Archivo local de revisión para Samuel: `data/2026-11-20_fase1_revision-llaves.csv` (UTF-8 con BOM; ignorado por git, verificado con `git check-ignore`), con las 200 llaves con más filas y sus marcas.
- Propuestas pendientes de decisión de Samuel; no se aplicó ninguna. Vacíos: Claude Code recomienda excluirlos de train, validación y test (CAMBIO DE DISEÑO). Frase "SIN NOMBRE": recomienda mantener esas filas y reportar aparte, en validación y test, la accuracy con y sin la frase (CAMBIO DE DISEÑO). Llaves iguales a un título o a `nombre_act`: recomienda mantenerlas. Nombres genéricos: recomienda mantenerlos sin desglose propio mientras Samuel no valide la definición en el archivo de revisión; son 50,134 filas (10.84 % de las filas con nombre) y 20,983 de ellas también tienen la frase "SIN NOMBRE".
- Mapa de carpetas de `CLAUDE.md`: se agregó `docs/figuras/`, a pedido de Samuel al aprobar el plan.
- Números en `results/2026-11-20_fase1_exploracion-clasificador.json`.

## 2026-11-21 - Fase 1, paso 5: validación y test congelados

- Empezó el 2026-11-20 y terminó el 2026-11-21.
- Población del clasificador (regla 8 nueva de `CLAUDE.md`, CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-20): las 212 filas sin nombre (llave vacía) se excluyen de train, validación y test; se mantienen las filas con la frase "SIN NOMBRE", las de llave igual a un título o a `nombre_act` y las de nombre genérico. La métrica principal es la del test con la distribución natural; en validación y test se reporta además la accuracy de las filas con y sin la frase "SIN NOMBRE" y de las genéricas y no genéricas. Las marcas solo sirven para reportar; nunca son entrada de un modelo. Claude Code lo propuso en el paso 4; el chat de Claude del proyecto redactó la instrucción y Samuel la aprobó.
- Regla 2 (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-20): decía "todas las filas con el mismo nombre normalizado van al mismo split"; ahora dice que ninguna llave aparece en dos conjuntos y que las demás filas de las llaves de validación y test se excluyen de train (regla 7). Claude Code señaló la contradicción en el plan; Samuel aprobó el texto.
- Precisiones de Samuel al aprobar el plan: "con y sin la frase" son dos subconjuntos (filas con la frase y filas sin ella), igual que genéricas y no genéricas; la entrada del vocabulario en `data/manifest.json` la escribe el código, con hash, tamaño y conteo calculados.
- Módulo único de subgrupos `src/subgrupos.py` (frase "SIN NOMBRE", genérico e igual a título); `src/exploracion_clasificador.py` ahora importa de ahí sus definiciones. Reproduce los conteos del paso 4: 43,979, 50,134 y 449 filas. El vocabulario de los genéricos (1,453 palabras de los títulos normalizados de todos los niveles de la estructura oficial 2023) quedó en `data/scian_2023_vocabulario.csv`, versionado, con su SHA-256 en `data/manifest.json`, crédito al INEGI en `README.md` y la excepción ampliada en la sección d de `CLAUDE.md`.
- Procedimiento (`src/splits.py`, semilla 42): universo de 462,520 filas con llave no vacía en orden de `id` numérico; permutación con `numpy.random.default_rng(42)`; test = las primeras 2,000 filas; validación = las siguientes 1,000 cuya llave no está en test; train candidato = las filas cuya llave no está en ninguno de los dos; el resto se excluye.
- Regla 7: 41,372 de 459,520 filas candidatas a train excluidas (9.0 %; 34,457 por llaves de test y 6,915 por llaves de validación), por debajo del 10 % y dentro del rango que estimó el paso 4 (8.63 % a 9.38 %). Las 20 llaves que más excluyen son todas de test y excluyen de 423 a 1,910 filas cada una. Están solo en `data/2026-11-21_fase1_regla7-llaves.csv` (ignorado por git, verificado con `git check-ignore`); el punto de control del 2026-11-20 dejó una copia idéntica con la fecha de ese día.
- Conjuntos: test 2,000 filas, 1,902 llaves y 318 clases; validación 1,000 filas, 990 llaves y 228 clases; train candidato 418,148 filas, 316,983 llaves y 929 clases; excluidas 41,372 filas, 989 llaves y 318 clases. Frase "SIN NOMBRE": 183 en test y 59 en validación; genéricas: 211 y 77; iguales a título: 5 y 0. Filas cuya clase no aparece en train candidato: 0 en test y 0 en validación.
- Validación no tiene la distribución natural: el 5.9 % de sus filas tiene la frase "SIN NOMBRE" y el 7.7 % son genéricas, contra el 9.51 % y el 10.84 % del universo (en test, 9.15 % y 10.55 %). Validación solo toma filas cuya llave no está en test, y test ya tocó casi todos los grupos grandes, que concentran esas marcas; es una propiedad del procedimiento, no de la semilla. Claude Code lo detectó en el punto de control; Samuel lo aceptó y decidió no cambiar el procedimiento ni volver a sortear.
- Splits en `data/splits/` (ignorado por git; JSONL en UTF-8 sin BOM, `\n`, ordenado por `id`): `test.jsonl` (SHA-256 `9799a9a6efd249e5ce9ca73940377ec00af402e2a719481642873df7d0d9550c`) y `val.jsonl` (`c78dc600e44a979e4fc993a072b95fa6fe02e179171a5644609598ed0203ba6d`) congelados; `train_candidatos.jsonl` (418,148 filas, `2d44ad971821ea8736170be5ac6db9c386258fa07328a2ba60108692857053c0`) sin congelar, porque el hard set y el tope por clase lo cambiarán. Commit del código `ef5be77`, numpy 2.5.5. Samuel aprobó escribirlos tras el punto de control. Cada archivo se releyó para comparar su hash y se regeneró en memoria con los mismos bytes.
- Pendiente: con el split por nombre, ninguna llave de validación o test está en train, así que ahí el baseline de diccionario siempre responde la clase más frecuente global; el experimento de split aleatorio contra split por nombre (regla 7) es el que lo pone a prueba.
- 37 pruebas nuevas de pytest pasan sobre datos sintéticos (223 en total en el repo).
- Números en `results/2026-11-21_fase1_splits-clasificador.json`.

## 2026-11-21 - Fase 2, paso 6a: muestra de prueba y dataset privado en Kaggle

- Datos para una prueba de punta a punta desechable con Qwen3-0.6B en Kaggle (el notebook es el paso 6b); sus números no se reportan. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Samuel aprobó el plan sin querer antes de que el chat lo revisara; Claude Code siguió en modo automático y se detuvo en el punto de control, como decía el plan. El chat lo revisó después, y Samuel aprobó el plan, el punto de control y tres cambios de Claude Code al plan (ver `docs/claude-code-log.md`).
- Candado de Kaggle (commit `f3e0b7d`): 20 reglas `deny` nuevas en `.claude/settings.json` (10 en Bash y 10 en PowerShell) contra `kaggle datasets create`, `version` y `delete` y `kaggle kernels push`, también con los alias de la CLI (`d`, `k`, `update`) y con cualquier prefijo (ruta completa, `kaggle.exe`, `.venv/Scripts/kaggle`, `python -m kaggle`). El chat de Claude del proyecto lo propuso y Samuel lo aprobó. Motivo: con el modo automático, todo lo que saca algo de la laptop (subir datos o lanzar corridas que gastan la cuota de GPU) debe tener un candado técnico, igual que `git push`, y no solo la regla escrita de `CLAUDE.md`. Desde ahora Samuel ejecuta desde PowerShell `kaggle datasets create`, `kaggle datasets version` y `kaggle kernels push`.
- Prueba del candado: las reglas rechazaron los 13 intentos de Claude Code (8 en Bash, uno de ellos compuesto con `&&`, y 5 en PowerShell). Todos apuntaban a una carpeta o a un dataset que no existen, así que nada se habría subido ni borrado aunque una regla fallara. Los comandos de solo lectura (`kaggle datasets status` y `files`, `kaggle kernels status` y `output`) siguen permitidos.
- Reglas nuevas en `CLAUDE.md` (sección e): los candados de Kaggle y que Claude Code no escribe ni ejecuta scripts que llamen a la API de Kaggle para crear, versionar o borrar datasets ni para lanzar kernels, porque las reglas `deny` solo revisan el texto del comando. Claude Code señaló ese límite y propuso mencionar los candados de Kaggle en la sección e; el chat de Claude del proyecto redactó la regla y Samuel la aprobó.
- Script `src/muestra_humo.py` (commit `90bfc11`): verifica el SHA-256 de `val.jsonl` y `train_candidatos.jsonl` contra `data/manifest.json` antes de leerlos y saca 2,000 y 200 filas con `numpy.random.default_rng(42)`: un generador nuevo por archivo, las primeras k posiciones de una permutación, ordenadas. Escribe las filas en orden de `id` numérico, solo con `id`, `nom_estab` y `codigo_act`, sin la llave ni las marcas. `test.jsonl` nunca se abre: de él solo se usa su hash del manifest. 34 pruebas nuevas de pytest pasan sobre datos sintéticos (257 en total en el repo); una de ellas espía `open` y confirma que el flujo completo no abre un `test.jsonl` inventado.
- Candado de la carpeta: en ella solo pueden estar `humo_train.jsonl`, `humo_val.jsonl` y `dataset-metadata.json`, y ninguno puede tener el SHA-256 de otro archivo del manifest, en particular el de test; si no, el script se detiene. Corre en el punto de control, al escribir y con `--verificar`, justo antes de subir.
- Muestra: `humo_train.jsonl` con 2,000 filas, 334 clases y 162,934 bytes (SHA-256 `b68789187127ea8280153724ca668776e9a32f953de9210d3d2103ef1c0e3d2c`); `humo_val.jsonl` con 200 filas, 94 clases y 16,290 bytes (`70db8e2a483e35ae7f851dbdbc88dd38937348028ecefc332c787e6917a5bcc0`). Ningún `id` se comparte entre las dos; 7 clases de la muestra de validación (7 filas) no están en la muestra de train. Puede incluir filas que luego vayan al hard set; el modelo se desecha.
- `dataset-metadata.json` (400 bytes, `d26dd65c2248dd296b096e128569d94007645a0831c728ee2151285da8e5ae4e`): en ASCII, con el título "DENUE SCIAN prueba de humo", licencia "Other" y el crédito "Fuente: INEGI, DENUE" con los Términos de Libre Uso del INEGI.
- Carpeta fuera del repo: `../kaggle-datasets/denue-scian-humo/`. Hashes, origen, semilla, procedimiento y commit del código en `data/manifest.json`, en una sección nueva de primer nivel (`kaggle_datasets.denue_scian_humo`) que escribe el código; las demás secciones quedaron idénticas (51 líneas agregadas, ninguna borrada). No va a `results/`: no hay métricas.
- Subida, hecha por Samuel desde PowerShell: `--verificar` confirmó los tres hashes y el candado, y `kaggle datasets create -p` con la ruta completa de la carpeta subió los dos JSONL ("Your private Dataset is being created"). `kaggle datasets status` dio `ready`, y `kaggle datasets files` lista los dos archivos con los mismos bytes que el manifest. La URL del dataset da 404 en una ventana privada, y la página con la sesión de Samuel dice "Private". Captura en `docs/evidencia/2026-11-21_fase2_dataset-privado-kaggle.png`; Claude Code la revisó: no muestra correos, llaves, tokens, teléfonos ni contraseñas, no trae metadatos de texto ni EXIF y solo muestra el nombre y el avatar de la cuenta de Kaggle de Samuel.
- Incidente sin efecto: antes de `--verificar`, Samuel escribió `kaggle datasets create` sin `-p` desde la raíz del repo. Sin `-p`, la CLI usa la carpeta actual; falló con "Metadata file not found" porque la raíz no tiene `dataset-metadata.json`, y no subió nada. Causa: el chat de Claude del proyecto listó los comandos con nombres cortos; desde ahora los da completos.
- Cuota de GPU de Kaggle, vista por Samuel con Gemini, sin iniciar sesión: 00:00 de 30 h usadas; se reinicia el viernes 2026-11-27 a las 18:00 (GMT-6), después de la charla, así que todo lo que se entrene antes de la charla sale de esas 30 h. Aceleradores disponibles: GPU T4 x2 y TPU v5e-8; no hay P100. La captura de la cuota no está en `../evidencia/`, así que no se copió.

## 2026-11-21 - Fase 2, paso 6b: kernel de la prueba de punta a punta

- Kernel de Kaggle que prueba toda la cadena con Qwen3-0.6B: QLoRA sobre `humo_train.jsonl`, inferencia greedy en las 200 filas de `humo_val.jsonl`, exportación a GGUF Q4_K_M y una medición corta del Qwen3-4B. Es desechable y no reportable: no escribe en `results/`. Las métricas de la regla 5 se calculan en la laptop en el paso 6c, a partir de `predicciones.jsonl`.
- Autoría: la instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Claude Code propuso el plan; el chat lo revisó con ocho ajustes y Samuel lo aprobó.
- Decisiones:
  - Modelo: opción B. Se usan los repos exactos `Qwen/Qwen3-0.6B` (revisión `c1899de2…`) y `Qwen/Qwen3-4B` (`1cfa9a72…`) con `use_exact_model_name=True`; las revisiones se verifican contra lo descargado y bitsandbytes cuantiza a NF4 al cargar. El chat de Claude del proyecto la recomendó y Samuel la aprobó. No es un CAMBIO DE DISEÑO y `CLAUDE.md` no cambia.
  - Mediciones de lote (solo tiempo, desechables): unos 20 pasos con lote 32 y 64 en el 0.6B, y con lote 32 en el 4B si cabe en memoria. Las propuso el chat de Claude del proyecto y Samuel las aprobó. No cambian el entrenamiento de la prueba: en el 0.6B corren sobre un modelo recién cargado, después de guardar todo.
  - Guardar el adaptador LoRA como respaldo: lo propuso Claude Code y Samuel lo aprobó.
  - La comprobación (b) (cómo difiere la plantilla oficial del texto de entrenamiento construido) solo se registra. (a), (c), (d) con conteo distinto de 0 y cualquier fila de más de 128 fichas abortan. El chat de Claude del proyecto precisó el texto y Samuel lo aprobó.
  - Estimación de GPU: de 20 a 35 minutos esperados, con tope de 1.5 horas (`-t 5400`); el 4B se omite si el reloj pasa de 60 minutos. Samuel la aprobó.
- Versiones fijadas: unsloth 2026.11.9, unsloth_zoo 2026.11.8, transformers 5.17.0, trl 1.15.0, datasets 4.8.5, peft 0.21.3, bitsandbytes 0.50.2 y accelerate 1.15.0; torch es el de la imagen de Kaggle si es menor que 2.15. Motivo: es el código de Unsloth que Claude Code revisó para el plan, y transformers, trl y datasets están en los topes que acepta esa versión.
- Base de 16 bits del GGUF: según el código de Unsloth, la fusión no aplica `use_exact_model_name` y resuelve `Qwen/Qwen3-0.6B` a `unsloth/Qwen3-0.6B` en su rama main. Por eso el kernel aplica el camino de verificación que pidió el chat:
  - antes de exportar, comprueba que la rama main del espejo sea la revisión fijada (`db4e90b8…`);
  - compara el SHA-256 de cada tensor con los de `Qwen/Qwen3-0.6B`, tomando en cuenta el `lm_head` atado a los embeddings;
  - si algo difiere, no exporta, registra los tensores distintos y sigue con el 4B.

  En la laptop, las cabeceras de los dos safetensors ya coinciden (310 tensores con el mismo nombre, tipo y forma; el `lm_head.weight` está solo en el repo de Qwen, y los dos `config.json` dicen `tie_word_embeddings: true`). La comparación byte a byte la hace el kernel.
- Prompt: una sola definición, en `src/prompt_clasificador.py`. Es un solo mensaje de usuario: "Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. Responde solo con el código de 6 dígitos, sin texto adicional." y, en otra línea, "Nombre: " seguido del `nom_estab` original. El prompt de generación con `enable_thinking=False` está escrito a mano y el kernel lo compara byte a byte con la plantilla oficial en las 2,201 filas (las 2,200 de la muestra y el nombre inventado). El texto de entrenamiento es el prompt de generación + el código + `<|im_end|>`, sin el `\n` que agrega la plantilla.
- Kernel en forma de script (`notebooks/kaggle-humo/humo_6b.py`), generado por `src/construir_kernel.py`:
  - lleva incrustados los módulos de `src/` con su SHA-256 y los hashes de las muestras leídos de `data/manifest.json`, porque la CLI de Kaggle solo sube el archivo de código;
  - `kernel-metadata.json` lo genera el mismo script: kernel privado, `machine_shape` `NvidiaTeslaT4` ("GPU T4 x2"), internet, el dataset como fuente e imagen fijada (`docker_image_pinning_type` `original`);
  - en la corrida se usa una sola GPU (`CUDA_VISIBLE_DEVICES=0`) y el kernel aborta si `nvidia-smi` no muestra T4.
- 82 pruebas nuevas de pytest pasan sobre datos sintéticos (339 en total en el repo). Una de ellas comprueba que el kernel escrito es exactamente lo que se generaría hoy. Además, Claude Code corrió fuera del repo un simulacro de todas las etapas con torch, Unsloth y Hugging Face falsos, en seis modos (normal, espejo distinto, sin memoria, reloj pasado, generación que no acepta `None` y tokens de más), y comprobó que el JSON de mediciones no lleva nombres.
- Pendiente para la Fase 5: la guía indica que Bedrock Custom Model Import pide guardar el modelo con transformers 4.51.3, y aquí se usa 5.17.0. No se resuelve ahora. Lo registró el chat de Claude del proyecto.
- PARADA: el kernel está escrito, probado y con commit, sin push. Falta que Samuel lo lance con `kaggle kernels push` desde PowerShell.

## 2026-11-21 - Fase 2, paso 6c: salida de la prueba de punta a punta y métricas locales

- Autoría: la instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Claude Code propuso el plan; el chat lo revisó con tres ajustes (la precisión de la regla 5 en `CLAUDE.md`, la captura de la corrida y la lista de commits pendientes de push) y Samuel lo aprobó.
- Corrida: Samuel lanzó `samcaballero/denue-scian-humo-6b` (versión 1) desde PowerShell y terminó con `COMPLETE`. El script corrió de 22:51:43 a 23:03:37 UTC (714 s); la última marca del log está en 723.6 s, y la página de Kaggle dice 16 minutos. Se registran 0.27 horas de GPU, la mayor de las mediciones; falta que Samuel confirme la cuota que descontó Kaggle.
- Descarga: con `kaggle kernels output` a `kaggle-output/humo-6b/`, ignorada por git. Bajaron 11 archivos (437,455,673 bytes, casi todo del GGUF). Los 9 de los que el kernel calculó un SHA-256 coinciden con lo bajado. Ninguno tiene un SHA-256 del manifest, o sea, no salieron datos.
- Revisión de nombres, hecha antes de mirar el log o las mediciones. Se buscaron las 2,200 filas de las dos muestras, con su hash verificado contra el manifest:
  - No aparece ningún `nom_estab` ni ninguna llave de varias palabras en el log, en `humo_mediciones.json`, en las predicciones ni en los demás archivos, y el kernel no tapó nada.
  - Hay coincidencias de una sola palabra (una llave de 5 letras): en `pip_freeze_antes.txt` (líneas 727 y 728), en `pip_freeze_despues.txt` (732 y 733) y en el README del adaptador (línea 36). Son listas de paquetes y la plantilla de la tarjeta del modelo, no datos del DENUE. Quedan para que Samuel las revise.
- Etapas: las 12 terminaron bien.
  - GPU: dos T4 en la sesión y solo una visible para torch.
  - Datos y versiones: hashes de los datos verificados y versiones fijadas instaladas; torch 2.11.0 viene de la imagen.
  - Revisiones: las tres de Hugging Face, y no se descargó nada más.
  - Plantilla y tokens: (a) a (d) bien; ninguna fila pasa de 128 fichas (el máximo es 104), y en la tabla del ejemplo inventado solo llevan etiqueta los 6 dígitos y `<|im_end|>`.
  - Entrenamiento: en fp16; la pérdida bajó de 2.30 a 0.88 (promedio de los primeros y de los últimos 10 pasos).
  - Espejo: sus 310 tensores son idénticos a los oficiales, con el `lm_head` atado a los embeddings.
  - GGUF: exportado, con la plantilla igual a la del tokenizador.
  - 4B: corrió 50 pasos con lote 16 y otra medición con lote 32.
  - Detalle en `results/2026-11-21_fase2_prueba-humo.json`.
- Métricas de la regla 5 (no reportables en la charla), sobre 200 respuestas:
  - 0 con formato inválido, 1 código inexistente (0.5 %) y 199 códigos existentes; formato válido, 100 %;
  - 15 aciertos: accuracy de 7.5 %.
  - El modelo respondió 461110 (abarrotes) en 199 de las 200 filas. Es la clase más frecuente de `humo_train` (162 de 2,000 filas) y aparece en 15 filas de la validación: el modelo colapsó a la clase mayoritaria.
  - La cadena de punta a punta funciona, pero con 2,000 filas y 125 pasos el modelo de prueba no aprendió nada más que esa clase.
- Rendimiento en una T4, sin los 5 pasos de calentamiento:
  - 0.6B: 42.9 ejemplos/s con lote 16, 50.9 con lote 32 y 50.4 con lote 64.
  - 4B: 7.9 ejemplos/s con lote 16 y 8.5 con lote 32.
  - VRAM pico reservada: 4,886 MiB como máximo (4B con lote 32).
- Horas de GPU estimadas para 1 época, con el lote más rápido (32 en los dos modelos):

  | Filas de train | 0.6B | 4B |
  |---:|---:|---:|
  | 50,000 | 0.27 h | 1.63 h |
  | 75,000 | 0.41 h | 2.45 h |
  | 130,000 | 0.71 h | 4.24 h |
  | 190,000 | 1.04 h | 6.20 h |

  A cada corrida se suma un costo fijo medido de 99 s en el 0.6B y 154 s en el 4B. Son solo estimaciones; el lote y el tope por clase los deciden el chat y Samuel.
- Precisión de la regla 5 en `CLAUDE.md`: en las orillas solo se quitan el espacio ASCII, `\r` y `\n`, y los dígitos válidos son solo los ASCII. Claude Code la propuso, el chat de Claude del proyecto la revisó y Samuel la aprobó.
- Código:
  - `src/metricas_clasificador.py` es la única implementación de la regla 5, para todas las evaluaciones;
  - `src/evaluar_humo.py` es el script del paso, que no abre nunca `test.jsonl`;
  - 76 pruebas nuevas de pytest con datos inventados; pasan las 415 del repo;
  - commits `9d10e3e` y `73dea2d`;
  - el resultado se generó dos veces, con bytes idénticos.
- Captura `docs/evidencia/2026-11-21_fase2_primera-corrida-kaggle.png`, copiada de `../evidencia/` sin cambios (mismo SHA-256). Claude Code la revisó: solo muestra el título del kernel, "Version #1 with GPU T4 x2", "Successful" y "16 minutes"; no muestra correos, llaves, tokens, teléfonos, contraseñas ni nombres, y no trae metadatos de texto ni EXIF.
- PARADA: falta probar el GGUF en la laptop (paso 6d). Los commits del 6b y del 6c están sin push.

## 2026-11-22 - Fase 1, paso 7: hard set del clasificador

- Las decisiones son del 2026-11-21; el trabajo se hizo el 2026-11-22.
- Autoría: la instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Claude Code propuso el plan; el chat lo revisó con cinco ajustes y Samuel lo aprobó. Las decisiones del 2026-11-21 las propuso el chat y las aprobó Samuel: dos fuentes de candidatos, una fila por llave, reparto por sector, dos tipos (ambiguo e indirecto) y descartar, sin corregirla, una fila cuya etiqueta del DENUE parezca equivocada.
- Regla 3 de `CLAUDE.md` (CAMBIO DE DISEÑO propuesto por el chat de Claude del proyecto y aprobado por Samuel el 2026-11-21): el hard set del clasificador se trata como el test. No se usa para elegir modelos ni hiperparámetros, no sube a Kaggle hasta la Fase 4 y se evalúa una sola vez, junto con el test.
- Candidatos (`src/hard_set.py`, semilla 42), sacados de `train_candidatos.jsonl` con su hash verificado:
  - De las 316,983 llaves se excluyen las que tienen alguna fila con la frase "SIN NOMBRE" (9,327) o igual a un título (77).
  - Fuente (i), clases mezcladas: llaves de 2 a 20 filas cuya clase más común no pasa del 60 % de sus filas. Hay 5,566 elegibles, 511 de ellas genéricas (se conservan y quedan marcadas), y 36 quedan fuera por tener más de 20 filas.
  - Fuente (ii), sin vocabulario: llaves de 1 fila sin ninguna palabra del vocabulario SCIAN, sin contar las palabras gramaticales. Hay 111,708 elegibles, contra 111,587 con la lista original y 85,286 sin ignorar ninguna palabra.
  - Palabras gramaticales: la lista original (artículos, preposiciones, conjunciones y posesivos) la propuso el chat de Claude del proyecto y la aprobó Samuel. MEDIANTE, QUE, SOBRE, SU y SUS las detectó Claude Code y las aprobó el chat.
  - Una ronda de 200 candidatos por fuente, con un tope de 30 por sector de 2 dígitos y por fuente. Los 400 candidatos son de 185 clases.
  - Tope de 20 filas por llave, para la regla 7. Con las 150 llaves que se planeaban, la cota garantizada era de 3,000 filas (44,372 excluidas en total, 9.66 %); en el peor caso de los 400 candidatos salían 423 filas (9.10 %).
- CSV de curación: `data/2026-11-22_fase1_hard-candidatos-r1.csv`, ignorado por git, con 400 filas y SHA-256 `4f1c7bbc737f8dd7512f7e2c53d8e8355622897f5bf55b7917ab3afe759c7820`. Samuel no tiene Excel, así que Claude Code adaptó las instrucciones a Google Sheets y agregó dos cosas en un commit extra (`09ebed2`):
  - un candado de ruta: el CSV curado debe estar dentro de `data/`, ignorado por git, y su nombre debe terminar en `-curado.csv`;
  - una prueba con lo que descarga Google Sheets.
- Curación (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-22): no la hizo Samuel a mano.
  - La hizo una IA (Gemini, en Google Sheets) con los criterios que definieron Samuel y el chat de Claude del proyecto. Samuel aprobó el cambio y no revisó la curación fila por fila.
  - El chat revisó las 400 filas y ajustó 23 con los mismos criterios.
  - El hard set quedó en 105 filas en vez de 150, sin ronda 2.
  - Sesgo reconocido: las filas las eligió un modelo de lenguaje con su propio conocimiento, sobre todo las de tipo indirecto.
  - Los nombres de los 400 candidatos pasaron por Google Drive, por Gemini y por el chat de Claude del proyecto. Samuel borró la copia de Google Drive.
- Ajustes del chat, por motivo:

  | Ajuste | Motivo | Filas |
  |---|---|---:|
  | de S a N | pista frecuente en nombres de CDMX (anglicismo o sigla) | 13 |
  | de S a N | el nombre contiene la palabra de la actividad (abreviada, en juego de palabras, con errata o cognado) | 4 |
  | de S a N | etiqueta no juzgable sin conocimiento técnico | 1 |
  | de indirecto a ambiguo | la deducción requiere conocimiento experto; opaco para una persona | 4 |
  | de indirecto a ambiguo | dos pistas hacia actividades distintas | 1 |

  Antes de aplicar los ajustes, el código comprobó que cada fila ajustada estaba en S y que cada ajuste cambiaba algo.
- Archivo de ajustes: como lleva ids del DENUE, vive en `data/2026-11-22_fase1_hard-set-ajustes-chat.json`, ignorado por git. El código versionado guarda solo su SHA-256, y `congelar` se niega si el archivo no coincide con él. El chat había pedido versionarlo; Claude Code detectó que eso subiría ids del DENUE a un repo que será público (sección d de `CLAUDE.md`), y el chat corrigió su instrucción.
- Hard set:
  - La IA dejó 123 filas con S (70 ambiguas y 53 indirectas); tras los ajustes quedaron 105, 74 ambiguas y 31 indirectas.
  - Por fuente: (i) da 47 ambiguas y 11 indirectas, y (ii) da 27 ambiguas y 20 indirectas.
  - Hay 3 filas genéricas entre las ambiguas y 1 entre las indirectas. Las filas son de 16 sectores y 84 clases.
  - Aviso: el tipo indirecto tiene 31 filas, menos del mínimo de 50, así que sus resultados se reportarán con su intervalo y una advertencia.
- Regla 7: salen de train las 192 filas de las 105 llaves, y la exclusión total queda en 41,564 de 459,520 filas candidatas (9.05 %). Train candidato sin hard tiene 417,956 filas y 929 clases: ninguna clase de train se queda sin ejemplos, y todas las del hard set tienen ejemplos en train.
- Archivos en `data/splits/`, ignorados por git:
  - `hard.jsonl`, congelado: 105 filas, 18,948 bytes, SHA-256 `13755fc7022d78b3e6b161425b4ee87bd7aec2eebf0ea42e4c7169f3288b08fa`.
  - `train_candidatos_sin_hard.jsonl`, sin congelar: 417,956 filas, 75,261,622 bytes, SHA-256 `db3dd09accc6715ffc844192d1a5a0f1ce0feee7029634f4871725575bc8cefc`. De él parte el tope por clase, que nunca abre `hard.jsonl`.
  - El manifest guarda también los hashes del CSV de la IA (`d0df988294d4c42bd71145c020724983842ae3ac7b1ea10401d0b12699584752`) y de los ajustes (`61b4dd8c8787d2856de98c6ff4caf93a0b7609649142476bffe88d45eb0ad390`).
  - Los archivos se generaron desde el commit `f0f3deb`. Antes de escribirlos se generaron dos veces desde cero, con los mismos bytes, y después se releyeron sus hashes.
- `CLAUDE.md`: la regla 3 lleva los dos CAMBIOS DE DISEÑO, y la sección b describe los dos archivos. El texto de la sección b con 105 filas lo propuso Claude Code; el chat de Claude del proyecto lo aprobó en una respuesta que pegó Samuel.
- Pruebas: 70 en `tests/test_hard_set.py`, con datos inventados; pasan las 485 del repo. Una de ellas espía `open` y confirma que el flujo completo, con los ajustes incluidos, nunca abre `test.jsonl`.
- Cuota de GPU de Kaggle: quedan 29 h 47 min de 30, o sea, Kaggle descontó 13 minutos por la prueba de punta a punta del paso 6c. El dato llegó en la instrucción del paso. `results/2026-11-21_fase2_prueba-humo.json` registró 0.27 h (16 minutos, la mayor de las mediciones) y no se modifica.
- Números en `results/2026-11-22_fase1_hard-set.json`.
- PARADA: los commits del paso están sin push; el push lo hace Samuel.

## 2026-11-22 - Fase 1, paso 8: tope por clase, train final y paquete para Kaggle

- Autoría: la instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Claude Code propuso el plan. El chat lo revisó con tres ajustes y Samuel lo aprobó:
  - dos columnas más en la tabla del punto de control: la distancia de variación total contra train sin tope y el % de validación en clases recortadas;
  - aplicar en el último commit el texto de la sección b de `CLAUDE.md` que propusiera Claude Code;
  - esperar la decisión del tope en la PARADA 1.
- Para qué sirve el tope: que las clases grandes no ahoguen a la cola larga del train. Validación y test conservan la distribución natural.
- Procedimiento (`src/tope_clase.py`, semilla 42), propuesto por Claude Code en el plan y aprobado con él:
  - Parte de `train_candidatos_sin_hard.jsonl`, con su SHA-256 verificado antes de parsear. Agrupa las filas por clase, las ordena por `int(id)` dentro de cada clase y recorre las clases en orden de código.
  - Un solo `numpy.random.default_rng(42)`. En cada clase, aunque no se recorte, calcula `perm = rng.permutation(n_clase)` y conserva las filas de `sort(perm[:min(n_clase, tope)])`.
  - Como todas las clases consumen el generador igual con cualquier tope, el sorteo de una clase no depende del tope y los trains quedan anidados: cada tope menor es subconjunto del mayor (el código lo comprueba).
  - El resultado no depende del orden de lectura; una prueba baraja las filas de entrada y obtiene los mismos bytes.
- Punto de control (PARADA 1), en `results/2026-11-22_fase1_train-final.json`. Horas de GPU por época, solo de entrenamiento, con las tasas de lote 32 de la prueba de punta a punta. TVD es la distancia de variación total: la fracción de filas que habría que cambiar de clase para que dos distribuciones fueran iguales.

  | Tope | Filas | Clases recortadas | h/época 0.6B | h/época 4B | TVD contra validación | TVD contra train sin tope | % de validación en clases recortadas |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 100 | 47,079 | 328 | 0.26 | 1.54 | 0.6504 | 0.5787 | 97.10 |
  | 200 | 74,325 | 238 | 0.41 | 2.42 | 0.6075 | 0.5269 | 94.80 |
  | 500 | 128,984 | 143 | 0.70 | 4.21 | 0.5318 | 0.4470 | 88.20 |
  | 1,000 | 187,350 | 95 | 1.02 | 6.11 | 0.4643 | 0.3760 | 80.70 |
  | sin tope | 417,956 | 0 | 2.28 | 13.63 | 0.2092 | 0 | 0 |

  - Con todos los topes quedan las 929 clases, y 209 tienen menos de 10 filas; ningún tope las toca.
  - Validación tiene 1,000 filas, así que la TVD contra validación tiene un piso de ruido de 0.21 (la fila sin tope).
  - Una sesión de GPU de Kaggle dura como máximo 12 h; Claude Code lo confirmó en la documentación oficial (https://www.kaggle.com/docs/notebooks). Sin tope, una época del 4B no cabe.
- Decisión: tope de 1,000 filas por clase. El chat de Claude del proyecto lo recomendó y Samuel lo aprobó.
  - Motivo: es el tope más alto con el que una época del 4B (6.11 h; 6.15 h con el costo fijo) cabe en una sesión de 12 h, y el que menos se aleja de la distribución natural (TVD 0.376).
  - Costo aceptado: el 80.7 % de las filas de validación está en clases recortadas. Los baselines entrenan con el mismo train.
- `data/splits/train_final.jsonl`, ignorado por git y congelado: no se regenera ni cambia, pero a diferencia de test y hard sí se lee para entrenar.
  - 187,350 filas, 158,687 llaves, 929 clases y 95 clases recortadas; 34,331,964 bytes; SHA-256 `fe6780a7e60400604238b601a4052dc55e0ab4ab9747f49a89166e217aaa9a9e`.
  - Por marca de subgrupo: 11,990 filas con "SIN NOMBRE", 15,317 genéricas y 89 iguales a un título.
  - Se generó desde el commit `7857e84`. Antes de escribirlo se generó dos veces desde cero, con los mismos bytes, y después se releyó su hash. El manifest lo registra en `splits.clasificador_train_final`.
- Paquete para Kaggle (`src/paquete_clasificador.py`), en `C:\sca\fine-tuning\proyecto-denue\kaggle-datasets\denue-scian-clasificador\`, fuera del repo. Lleva `train_final.jsonl` y `val.jsonl` completos, solo con `id`, `nom_estab` y `codigo_act`, más `dataset-metadata.json` con el crédito al INEGI.
  - `train_final.jsonl`: 187,350 filas, 15,583,913 bytes, SHA-256 `6f3d0b1ce5472ad96581d27605b1a19e868431e537686da72825da0ef136b917`.
  - `val.jsonl`: 1,000 filas, 81,500 bytes, SHA-256 `5d778ba77b2ceee3f2ad5b9d094970c6ac6aaf6a529e9e20df9131d38d8723f3`.
  - `dataset-metadata.json`: 469 bytes en ASCII, SHA-256 `1378b45709007716a83b399cdee5736b09a9f1dd2ffa1149356be9e5e800abe2`; id `samcaballero/denue-scian-clasificador`, privado.
  - El candado rechaza archivos fuera de la lista y cualquier archivo con el hash de test, de hard o de otra entrada del manifest; de test y hard solo lee el hash del manifest. `--verificar` pasa.
  - Train y validación no comparten ids ni llaves, y ninguna clase de validación falta en train.
  - Los hashes están en `kaggle_datasets.denue_scian_clasificador`. Claude Code no subió nada; la subida la hace Samuel.
- `src/muestra_humo.py`: Claude Code le agregó parámetros con valores por omisión, para reutilizar su candado y su metadata. La prueba de humo no cambia.
- `CLAUDE.md`, sección b: describe `train_final.jsonl` y los datasets de Kaggle. Claude Code redactó el texto y lo aplicó por instrucción del chat, aprobada por Samuel; se muestra en el reporte final para revisarlo.
- Pruebas: 39 nuevas con datos inventados; pasan las 524 del repo. Dos de ellas espían `open` y confirman que ni el tope ni el paquete abren `test.jsonl` ni `hard.jsonl`.
- PARADA: los commits del paso están sin push; el push y la subida a Kaggle los hace Samuel.
- Subida, hecha por Samuel después de la PARADA y registrada en el paso 9:
  - creó el dataset privado `samcaballero/denue-scian-clasificador` con `kaggle datasets create`, sin `-u`;
  - el dataset quedó en estado ready, y los tamaños en Kaggle (15,583,913 bytes de `train_final.jsonl` y 81,500 de `val.jsonl`) son iguales a los del manifest;
  - en una ventana privada, la página del dataset da 404, así que no es público.
- Captura de la tabla del tope: `docs/evidencia/2026-11-22_fase1_tabla-tope-clase.png`, copiada en el paso 9 desde `../evidencia/` sin cambios (mismo SHA-256, `61709975…`).
  - Claude Code la revisó: solo muestra la tabla de topes (números y códigos SCIAN) y la nota de las 209 clases; no muestra correos, llaves, tokens, teléfonos, contraseñas ni nombres de negocios.
  - No trae metadatos de texto ni EXIF: el PNG solo tiene los bloques de imagen, gamma, resolución y sRGB.

## 2026-11-22 - Fase 2, paso 9: kernel de la primera corrida real del clasificador

- Autoría: la instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Claude Code propuso el plan. El chat lo revisó con tres ajustes y Samuel lo aprobó, junto con la estimación de GPU:
  - comprobar que el train se recorre barajado y agregar una prueba;
  - revisar antes del commit las condiciones que harían abortar el kernel;
  - fijar el largo máximo de 256 como estándar del clasificador.
- Qué es: el kernel de la primera corrida que cuenta, Qwen3-0.6B con la configuración inicial de la guía sobre `train_final.jsonl` (187,350 filas) y `val.jsonl` (1,000). Claude Code no lo lanzó.
  - Su resultado en validación se reporta en el paso siguiente, cuando se evalúe en la laptop: descarga, revisión de nombres, regla 5 y subgrupos de la regla 8 en `results/`.
- Largo máximo de secuencia: **256 fichas, en vez de las 128 de la guía**. Es el estándar de todas las corridas del clasificador: entrenamiento, barrido, 4B e inferencia.
  - Claude Code lo propuso al ver que el train final tiene nombres de hasta 151 bytes (156 filas de más de 100 bytes); en la prueba, el máximo era de 97. Con la regla validada, una fila de más del máximo aborta el kernel, porque truncarla cortaría el código de la respuesta. Samuel aprobó subirlo a 256.
  - El chat de Claude del proyecto lo fijó como estándar y Samuel lo aprobó. Es el valor por omisión de las corridas en `src/construir_kernel.py` (`LARGO_MAXIMO_CLASIFICADOR`), y una prueba lo comprueba.
  - Por la cota de bytes, ninguna fila puede pasar de 223 fichas: no se trunca ni se excluye ninguna. Las filas de más de 128 fichas solo se cuentan.
- Configuración: va en el diccionario `CORRIDA`, incrustado en el kernel y validado al arrancar.
  - Modelo `Qwen/Qwen3-0.6B` en la revisión `c1899de2…`, en NF4 y con `use_exact_model_name=True`.
  - LoRA: rank 16, alpha 16, dropout 0, en las 7 proyecciones lineales.
  - Tasa de aprendizaje 2e-4, 1 época, lote 32 sin acumulación (5,855 pasos), AdamW de 8 bits, fp16 y semilla 42.
  - Calentamiento de la tasa (warmup): 3 % de los pasos (176), lineal desde 0. Después, decaimiento lineal hasta 0, como en la prueba. Claude Code los propuso y se aprobaron con el plan.
  - Sin early stopping ni elección de punto de control: el adaptador es el del final de la época.
- Del lote 16 de la prueba al 32:
  - los pasos por época bajan de 11,710 a 5,855;
  - como el paso de AdamW depende poco del tamaño del lote, con la misma tasa los pesos se mueven más o menos la mitad por época; se mantiene la tasa de la guía, y el barrido puede revisarlo;
  - velocidad medida: de 42.9 a 50.9 ejemplos/s;
  - VRAM pico reservada: de 892 a 1,458 MiB;
  - relleno: de 11.9 % a 16.8 %.
- Curvas:
  - pérdida de train en cada paso;
  - pérdida de validación en las 1,000 filas en el paso 0 (antes de entrenar), cada 500 pasos y en el último paso, 13 puntos en total;
  - en cada evaluación se reescribe `mediciones.json`, así que si la corrida se corta, la curva hasta ese punto queda.
- Barajado (ajuste del chat): Claude Code leyó el código de transformers 5.17.0, accelerate 1.15.0, unsloth 2026.11.9 y unsloth_zoo 2026.11.8, de los wheels bajados sin instalarlos.
  - Con `trainer.train()`, el Trainer usa un `RandomSampler` (`train_sampling_strategy="random"`), y accelerate lo cambia por un `SeedableRandomSampler` con `data_seed=42`.
  - El parche de Unsloth al ciclo de entrenamiento no toca el sampler. El ciclo propio de Unsloth que recorre en orden (`unsloth_train`, con `SequentialSampler`) solo corre si se llama a mano, y el kernel no lo llama.
  - El colador propio recibe los lotes ya en el orden del sampler.
  - El kernel lo comprueba en la corrida:
    - el colador registra la posición de cada fila;
    - en el paso 20, el kernel aborta si las primeras 640 filas llegaron en orden de id;
    - al final, registra si el train se recorrió completo, una vez cada fila y barajado, con el SHA-256 del orden.
  - Las pruebas cubren los argumentos del Trainer, la revisión del orden y el aborto con filas en orden.
- Revisión antes del commit (ajuste del chat), solo con conteos:
  - hay 0 nombres con `<|`, tanto en train como en validación;
  - ninguna fila puede pasar de 256 fichas: la cota máxima es de 223 en train y de 164 en validación;
  - el log y las mediciones de la prueba no registraron qué había en `/kaggle/input`, así que no hay evidencia directa de lo que agrega Kaggle. La API de Kaggle lista solo `train_final.jsonl` y `val.jsonl` en el dataset. Si la regla de solo dos archivos abortara por un archivo de Kaggle, sería en la etapa 2, antes de instalar, y el listado quedaría en `mediciones.json`.
- Código:
  - `src/clasificador_kaggle.py`, nuevo:
    - reutiliza de `src/humo_kaggle.py` la verificación de la T4, la lectura de datos con su SHA-256, la instalación de versiones fijadas, los imports, la descarga verificada, el colador, la inferencia y el registro sin nombres;
    - reescribe, partiendo de ese código, la carga, LoRA y los tokens con el largo de la corrida, y agrega el entrenamiento con validación.
  - `src/construir_kernel.py`, con una especificación por kernel (`humo-6b` y `06b-inicial`):
    - el kernel de la prueba sale idéntico byte a byte, con el mismo SHA-256 que el que se lanzó desde `1b4e41f`;
    - el candado de cada kernel rechaza `test.jsonl`, `hard.jsonl`, sus SHA-256 y cualquier otro hash del manifest.
  - `src/humo_kaggle.py`, `src/prompt_clasificador.py`, `src/normalizacion.py` y `src/__init__.py` no cambian.
- Kernel en `notebooks/kaggle-06b-inicial/`:
  - `clasificador_06b_inicial.py`: 134,875 bytes, SHA-256 `570a3aa8…`;
  - `kernel-metadata.json`: id `samcaballero/denue-scian-06b-inicial`, privado, T4, internet, imagen fijada y el dataset del clasificador como única fuente.
- Pruebas:
  - 57 nuevas con datos inventados; pasan las 581 del repo;
  - Claude Code corrió fuera del repo un simulacro de todas las etapas con dependencias falsas, en seis modos: normal, sin barajar, fila larga, archivo de más, SHA-256 distinto y falla de la validación en el paso 0;
  - cada modo terminó como se esperaba, y no apareció ningún nombre en el log ni en las mediciones.
- Estimación de GPU, aprobada por Samuel:

  | Etapa | Estimado |
  |---|---:|
  | Arranque del contenedor y guardado de la salida | ~3 min |
  | GPU y datos | <0.3 min |
  | Instalación, imports, descarga y carga | ~1.7 min |
  | Tokens y plantilla (188,351 textos) | ~2 min |
  | Entrenamiento (5,855 pasos) | 61-63 min |
  | Validación (13 × 1,000 filas) | ~2 min |
  | Inferencia (1,000 filas) | ~4 min |
  | **Total** | **≈1.25 h (rango de 1.2 a 1.4 h)** |

  El tope es `-t 9000` (2.5 h).
- Commits: `54374f9` (código, pruebas y kernel) y el de esta bitácora, el registro y la captura.
- PARADA: el kernel está listo y sin lanzar. Falta que Samuel corra `--verificar` y `kaggle kernels push` desde PowerShell.

## 2026-11-22 - Fase 1, paso 10: evaluador, baselines y experimento de fuga (plan y rejilla de C)

- Autoría:
  - El chat de Claude del proyecto redactó la instrucción y Samuel la aprobó.
  - Claude Code propuso el plan. El chat lo revisó con cuatro ajustes y Samuel lo aprobó:
    - el desempate del diccionario;
    - el vectorizador se ajusta solo con train, con una prueba;
    - los resultados se escriben al terminar cada ajuste;
    - el commit del evaluador va primero.
- Punto de control de costo (PARADA 1). TF-IDF + regresión logística con saga, en submuestras de `train_final`:
  - con 10,000 filas, C = 1 tardó 69 s (15 iteraciones) y C = 10 tardó 163 s (36);
  - C = 100 tardó 457 s y no convergió en el tope de 100 iteraciones;
  - con 30,000 filas, C = 1 tardó 310 s (16 iteraciones).
  - La extrapolación al train completo (187,350 filas y 929 clases) da unos 140 s por iteración: 37 min con C = 1, 84 min con C = 10 y 3.9 h o más con C = 100. El plan de 4 ajustes completos costaba de 6.5 a 10 h, por encima del límite de 2 h.
  - Claude Code detuvo el punto de control durante C = 10 con 30,000 filas, porque a la laptop le faltaba memoria (Chrome ocupaba unos 9 GB). No se relanzó.
- Decisión sobre la rejilla de C:
  - Claude Code propuso elegir C en una submuestra de 30,000 filas.
  - El chat de Claude del proyecto la descartó y propuso barrer el train completo del split por nombre con **C ∈ {1, 10}**. Samuel lo aprobó.
  - Por qué se descarta C = 100: no convergió en 100 iteraciones ni con 10,000 filas, y en el train completo costaría 3.9 h o más.
  - Elección: gana el C con mejor accuracy en validación y, si empatan, el menor. Ese C se ajusta en el split aleatorio.
  - Tiempo estimado: de 2.6 a 3.4 h.
  - Queda en `src/baselines.py` (`VALORES_C` y `DECISION_VALORES_C`) y en la configuración de los dos JSON de TF-IDF.
- Los ajustes largos corren en la terminal de Samuel, no como tarea de Claude Code.
- Commits hasta aquí:
  - `9a4118e`: el evaluador;
  - `afa8cfd`: el split aleatorio y los baselines;
  - `5fb9ebd`: el split aleatorio en el manifest y en `results/`;
  - el de esta rejilla de C.
- Los resultados están en la entrada del 2026-11-23.

## 2026-11-23 - Fase 1, paso 10: resultados de los baselines y de la fuga

- Autoría:
  - Samuel corrió los ajustes en su terminal, con `src/baselines.py --escribir` sobre el commit `d5e7a84`. Empezaron a las 04:10 (la hora de los primeros JSON) y terminaron a las 06:25.
  - El chat de Claude del proyecto redactó la lectura de la fuga y Samuel la aprobó.
  - Claude Code revisó los cinco JSON: son del mismo commit con el árbol limpio y solo tienen códigos, sin ids, nombres ni llaves. Después hizo los commits.
- Resultados en validación, 1,000 filas por split. El IC 95 % sale de un bootstrap de percentiles con 1,000 remuestreos y semilla 42. Todas las respuestas tienen formato válido y código existente:

  | Modelo | Split | Accuracy (IC 95 %) | Accuracy de sector | Macro-F1 de sector |
  |---|---|---|---:|---:|
  | Diccionario | por nombre | 0.083 (0.066-0.099) | 0.443 | 0.034 |
  | Diccionario | aleatorio | 0.236 (0.210-0.261) | 0.530 | 0.235 |
  | TF-IDF + LR, C = 10 | por nombre | 0.682 (0.652-0.711) | 0.845 | 0.780 |
  | TF-IDF + LR, C = 10 | aleatorio | 0.681 (0.652-0.710) | 0.823 | 0.770 |

- **Diccionario en el split por nombre:**
  - ninguna llave de validación está en train, así que siempre responde la clase global 461110;
  - su accuracy, 0.083, es la proporción de esa clase en validación: equivale al baseline de clase mayoritaria.
- **Diccionario en el split aleatorio:**
  - 212 filas (21.2 %) tienen su llave en train;
  - responde 461110 en el 80.1 % de las filas y usa 104 clases distintas.
- **C = 1 y C = 10 quedaron prácticamente iguales:** 0.681 (IC 0.651-0.710) contra 0.682 (IC 0.652-0.711).
  - Es un acierto de diferencia, dentro del intervalo.
  - Ganó C = 10 por la regla (más aciertos y, en empate, el menor).
  - En sector, C = 10 da 0.845 y C = 1 da 0.837.
  - C = 10 tardó 2.2 veces más.
  - Como C se eligió en esta misma validación, la accuracy del C elegido es un poco optimista; con esta diferencia, el sesgo es mínimo.
- **Fuga** (accuracy del split aleatorio menos la del split por nombre; IC por bootstrap independiente de las dos validaciones):

  | Modelo | Fuga en clase (IC 95 %) | Fuga en sector (IC 95 %) |
  |---|---|---|
  | Diccionario | +0.153 (+0.120 a +0.186) | +0.087 (+0.043 a +0.130) |
  | TF-IDF + LR | -0.001 (-0.040 a +0.040) | -0.022 (-0.056 a +0.012) |

  - **Lectura** (del chat; Samuel la aprobó): la fuga infla al modelo que memoriza nombres (el diccionario) y no al que generaliza por n-gramas (TF-IDF).
  - **Precisiones de Claude Code:**
    - El IC de TF-IDF va de -4 a +4 puntos. Con 1,000 filas, el experimento descarta una fuga grande, pero no una menor de 4 puntos.
    - En el split aleatorio solo el 21.2 % de la validación tiene su llave en train (32.5 % antes del tope). Eso acota cuánto puede inflar la fuga.
  - **Subgrupos del diccionario** (muestras distintas en cada validación, con intervalos anchos): la ganancia se concentra en los nombres que se repiten mucho.
    - Filas "SIN NOMBRE": 0.017 en el split por nombre (59 filas) y 0.507 en el aleatorio (69).
    - Filas genéricas: 0.091 (77 filas) y 0.525 (80).
- **Tiempos reales**, en CPU y sin GPU (`horas_gpu: 0`):

  | Ajuste | Iteraciones | Tiempo real | Estimado en la PARADA |
  |---|---:|---:|---:|
  | Diccionario, split por nombre | - | 3.9 s | - |
  | Diccionario, split aleatorio | - | 2.6 s | - |
  | Vectorizador, split por nombre | - | 33 s | - |
  | TF-IDF + LR, split por nombre, C = 1 | 18 | 1,474 s (24.6 min) | ~37 min |
  | TF-IDF + LR, split por nombre, C = 10 | 43 | 3,301 s (55.0 min) | ~84 min |
  | Vectorizador, split aleatorio | - | 10 s | - |
  | TF-IDF + LR, split aleatorio, C = 10 | 43 | 3,290 s (54.8 min) | ~84 min |
  | **Total** | | **~2.25 h (04:10 a 06:25)** | **3 a 3.5 h** |

  - Los tres ajustes convergieron (`tol` 1e-3, tope de 100 iteraciones).
  - Cada iteración costó unos 80 s, contra los 140 s extrapolados de la submuestra de 30,000 filas. Hicieron falta algunas iteraciones más de las observadas en la submuestra (18 contra 16 con C = 1 y 30,000 filas; 43 contra 36 con C = 10 y 10,000 filas).
  - Memoria pico del ajuste (`tracemalloc`): 4,171 MB. Pico del proceso: 5,300 MB. Se había estimado unos 5 GB.
- Archivos:
  - `results/2026-11-23_fase1_baseline-diccionario_split-nombre.json` y `…_split-aleatorio.json`;
  - `results/2026-11-23_fase1_baseline-tfidf-lr_split-nombre.json` (con la tabla del barrido) y `…_split-aleatorio.json`;
  - `results/2026-11-23_fase1_fuga-split-aleatorio.json`.
  - Los registros de cada ajuste quedan en `data/baselines/d5e7a84b2cc4/` (ignorado por git).
- `CLAUDE.md`, sección b: se agregaron los dos archivos del split aleatorio, como "solo experimento", y la carpeta de registros.
- Commits:
  - `9a4118e`: el evaluador;
  - `afa8cfd`: el split aleatorio y los baselines;
  - `5fb9ebd`: el split aleatorio en el manifest;
  - `d5e7a84`: la rejilla de C;
  - `98fa843`: los resultados;
  - el de esta bitácora, el registro y `CLAUDE.md`.

## 2026-11-23 - Fase 2, paso 11 (Parte A): evaluación de la corrida 06b-inicial

- Autoría:
  - El chat de Claude del proyecto redactó la instrucción y Samuel la aprobó.
  - Claude Code propuso el plan. Samuel lo aprobó con ocho ajustes que redactó el chat:
    - alpha = rank;
    - el signo y la alineación de la diferencia pareada;
    - los metadatos de las capturas;
    - el permiso de `kaggle kernels pull`;
    - las versiones del barrido;
    - los cuelgues;
    - los comandos de la PARADA 2;
    - la autoría de la diferencia pareada.
  - **Alpha = rank en el barrido** (escala alpha/rank = 1, como en la inicial): lo propuso el chat de Claude del proyecto y Samuel lo confirmó el 2026-11-23 a las 2:36 pm.
  - **Diferencia pareada:** la pidió el chat en la instrucción ("si el evaluador ya lo permite"), pero el evaluador solo tenía la diferencia independiente. Claude Code propuso implementarla, el chat lo recomendó y Samuel lo aprobó.
  - **Precisión de la revisión de nombres:** Claude Code la propuso, el chat la acotó y Samuel la aprobó (se describe abajo).
- Corrida:
  - Samuel lanzó `samcaballero/denue-scian-06b-inicial` (versión 1) el 2026-11-22; el `lastRunTime` de la CLI es 22:19:31.
  - Terminó Successful, con 1 h 12 min 43 s en la página de Kaggle; el script midió 4,351 s.
  - La cuota pasó de 29 h 47 min a 28 h 35 min, así que Kaggle descontó 1 h 12 min. El resultado registra `horas_gpu` de 1.212.
- Capturas, de `../evidencia/` a `docs/evidencia/`:
  - Ninguna de las seis revisadas trae metadatos de texto ni EXIF: solo los bloques de imagen, gamma, resolución y sRGB.
  - Copiadas sin cambios, con el mismo SHA-256:
    - `2026-11-21_fase1_splits-congelados.png` (conteos de los splits y de la regla 7);
    - `2026-11-21_fase2_cuota-despues-prueba.png` (cuota de 29 h 47 min);
    - `2026-11-22_fase2_cuota-despues-06b.png` (la corrida en estado Successful);
    - `2026-11-23_fase1_tabla-baselines-fuga.png` (tabla de los baselines y de la fuga).
  - No copiadas:
    - `2026-11-22_fase2_corrida-06b-inicial.png`: muestra el nombre de la cuenta de Kaggle y su avatar;
    - `2026-11-23_fase1_baselines-fuga.png`: dudosa, porque muestra la ruta de la carpeta de usuario de Windows;
    - `2026-11-17_fase0_gpu-kaggle-1.png`: Samuel decidió el 2026-11-18 no copiarla porque es casi un duplicado.
- Descarga a `kaggle-output/06b-inicial/`, ignorada por git:
  - 8 archivos y 43.0 MB, casi todo del adaptador (40.4 MB).
  - Coinciden los 6 de los que el kernel calculó un SHA-256; el log y `mediciones.json` no lo traen.
  - Ninguno tiene un SHA-256 del manifest.
- Revisión de nombres, antes de mirar el log o las mediciones:
  - Método del paso 6c, con búsqueda indexada, contra los 188,350 nombres de `train_final.jsonl` y `val.jsonl` del paquete.
  - No apareció ninguna llave de varias palabras en ningún archivo. `mediciones.json` y `predicciones.jsonl` no tienen nombres crudos con letras.
  - Aparecieron tres nombres crudos de 3 y 4 letras, siempre dentro de un token más largo:
    - dos en el log, dentro de identificadores en mayúsculas de 22 y 23 caracteres;
    - uno en los dos pip freeze, en el mismo paquete.

    Samuel los aceptó como falsos positivos.
  - **Precisión del método** (la propuso Claude Code, la acotó el chat de Claude del proyecto y la aprobó Samuel):
    - un nombre crudo de 4 caracteres o menos que solo aparece dentro de un token más largo (letras, dígitos, `_` o `-`) no es posible fuga y se cuenta aparte;
    - uno de 5 caracteres o más dentro de cualquier token sigue siendo PARADA;
    - lo que, normalizado, es solo dígitos se acepta, como en `tapar_llaves` y `verificar_privacidad`.
  - Las coincidencias de una sola palabra solo se cuentan: de 0 a 16 llaves por archivo.
  - El kernel tapó dos valores de configuración, `save_strategy` y `train_sampling_strategy`: sus valores ("no" y "random"), normalizados, son llaves de negocios. La evaluación los compara con lo que arma el código.
- Revisión por etapa: las 9 etapas terminaron bien y sin fallas.
  - **GPU:** dos T4 en la sesión y una visible para torch. Sin bf16 nativo, así que fp16.
  - **Datos:**
    - en la entrada solo están los dos archivos, con su SHA-256;
    - 187,350 filas de train (929 clases) y 1,000 de validación (228 clases);
    - el kernel lanzado es idéntico al que genera hoy `src/construir_kernel.py`, y sus fuentes incrustadas son las de `src/`.
  - **Versiones:**
    - las 8 fijadas quedaron instaladas;
    - de la imagen: torch 2.11.0+cu130, torchvision 0.26.0+cu130 y triton 3.6.0, con Python 3.13.16;
    - pip agregó xformers 0.0.36, cut-cross-entropy, hf-transfer, msgspec, mslk, structlog y tyro;
    - pip cambió transformers (de 5.20.2 a 5.17.0), peft (de 0.21.2 a 0.21.3) y torchao (de 0.10.0 a 0.18.0).
  - **Modelo:** `Qwen/Qwen3-0.6B` en la revisión `c1899de2…`, con sus 2 archivos LFS verificados y nada más en la caché. NF4, con 196 capas de 4 bits. LoRA: 10,092,544 parámetros entrenables.
  - **Tokens:** (a) a (d) bien. La fila más larga tiene 118 fichas en train y 98 en validación. Cada fila etiqueta 7 fichas: 6 dígitos y `<|im_end|>`.
  - **Entrenamiento:**
    - 5,855 pasos y 176 de warmup;
    - el train se recorrió completo y barajado; el SHA-256 del orden, `b6b53a85…`, queda para compararlo en el barrido;
    - 49.8 ejemplos/s y 0.659 s/paso de mediana;
    - VRAM pico reservada: 1,152 MiB.
  - **Inferencia:** 1,000 filas en 228 s, greedy. Todas terminan en el token de fin, con 7 fichas como máximo.
- Tiempos por etapa:

  | Etapa | Segundos |
  |---|---:|
  | GPU y datos | 3 |
  | Instalación | 48 |
  | Imports | 46 |
  | Descarga y carga | 21 |
  | Tokens y plantilla | 115 |
  | LoRA | 5 |
  | Entrenamiento (con 13 evaluaciones) | 3,883 |
  | Adaptador y cierre | 0.4 |
  | Inferencia | 228 |
  | **Total del script** | **4,351** |

- Curvas, en `docs/figuras/2026-11-23_fase2_curvas-perdida-06b-inicial.png`:
  - la pérdida de train baja de 3.118 (media de los primeros 10 pasos) a 0.287 (media de los últimos 10);
  - la de validación baja de 3.324 en el paso 0 a 0.226 en el paso 5,855. Nunca sube, pero baja cada vez menos: de 0.229 en el paso 5,000 a 0.226 al final;
  - la de validación queda por debajo de la de train en toda la época. Una explicación posible, sin comprobar: el train tiene el tope de 1,000 filas por clase, que da más peso a las clases raras, y la validación conserva la distribución natural.
- **Bug del kernel del paso 9, introducido por Claude Code:**
  - la pérdida de validación del paso 0 no quedó en `mediciones.json`, porque `trainer.train()` crea un estado nuevo y descarta el `log_history` de la evaluación previa;
  - el valor se recuperó de la línea del log que imprime el callback; los otros 12 valores del log son iguales a los de `mediciones.json`;
  - el barrido debe guardar ese punto en cada configuración.
- Resultados en validación: 1,000 filas, con el IC 95 % de un bootstrap de 1,000 remuestreos y semilla 42.

  | Modelo | Accuracy (IC 95 %) | Accuracy de sector (IC 95 %) | Macro-F1 de sector | Clases distintas predichas |
  |---|---|---|---:|---:|
  | 06b-inicial | 0.678 (0.646-0.707) | 0.847 (0.824-0.869) | 0.731 | 194 |
  | TF-IDF + LR, C = 10 | 0.682 (0.652-0.711) | 0.845 (0.824-0.867) | 0.780 | 213 |
  | Diccionario | 0.083 (0.066-0.099) | 0.443 (0.412-0.473) | 0.034 | 1 |

  - Regla 5: 0 respuestas con formato inválido, 0 con código inexistente y 1,000 con código existente.
  - Accuracy por nivel: sector 0.847, subsector 0.786, rama 0.758, subrama 0.732 y clase 0.678.
  - Subgrupos:
    - filas con "SIN NOMBRE": 0.746 (59 filas); sin la frase: 0.674 (941);
    - genéricas: 0.701 (77 filas); no genéricas: 0.676 (923).
  - Sin señales de colapso: responde 194 clases distintas. La más predicha, 461110, sale en 70 respuestas (7.0 %); en validación son 83 filas (8.3 %).
  - **Diferencia pareada 06b-inicial − TF-IDF** (positiva si 06b-inicial acierta más):
    - clase: −0.004 (IC −0.025 a +0.019); solo TF-IDF acierta en 67 filas y solo 06b-inicial en 63;
    - sector: +0.002 (IC −0.017 a +0.022);
    - en los dos niveles el cero queda dentro del IC. El macro-F1 de sector de 06b-inicial es más bajo (0.731 contra 0.780).
  - **Alineación de los aciertos de TF-IDF** (`data/baselines/d5e7a84b2cc4/tfidf_nombre_C10.json`, ignorado por git):
    - el registro solo trae aciertos de clase y de sector, sin códigos predichos ni ids, así que no se pudieron recalcular contra las etiquetas;
    - se comprobó que es del mismo commit y tiene las mismas métricas que su JSON de `results/`;
    - coinciden los conteos de clase, de sector y de los cuatro subgrupos, y se reproducen exactamente sus IC por bootstrap;
    - como el bootstrap con semilla fija depende de la posición de cada acierto, eso confirma el orden con alta probabilidad, pero no fila por fila.

    Los aciertos de 06b-inicial sí se alinearon fila por fila por `int(id)`.
- Archivos:
  - `results/2026-11-23_fase2_06b-inicial.json`: 54,602 bytes, generado dos veces desde `adaad20` con bytes idénticos;
  - la figura de las curvas.
- Código:
  - `diferencia_pareada_bootstrap` en `src/evaluador_clasificador.py`;
  - `src/evaluar_corrida_clasificador.py`;
  - 31 pruebas nuevas; pasan las 667 del repo.
- Commits:
  - `adaad20`: código y pruebas;
  - el de esta bitácora, el registro, el resultado, la figura y las capturas.
- PARADA 1: falta el visto bueno de Samuel para la Parte B (barrido).

## 2026-11-23 - Fase 2, paso 11 (Parte B): kernel del barrido del 0.6B

- Autoría:
  - Samuel dio el visto bueno a la Parte A y agregó cuatro puntos que redactó el chat de Claude del proyecto:
    - copiar las dos capturas pendientes;
    - fijar xformers;
    - recuperar del log la pérdida de validación del paso 0;
    - confirmar la máscara y el promedio de las dos pérdidas.
  - Claude Code diseñó y escribió el kernel con los ocho ajustes del plan y esos cuatro puntos.
- Capturas, por decisión de Samuel:
  - se copiaron `2026-11-22_fase2_corrida-06b-inicial.png` (la cuenta samcaballero ya es pública en el repo) y `2026-11-23_fase1_baselines-fuga.png` (la ruta solo muestra el usuario Invitado);
  - Claude Code las volvió a revisar: la primera muestra el nombre y el avatar de la cuenta, el docstring del kernel y la duración; la segunda, rutas, memoria libre y conteos. No muestran nada más y no traen metadatos;
  - se copiaron con el mismo SHA-256 (`7abdd6d9…` y `2ea04c02…`).
- Configuraciones del barrido, con alpha = rank (escala alpha/rank = 1, como en la inicial):

  | Configuración | Tasa | Rank | Alpha | Diferencias con 06b-inicial |
  |---|---:|---:|---:|---|
  | `06b-lr1e-4-r8` | 1e-4 | 8 | 8 | tasa, rank y alpha |
  | `06b-lr1e-4-r32` | 1e-4 | 32 | 32 | tasa, rank y alpha |
  | `06b-lr2e-4-r8` | 2e-4 | 8 | 8 | rank y alpha |
  | `06b-lr2e-4-r32` | 2e-4 | 32 | 32 | rank y alpha |

  - Lo demás es idéntico a 06b-inicial, y una prueba lo comprueba clave por clave:
    - datos: el dataset y sus SHA-256;
    - modelo: la revisión fijada y NF4;
    - LoRA: los 7 módulos objetivo y dropout 0;
    - entrenamiento: fp16, lote 32, AdamW de 8 bits, warmup de 3 %, decaimiento lineal, 1 época, semilla 42 y `data_seed` 42 (el mismo barajado);
    - largo máximo de 256, con el mismo prompt y la misma inferencia greedy.
  - El orden del train de cada configuración se compara con el de 06b-inicial (`b6b53a85…`).
- Versiones:
  - **Fijadas con pip** (20):
    - las 8 de 06b-inicial;
    - lo que pip agregó o cambió en 06b-inicial: xformers 0.0.36, torchao 0.18.0, cut-cross-entropy 25.1.1, hf-transfer 0.1.9, msgspec 0.22.0, mslk 0.0.0, structlog 26.1.0 y tyro 1.0.17;
    - las de la imagen: huggingface_hub 1.33.0, tokenizers 0.23.3, safetensors 0.8.0 y numpy 2.1.3.

    Las 12 extra están en PyPI con un wheel para Linux x86_64 y Python 3.13, y ninguna está retirada. Si una fijada con pip no queda instalada, el kernel aborta antes de entrenar.
  - **No se pueden fijar con pip:** torch 2.11.0+cu130, torchvision 0.26.0+cu130, triton 3.6.0 y el controlador de la GPU (580.178.04). Son compilaciones CUDA de la imagen.
    - Se fija la imagen de Docker de 06b-inicial: `gcr.io/kaggle-private-byod/python@sha256:fc720c15…`, leída con `kaggle kernels pull … -m`.
    - Si alguna difiere, queda como advertencia en `barrido.json` y la corrida sigue (ajuste del chat).
  - El código que bajó `pull` es idéntico al kernel de 06b-inicial que generó el repo, salvo los finales de línea CRLF.
- Ejecución:
  - cada configuración corre en su propio proceso, que llama a `src/clasificador_kaggle.py` sin cambios; al salir el proceso, se registra la memoria usada de la GPU 0;
  - salidas: la de cada configuración en `/kaggle/working/<nombre>/` y su log completo en `/kaggle/working/logs/<nombre>.log`;
  - **cuelgues** (ajuste del chat): cada proceso tiene un tiempo máximo de 2 veces su estimación, sin pasar del presupuesto. Si se pasa, se mata, se registra como falla y sigue la siguiente;
  - un candado omite una configuración si no cabe;
  - el orquestador sale con 0 aunque una configuración falle, para que Kaggle guarde la salida.
- **Pérdida de validación del paso 0** (decisión de Samuel): `src/clasificador_kaggle.py` no se toca. En la evaluación del barrido se recupera del log de cada configuración con el mismo método de la Parte A (`validacion_del_log`), y se comprueba que los otros 12 puntos coinciden con su `mediciones.json`.
- **Máscara y promedio de las dos pérdidas:** coinciden. Claude Code lo confirmó leyendo el código de transformers 5.17.0, unsloth 2026.11.9 y unsloth_zoo 2026.11.8, sin GPU:
  - train y validación pasan por el mismo colador (relleno con -100) y por la misma pérdida de Unsloth;
  - las dos dividen la suma de la entropía cruzada de las fichas con etiqueta desplazada distinta de -100 entre el número de esas fichas del lote (`num_items_in_batch`, que transformers 5.17 también cuenta en `prediction_step`);
  - la validación promedia los lotes ponderando por filas; como cada fila tiene 7 fichas etiquetadas, eso equivale al promedio por ficha.
- **Sesiones de GPU paralelas:** sin fuente oficial clara, así que va un solo kernel secuencial.
  - La página de Notebooks de Kaggle solo confirma el límite de 12 h por sesión.
  - La guía de uso eficiente de GPU dice que un notebook "can have multiple concurrent batch sessions", pero no dice si dos sesiones de GPU corren a la vez ni cuál es el límite.
  - Claude Code leyó las dos páginas renderizadas con la extensión de Chrome, porque sin JavaScript solo traen el título.
- Estimación, con los tiempos reales de 06b-inicial (rank 8 igual que rank 16; rank 32 con un 5 % más de entrenamiento):

  | Configuración | Estimado | Empieza hacia | Tiempo máximo |
  |---|---:|---:|---:|
  | `06b-lr1e-4-r8` | 72.5 min | 0.04 h | 2.42 h |
  | `06b-lr1e-4-r32` | 75.7 min | 1.25 h | 2.52 h |
  | `06b-lr2e-4-r8` | 72.5 min | 2.51 h | 2.42 h |
  | `06b-lr2e-4-r32` | 75.7 min | 3.72 h | 2.52 h |
  | **Total** (con 150 s del orquestador) | **4.98 h** | | |

  - `-t 27000` (7.5 h, 1.51 veces el total), con 4.5 h de margen contra el límite de 12 h. El presupuesto interno es de 26,100 s.
  - Cuota: quedan 28 h 35 min. Después del barrido quedarían unas 23 h 36 min, o 21 h 5 min en el peor caso (las 7.5 h completas).
- **Regla de elección, escrita antes de lanzar:**
  - Gana la mayor accuracy de clase en validación entre las 5 configuraciones: 06b-inicial y las 4 del barrido.
  - En empate exacto con la inicial (los mismos aciertos), gana la inicial.
  - Se reportan el IC 95 % de cada una y la diferencia pareada contra la inicial (configuración − 06b-inicial).
  - Advertencia: como se elige en la misma validación, la accuracy de la elegida es algo optimista.
  - La regla la redactó el chat de Claude del proyecto en la instrucción y Samuel la aprobó.
  - **Complemento, escrito antes de lanzar.** Claude Code señaló los dos casos que la regla no cubría; el chat de Claude del proyecto propuso el complemento y Samuel lo aprobó:
    - **Solo cuentan las configuraciones completas:** 5,855 pasos de entrenamiento e inferencia de las 1,000 filas de validación. Una configuración incompleta se reporta y no se repite antes del 4B.
    - **Empate exacto en accuracy de clase:**
      - si la inicial está entre las empatadas, gana la inicial;
      - si no, gana la de menor rank;
      - si siguen empatadas, gana la de lr 2e-4.
- Código:
  - `src/barrido_kaggle.py`, nuevo;
  - `src/construir_kernel.py`, con dos campos opcionales: `corridas` y `docker_image`;
  - `notebooks/kaggle-06b-barrido/`: el kernel (168,603 bytes) y su metadata.
- Pruebas:
  - 39 nuevas; pasan las 706 del repo;
  - los kernels de humo-6b y 06b-inicial siguen idénticos byte a byte a los lanzados;
  - la prueba en seco de la orquestación corre procesos reales en CPU con un proceso falso: camino normal, configuración que falla, cuelgue que se mata a los 2 s, candado de tiempo, memoria no liberada, versión fijada distinta (aborta) y versión no fijable distinta (solo advierte).
- Commits:
  - `41e324a`: código, pruebas y kernel;
  - el de esta bitácora, el registro y las dos capturas.
- PARADA 2: el kernel está listo y sin lanzar. Falta que Samuel corra `--verificar`, el push de git y `kaggle kernels push` desde PowerShell.

## 2026-11-23 - Fase 2, paso 12 (Parte A): evaluación del barrido, regla de elección y kernel del 4B, preparados sin GPU

- Autoría:
  - El chat de Claude del proyecto redactó la instrucción y Samuel la aprobó.
  - Claude Code propuso el plan. Samuel lo aprobó con dos ajustes para la PARADA B:
    - comandos completos que funcionen desde cualquier carpeta de PowerShell;
    - qué revisar en el log entre 15 y 20 minutos después de lanzar el 4B.
- **Lanzamiento del barrido:**
  - Samuel lanzó `samcaballero/denue-scian-06b-barrido` (versión 1) el 2026-11-23 a las 6:55 pm (hora de la Ciudad de México), con `-t 27000`. El `lastRunTime` de la CLI es 2026-11-24 00:56:39 UTC.
  - Kaggle aceptó la imagen fijada: la metadata que bajó `kaggle kernels pull -m` (a una carpeta fuera del repo, sin ejecutar nada) trae `docker_image` igual a la de 06b-inicial (`fc720c15…`).
  - El código bajado es idéntico al kernel del repo salvo los CRLF de sus 3,044 líneas (SHA-256 normalizado `a3d30180…`). Sus SHA-256 de kernel y metadata quedan en la prueba de kernels lanzados, junto a los de humo-6b y 06b-inicial.
  - A las 7:46 pm, `kaggle kernels status` daba RUNNING. Debe terminar hacia la medianoche (4.98 h estimadas).
- **Regla de elección** (`src/eleccion_barrido.py`), como función pura, tal como está escrita en el paso 11 con su complemento:
  - solo cuentan las configuraciones completas (5,855 pasos e inferencia de las 1,000 filas); las incompletas se reportan;
  - gana el mayor número de aciertos de clase (se comparan enteros, no flotantes);
  - en empate exacto: la inicial si está entre las empatadas; si no, la de menor rank; si siguen empatadas, la de lr 2e-4;
  - devuelve la ganadora, la cláusula que decidió, las empatadas, las excluidas y el orden de todas.
- **Evaluación del barrido** (`--barrido 06b-barrido` en `src/evaluar_corrida_clasificador.py`), con las revisiones del paso 11:
  - la descarga contra `barrido.json` y contra el `mediciones.json` de cada configuración;
  - los nombres, con la regla acotada, sobre todos los textos (logs de cada configuración, `barrido.json`, mediciones, predicciones, pip freeze y adaptador);
  - el orquestador: procesos, memoria liberada, versiones fijadas y advertencias de las no fijables;
  - por configuración: las etapas, el orden del train igual al de 06b-inicial (`b6b53a85…`), el punto 0 de validación sacado de su log y las versiones contra 06b-inicial;
  - las métricas con el evaluador y la diferencia pareada contra 06b-inicial (configuración − 06b-inicial), alineada fila por fila por `int(id)`. Los aciertos de 06b-inicial salen de su `predicciones.jsonl`, con el SHA-256 de su resultado, y deben reproducir sus conteos e IC;
  - salidas: un JSON de `results/` por configuración, el resumen del barrido con la regla y la figura de las curvas de las 5;
  - una configuración incompleta es PARADA; solo con la decisión de Samuel, `--aceptar-incompletas` la deja fuera de la regla.
  - El refactor del evaluador no cambia el resultado de 06b-inicial: regenerado con la fecha y el commit de su JSON, da los mismos 54,602 bytes.
- **Figura del barrido:** dos paneles, train por ventanas de 50 pasos y validación desde el paso 1,000 (con el eje desde 0, las diferencias finales no se verían). Colores: las posiciones 1 a 5 de la paleta de referencia, validadas con el script de la skill de visualización. Tres tienen poco contraste con el fondo, así que cada serie lleva su propio marcador y la leyenda escribe el valor final de validación.
- **Kernel del 4B** (`src/corrida_4b_kaggle.py` y `espec_4b` en `src/construir_kernel.py`), sin escribir todavía:
  - usa el orquestador del barrido con una sola configuración: los mismos 20 pins, la revisión de versiones, la imagen de Docker y `src/clasificador_kaggle.py` sin cambios en su propio proceso;
  - `src/barrido_kaggle.py` tampoco cambia: va incrustado en el barrido lanzado;
  - parámetros: tasa y rank, con alpha = rank; si gana la inicial, rank 16, alpha 16 y tasa 2e-4. Modelo `Qwen/Qwen3-4B` en la revisión `1cfa9a72…`; id `samcaballero/denue-scian-4b`;
  - el adaptador ya se guarda al terminar el entrenamiento y antes de la inferencia (etapa `adaptador` de `src/clasificador_kaggle.py`); una prueba lo comprueba sobre el código;
  - el orquestador mata el proceso al pasar el presupuesto (el `-t` menos 15 min) y sale con 0, así que Kaggle guarda la salida con el adaptador aunque la inferencia no termine;
  - las pruebas usan la configuración inicial. El kernel se registra y se escribe con la ganadora en la Parte B.
- **De dónde salen las ~6.1 h por época del 4B:**
  - de la tabla del tope por clase del paso 8 (`results/2026-11-22_fase1_train-final.json`): 187,350 filas ÷ 8.515 ejemplos/s ÷ 3,600 = 6.11 h (6.15 h con 154 s de costo fijo);
  - los 8.515 ejemplos/s se midieron en la prueba de punta a punta del paso 6 (`results/2026-11-21_fase2_prueba-humo.json`): Qwen3-4B, lote 32, 25 pasos (20 medidos), largo 128, rank 16;
  - solo cuentan los pasos de entrenamiento: ni las 13 validaciones, ni la inferencia, ni el arranque.
- **Lote 32 en la T4 con el 4B y largo 256: cabe.** No hace falta acumulación de gradiente y no hay CAMBIO DE DISEÑO.
  - En el paso 6, el lote 32 del 4B usó 4,886 MiB reservados como máximo (4,581 asignados) de los 14,912 MiB que ve torch: un 33 %.
  - El largo 256 no cambia la memoria: el colador rellena hasta la fila más larga del lote. La fila más larga del train real tiene 118 fichas, contra 104 en la muestra (13 % más de activaciones).
  - En el 0.6B, pasar del largo 128 (prueba) al 256 (06b-inicial) no subió el pico: de 1,458 a 1,152 MiB.
  - Rank 32 agrega unos 33 millones de parámetros entrenables, unos 0.4 GB con gradientes y AdamW de 8 bits. El peor caso queda cerca de 6 GB.
- **Tiempo estimado del 4B** (`estimacion_s`): tiempos reales de 06b-inicial × la razón de segundos por paso 4B/0.6B medida en la prueba de punta a punta con el mismo lote 32 en la misma T4 (3.758 / 0.629 = 5.98). La descarga y la carga del 4B también se midieron en esa prueba (63.6 + 11.8 s). El rank usa el factor del barrido (8 y 16: 1.0; 32: 1.05).

  | Parte | Rank 8 | Rank 16 | Rank 32 |
  |---|---:|---:|---:|
  | Orquestador (GPU, hashes, pip común) | 2.5 min | 2.5 min | 2.5 min |
  | Antes de entrenar (pip, imports, descarga y carga, tokens, LoRA) | 4.9 min | 4.9 min | 4.9 min |
  | 5,855 pasos de entrenamiento | 6.31 h | 6.31 h | 6.62 h |
  | 13 validaciones de 1,000 filas | 8.5 min | 8.5 min | 9.0 min |
  | Adaptador e inferencia de 1,000 filas | 22.7 min | 22.7 min | 22.7 min |
  | **Total** | **6.95 h** | **6.95 h** | **7.27 h** |

  - La inferencia del 4B no se midió: la cifra es una cota (la de 06b-inicial × 5.98).
  - Las 6.31 h de pasos son las 6.11 h de la tabla del paso 8 más un 3 %: lo que el 0.6B real tardó de más frente a la prueba (2 % por paso, más los pasos de calentamiento).
  - **`-t` propuesto: 39,600 s (11 h).** Es 1.58 veces el total de rank 8 y 16 y 1.51 el de rank 32, y queda 1 h por debajo del límite de 12 h. El candado del orquestador deja lanzar los tres ranks; el proceso se mata hacia los 38,550 s si se cuelga.
  - **Cuota:** antes del barrido quedaban 28 h 35 min. Después del barrido quedarían unas 23 h 36 min (21 h 5 min en el peor caso). Después del 4B, unas 16 h 39 min con rank 8 o 16, o 16 h 20 min con rank 32; en el peor caso de los dos, 10 h 5 min. En la Parte B se usa la cuota real.
  - En la Parte B la estimación se contrasta con los tiempos reales del barrido.
- Código:
  - `src/eleccion_barrido.py` y `src/corrida_4b_kaggle.py`, nuevos;
  - `src/evaluar_corrida_clasificador.py`, con el modo `--barrido`;
  - `src/construir_kernel.py`, con `corrida_4b`, `espec_4b` y el campo opcional `corrida_unica`. Los kernels de humo-6b, 06b-inicial y barrido siguen idénticos byte a byte.
- Pruebas: 48 nuevas con datos inventados; pasan las 754 del repo.
- Commit: el de esta entrada (código, pruebas, bitácora y registro).
- PARADA A: falta que Samuel confirme que el barrido terminó (0 Active Events) y dé la duración de la página del kernel y la cuota que queda.

## 2026-11-24 - Fase 2, paso 12 (Parte B): resultados del barrido del 0.6B, configuración ganadora y kernel del 4B

- Autoría:
  - El chat de Claude del proyecto redactó la instrucción de la Parte B y Samuel la aprobó.
  - La regla de elección es la del paso 11, escrita antes de lanzar; Claude Code solo la aplicó.
- **Fin del barrido:**
  - Samuel vio en Kaggle "Version #1 Successful" y "0 Active Events"; no abrió la página del kernel.
  - `kaggle kernels status` dio COMPLETE el 2026-11-23 a las 11:55 pm.
- **Duración**, de `barrido.json` y del `lastRunTime` de la CLI:
  - el orquestador corrió de 00:56:47 a 05:42:03 UTC (17,115.8 s, 4.754 h); el contenedor arrancó 7.6 s después del `lastRunTime`;
  - del `lastRunTime` al fin del orquestador pasaron 17,123 s (4.756 h). Es el valor de `horas_gpu` del resumen;
  - estaban estimadas 4.98 h: cada configuración terminó entre un 2 y un 6 % antes de lo estimado.
- **Cuota:** 28 h 35 min antes del barrido y 23 h 49 min después (observada por Samuel hacia las 11:45 pm). Kaggle descontó 4 h 46 min, igual que la duración medida.
- **Sesión interactiva:** hacia las 7:50 pm, Gemini abrió sin querer una sesión interactiva de GPU desde la vista /edit al revisar el log, y quedó cancelada. La cuota a las 9:21 pm (26 h 12 min) coincide con el tiempo del barrido, así que no gastó cuota apreciable.
- **Capturas**, de `../evidencia/` a `docs/evidencia/`, copiadas sin cambios (mismo SHA-256):
  - `2026-11-23_fase2_barrido-lanzado.png` (`b5f54be5…`): el kernel en Running a los 2 min, con 1 Active Event;
  - `2026-11-23_fase2_configuraciones-barrido.png` (`a5505743…`): la tabla de las 4 configuraciones;
  - `2026-11-23_fase2_tabla-06b-inicial.png` (`97591acd…`): la tabla de métricas de 06b-inicial y los baselines;
  - `2026-11-23_fase2_barrido-successful.png` (`cab7d897…`): Version #1 Successful y 0 Active Events;
  - `2026-11-23_fase2_cuota-despues-barrido.png` (`fcb86813…`): 23 h 49 min de GPU disponibles.

  Claude Code las revisó: solo muestran títulos, estados, la cuota y tablas de números y configuraciones. No muestran cuentas, avatares, rutas, correos, llaves, tokens ni nombres de negocios, y no traen metadatos de texto ni EXIF.
- **Descarga** a `kaggle-output/06b-barrido/`, ignorada por git:
  - 36 archivos y 213.6 MB, casi todo de los cuatro adaptadores (202.1 MB);
  - coinciden los 34 que lista `barrido.json` con su SHA-256 (no lo traen `barrido.json` ni el log de la CLI);
  - cada configuración trae exactamente lo que lista su `mediciones.json`, y sus predicciones son las de su inferencia;
  - ningún archivo tiene un SHA-256 del manifest.
- **Revisión de nombres**, con la regla acotada del paso 11 y antes de mirar logs o mediciones:
  - sin posible fuga en ningún archivo, y ninguna llave de varias palabras;
  - los nombres crudos con letras son de 4 caracteres o menos y siempre van dentro de un token más largo (pip freeze y logs); los demás son solo dígitos;
  - el kernel tapó en cada configuración los mismos dos valores de siempre (`save_strategy` y `train_sampling_strategy`).
- **Etapas:** todas bien en las 4 configuraciones.
  - Orquestador: dos T4 en la sesión y una en uso; las 20 versiones fijadas con pip quedaron instaladas; torch, torchvision, triton y el controlador son los de 06b-inicial, así que no hay advertencias.
  - Cada proceso terminó con código 0 y dejó la GPU 0 en 0 MiB.
  - El orden del train es el de 06b-inicial (`b6b53a85…`) en las 4: el barajado no depende del rank ni de la tasa.
  - El punto 0 de validación sale del log de cada configuración (3.3235 en las cuatro, como en 06b-inicial).
- Resultados en validación: 1,000 filas, IC 95 % por bootstrap de 1,000 remuestreos con semilla 42. La diferencia pareada es configuración − 06b-inicial, alineada fila por fila por `int(id)`.

  | Configuración | Tasa | Rank | Accuracy (IC 95 %) | Sector (IC 95 %) | Macro-F1 de sector | Pareada de clase (IC 95 %) | Pareada de sector (IC 95 %) | Proceso |
  |---|---:|---:|---|---|---:|---|---|---:|
  | 06b-inicial | 2e-4 | 16 | 0.678 (0.646-0.707) | 0.847 (0.824-0.869) | 0.731 | — | — | 72.5 min |
  | 06b-lr1e-4-r8 | 1e-4 | 8 | 0.643 (0.612-0.673) | 0.830 (0.805-0.853) | 0.709 | −0.035 (−0.050 a −0.020) | −0.017 (−0.033 a −0.002) | 71.0 min |
  | 06b-lr1e-4-r32 | 1e-4 | 32 | 0.664 (0.632-0.694) | 0.839 (0.817-0.862) | 0.726 | −0.014 (−0.026 a −0.001) | −0.008 (−0.019 a +0.003) | 71.8 min |
  | 06b-lr2e-4-r8 | 2e-4 | 8 | 0.665 (0.634-0.694) | 0.839 (0.817-0.862) | 0.712 | −0.013 (−0.028 a +0.001) | −0.008 (−0.021 a +0.004) | 70.3 min |
  | **06b-lr2e-4-r32** | 2e-4 | 32 | **0.687 (0.656-0.715)** | 0.849 (0.826-0.870) | 0.727 | +0.009 (−0.005 a +0.024) | +0.002 (−0.010 a +0.015) | 71.5 min |

  - Regla 5: en las 4, 0 respuestas con formato inválido, 0 con código inexistente y 1,000 con código existente.
  - Sin colapso: cada una responde entre 195 y 199 clases distintas, y la más predicha (461110) sale en un 7 % de las respuestas.
  - Pérdida final de validación: 0.251, 0.233, 0.236 y 0.217, en el orden de la tabla (0.226 en 06b-inicial). La ganadora también tiene la menor.
  - Con tasa 1e-4, las dos configuraciones quedan por debajo de la inicial y el IC de la pareada de clase excluye el cero. Con rank 32, las dos tasas superan a su par de rank 8.
- **Regla de elección: gana `06b-lr2e-4-r32`**, la única con la mayor accuracy de clase (687 aciertos, contra 678 de la inicial); no hubo empate ni configuraciones incompletas.
  - Advertencia: se eligió con la misma validación con la que se mide, así que su 0.687 es algo optimista.
  - Además, la ventaja sobre la inicial no es clara: el IC de la pareada de clase (−0.005 a +0.024) incluye el cero.
- Archivos:
  - un JSON de `results/` por configuración (`2026-11-24_fase2_06b-lr1e-4-r8.json` y los otros tres);
  - el resumen `results/2026-11-24_fase2_06b-barrido.json` (149,566 bytes), con la tabla, la regla y su resultado;
  - la figura `docs/figuras/2026-11-24_fase2_curvas-barrido-06b.png`.

  Se generaron dos veces desde `9584237`, con bytes idénticos. Las 28 coincidencias con ids del DENUE en esos JSON son cifras decimales de valores con punto flotante, no ids.
- **Bug del evaluador del barrido, introducido por Claude Code en la Parte A:**
  - la comprobación de los argumentos del Trainer falló en las 4 configuraciones, porque esperaba la carpeta de trabajo de 06b-inicial (`/tmp/denue_trabajo/trainer_<nombre>`);
  - el orquestador le da a cada proceso su propia carpeta (`/tmp/denue_trabajo/<nombre>/`). Era un error de la evaluación, no de la corrida: los demás argumentos coincidían;
  - se corrigió en `9584237`, con una prueba, antes de escribir los resultados.
- **Contraste de la estimación del 4B con el barrido:**
  - en el 0.6B, el rank 32 tardó un 2 % más que el rank 8 en el entrenamiento; la estimación del 4B usa un 5 %;
  - las validaciones (81-84 s) y la inferencia (224-229 s) duraron lo mismo que en 06b-inicial.

  La estimación queda como cota conservadora y el `-t` no cambia.
- **Kernel del 4B** (`notebooks/kaggle-4b/`), con la ganadora:
  - Qwen/Qwen3-4B en la revisión `1cfa9a72…`, tasa 2e-4, rank 32 y alpha 32; lo demás, como el barrido;
  - `clasificador_4b.py`: 177,054 bytes, SHA-256 `ae2313ce…`;
  - `kernel-metadata.json`: id `samcaballero/denue-scian-4b`, privado, T4, el dataset del clasificador y la imagen de Docker de 06b-inicial; SHA-256 `4f0078f1…`;
  - `--verificar` pasa en los 4 kernels, y una prueba ata el kernel a la ganadora que registra el resumen.
- **GPU y cuota del 4B:**
  - estimado: 7.27 h con rank 32 (probablemente algo menos, por lo que se vio en el barrido);
  - `-t 39600` (11 h);
  - quedarían unas 16 h 33 min de cuota, o 12 h 49 min en el peor caso.
- Pruebas: pasan las 762 del repo.
- Commits:
  - `9584237`: observaciones del barrido y corrección del evaluador;
  - `e92718d`: resultados y figura;
  - `2c7198f`: kernel del 4B y pruebas;
  - el de esta bitácora, el registro y las capturas.
- PARADA B: el kernel del 4B está listo y sin lanzar. Falta que Samuel corra `--verificar`, el push de git y `kaggle kernels push` desde PowerShell.

## 2026-11-24 - Fase 2, paso 13: cierre de la Fase 2

- Autoría:
  - El chat de Claude del proyecto redactó la instrucción y Samuel la aprobó.
  - Claude Code propuso el plan. Samuel lo aprobó con tres ajustes:
    - la decisión sobre el modelo base en la Fase 4;
    - la nota sobre TF-IDF para la Fase 4;
    - los segundos por fila de la inferencia y la carga del modelo en la PARADA.
  - **Regla de las horas de GPU con cola:** la propuso Claude Code en el plan y Samuel la aprobó con él. Si el contenedor arranca más de 5 min después del `lastRunTime`, las horas son las del orquestador; si no, se cuentan desde el `lastRunTime`, como en el barrido.
- **Corrida:**
  - Samuel lanzó `samcaballero/denue-scian-4b` (versión 1) hacia la 1:05 am, con `-t 39600`.
  - Estuvo en Queued hasta el `lastRunTime` de la CLI, 08:23:28 UTC (2:23 am, hora de la Ciudad de México). El contenedor arrancó 2.3 s después, así que la cola, de unos 78 min, quedó antes de esa marca.
  - El orquestador corrió de 08:23:31 a 15:35:59 UTC (de 2:23 am a 9:36 am): 25,948.5 s.
  - `kaggle kernels status` dio COMPLETE a las 12:54 pm.
  - `kaggle kernels pull -m` bajó el código y la metadata a una carpeta nueva del scratchpad; no se ejecutó nada.
    - El código es el kernel del repo salvo los CRLF de sus 3,164 líneas (SHA-256 `ae2313ce…`).
    - La metadata trae la imagen de 06b-inicial (`fc720c15…`).
    - Sus SHA-256 quedan en la prueba de kernels lanzados.
- **Horas de GPU y cuota:**
  - 7.209 h (25,950.8 s), del `lastRunTime` al fin del orquestador. Es el valor de `horas_gpu`.
  - Kaggle descontó 7 h 12 min (432 min): 23 h 49 min antes y 16 h 37 min después, observadas por Samuel. La medida pasa en 0.5 min a lo descontado: la cola no se cobró.
  - Estaban estimadas 7.27 h.
- **Captura**, de `../evidencia/` a `docs/evidencia/`, copiada sin cambios: `2026-11-24_fase2_tabla-barrido.png` (`02a9f402…`), la tabla de las 5 configuraciones del barrido en validación.
  - Claude Code la revisó: solo muestra números y nombres de configuración, sin metadatos de texto ni EXIF.
  - `2026-11-17_fase0_gpu-kaggle-1.png` sigue fuera, por la decisión de Samuel del 2026-11-18.
- **Descarga** a `kaggle-output/4b/`, carpeta nueva e ignorada por git:
  - 12 archivos y 255 MB, casi todo del adaptador (264,308,896 bytes);
  - coinciden los 10 que lista `barrido.json` con su SHA-256; no lo traen `barrido.json` ni el log de la CLI;
  - la configuración trae exactamente lo que lista su `mediciones.json`, y sus predicciones son las de su inferencia;
  - ningún archivo tiene un SHA-256 del manifest.
- **Revisión de nombres**, con la regla acotada del paso 11 y antes de mirar logs o mediciones:
  - sin posible fuga en ningún archivo, y ninguna llave de varias palabras;
  - los nombres crudos con letras son de 4 caracteres o menos y siempre van dentro de un token más largo (pip freeze y logs); los demás son solo dígitos;
  - el kernel tapó los mismos dos valores de siempre.
- **Orquestador:**
  - dos T4 en la sesión y una en uso;
  - las 20 versiones fijadas con pip quedaron instaladas;
  - torch 2.11.0+cu130, torchvision 0.26.0+cu130, triton 3.6.0 y el controlador 580.178.04 son los de 06b-inicial, así que no hay advertencias;
  - el proceso terminó con código 0, sin que lo mataran, en 25,906 s (tiempo máximo: 38,658.5 s), y dejó la GPU 0 en 0 MiB.
- **Etapas:** todas bien, incluida la nueva de la referencia.
  - **GPU:** una T4 visible para torch (14,912 MiB). Sin bf16 nativo, así que fp16.
  - **Datos:** 187,350 filas de train (929 clases) y 1,000 de validación (228), con su SHA-256.
  - **Modelo:**
    - `Qwen/Qwen3-4B` en la revisión `1cfa9a72…`, con sus 4 archivos LFS verificados y nada más en la caché;
    - NF4, con 252 capas de 4 bits; 2,538 MiB de VRAM tras cargar;
    - LoRA de rank 32: 66,060,288 parámetros entrenables. Es lo que da el cálculo con las dimensiones de Qwen3-4B, y en fp32 son unos 264 MB, el tamaño del adaptador.
  - **Tokens:** la fila más larga tiene 118 fichas en train y 98 en validación. Cada fila etiqueta 7 fichas.
  - **Entrenamiento:**
    - 5,855 pasos y 176 de warmup;
    - el orden del train es el de 06b-inicial (`b6b53a85…`);
    - 4.198 s/paso de media (mediana 4.240) y 7.62 ejemplos/s;
    - VRAM pico: 4,604 MiB reservados (4,323 asignados), un 31 % de lo que ve torch.
  - **Pérdida de validación del paso 0:** 2.1778, sacada del log; en el 0.6B era 3.3235. Los otros 12 puntos coinciden con `mediciones.json`.
  - **Adaptador:** rank 32, alpha 32, dropout 0 y base `Qwen/Qwen3-4B`.
  - **Inferencia:** 1,000 filas en 430 s (0.430 s/fila), greedy. Todas terminan en el token de fin, con 7 fichas como máximo.
  - **Referencia:**
    - `06b-lr2e-4-r32` es la ganadora del barrido, y su resultado es el que registra el resumen;
    - la receta es la misma salvo nombre, descripción y modelo;
    - su adaptador es el de su resultado.
- **Curvas**, en `docs/figuras/2026-11-24_fase2_curvas-4b-vs-06b.png`, junto a las de `06b-lr2e-4-r32`:
  - la pérdida de train baja de 2.061 (media de los primeros 10 pasos) a 0.253 (media de los últimos 10);
  - la de validación baja de 2.178 en el paso 0 a 0.193 en el paso 5,855. Sube un poco dos veces: de 0.248 a 0.252 en el paso 2,000 y de 0.193 a 0.194 en el 5,500;
  - la del 0.6B ganador termina en 0.217.
- **Resultados en validación de todos los modelos.** 1,000 filas, con el IC 95 % de un bootstrap de 1,000 remuestreos y semilla 42. Todo sale de `results/2026-11-24_fase2_4b-lr2e-4-r32.json` (`tabla_validacion`), que guarda la ruta y el SHA-256 del JSON de cada fila.

  | Modelo | Base | Tasa | Rank | Accuracy (IC 95 %) | Sector (IC 95 %) | Macro-F1 de sector | Clases distintas predichas | 461110 (% de respuestas) |
  |---|---|---:|---:|---|---|---:|---:|---:|
  | 06b-inicial | 0.6B | 2e-4 | 16 | 0.678 (0.646-0.707) | 0.847 (0.824-0.869) | 0.731 | 194 | 7.0 |
  | 06b-lr1e-4-r8 | 0.6B | 1e-4 | 8 | 0.643 (0.612-0.673) | 0.830 (0.805-0.853) | 0.709 | 196 | 6.9 |
  | 06b-lr1e-4-r32 | 0.6B | 1e-4 | 32 | 0.664 (0.632-0.694) | 0.839 (0.817-0.862) | 0.726 | 195 | 6.9 |
  | 06b-lr2e-4-r8 | 0.6B | 2e-4 | 8 | 0.665 (0.634-0.694) | 0.839 (0.817-0.862) | 0.712 | 198 | 7.0 |
  | 06b-lr2e-4-r32 | 0.6B | 2e-4 | 32 | 0.687 (0.656-0.715) | 0.849 (0.826-0.870) | 0.727 | 199 | 6.9 |
  | **4b-lr2e-4-r32** | 4B | 2e-4 | 32 | **0.707 (0.676-0.735)** | 0.862 (0.840-0.883) | 0.760 | 209 | 6.9 |
  | TF-IDF + LR, C = 10 | — | — | — | 0.682 (0.652-0.711) | 0.845 (0.824-0.867) | 0.780 | 213 | 6.5 |
  | Diccionario | — | — | — | 0.083 (0.066-0.099) | 0.443 (0.412-0.473) | 0.034 | 1 | 100.0 |

  - **Regla 5:** en los 8, 0 respuestas con formato inválido y 0 con código inexistente. El 4B tiene 1,000 con código existente.
  - **Accuracy del 4B por nivel:** sector 0.862, subsector 0.802, rama 0.775, subrama 0.753 y clase 0.707.
  - **Subgrupos del 4B:**
    - filas con "SIN NOMBRE": 0.814 (59 filas); sin la frase: 0.700 (941);
    - genéricas: 0.714 (77 filas); no genéricas: 0.706 (923).
  - **Diferencia pareada 4B − 06b-lr2e-4-r32** (positiva si el 4B acierta más), alineada fila por fila por `int(id)`:
    - clase: +0.020 (IC +0.004 a +0.034); solo 06b-lr2e-4-r32 acierta en 21 filas y solo el 4B en 41;
    - sector: +0.013 (IC 0.000 a +0.026); solo 06b-lr2e-4-r32 acierta en 16 filas y solo el 4B en 29. El límite inferior del IC es exactamente 0.
    - En clase, el IC excluye el cero.
  - **Diferencia pareada 4B − TF-IDF** (positiva si el 4B acierta más):
    - clase: +0.025 (IC +0.004 a +0.046); solo TF-IDF acierta en 45 filas y solo el 4B en 70;
    - sector: +0.017 (IC −0.001 a +0.036); solo TF-IDF acierta en 36 filas y solo el 4B en 53. El IC incluye el cero.
    - Con la misma nota del paso 11 sobre la alineación: los aciertos de TF-IDF salen del registro de `data/baselines/d5e7a84b2cc4/tfidf_nombre_C10.json`, que no trae códigos ni ids. El orden se confirmó con alta probabilidad (conteos e IC reproducidos), pero no fila por fila.
    - El macro-F1 de sector del 4B (0.760) sigue por debajo del de TF-IDF (0.780).
  - **Sin señales de colapso:**
    - el 4B responde 209 clases distintas (199 el 0.6B ganador y 213 TF-IDF) y 209 respuestas distintas;
    - la más predicha, 461110, sale en 69 respuestas (6.9 %); en validación son 83 filas (8.3 %);
    - ninguna respuesta queda fuera del formato.
- **Advertencias:**
  - La configuración (tasa 2e-4, rank 32, alpha 32) se eligió en el 0.6B y se transfirió al 4B sin barrido propio, así que puede no ser la mejor para el 4B.
  - La accuracy en validación del 0.6B elegido (0.687) es algo optimista, porque se eligió con la misma validación con la que se mide.
  - La del 4B no lo es: es una sola corrida, no se eligió entre varias con su validación. Claude Code agrega un matiz: la configuración sí salió de esta validación, pero en el 0.6B.
  - Por eso la pareada 4B − 06b-lr2e-4-r32 queda, si acaso, sesgada en contra del 4B.
- **Tiempo estimado contra real**, de `barrido.json` y `mediciones.json`:

  | Parte | Estimado | Real | Real / estimado |
  |---|---:|---:|---:|
  | Antes de entrenar (pip, imports, descarga y carga, tokens, LoRA) | 292.7 s | 287.6 s | 0.98 |
  | Pasos de entrenamiento | 23,833.7 s (6.62 h) | 24,620.8 s (6.84 h) | 1.03 |
  | 13 validaciones | 538.0 s | 555.9 s | 1.03 |
  | Adaptador, inferencia y cierre | 1,363.1 s | 431.9 s | 0.32 |
  | Proceso | 26,027.6 s | 25,906.0 s | 0.995 |
  | Orquestador | 150 s | 42.3 s | 0.28 |
  | **Total** | **26,177.6 s (7.27 h)** | **25,948.5 s (7.21 h)** | **0.99** |

  - **Pasos de entrenamiento:** 4.198 s/paso de media, contra 4.071 estimados. La razón real de s/paso entre el 4B y el 0.6B de rank 32 es 4.198 / 0.650 = 6.46, contra los 5.98 de la prueba de humo.
  - **Inferencia:** la estimación era una cota (la de 06b-inicial × 5.98). Tardó 1.92 veces la del 0.6B, no 5.98.
  - Las dos desviaciones casi se compensan.
- **Para estimar la GPU de la Fase 4** (ajuste de Samuel), de `mediciones.json` y a través de los resultados:

  | | 4b-lr2e-4-r32 | 06b-lr2e-4-r32 |
  |---|---:|---:|
  | Inferencia (s/fila) | 0.430 | 0.224 |
  | Inferencia de 1,000 filas | 430.2 s | 223.9 s |
  | VRAM pico de la inferencia | 3,168 MiB | 749 MiB |
  | Descarga del modelo | 66.7 s | 4.3 s |
  | Carga del modelo | 13.7 s | 8.1 s |

  - Es inferencia greedy, una fila a la vez, con el adaptador y el prompt del clasificador, sin ejemplos.
  - Un prompt few-shot será más largo y más lento.
- **Decisión de Samuel para la Fase 4** (24 nov, 12:51 pm; la propuso el chat de Claude del proyecto):
  - en la Fase 4 se medirá también el modelo base sin afinar (Qwen3-0.6B y Qwen3-4B), 0-shot y few-shot, sobre test y hard set, como pide la guía;
  - el número de ejemplos del few-shot y cómo se eligen se fijan en el plan de la Fase 4, antes de abrir test.
- **Nota para la Fase 4: TF-IDF + LR con C = 10** (ajuste de Samuel):
  - **El modelo no quedó guardado.** `src/baselines.py` no guarda los modelos y no hay `.joblib` ni `.pkl`. Solo existe el registro de aciertos `data/baselines/d5e7a84b2cc4/tfidf_nombre_C10.json` (30,189 bytes, commit `d5e7a84`, ignorado por git).
  - Hay que reentrenarlo: 33.4 s de vectorizador más 3,300.7 s de ajuste, unos 56 min de CPU (`results/2026-11-23_fase1_baseline-tfidf-lr_split-nombre.json`).
  - Al reentrenarlo, comprobar que reproduce fila por fila los aciertos de validación de ese registro.
  - Además, guardar sus predicciones (id y código) de validación, test y hard set en una carpeta ignorada, para que las pareadas contra TF-IDF queden comprobadas fila por fila.
- **Adaptadores completos en `kaggle-output/`**, ignorado por git:
  - el del 4B (`kaggle-output/4b/4b-lr2e-4-r32/adaptador/adapter_model.safetensors`): 264,308,896 bytes y SHA-256 `68898fd8…`, igual al de `barrido.json` y al de su `mediciones.json`;
  - el de 06b-lr2e-4-r32 (`kaggle-output/06b-barrido/06b-lr2e-4-r32/adaptador/adapter_model.safetensors`): 80,792,456 bytes y SHA-256 `19d9f15e…`, igual al de su resultado. Se volvió a hashear.
- **Bug del ayudante de la figura, introducido por Claude Code:**
  - al extraer de `figura_barrido` un ayudante de dos paneles, el bucle de los títulos de los paneles (`titulo`) pisó el parámetro con el título de la figura;
  - la regeneración byte a byte de la figura del barrido lo detectó antes del commit: el título grande salía como el del panel de validación;
  - se corrigió renombrando la variable del bucle. La figura del barrido regenerada vuelve a ser idéntica a la versionada.
- Archivos:
  - `results/2026-11-24_fase2_4b-lr2e-4-r32.json`: 91,112 bytes, generado dos veces desde `b45dfd4` con bytes idénticos (`e4cba27b…`). Pasa el candado de llaves;
  - de los 188,350 ids del paquete buscados como números en ese JSON aparecen 6: 5 son cifras decimales y 1 va dentro de un SHA-256;
  - la figura de las curvas, también idéntica en las dos escrituras.
- Código:
  - `--barrido 4b` en `src/evaluar_corrida_clasificador.py`: las revisiones del barrido con una sola configuración, sin regla de elección;
  - la pareada contra la referencia, la tabla de todos los modelos, el tiempo estimado contra el real, las horas de GPU, los adaptadores, los datos para la Fase 4 y la figura;
  - `src/*_kaggle.py` y `notebooks/` no cambian;
  - los resultados de 06b-inicial y del barrido, regenerados con la fecha y el commit de sus JSON, dan los mismos bytes, y la figura del barrido también.
- Pruebas: 16 nuevas (15 del evaluador, con datos inventados, y el 4B en la prueba de kernels lanzados); pasan las 778 del repo.
- Commits:
  - `b45dfd4`: código y pruebas;
  - `91d9fe7`: resultado y figura;
  - el de esta bitácora, el registro y la captura.
- PARADA: la Fase 2 queda cerrada. Falta el push de Samuel desde PowerShell.

## 2026-11-24 15:27 - Fase 4, paso 14 (Tema A): pre-registro de la evaluación final

- **Qué es.** Antes de mirar test y hard set se escriben las reglas de la evaluación final: qué modelos se miden, con qué prompt, cómo se decodifica, qué métricas se reportan y qué comparaciones son las principales. Así ningún resultado de test puede cambiar las reglas después. A esto se le llama pre-registro.
- **Dónde quedó.** En `evals/fase4_preregistro.json`. Los scripts de la Fase 4 leen las reglas de ese archivo en vez de repetirlas, y cada uno registra su SHA-256 (su huella digital).
- **Quién.** El chat de Claude del proyecto propuso las reglas; Claude Code redactó el archivo y esta sección y calculó la selección few-shot y el límite de largo; Samuel lo aprobó.
- **Los 8 modelos que van a test y al hard set:**
  - los dos baselines: el diccionario y TF-IDF + regresión logística con C = 10. TF-IDF se reentrena (el modelo no se guardó) y debe reproducir fila por fila sus aciertos de validación del registro `d5e7a84`. El diccionario también. Si alguno no los reproduce, se detiene y decide Samuel;
  - los dos afinados: 06b-lr2e-4-r32 (la ganadora del barrido del 0.6B) y el 4B afinado;
  - Qwen3-0.6B y Qwen3-4B sin afinar (modelos base), cada uno 0-shot (sin ejemplos) y few-shot (con 5 ejemplos en el prompt).
- **Lo que no va a test** queda escrito en el archivo: 06b-inicial, las otras 3 configuraciones del barrido, TF-IDF con C = 1 y los dos modelos del split aleatorio.
- **Orden en el kernel de Kaggle:** primero los dos afinados, después los dos base 0-shot y al final los dos base few-shot; dentro de cada grupo, el 0.6B antes que el 4B.
- **Prompt y decodificación (cómo el modelo elige sus fichas):**
  - todos los LLM usan el mismo prompt de `src/prompt_clasificador.py`;
  - todos generan en modo greedy (siempre la ficha más probable, sin azar);
  - los afinados repiten exactamente la decodificación de su validación (hasta 16 fichas nuevas);
  - los base pueden generar hasta 32 fichas nuevas, con los mismos tokens de fin y el mismo corte de respuesta. Se reporta qué % de salidas llega a ese tope.
- **Few-shot (ejemplos en el prompt):**
  - son 5 ejemplos de train final, uno por sector;
  - se baraja train final con la semilla 42 y se toma la primera fila de cada sector que aún no tenga ejemplo, hasta juntar 5;
  - salieron los códigos 461122 (sector 46), 812210 (81), 722513 (72), 712112 (71) y 541110 (54), en ese orden;
  - manda la copia de train final que está en Kaggle, porque es la que verá el kernel. La copia local da exactamente la misma selección;
  - las llaves de los 5 ejemplos no están en validación, test ni hard;
  - los nombres no entran al repo: se guarda solo una huella de sus ids (`0f14aa91…`), y el kernel volverá a elegirlos con la misma regla y comprobará la huella.
- **Largo máximo (CAMBIO DE DISEÑO aprobado por Samuel el 24 nov):**
  - 256 fichas para todo, salvo el prompt few-shot;
  - para el few-shot, la laptop no tiene el tokenizador de Qwen. Pero cada ficha ocupa al menos un byte, así que los bytes del prompt son un tope seguro de sus fichas;
  - el prompt few-shot más largo de test y hard mide 1,526 bytes. Con 32 fichas nuevas y redondeado a múltiplo de 64, el límite queda en 1,600. Las fichas reales serán bastantes menos;
  - antes de generar la primera respuesta de cada modelo, el kernel cuenta las fichas reales de todas sus filas. Si alguna no cabe, se detiene antes de empezar ese modelo, nunca a la mitad. Nunca recorta un prompt.
- **Métricas:**
  - la principal es la accuracy de clase (regla 5) en las 2,000 filas de test, con el mismo evaluador para los 8 modelos;
  - además, la accuracy por nivel del SCIAN, el macro-F1 de sector y los subgrupos de la regla 8;
  - los intervalos de confianza salen de un bootstrap (remuestreo) de 1,000 repeticiones con semilla 42;
  - **métrica tolerante (CAMBIO DE DISEÑO aprobado por Samuel el 24 nov), solo secundaria:** toma el primer grupo de exactamente 6 dígitos de la salida completa, aunque venga con texto alrededor. Se calcula para los 6 LLM y se reporta aparte.
- **Comparaciones (siempre sobre las mismas filas):**
  - principales: 06b-lr2e-4-r32 contra TF-IDF y 4B afinado contra TF-IDF, en accuracy de clase en test;
  - secundarias: 4B afinado contra 06b-lr2e-4-r32, y cada afinado contra su base 0-shot y contra su base few-shot;
  - las mismas 7 pareadas en accuracy de sector, también secundarias;
  - no hay corrección por comparaciones múltiples, y el reporte lo dirá;
  - en el hard set se miden el total y cada tipo, con las mismas pareadas. Es exploratorio y lleva dos advertencias:
    - la de la curación: las 105 filas las pre-curó Gemini con criterios de Samuel y del chat; Samuel no las revisó fila por fila; el chat hizo 23 ajustes; sesgo reconocido (CAMBIO DE DISEÑO del 22 nov);
    - la del tamaño: el tipo indirecto tiene 31 filas, menos de 50, y se reporta con su intervalo.
- **Uso único de test:**
  - "abrir test" es calcular cualquier métrica con las etiquetas de test o de hard. Pasa una sola vez, en la laptop, cuando ya estén guardadas las predicciones de los 8 modelos;
  - si el kernel falla por algo técnico, se relanza sin cambios que dependan de resultados;
  - si test contradice validación, se reporta tal cual;
  - antes de test, el kernel repite la validación con los dos afinados. Cuándo debe detenerse por no reproducirla se decide en el paso 15.
- **50 errores para analizar:** del 4B afinado, sorteados con la semilla 42 entre sus errores de test. Quién los clasifica se decide aparte.
- **GGUF:** su concordancia con el modelo de Kaggle se mide en validación, nunca en test.
- **Observación al calcular:** ninguna llave de test ni de hard está en train final (0 y 0), así que el diccionario responderá la clase global (la más frecuente) en todas sus filas, igual que en validación.
- **Código:**
  - `src/preregistro_fase4.py`: elige los ejemplos, arma el prompt few-shot, calcula el límite y comprueba el config contra el manifest, los archivos locales y las revisiones;
  - de test y hard solo lee el nombre y la llave: la etiqueta se descarta al leer el archivo y no se calcula ninguna métrica;
  - `--comprobar` pasa, incluida la revisión de privacidad del config;
  - `tests/test_preregistro_fase4.py`: 12 pruebas con datos inventados.

## 2026-11-24 21:09 - Fase 4, paso 14 (Tema B): test y hard set sin etiquetas en Kaggle

- **Qué es.** Una copia de test y del hard set con solo el id y el nombre del negocio, sin la clase SCIAN (`codigo_act`) ni el tipo del hard set. Con ella el kernel de Kaggle genera las respuestas sin poder ver las etiquetas. Los baselines también predicen desde estas mismas copias.
- **Quién.** El chat de Claude del proyecto propuso; Claude Code escribió el código y esta sección; Samuel lo aprobó y subió el dataset desde su PowerShell.
- **Código:** `src/paquete_evaluacion.py` (commit `b765755`), con 8 pruebas de datos inventados. Sigue el patrón del paquete del clasificador.
- **Candado** (una revisión automática que se niega a escribir o a verificar si algo no cuadra). Rechaza:
  - archivos de más, ocultos, subcarpetas o enlaces, y una carpeta dentro del repo;
  - cualquier archivo con el SHA-256 de test o de hard completos, o con el de otra entrada del manifest;
  - nuevo: cualquier línea cuyas claves no sean exactamente `id` y `nom_estab`.
- **Carpeta** `../kaggle-datasets/denue-scian-evaluacion/`, fuera del repo:

  | Archivo | Filas | Bytes | SHA-256 |
  |---|---:|---:|---|
  | `test.jsonl` | 2,000 | 114,708 | `a066b066…` |
  | `hard.jsonl` | 105 | 5,044 | `848f8235…` |
  | `dataset-metadata.json` | | 470 (ASCII) | `3434fde1…` |

- **Manifest:** la entrada `kaggle_datasets.denue_scian_evaluacion` lleva filas, bytes y SHA-256, sin clases. Las demás secciones no cambiaron.
- **Subida:**
  - Samuel la hizo con `kaggle datasets create -p` y la ruta absoluta, sin `--public` ni `-u`;
  - Kaggle registra los archivos a las 03:04 UTC del 25 nov (21:04 en la Ciudad de México);
  - `kaggle datasets status` dice ready;
  - `kaggle datasets files` da 114,708 bytes para `test.jsonl` y 5,044 para `hard.jsonl`, iguales al manifest. `dataset-metadata.json` no aparece en la lista porque Kaggle lo toma como metadato, no como archivo;
  - un curl sin credenciales a la página del dataset da 404, sin seguir redirecciones y siguiéndolas: no es público.
- **CLAUDE.md** (sección b) describe el nuevo dataset, con la aprobación de Samuel.
- Commits del paso 14:
  - `47c0080`: código del pre-registro;
  - `57b0d68`: pre-registro y bitácora del Tema A;
  - `b765755`: código del paquete de evaluación;
  - el de esta sección, el manifest, el registro de Claude Code y CLAUDE.md.
- PARADA: falta el push de Samuel desde PowerShell. Sigue el paso 15, el kernel de la Fase 4.

## 2026-11-25 01:05 - Fase 4, paso 15: kernel de inferencia en test y hard set

- **Qué es.** El kernel de Kaggle que genera las respuestas de los 6 LLM del pre-registro en test (2,000 filas) y en el hard set (105): los dos afinados, con su adaptador LoRA, y los cuatro modelos base, en 0-shot y few-shot. Lee test y hard solo desde las copias sin etiquetas, y no calcula ninguna métrica: las respuestas se evalúan una sola vez, en la laptop.
- **Autoría:**
  - El chat de Claude del proyecto propuso el paso y el criterio de reproducción; Samuel los aprobó al pegar la instrucción.
  - Claude Code propuso el plan, con 5 decisiones de diseño. Samuel lo aprobó con 3 ajustes:
    - el respaldo de la carga del adaptador se prueba también si falla la comparación de tensores, comparando cada tensor ya convertido al dtype del parámetro y registrando los dos dtypes;
    - en el punto 9, si el estado es error, se baja la salida y se reportan solo las líneas del error;
    - en la PARADA 2, el estimado de GPU separado en arranque, cada etapa en mejor y peor caso y total, con la cuota que quedaría.
  - Claude Code redactó el código, las pruebas, el cambio al config y esta sección.
  - Samuel lanzó el kernel desde su PowerShell.
- **Criterio de reproducción de validación** (punto 1). En `uso_unico.reproduccion_validacion_en_el_kernel.criterio_de_paro`:
  - cada afinado repite su inferencia en validación, y su respuesta cruda se compara, por id, con la de su `predicciones.jsonl` de referencia;
  - con al menos 980 de 1,000 idénticas, sigue con test y hard; con menos, o si validación no termina completa, ese modelo no corre en test ni en hard y el kernel sigue con el siguiente;
  - queda constancia en `registro.actualizaciones` (2026-11-25, 00:26). El SHA-256 del config pasa de `559ab3bc…` a `7de9d3fc…` (27,038 bytes);
  - `--comprobar` ahora revisa el criterio, así que `registro.commit_codigo` pasa de `47c0080` a `a4330d1`. La selección few-shot y el límite de largo no cambian.
- **Tokenizador real en la laptop** (punto 2). Se usó un entorno aparte, fuera del repo, con los pins de Kaggle: transformers 5.17.0, tokenizers 0.23.3, huggingface_hub 1.33.0 y numpy 2.1.3. Se bajaron solo los archivos del tokenizador de las dos revisiones fijadas.
  - (a) Los dos tokenizadores son idénticos: sus 4 archivos son iguales y los 7,315 prompts dan los mismos ids.
  - (b) El prompt 0-shot es, byte a byte, el de `apply_chat_template` con `enable_thinking=False` en las 3,105 filas de test, hard y validación.
  - (c) El prompt few-shot, con los 5 ejemplos, también lo es en las 2,105 filas de test y hard.
  - (d) Las fichas reales caben en todos los casos:

    | Modo | Máximo del prompt | + nuevas | Límite |
    |---|---:|---:|---:|
    | Afinados | 95 (91 en validación) | 111 | 256 |
    | Base 0-shot | 95 | 127 | 256 |
    | Base few-shot | 436 (media 404) | 468 | 1,600 |

  - La cota por bytes del paso 14 (1,526 para el few-shot) resultó unas 3.5 veces las fichas reales.
- **Cómo corre el kernel:**
  - 14 etapas, cada una en su propio proceso: los afinados en validación, test y hard; los base en test y hard. El 0.6B va antes que el 4B en cada grupo.
  - Antes de cargar ningún modelo revisa:
    - el SHA-256 del config incrustado;
    - el SHA-256 de cada insumo: las dos copias, train final, validación, adaptadores, `adapter_config` y predicciones de referencia;
    - que ningún archivo de la entrada tenga el SHA-256 de test o de hard con etiquetas;
    - las versiones fijadas;
    - la selección few-shot rehecha;
    - la descarga verificada de las dos bases;
    - la plantilla y los largos con los tokenizadores reales en todas las filas de cada modelo.

    Si algo falla, sale antes de la primera etapa.
  - Cada etapa tiene un tiempo máximo y un candado de tiempo. Si falla o se pasa, queda incompleta, sus salidas no se evalúan y el kernel sigue con la siguiente.
  - El adaptador se carga como en el entrenamiento: la LoRA de su `adapter_config` con la misma llamada de Unsloth y los pesos del archivo. Si eso falla, se usa el respaldo, que es el mismo camino del cargador de Unsloth. En los dos casos se compara cada tensor con el del archivo.
  - Las salidas van en `fase4/`: un JSONL de predicciones por etapa completa, el JSON de cada etapa, `comprobaciones.json` y `mediciones.json`. En los JSON y en el log van solo estados, conteos, tiempos y hashes.
- **Kernel** `samcaballero/denue-scian-fase4-inferencia`:
  - privado, T4, internet y la imagen de 06b-inicial;
  - fuentes: los datasets de evaluación y del clasificador y las salidas de los kernels del barrido y del 4B;
  - `fase4_inferencia.py`: 338,408 bytes, SHA-256 `c13dfb94…`;
  - `kernel-metadata.json`: 760 bytes, SHA-256 `5cc5b0e7…`.
- **GPU estimada:**
  - arranque y comprobaciones: 450 s;
  - mejor caso: 2.14 h; peor caso, con los base siempre al tope de 32 fichas: 5.12 h;
  - `-t 28800` (8 h);
  - cuota: hay 16 h 37 min. En el peor caso quedarían unas 11 h 30 min; con el tope de 8 h, 8 h 37 min.
- **Lanzamiento:**
  - Samuel lanzó la versión 1 con `kaggle kernels push -p` y la ruta absoluta, con `-t 28800`;
  - `kaggle kernels status` a las 07:05 UTC (01:05 en la Ciudad de México): QUEUED;
  - `kaggle kernels pull -m` bajó el código a una carpeta nueva del scratchpad, sin ejecutarlo. Es el del repo salvo los CRLF de sus 5,787 líneas. La metadata trae la misma imagen y las mismas fuentes de kernels;
  - sus SHA-256 quedan en la prueba de kernels lanzados.
- **Pruebas:** pasan las 874 del repo. `--verificar` pasa en los 5 kernels, y los 4 anteriores siguen idénticos byte a byte.
- **Commits del paso 15:**
  - `a4330d1`: `--comprobar` revisa el criterio;
  - `e43fe76`: criterio en el config;
  - `0081b86`: comprobaciones de plantilla y largos;
  - `32a9f02`: orquestador y procesos del kernel;
  - `99de3c4`: generador y kernel;
  - el de esta sección, el registro de Claude Code y la prueba de kernels lanzados.
- PARADA: el kernel está en cola. Falta el push de Samuel desde PowerShell. Cuando el kernel termine, siguen la descarga de la salida y la revisión de nombres, antes de mirar logs o mediciones.

## 2026-11-25 02:29 - Fase 4, paso 16: baselines en test y hard set

- **Qué es.** Las predicciones del diccionario y de TF-IDF + LR (C = 10) en test y en el hard set, sacadas de test y hard sin `codigo_act` (los archivos que subieron a Kaggle). Antes, cada baseline reproduce fila por fila sus aciertos de validación. No se calcula ninguna métrica de test ni de hard: eso pasa una sola vez, en la evaluación final.
- **Autoría:**
  - El chat de Claude del proyecto propuso el paso; Samuel lo aprobó.
  - Al aprobar el plan, Samuel pidió reusar un modelo ya guardado sin reentrenar, solo si coinciden su SHA-256, la huella de la configuración, el SHA-256 de train final y las versiones de numpy, scipy, scikit-learn y joblib.
  - Claude Code redactó el código, las pruebas y esta sección.
- **Código:** `src/baselines_fase4.py`, que usa `src/baselines.py` sin cambios (commit `40e4671`).
- **Mínimo de memoria libre** para `--escribir` (commit `57b28b4`): baja de 7.0 a 5.0 GB.
  - El chat de Claude del proyecto lo propuso; Samuel lo aprobó antes de lanzar.
  - Motivo: la laptop no pasa de unos 5.6 GB libres aunque se cierre todo, y con 7.0 GB la corrida nunca arrancaría. La del 23 nov terminó bien con esa memoria. Si falta memoria, Windows usa el archivo de paginación y la corrida solo se hace más lenta.
- **Corrida `--escribir`:** de 2:29 a 3:37 (68.5 min), con el commit `57b28b4`, el árbol limpio y la suspensión del equipo bloqueada.
  - Memoria: 5.6 GB libres al inicio y pico del proceso de 5,069 MB.
  - Config: SHA-256 `7de9d3fc…`, el esperado. Los dos registros de validación tienen el SHA-256, el commit y la huella del config.
  - **Diccionario:** reproduce validación en 1,000 de 1,000 filas (83 aciertos). Ninguna llave de test ni de hard está en train final (0 y 0), así que responde la clase global en todas sus filas, como esperaba el config.
  - **TF-IDF + LR:**
    - vectorizador: 13 s, 100,000 rasgos;
    - ajuste: 68.0 min, contra 55 en validación, con las mismas 43 iteraciones y el mismo pico de 4,171 MB. No se midió por qué tardó más. Lo probable es la laptop con menos memoria libre (4.9 GB durante el ajuste), pero no está comprobado;
    - el modelo se guardó antes de comparar: `tfidf-lr-C10_modelo.joblib`, 746,673,129 bytes, SHA-256 `88c3138c…`. Se recargó desde el archivo, y de él salen las predicciones;
    - reproduce validación en 1,000 de 1,000 filas (682 aciertos).
  - **Predicciones** en `data/fase4/baselines/` (ignorado), con su SHA-256:

    | Archivo | Filas | SHA-256 |
    |---|---:|---|
    | `diccionario_validacion.jsonl` | 1,000 | `ef66c913…` |
    | `diccionario_test.jsonl` | 2,000 | `7072e5d9…` |
    | `diccionario_hard.jsonl` | 105 | `92f8cd2f…` |
    | `tfidf-lr-C10_validacion.jsonl` | 1,000 | `924489a6…` |
    | `tfidf-lr-C10_test.jsonl` | 2,000 | `68fd36e2…` |
    | `tfidf-lr-C10_hard.jsonl` | 105 | `339c9eb1…` |

  - **Registro de la corrida:** `results/2026-11-25_fase4_baselines-predicciones.json`, sin ids, nombres ni métricas de test o hard. HEAD no cambió durante la corrida.
- **Revisión de privacidad del registro** (paso 17, punto 2, antes de su commit):
  - `verificar_privacidad` con 319,875 llaves (train final, validación, train candidato sin hard, y test y hard sin etiquetas) y `verificar_sin_ids` con 421,061 ids: limpias.
  - La regla acotada contra los 4 conjuntos que vio el kernel marca una llave de dos palabras de train final en el campo `split`. Ese campo es un texto fijo de `src/baselines_fase4.py`, ya versionado, y la frase también está en ese código.
  - Por la regla era PARADA, así que el registro se dejó fuera del commit hasta que decidiera Samuel. El log y el JSON del modelo no tienen posible fuga.
  - **Decisión de Samuel en la PARADA 2 del paso 17:** es un falso positivo, por tres razones:
    - la coincidencia es texto fijo de código ya versionado, no un dato del DENUE;
    - el archivo lo escribió el script;
    - no se edita a mano.

    Se hizo su commit tal cual, sin cambiar un byte. La frase no se repite aquí ni en ningún otro archivo.

## 2026-11-25 13:49 - Fase 4, paso 17: evaluación final del clasificador en test y hard set

- **Qué es.** La salida del kernel de inferencia revisada sin etiquetas, el ensayo en validación y la evaluación final de los 8 modelos del pre-registro en test y en el hard set, una sola vez, con su tabla para la charla.
- **Autoría:**
  - El chat de Claude del proyecto propuso el paso; Samuel lo aprobó.
  - Claude Code propuso el plan, con 5 precisiones. Samuel lo aprobó con 3 ajustes:
    - una copia CSV de los errores (UTF-8 con BOM, para Google Sheets);
    - en la PARADA 2, primero el resumen del kernel y después el ensayo;
    - en la PARADA 3, la tabla de validación contra test y los 5 niveles del SCIAN del 4B y de TF-IDF.
  - Samuel aprobó además la D7: en el paso 18, Claude Code propone un tipo de error por fila y Samuel confirma o cambia cada una.
  - Claude Code redactó el código, las pruebas y esta sección.
- **Salida del kernel** (`src/salida_fase4.py`; de Kaggle, solo lectura: `kernels status`, `kernels output` y `kernels pull -m`):
  - **Descarga:** 48 archivos en `kaggle-output/fase4/` (ignorado). Los 46 que lista el kernel coinciden en bytes y SHA-256 con `mediciones.json`, y cada uno de los 14 JSONL coincide también con el JSON de su etapa. Ninguno tiene un SHA-256 del manifest ni el de test o hard con etiquetas.
  - **Revisión de nombres con la regla acotada**, antes de leer los JSON en detalle: 50 textos contra los 4 archivos que vio el kernel (copias de test y hard, train final y validación). 0 posibles fugas; solo 32 nombres crudos de 4 caracteres o menos dentro de tokens más largos.
  - **Ids:** los de cada JSONL son exactamente los de su copia, en orden de `int(id)`.
  - **Etapas:** 14 de 14 completas, 0 fallas. Respuestas de validación idénticas a la referencia: 1,000 de 1,000 en los dos afinados (mínimo, 980), recalculado en la laptop.
  - **Tiempo:** 1.35 h de GPU (4,872 s), contra 2.14 h estimadas en el mejor caso y 5.12 h en el peor. Ninguna respuesta llegó al tope de fichas: los base generaron 9 como máximo.
  - **Imagen:** se usó la fijada (`fc720c15…`), aunque la página de Kaggle decía "Latest Container Image".
    - La metadata de `pull -m` trae esa imagen.
    - El `pip freeze` de antes de instalar es idéntico byte a byte al del kernel del 4B.
    - Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0, cudnn 92700 y el controlador 580.178.04 coinciden con los del 4B.
    - Las 20 versiones fijadas quedaron instaladas.
  - El código que bajó `pull -m` es el del repo salvo los CRLF.
- **Commit del paso 16.** Ver la sección del paso 16: el registro de `results/` entró al commit después de que Samuel declaró falso positivo la coincidencia de la revisión de nombres.
- **Código nuevo:**
  - `src/metrica_tolerante.py`: la métrica tolerante del config.
  - `src/evaluacion_final_fase4.py`. Usa el evaluador único sin cambios y tiene tres modos:
    - punto de control;
    - `--validacion`: ensayo que no escribe nada;
    - `--test`: la evaluación final. Es el único modo que lee `codigo_act` de test y hard y escribe una vez: se niega si el resultado ya existe, calcula todo dos veces y escribe solo con bytes idénticos y candados limpios.
- **Maqueta:** `--validacion` arma el resultado y la tabla completos con números de validación y les pasa los candados de `--test`. Claude Code la propuso durante el paso.
  - Encontró que el texto fijo de la regla 5 en `src/metricas_clasificador.py` (también presente en los `results/` anteriores con `definiciones`) tiene dos palabras seguidas que, por casualidad, son la llave de un nombre de test.
  - En el resultado final, `definiciones.regla5` es una referencia a donde vive la definición, sin su texto. La frase no se repite en ningún archivo.
- **Ensayo en validación:** todo idéntico a lo registrado, campo por campo, en los 4 modelos (06b, 4B, TF-IDF y diccionario) y en las 6 pareadas.
  - La pareada 4B − TF-IDF, alineada ahora por id con las predicciones de validación de TF-IDF del paso 16, da +0.025 (+0.004 a +0.046) en clase y +0.017 (−0.001 a +0.036) en sector: igual a la registrada, y queda como valor de referencia.
- **Primera corrida de `--test`** (hacia la 1:00 pm, commit `176378c`):
  - leyó las etiquetas de test y hard y calculó todo en memoria; pasaron el doble cálculo y `verificar_privacidad`;
  - se detuvo antes de escribir, en el candado de ids: 10 campos de la concentración de respuestas tenían códigos SCIAN respondidos (6 dígitos) que coinciden por casualidad con ids de 6 dígitos;
  - en la terminal solo salieron las rutas de esos campos: nadie vio ningún número ni métrica, y no se escribió ningún archivo;
  - la corrección (`0c0d552`) saca del chequeo de ids solo esos dos campos, que por construcción son códigos. No depende de los resultados, y el cálculo es determinista;
  - Samuel dio el visto bueno para relanzar una vez.
- **Segunda corrida de `--test`** (1:49 pm, commit `0c0d552`, árbol limpio): bytes idénticos en el doble cálculo, candados limpios y HEAD sin cambios. Escribió:

  | Archivo | Bytes | SHA-256 |
  |---|---:|---|
  | `results/2026-11-25_fase4_evaluacion-final.json` | 233,740 | `4b990601…` |
  | `results/2026-11-25_fase4_tabla-final.md` | 10,150 | `5e6471b0…` |
  | `data/fase4/errores_4b-lr2e-4-r32_test.jsonl` (ignorado) | 22,815 | `8958bcfd…` |
  | `data/fase4/errores_4b-lr2e-4-r32_test.csv` (ignorado) | 10,972 | `dc46b0c5…` |

- **Resultados en test** (2,000 filas, distribución natural; de `results/2026-11-25_fase4_evaluacion-final.json`):

  | Modelo | Clase (IC 95 %) | Sector | Macro-F1 de sector | % inválido | % inexistente |
  |---|---|---:|---:|---:|---:|
  | diccionario | 8.3 (7.1–9.6) | 44.1 | 0.034 | 0.0 | 0.0 |
  | tfidf-lr-C10 | 69.5 (67.5–71.5) | 84.5 | 0.678 | 0.0 | 0.0 |
  | 06b-lr2e-4-r32 | 68.8 (66.8–70.9) | 84.4 | 0.679 | 0.0 | 0.0 |
  | 4b-lr2e-4-r32 | 71.6 (69.6–73.5) | 86.1 | 0.724 | 0.0 | 0.0 |
  | 06b-base-0shot | 0.0 | 0.0 | 0.000 | 78.0 | 21.9 |
  | 4b-base-0shot | 0.0 | 0.0 | 0.000 | 22.0 | 78.0 |
  | 06b-base-fewshot | 0.1 | 2.5 | 0.005 | 0.0 | 17.4 |
  | 4b-base-fewshot | 8.9 (7.7–10.2) | 23.7 | 0.093 | 0.3 | 38.0 |

  - **Pareadas principales** (b − a, puntos porcentuales): 4B − TF-IDF, +2.1 (+0.8 a +3.6); 06b − TF-IDF, −0.7 (−2.1 a +0.7).
  - **Secundarias:** 4B − 06b, +2.8 (+1.6 a +3.9). Cada afinado contra su base, de +62.6 a +71.6.
  - **Validación contra test:** en los 4 modelos que tienen las dos, la accuracy de test cae dentro del IC de validación.
  - **Métrica tolerante:** igual a la de la regla 5 en los 6 LLM; no rescata ninguna fila.
- **Hard set** (exploratorio; accuracy de clase total, ambiguo e indirecto):
  - TF-IDF 10.5, 6.8 y 19.4;
  - 06b 11.4, 5.4 y 25.8;
  - 4B 9.5, 4.1 y 22.6;
  - diccionario 1.9, 2.7 y 0.0;
  - los 4 base, 0.0.

  Con sus dos advertencias: la curación y el n de indirecto (31).
- **Hallazgos:**
  - El 06b few-shot responde 541110, el código del último ejemplo del prompt, en el 68 % de las filas de test.
  - El 4B 0-shot responde sobre todo códigos inexistentes.
  - El macro-F1 de sector del 4B queda arriba del de TF-IDF en test (0.724 contra 0.678); en validación era al revés.
- **Revisión extra contra train final**, más allá de los candados del paso (que son contra test y hard y salieron limpios). Claude Code pasó la regla acotada a los archivos de `results/` y al texto nuevo de los documentos contra los 4 conjuntos.
  - En el resultado y la tabla marca coincidencias con nombres de train final en tres textos fijos:
    - la atribución exigida por CLAUDE.md, idéntica a una constante del código;
    - la nota del 06b que viene del config;
    - una nota que escribe el código.
  - Con el criterio de Samuel de la PARADA 2 (texto fijo de código o config ya versionado, en un archivo que escribió el script y no se edita a mano), van tal cual.
  - En el texto que Claude Code agregó hoy a la bitácora, al registro de Claude Code y a CLAUDE.md hubo coincidencias en su redacción, también en dos líneas de la sección del paso 16 que ya tenían commit. Se reescribieron esas líneas hasta que la revisión quedó limpia.
- **Errores para el análisis:** 568 errores de clase del 4B en test. Se sortearon 50 con la convención del config, en JSONL y CSV en `data/fase4/`. En `results/` solo van el conteo y los hashes. Su clasificación va en el paso 18.
- **Pruebas:** pasan las 950 del repo.
- **Commits del paso 17:**
  - `9e9208e` y `170640c`: paso 16;
  - `46432f4`: la métrica tolerante;
  - `721e87e`: la revisión de la salida del kernel;
  - `176378c`: la evaluación final;
  - `0c0d552`: el candado de ids;
  - el de esta sección, con el resultado, la tabla, el registro de Claude Code y `CLAUDE.md`.

## 2026-11-25 14:24 - Fase 4, paso 18: tipos de error del 4B en test

- **Qué es.** Claude Code propone un tipo para cada uno de los 50 errores del 4B en test sorteados en el paso 17. Samuel confirma o cambia cada fila en Google Sheets.
- **Autoría:**
  - El chat de Claude del proyecto propuso el paso y los tipos; Samuel lo aprobó.
  - Claude Code propuso el plan. Samuel lo aprobó con 3 ajustes:
    - el CSV para Google Sheets lleva la posición (1 a 50) en lugar del id;
    - la constancia de los falsos positivos del paso 17 (ver abajo);
    - al terminar, exportar el CSV revisado a `data/fase4/` y borrar la hoja de Google Drive.
  - Claude Code redactó los criterios y propone el tipo de cada fila.
- **Criterios:** en `docs/criterios-errores.md`, con su commit antes de clasificar.
  - **Tipos, en orden de precedencia:**
    - a. `codigo_inexistente` (automático);
    - b. `clases_hermanas`, con la rama de 4 dígitos en común (automático);
    - c. `etiqueta_dudosa`;
    - d. `nombre_ambiguo`;
    - `otro`, con su razón.
  - Los criterios incluyen además los desempates, la razón de 15 palabras como máximo y los candidatos para la charla.
  - Samuel no cambia los tipos automáticos: si una de esas filas merece otra lectura, deja una nota. Lo propuso Claude Code y Samuel lo aprobó.
  - **Sesgo de anclaje** (D7): Samuel ve la propuesta antes de decidir. El conteo registra la matriz propuesto → final.
- **Falsos positivos del paso 17.** La revisión extra contra train final encontró coincidencias en tres textos fijos del resultado y la tabla: la línea de crédito de los datos, la nota del 06b del config y una nota del código.
  - Samuel confirmó que son falsos positivos, con el criterio de la PARADA 2: texto fijo de código o config ya versionado, en un archivo que escribió el script y que no se edita a mano.
  - Esos archivos ya están en el push de `7e11b0e`.
