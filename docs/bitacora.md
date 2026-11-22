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
