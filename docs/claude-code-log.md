# Registro de Claude Code

Momentos notables del trabajo con Claude Code: bugs que Claude encontró, bugs que Claude introdujo y decisiones que Samuel corrigió. Una sección por fecha.

## 2026-11-18 — Fase 0: estructura inicial del repo

**Qué se hizo:** revisión de seguridad de las capturas de `../evidencia/`; creación de `CLAUDE.md`, de las carpetas del proyecto y de este registro; ampliación de `.gitignore`; copia de la evidencia a `docs/evidencia/`; conversión de `bitacora.txt` a `docs/bitacora.md`; sección del proyecto en `README.md`. Sin commit ni push.

### Hallazgos

- **Correo personal en el historial de git.** El commit inicial (`1f60028`), creado desde la web de GitHub, tiene como autor un correo personal en lugar del noreply de GitHub. La configuración local de git sí usa el noreply. Corregido por Samuel en otra terminal (Claude no ejecuta comandos de git que cambien el historial): se reescribió el autor con el noreply de GitHub (commit `012e43a`, mismo contenido; Claude verificó que el árbol de archivos es idéntico al de `1f60028`). Pendiente: borrar y recrear el repo en GitHub para eliminar el commit viejo, y subir el historial limpio.
- **El commit viejo puede seguir accesible después de corregirlo.** Tras un push forzado, GitHub puede seguir mostrando el commit original a quien abra su SHA (identificador del commit) directamente, y el SHA corto aparece en este registro y en la captura `2026-11-18_fase0_hallazgos-claude-code.png`. Opciones antes de hacer público el repo: borrar y volver a crear el repo en GitHub con el historial ya corregido, o pedir a GitHub Support que elimine el commit viejo.
- **Ruta del CSV.** El archivo está en `../datos-originales/conjunto_de_datos/`, no directamente en `../datos-originales/` como decía la instrucción inicial.
- **Tamaño del CSV.** 260,022,563 bytes ≈ 248 MB (253,929 KB), no 254 MB.
- **Codificación del CSV.** Está en latin-1 (ISO-8859-1), no en UTF-8. Leerlo sin `encoding="latin-1"` falla o rompe los acentos.
- **Datos personales en el CSV.** Incluye `telefono`, `correoelec` y `www`. Se agregó a `CLAUDE.md` la regla de no cargarlas ni mostrarlas.
- **Revisión de imágenes.** Ninguna de las 6 capturas de `../evidencia/` (4 del 2026-11-17 y 2 del 2026-11-18) muestra correos, llaves, tokens, teléfonos ni contraseñas, y ninguna trae metadatos de texto o EXIF. Se copiaron 5 a `docs/evidencia/` con SHA-256 verificado; `2026-11-17_fase0_gpu-kaggle-1.png` se quedó fuera por decisión de Samuel.

### Decisiones que Samuel corrigió en el plan

- No copiar `2026-11-17_fase0_gpu-kaggle-1.png` (casi un duplicado); se queda solo en `../evidencia/`.
- Restaurar los acentos en la parte 2 de la bitácora (se habían quitado por miedo a la codificación).
- Agregar `raz_social` a las columnas que no se cargan salvo decisión explícita (puede contener nombres de personas físicas).
- En `CLAUDE.md`: nombrar los modelos y `enable_thinking=False`; dejar para la Fase 1 la definición de las columnas de domicilio y la verificación de la versión del SCIAN de `codigo_act`.
- En `README.md`: aviso de que los datos fueron transformados y de que el proyecto no es un producto oficial del INEGI.
- En `README.md`: enlace a los Términos de Libre Uso del INEGI con https en lugar de http.
- `docs/sesiones/` sirve para guardar con `/export` las sesiones importantes de Claude Code.

### Correcciones de Samuel a `CLAUDE.md`

- La nota de PowerShell 5.1 (sin `&&` ni `||`) mezclaba entornos: Claude Code ejecuta sus comandos en Git Bash; la nota aplica a los comandos que se le dan a Samuel para su terminal.
- La normalización de nombres también quita la puntuación, y se implementa una sola vez en `src/`.
- Los conjuntos reservados solo para test (hard set de nombres ambiguos, 100 direcciones escritas a mano y dos familias de ruido) también quedan congelados, con su hash en `data/manifest.json`.
- Cada JSON de `results/` incluye el hash del split usado y, si hubo entrenamiento, las horas de GPU consumidas.
- Ninguna corrida en Kaggle (`kaggle kernels push`) sin aprobación de Samuel y sin estimar antes las horas de GPU.
- Entorno de Python: `.venv/` dentro del repo, con dependencias fijadas en `requirements.txt` (Fase 1).

### Errores detectados en la revisión del plan

- **`!data/.gitkeep` en el bloque de `.gitignore`.** La revisión del plan, hecha sobre una copia incompleta del texto de la terminal, señaló que faltaba `!data/.gitkeep`; sí estaba; se verificó con `git check-ignore`. Lección: revisar los archivos, no copias de la terminal.
- **Bug de Claude: lista de acentos incompleta.** Al listar las palabras sin acento de la bitácora, Claude omitió "tambien" (línea 16 del original). Samuel decidió corregirlo a "también".

## 2026-11-18 — Fase 1, paso 1: entorno local de Python

**Qué se hizo:** entorno virtual `.venv/` con Python 3.12.10 para trabajo en CPU; `requirements.txt` con versiones exactas; prueba de lectura del CSV y del diccionario de datos; registro del CSV en `data/manifest.json`; reglas nuevas en `CLAUDE.md`. Commit sin push.

### Hallazgos

- **"% códigos inexistentes" no tenía denominador definido.** Al redactar en el plan la regla de métricas del clasificador, Claude Code detectó que no estaba definido si el porcentaje se calcula sobre el total de respuestas o solo sobre las de formato válido, y lo preguntó antes de ejecutar nada. Claude Code propuso calcularlo sobre el total de respuestas y Samuel lo aprobó; el texto final de la regla (`CLAUDE.md`, sección c, regla 5) lo redactó el chat de Claude del proyecto y Samuel lo aprobó: tres categorías que suman 100%, formato válido con `^[0-9]{6}$` en dígitos ASCII, accuracy sobre el total y conteos de las tres categorías en cada `results/*.json`.

## 2026-11-19 — Fase 1, paso 2a: cargador único del DENUE

**Qué se hizo:** diagnóstico de finales de línea; lectura completa del diccionario de datos; verificación de la codificación en el archivo completo; cargador único `src/denue.py` con pruebas de pytest sobre CSV sintéticos; comparación de `clee` con `codigo_act`, conteo de `nom_estab` vacíos o tipo "NA" y conteo de `nom_estab` iguales a `nombre_act` (solo conteos, en `results/2026-11-19_fase1_diagnostico-denue.json`). La instrucción la redactó el chat de Claude del proyecto con la estructura que pidió Samuel; Samuel la aprobó sin cambios. Los ajustes al plan también los redactó el chat de Claude del proyecto y Samuel los aprobó. Commits sin push.

### Hallazgos

- **El diccionario dice SCIAN 2018, pero no está confirmado.** Claude Code encontró en el diccionario de datos que `codigo_act` y `nombre_act` usan el SCIAN 2018; `CLAUDE.md` lo tenía pendiente porque `metadatos_denue.txt` no lo dice. En los ajustes al plan, el chat de Claude del proyecto señaló la contradicción: la página del DENUE en el INEGI dice que esta edición se basa en los Censos Económicos 2024, que usan el SCIAN 2023. Samuel lo aprobó y la versión se decide en el paso 2b por los códigos presentes.
- **El diccionario no dice que `clee` contenga el código de actividad.** La instrucción pedía proponer la regla de `clee` si el diccionario lo confirmaba; Claude Code reportó que no lo confirma y preguntó qué hacer. El chat de Claude del proyecto propuso agregar la regla de todos modos, con otra razón: `clee` es un identificador que un negocio nuevo no tendría al momento de clasificarlo y la única entrada del clasificador es `nom_estab`; Samuel lo aprobó. Comparación medida: los caracteres 6 a 11 de `clee` son iguales a `codigo_act` en 400,328 de 462,732 filas (86.51 %), así que `clee` sí lleva la etiqueta, y ese dato quedó en la regla 6 de la sección c. En las 62,404 discrepancias, esos caracteres también son 6 dígitos, y los pares más frecuentes van de códigos 7222xx a 7225xx (por ejemplo, 722219 → 722513 en 1,857 filas). Hipótesis para el paso 2b: `clee` conserva el código con el que se registró el establecimiento, de una versión anterior del SCIAN.
- **`nom_estab` puede llevar la etiqueta dentro de la entrada.** Al leer el diccionario, Claude Code señaló en el plan que, cuando un negocio no tiene nombre comercial, `nom_estab` lleva el nombre genérico o la descripción de la actividad. El chat de Claude del proyecto pidió medirlo y Samuel lo aprobó. Resultado: 0 `nom_estab` iguales a `nombre_act` de forma exacta y 32 (0.0069 %) tras quitar espacios al inicio y al final y pasar a mayúsculas. La comparación no quita acentos ni puntuación, así que es una cota inferior. Qué hacer con esas filas se decide en el análisis exploratorio.
- **Finales de línea y hashes.** Claude Code explicó el aviso perdido de `git add` del paso 1 (`core.autocrlf=true` y archivos escritos con LF) y lo reprodujo en un repo de prueba fuera del proyecto. También señaló un riesgo futuro: un conjunto congelado guardado como texto en git tendría un SHA-256 distinto en Windows (CRLF) y en Kaggle (LF). Queda pendiente proponer un `.gitattributes` con LF antes de crear el primer conjunto congelado versionado.
- **Bug de Claude Code: acentos en la terminal.** La primera corrida de `src/diagnostico_denue.py` imprimió mal los acentos, porque Python escribe en cp1252 cuando la salida va por una tubería, como en la terminal de Claude Code. Se corrigió antes del commit con `sys.stdout.reconfigure(encoding="utf-8")`. El JSON no se vio afectado, porque se escribe en UTF-8.

## 2026-11-20 — Fase 1, paso 2b: catálogo SCIAN y versión de `codigo_act`

**Qué se hizo:** descarga de las estructuras del SCIAN 2023 y 2018 y de las tablas comparativas 2023-2018 a `../datos-originales/scian/`; extractor `src/scian.py` (pypdfium2) con pruebas sobre texto sintético; catálogos de clases de ambas versiones con candado de conteos; decisión de la versión de `codigo_act` por los códigos presentes; `.gitattributes` con LF. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Empezó el 2026-11-19. Commits sin push.

### Hallazgos

- **`../evidencia/` no tenía archivos del SCIAN.** La instrucción pedía inventariarlos y compararlos por SHA-256 con las descargas. Claude Code encontró 8 archivos (7 capturas y `bitacora.txt`), ninguno del SCIAN, y lo reportó en el plan; no hubo nada que comparar ni copiar.
- **Los PDF del INEGI están cifrados.** Claude Code encontró que los tres PDF tienen cifrado sin contraseña de apertura (AES-128 en la estructura 2023 y en las tablas comparativas; RC4 en la de 2018) y que sus permisos permiten extraer texto. Con pypdf, el AES exige instalar además `cryptography` o `pycryptodome`. Claude Code propuso pypdfium2, que lo resuelve con un solo paquete sin dependencias; Samuel lo aprobó.
- **En el PDF de 2018 la versión en inglés va primero.** Cada estructura trae tres secciones (español, inglés y español-inglés), y en la de 2018 la de inglés está antes que la de español. Si el extractor tomara el título de la primera aparición de cada código, el catálogo 2018 saldría en inglés. Claude Code lo detectó al revisar el texto de los PDF durante el plan; el extractor elige las páginas por su encabezado ("Versión en Español") y una prueba cubre el caso.
- **Con HEAD, el INEGI responde HTML.** Con `curl -I`, el servidor responde `text/html` para las tres URL; con GET entrega el PDF. Las descargas se validan sobre el archivo descargado (`%PDF-`, tipo de contenido, tamaño y SHA-256), no con HEAD.
- **En el PDF de 2018, pypdfium2 pega el pie de página al número de página.** En el punto de control del plan, Claude Code vio que pypdfium2 entrega líneas como "55 INEGI. Sistema de Clasificación…"; sin un ajuste, el parser las habría leído como una entrada del sector 55, que existe. El parser ignora las líneas que empiezan con un número de página opcional seguido de "INEGI.", y una prueba cubre el caso.
- **Doble marca "T" en el PDF oficial 2023.** La validación de títulos detectó que la subrama 23611 trae "Edificación residencialT T" (dos marcas, una pegada y otra con espacio) en el PDF oficial de 2023; en el de 2018 trae una sola. Claude Code se detuvo y propuso cambiar la limpieza (quitar todas las marcas "T" finales, con una prueba) en vez de relajar la validación; Samuel lo aprobó. Las demás validaciones de 2023 (jerarquía, códigos únicos y candado) ya pasaban.
- **El diccionario de datos dice SCIAN 2018, pero `codigo_act` usa el SCIAN 2023.** Los 931 códigos distintos existen en 2023; 31 de ellos (1,731 filas) no existen en 2018, por ejemplo 931412 (698 filas) y 519212 (290).
- **`nombre_act` conserva títulos del SCIAN 2018.** En las 30 clases del DENUE que mantienen el código pero cambiaron de título en 2023, `nombre_act` coincide con el título 2018 en 9,469 filas y con el 2023 en ninguna. Refuerza la regla de la sección a de `CLAUDE.md`: la descripción de cada clase sale del catálogo, no de `nombre_act`.
- **La hipótesis de `clee` del paso 2a solo se sostiene en parte.** En las 62,404 filas en que `clee` discrepa de `codigo_act`, su código no existe en ninguna de las dos versiones en el 21.11 % y solo existe en 2018 en el 2.11 %; en el 76.72 % existe en ambas versiones, así que esa diferencia no se explica por el cambio de versión.

### Precisiones al plan

Las redactó el chat de Claude del proyecto y Samuel las aprobó.

- Si falla una validación de títulos (mayúscula final, caracteres fuera de Latin-1 o título vacío), el error lista código y título de cada caso, y Claude Code se detiene a reportarlo en vez de relajar la validación.
- Aprobación explícita para instalar `pypdfium2==5.15.2` y agregarlo a `requirements.txt`.
- No versionar el catálogo de la otra versión (2018), con las razones que Claude Code dio en el plan.

## 2026-11-20 — Fase 1, paso 3: normalización de nombres para agrupar

**Qué se hizo:** función única de normalización `src/normalizacion.py` con pruebas sobre texto sintético; comparación de tres políticas de puntuación sobre los `nom_estab` reales (solo conteos); script de llaves y tamaños de grupo con resultados en `results/2026-11-20_fase1_normalizacion-nombres.json`; regla 2 de `CLAUDE.md` precisada; dos decisiones del 2026-11-18 registradas en `CLAUDE.md`; captura del paso 2b copiada a `docs/evidencia/`. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Commits sin push.

### Hallazgos

- **La política de puntuación casi no cambia los grupos.** `nom_estab` ya viene en mayúsculas y casi sin signos: no tiene minúsculas ASCII, apóstrofos, º, ª ni espacios no separables. Entre las tres variantes, el número de llaves cambia en 24 como máximo (de 319,876 a 319,900, sobre 320,440 nombres distintos). Claude Code lo midió antes de recomendar una variante.
- **Borrar el guion separa más nombres de los que junta.** Comparado con cambiarlo por espacio, borrarlo junta 110 llaves en 55, pero separa en 166 llaves lo que el espacio junta en 83. Con el punto pasa al revés: borrarlo junta 52 en 26 y separa solo 18 en 9. Por eso Claude Code recomendó la variante C (borrar el punto y el apóstrofo, y cambiar el resto por espacio) y no la B (borrarlo todo).
- **Con NFKD, el acento suelto ´ se vuelve espacio.** NFKD descompone ´ (U+00B4) en un espacio y un acento combinante, así que "D´LUPITA" queda como "D LUPITA", mientras que "D'LUPITA" pierde el apóstrofo y queda "DLUPITA" (nombres sintéticos). No hay ´ en `nom_estab`; una prueba documenta el comportamiento.
- **El baseline de diccionario no estaba en `CLAUDE.md`.** La instrucción del paso 3 mencionaba un "baseline de diccionario", pero `CLAUDE.md` solo nombraba el baseline TF-IDF + regresión logística. Claude Code lo señaló en el plan y no lo agregó por su cuenta, porque agregar un baseline es un CAMBIO DE DISEÑO. Samuel lo había aprobado el 2026-11-18, junto con la regla de reportar las filas de train excluidas por compartir llave con validación o test, que también faltaba. El chat de Claude del proyecto redactó los dos textos y Samuel los aprobó; quedaron en `CLAUDE.md` (sección a y regla 7 de la sección c) marcados como "CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-18; registrado el 2026-11-20".

## 2026-11-20 — Fase 1, paso 3b: candados de configuración y ajustes de reglas

**Qué se hizo:** candados en `.claude/settings.json` (reglas `deny` para `git push` y para `git add .`, `-A`, `--all` y `-f`, en Bash y en PowerShell) y prueba de que bloquean; hook `pre-push` local que rechaza el push desde Claude Code, con su prueba; reglas nuevas en `CLAUDE.md` sobre el push, los candados y el modo automático; precisiones de las reglas 2, 6 y 7; regla de privacidad de nombres en la sección d. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Commits sin push.

### Hallazgos

- **La frase de la regla 2 contradecía al baseline de diccionario.** Al registrar ese baseline en el paso 3, Claude Code vio que la frase aprobada en el plan ("los modelos reciben el `nom_estab` original") lo contradecía, porque el baseline de diccionario usa la llave normalizada. La ajustó a "los modelos Qwen reciben…" y lo reportó. En el paso 3b, el chat de Claude del proyecto redactó la frase final, que además fija que el baseline TF-IDF recibe el `nom_estab` original, y Samuel la aprobó.
- **La regla 7 chocaba con la privacidad.** En el reporte del paso 3, Claude Code señaló que reportar "las 20 llaves que más filas excluyen" pondría nombres de negocios en `results/`, que se versiona en un repo público, contra la sección d (un `nom_estab` puede contener nombres de personas). Samuel aprobó como CAMBIO DE DISEÑO (privacidad) que esas llaves se le muestren solo a él, en la terminal o en un archivo de `data/`, y que en `results/` y en git solo vayan sus conteos y tamaños.
- **Una regla `deny` de prefijo no cubre todas las formas de un comando.** La documentación de Claude Code dice que `Bash(git push *)` no detiene `git -C . push` ni `sh -c '…'`. Claude Code agregó reglas para las opciones globales de git antes del subcomando y para cada bandera de `git add` en cualquier posición, y dejó documentado lo que no cubren: `bash -c`, programas que llamen a git y banderas combinadas como `-fA`. La prueba del hook lo confirmó: `bash -c 'git push --dry-run …'` pasó las reglas `deny`, y solo lo detuvo el hook `pre-push` que Claude Code propuso como segundo candado y Samuel aprobó.
- **Algunas frases del repo coinciden con nombres reales.** Al revisar los archivos versionados contra las llaves de `nom_estab`, sin imprimir nombres (solo conteos, líneas y posiciones), Claude Code encontró coincidencias de 3 o más palabras en títulos del catálogo SCIAN, en nombres inventados de las pruebas y en dos frases de la bitácora y del registro. Ninguna se copió del DENUE. Claude Code propuso aclarar en la sección d que esas coincidencias no cuentan como copia, y Samuel lo aprobó.
