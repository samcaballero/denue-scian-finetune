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

## 2026-11-20 — Fase 1, paso 4: análisis exploratorio para el clasificador

**Qué se hizo:** script `src/exploracion_clasificador.py` con pruebas sobre datos sintéticos; conteos de clases, nombres problemáticos, techo de ambigüedad, estimación de la regla 7 y topes por clase en `results/2026-11-20_fase1_exploracion-clasificador.json`; cuatro figuras para la charla en `docs/figuras/`; archivo local de revisión en `data/` (ignorado por git). Solo mide y propone: no filtra, no congela y no cambia `CLAUDE.md`. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Commits sin push. Al aprobar el plan, Samuel pidió además contar, en los grupos de 100 o más filas y en el aporte a la regla 7, los grupos con la frase "SIN NOMBRE" y los genéricos, y agregar `docs/figuras/` al mapa de carpetas de `CLAUDE.md`, el único cambio a `CLAUDE.md` del paso.

### Hallazgos

- **Los negocios sin nombre comercial no tienen llave vacía: llevan una descripción y la frase "SIN NOMBRE".** La instrucción pedía proponer marcadores de "sin nombre" tras revisar las llaves más frecuentes, sin imprimirlas. Claude Code comparó las llaves con una lista de marcadores escrita a mano y contó cuántas llaves contienen cada frase, sin mostrar ninguna. Ninguna llave es solo un marcador, pero 43,979 filas (9.51 % de las filas con nombre) tienen "SIN NOMBRE" después de una descripción de la actividad, casi siempre al final. No son filas sin información: su entrada es una descripción escrita en lugar del nombre, y en 2,191 filas esa descripción es igual al título de su clase. Por eso Claude Code propuso tratarlas como un subgrupo que se mide aparte, y no como filas vacías.
- **El catálogo versionado solo trae clases.** Los conteos y la figura por sector necesitan los títulos de los sectores. Claude Code los tomó de la estructura oficial 2023 con el extractor `src/scian.py`, que verifica el hash del PDF y el candado de conteos, y comprobó que sus 1,086 clases son idénticas a las de `data/scian_2023_clases.csv`, en vez de escribir los títulos a mano.

## 2026-11-21 — Fase 1, paso 5: validación y test congelados

**Qué se hizo:** regla 8 nueva y regla 2 precisada en `CLAUDE.md`; módulo único de subgrupos `src/subgrupos.py` con el vocabulario de los genéricos versionado; split del clasificador `src/splits.py` con semilla 42, punto de control con solo conteos y, tras la aprobación de Samuel, test y validación congelados con su hash en `data/manifest.json`. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Empezó el 2026-11-20. Commits sin push.

### Hallazgos

- **La regla 2 contradecía al procedimiento del split.** Decía que todas las filas de una llave van al mismo split, pero validación y test se sacan fila por fila y las demás filas de sus llaves se excluyen (la regla 7 ya lo suponía). Claude Code lo marcó en el plan como CAMBIO DE DISEÑO y propuso el texto nuevo; Samuel lo aprobó.
- **El vocabulario en `.txt` habría cambiado de hash entre Windows y Kaggle.** `.gitattributes` solo fija LF para `*.csv`, `*.jsonl` y `*.json`; con `core.autocrlf=true`, un `.txt` versionado se escribe en Windows con CRLF. Claude Code propuso guardarlo como CSV para heredar la regla de LF.
- **El CSV no viene ordenado por `id`, y los `id` tienen de 5 a 8 dígitos.** Ordenarlos como texto daría otro orden; el orden canónico usa el valor entero y una prueba cubre el caso.
- **La marca de igual a título usa `nombre_act`, que describe la etiqueta.** Las marcas van en los splits solo para reportar; Claude Code propuso dejar escrito en la regla 8 que nunca son entrada de un modelo.
- **Con el split por nombre, el baseline de diccionario siempre cae en la clase más frecuente global en validación y test,** porque ninguna llave de esos conjuntos está en train. No es un error: es lo que mide el experimento de split aleatorio contra split por nombre.
- **`git check-ignore -v` devuelve 0 también cuando coincide una excepción.** Al comprobar la excepción del vocabulario en `.gitignore`, Claude Code vio que `git check-ignore -v` imprime la regla `!data/scian_2023_vocabulario.csv` y termina con código 0, como si el archivo estuviera ignorado; sin `-v`, el código 1 dice que no lo está. La función `esta_ignorado()` de `src/splits.py` usa `-q`, y una prueba cubre el caso del vocabulario.
- **Bug de Claude Code: la revisión de privacidad iba después de escribir los splits.** En el primer borrador de `src/splits.py`, el resultado se revisaba contra las llaves del DENUE al final, después de escribir los splits y el manifest; un fallo habría dejado conjuntos congelados sin su resultado. Claude Code lo corrigió antes del commit: los hashes se calculan en memoria, el resultado se revisa antes de escribir nada y la revisión corre también en el punto de control.
- **Validación no tiene la distribución natural.** En el punto de control, Claude Code vio que el 5.9 % de las filas de validación tiene la frase "SIN NOMBRE" y el 7.7 % son genéricas, contra el 9.51 % y el 10.84 % del universo (en test, 9.15 % y 10.55 %). La causa es el procedimiento: validación solo toma filas cuya llave no está en test, y test ya tocó casi todos los grupos grandes, que concentran esas marcas. Así, las decisiones que se tomen con validación se basan en una mezcla algo distinta de la del test. Claude Code lo señaló sin proponer cambios; Samuel lo aceptó y decidió no cambiar el procedimiento ni volver a sortear.

### Precisiones al plan

Las aprobó Samuel al aprobar el plan.

- "Con y sin la frase" son dos subconjuntos (filas con la frase y filas sin ella), igual que genéricas y no genéricas; confirma la interpretación que Claude Code propuso en el plan.
- La entrada del vocabulario en `data/manifest.json` la escribe el código (hash, tamaño y conteo calculados), no se copia a mano; el plan proponía registrarla a mano, como en el paso 2b.

## 2026-11-21 — Fase 2, paso 6a: muestra de prueba y dataset privado en Kaggle

**Qué se hizo:** candado de Kaggle en `.claude/settings.json` con su prueba (commit `f3e0b7d`); script `src/muestra_humo.py` con pruebas sobre datos sintéticos (commit `90bfc11`); muestra de 2,000 y 200 filas en una carpeta fuera del repo, con sus hashes en `data/manifest.json`; reglas de Kaggle en la sección e de `CLAUDE.md`. Samuel subió el dataset privado desde PowerShell y registró la cuota de GPU (ver la bitácora). La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Commits sin push.

### Proceso

- **El plan se aprobó sin querer antes de la revisión del chat.** Samuel aprobó el plan antes de que el chat de Claude del proyecto lo revisara. Claude Code siguió en modo automático y se detuvo en el punto de control, como decía el plan, sin escribir la carpeta ni tocar el manifest. El chat revisó después, y Samuel aprobó el plan, el punto de control y los tres cambios de Claude Code al plan.
- **El candado de Kaggle lo propuso el chat de Claude del proyecto y Samuel lo aprobó.** Motivo: con el modo automático, todo lo que saca algo de la laptop (subir datos o lanzar corridas que gastan la cuota de GPU) debe tener un candado técnico, igual que `git push`, y no solo la regla escrita de `CLAUDE.md`. Desde ahora Samuel ejecuta desde PowerShell `kaggle datasets create`, `kaggle datasets version` y `kaggle kernels push`. Claude Code escribió las reglas y las probó contra una carpeta y un dataset que no existen: rechazaron los 13 intentos.
- **Regla de la API de Kaggle.** Claude Code señaló que las reglas `deny` solo revisan el texto del comando: un script de Python que llame a la API de Kaggle no queda cubierto. Propuso mencionar los candados de Kaggle en la sección e de `CLAUDE.md`; el chat redactó la regla ("Claude Code no escribe ni ejecuta scripts que llamen a la API de Kaggle para crear, versionar o borrar datasets ni para lanzar kernels") y Samuel la aprobó.

### Cambios de Claude Code al plan

Claude Code los hizo al implementar y los reportó en el punto de control; Samuel los aprobó.

- **Candado más estricto.** El plan decía detenerse solo si un archivo de la carpeta tenía el hash de test e imprimir cualquier otra coincidencia; el script se detiene ante cualquier coincidencia con un hash del manifest.
- **`--verificar` contra lo registrado.** Además del candado, compara los hashes de la carpeta con los de la entrada `kaggle_datasets.denue_scian_humo` del manifest, para correrlo justo antes de subir.
- **Precondición del módulo versionado.** Ver el primer hallazgo.

### Hallazgos

- **El chequeo de árbol limpio no detecta un script nuevo sin commit.** Los scripts del proyecto comprueban el árbol con `git status --porcelain --untracked-files=no`, que no muestra los archivos sin versionar: un script nuevo, corrido antes de su commit, pasaría como "árbol limpio" y el manifest registraría un commit que no lo contiene. `src/muestra_humo.py` exige además `git ls-files --error-unmatch` y un status vacío del propio archivo antes de escribir. En los pasos anteriores no tuvo efecto: cada script tenía commit antes de escribir. Los scripts anteriores no se cambiaron.
- **La ayuda de `kaggle datasets create` nombra mal el archivo de metadatos.** Dice `datasets-metadata.json`; el código de la CLI 2.2.5 busca `dataset-metadata.json`.
- **`-t` (`--keep-tabular`) no hace nada en la CLI 2.2.5:** la línea que lo enviaría está comentada en el código.
- **La CLI lee el metadata con la codificación por omisión de Windows (cp1252), no en UTF-8.** Por eso `dataset-metadata.json` se escribe en ASCII, con los acentos escapados.
- **La privacidad la decide la bandera, no el JSON.** `kaggle datasets create` crea el dataset privado salvo que se pase `-u` (`--public`); un `isPrivate` en el metadata no lo lee.
- **La CLI sube todos los archivos de la carpeta** salvo los de metadatos; de ahí el candado de la lista permitida.
- **Alias de la CLI.** `d` (datasets), `k` (kernels) y `update` (para `kernels push`) hacen lo mismo que la forma larga; las reglas `deny` cubren las dos formas.
- **La API de Kaggle responde 403, no 404, a un dataset o kernel que no existe** (`status` y `files`). Antes de subir, el 403 no distingue entre "no existe" y "no hay permiso".
- **El `kaggle` que se usa es el global** (`Python312\Scripts\kaggle.exe`, versión 2.2.5), no uno del `.venv`, y no está en `requirements.txt`. Claude Code lo comprobó con `command -v kaggle`, `Get-Command kaggle -All` y `Test-Path .venv\Scripts\kaggle.exe`.

### Incidente sin efecto

- Antes de `--verificar`, Samuel escribió `kaggle datasets create` sin `-p` desde la raíz del repo; falló con "Metadata file not found" y no subió nada. Causa, según Samuel: el chat de Claude del proyecto listó los comandos con nombres cortos; desde ahora los da completos.
- Claude Code confirmó en el código de la CLI que, sin `-p`, se usa la carpeta actual (`os.getcwd()`): falló solo porque la raíz del repo no tiene `dataset-metadata.json`. Si la carpeta actual hubiera tenido ese archivo, habría subido sus archivos. Las reglas `deny` no cubren este caso: solo actúan sobre los comandos de Claude Code, y este lo escribió Samuel en su terminal.

## 2026-11-21 — Fase 2, paso 6b: kernel de la prueba de punta a punta

**Qué se hizo:** módulo único del prompt (`src/prompt_clasificador.py`), módulo del kernel (`src/humo_kaggle.py`) y generador (`src/construir_kernel.py`), con 82 pruebas nuevas; el kernel generado y su metadata quedaron en `notebooks/kaggle-humo/`. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Commits sin push; el lanzamiento lo hace Samuel.

### Proceso

- **Revisión previa.** Antes de planear, Claude Code revisó la CLI de Kaggle 2.2.5 instalada y el código de unsloth 2026.11.9 y unsloth_zoo 2026.11.8, leído en memoria desde PyPI sin instalarlo. También revisó la documentación de Unsloth, con un agente de lectura web, y los repos y plantillas de Hugging Face.
- **Ajustes al plan.** El chat de Claude del proyecto revisó el plan con ocho ajustes y Samuel lo aprobó. El chat recomendó la opción B (repos exactos de Qwen) y propuso las mediciones de lote; Samuel aprobó las dos. Claude Code propuso guardar el adaptador LoRA; Samuel lo aprobó.
- **Simulacro.** Las etapas que necesitan GPU no se pueden probar con pytest en la laptop. Claude Code escribió, fuera del repo, un simulacro que corre `_correr()` completo con torch, Unsloth, Hugging Face y tokenizador falsos y con nombres inventados, en seis modos: normal, espejo distinto, sin memoria, reloj pasado, generación que no acepta `None` y tokens de más.
  - El primer intento abortó en la comprobación de filas de más de 128 fichas, porque el tokenizador falso usaba un token por carácter. Era un error del simulacro, y confirmó que el kernel aborta antes de entrenar.

### Hallazgos

- **Unsloth cambia de repo y descarta la revisión.** Con `load_in_4bit=True`, `from_pretrained("Qwen/Qwen3-0.6B")` carga `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (`FLOAT_TO_INT_MAPPER`) e ignora `revision` con solo un aviso. Al 4B le pasa lo mismo. La documentación no lo menciona, ni menciona `use_exact_model_name`, que lo evita.
- **La fusión para el GGUF no respeta `use_exact_model_name`.** `merge_and_overwrite_lora` llama a `get_model_name(config._name_or_path, load_in_4bit=False)`, que convierte `Qwen/Qwen3-0.6B` en `unsloth/Qwen3-0.6B` (`MAP_TO_UNSLOTH_16bit`). Además descarga la rama main de ese repo, sin fijar la revisión. Por eso aplica el camino de verificación tensor por tensor que pidió el chat.
- **Cabeceras de los safetensors.** `Qwen/Qwen3-0.6B` y `unsloth/Qwen3-0.6B` tienen los mismos 310 tensores (nombre, tipo BF16 y forma). Solo el de Qwen trae `lm_head.weight` aparte (311 MB), y los dos `config.json` dicen `tie_word_embeddings: true`. Claude Code lo comprobó leyendo solo las cabeceras, con peticiones de rango.
- **La plantilla de chat de los repos de Unsloth está reescrita.** Cambia `[::-1]` y `startswith` por otras construcciones, aunque para un mensaje de usuario con su respuesta da el mismo texto. El token de relleno también cambia: `<|vision_pad|>` en Unsloth y `<|endoftext|>` en Qwen.
- **`kaggle kernels push` solo envía el archivo de código** (`kaggle_api_extended.py`, líneas 7242-7262). Por eso los módulos de `src/` van incrustados en el kernel.
- **La T4 se pide con `machine_shape` `NvidiaTeslaT4`**, que es "GPU T4 x2" (línea 7279; `cli.py:3039`). En kagglesdk, `enable_gpu` está marcado como DEPRECATED.
- **La documentación de Unsloth no tiene un procedimiento de instalación para Kaggle**, y `docs.unsloth.ai` redirige a `unsloth.ai/docs`.

### Bug que Claude Code introdujo y encontró

- **Entrenar en bf16 sobre la T4.** La primera versión de `src/humo_kaggle.py` decidía entre fp16 y bf16 con `torch.cuda.is_bf16_supported()`. En torch reciente esa función cuenta la emulación por omisión y puede responder `True` en una T4 (capacidad de cómputo 7.5): el kernel habría pedido bf16 en una GPU que no lo tiene nativo. Claude Code lo encontró al revisar el código, antes del commit. Ahora se decide por la capacidad de cómputo (bf16 nativo desde la 8.0), y una prueba falla si se vuelve a usar `is_bf16_supported`.

### Pendiente

- **Fase 5:** la guía indica que Bedrock Custom Model Import pide guardar el modelo con transformers 4.51.3, y aquí se usa 5.17.0. Lo registró el chat de Claude del proyecto; no se resuelve ahora.

## 2026-11-21 — Fase 2, paso 6c: salida de la prueba de punta a punta y métricas locales

**Qué se hizo:** descarga de la salida del kernel, revisión de nombres, resumen por etapa, métricas de la regla 5, rendimiento y estimación de horas de GPU. El código nuevo es `src/metricas_clasificador.py` y `src/evaluar_humo.py`, con 76 pruebas nuevas. El resultado es `results/2026-11-21_fase2_prueba-humo.json`, no reportable. La instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Claude Code propuso el plan; el chat lo revisó con tres ajustes y Samuel lo aprobó.

### Proceso

- **Primero la privacidad.** Claude Code comprobó con `git check-ignore -q` que la carpeta de descarga estaba ignorada antes de bajar nada. Después de bajar, verificó los hashes contra los que calculó el kernel y buscó los nombres de las muestras antes de mostrar cualquier contenido del log o de las mediciones.
- **Precisión de la regla 5.** Claude Code propuso que "quitar espacios y saltos de línea" significara solo el espacio ASCII, `\r` y `\n`. El chat de Claude del proyecto la revisó y Samuel la aprobó; quedó escrita en `CLAUDE.md`.
- **Un commit más de los dos del plan.** Al ver que el modelo respondía casi siempre lo mismo, Claude Code agregó la concentración de las respuestas al resultado, para que ese número quede en `results/` (regla 4). Fue un commit de código aparte (`73dea2d`), además de los dos aprobados, y se avisa aquí y en el reporte.
- **El texto de la bitácora y del registro se agregó con la herramienta de edición.** El primer intento, con un solo comando de shell, falló antes de escribir, igual que en el 6b.

### Hallazgos

- **El 0.6B de prueba colapsó a la clase mayoritaria.** Respondió 461110 en 199 de 200 filas y 461111 (inexistente) en la otra. La accuracy de 7.5 % es exactamente la frecuencia de 461110 en la muestra de validación. El formato quedó bien aprendido (100 % válido), pero no la clasificación. Conviene vigilarlo en el entrenamiento real.
- **pip bajó transformers.** La imagen de Kaggle traía transformers 5.20.2 y la versión fijada 5.17.0 la reemplazó. También cambiaron peft (0.21.2 a 0.21.3) y torchao (0.10.0 a 0.18.0). torch (2.11.0+cu130) quedó el de la imagen. En Kaggle se usa Python 3.13.16; en la laptop, 3.12.10.
- **En el 0.6B, el lote 64 no es más rápido que el 32** (50.4 frente a 50.9 ejemplos/s): la GPU ya va saturada. La VRAM pico quedó muy por debajo de los 15 GB de la T4 (4,886 MiB como máximo, en el 4B con lote 32). Lotes mayores caben, pero no se midieron.
- **Las duraciones no cuadran del todo.** El script duró 11.9 minutos, entre el `lastRunTime` de la CLI y el primer `COMPLETE` observado pasaron 14.3 minutos, y la página de Kaggle dice 16 minutos. Se registra la mayor (0.27 h) hasta que Samuel vea la cuota.
- **Avisos del log sin efecto:** Hugging Face advierte de peticiones sin token y de que `HF_HUB_ENABLE_HF_TRANSFER` está obsoleto, y nbconvert da dos `SyntaxWarning` al convertir el script a HTML. No hubo `Traceback` ni errores de memoria.
- **Revisión de duplicados pedida durante la corrida:** `docs/bitacora.md` y este registro no tenían secciones ni viñetas duplicadas. Lo repetido en la terminal era la salida de un comando que falló antes de escribir.

### Bug que Claude Code introdujo y encontró

- **El estado de una etapa quedaba pisado.** En `resumen_etapas`, la etapa `estado_final` pasaba el estado del kernel como dato con el nombre `estado`, que sobrescribía el campo `estado` ("ok" o "fallo") de la etapa. Claude Code lo vio en la primera corrida en seco, antes del commit, porque la salida decía `estado_final: completo` en vez de "ok". Ahora el dato se llama `estado_del_kernel`, `_etapa` rechaza los nombres reservados y una prueba lo comprueba.

## 2026-11-22 — Fase 1, paso 7: hard set del clasificador

**Qué se hizo:**
- Se generaron los candidatos, se escribió el CSV de curación y se leyó de vuelta con los ajustes del chat. Después se congeló el hard set (105 filas) y se escribió train candidato sin hard.
- El código está en `src/hard_set.py`, con 70 pruebas.
- Autoría: la instrucción la redactó el chat de Claude del proyecto y Samuel la aprobó. Claude Code propuso el plan; el chat lo revisó con cinco ajustes y Samuel los aprobó.
- Los commits están sin push.

### Proceso

- **Corte de conexión.** La conexión se cortó mientras Claude Code escribía `src/hard_set.py`. Con `git status` comprobó que no había quedado nada escrito, y escribió el archivo completo.
- **Palabras gramaticales en la fuente (ii).** Claude Code señaló en el plan que el vocabulario SCIAN incluye palabras cortas como DE, LA, EL, Y o CON, así que ningún nombre que las tuviera entraba en (ii).
  - El chat de Claude del proyecto propuso ignorar una lista cerrada de artículos, preposiciones, conjunciones y posesivos, y Samuel la aprobó.
  - Claude Code buscó qué otras palabras de clase cerrada pasaban `en_vocabulario` y encontró MEDIANTE, QUE, SOBRE, SU y SUS; el chat las aprobó.
  - Dejó fuera VIA, porque tiene contenido (vía = camino), y ESTOS, porque es un demostrativo, fuera de las categorías aprobadas.
- **Google Sheets.** Samuel no tiene Excel, y el chat pidió adaptar las instrucciones. Claude Code agregó, en un commit extra (`09ebed2`):
  - un candado de ruta para el CSV curado: dentro de `data/`, ignorado por git y con un nombre que termine en `-curado.csv`;
  - una prueba con lo que descarga Google: UTF-8 sin BOM, CRLF o LF, comillas, saltos de línea dentro de celdas y el nombre de archivo con " - Hoja 1".
- **Curación por una IA.** La hizo Gemini, el chat ajustó 23 filas y Samuel aprobó el CAMBIO DE DISEÑO sin revisar la curación fila por fila. El detalle está en `docs/bitacora.md`.
- **Meta de 105 y tamaño planeado.** Claude Code puso `N_HARD = 105` como meta de `leer` y `congelar`. Conservó `N_HARD_PLANEADO = 150` para el peor caso de la PARADA 1, para que esa parada se pueda reproducir tal como se aprobó.
- **Contradicción en `CLAUDE.md`.** La instrucción pedía cambiar solo la regla 3, pero la sección b seguía diciendo "150 filas curadas a mano por Samuel". Claude Code no la tocó: señaló la contradicción en el reporte y propuso el texto nuevo, que el chat aprobó.
- **Ids del DENUE en git.** El chat pidió guardar sus ajustes (id, incluir, tipo y motivo) en un archivo versionado.
  - Claude Code lo preparó en `evals/`, sin commit, y señaló en el reporte que eso subiría ids del DENUE a un repo que será público (sección d), aunque no lleva nombres.
  - El chat corrigió su instrucción: el archivo pasó a `data/`, ignorado por git, y el código guarda solo su SHA-256, que `congelar` exige.
  - El archivo nunca entró en un commit.
  - Registro: el riesgo lo detectó Claude Code y el chat corrigió su instrucción.
- **Ajustes verificados antes de aplicarse.** Los 23 ids estaban entre los 400 candidatos y en S en el CSV de la IA, y cada ajuste cambiaba algo; si no, el código se habría detenido.
- **Commits, con los archivos agregados por nombre:**
  - `0862de1`: regla 3, el hard set se trata como el test;
  - `dc8564f`: código y pruebas;
  - `09ebed2`: candado de ruta y prueba de Google Sheets;
  - `37f7553`: `CLAUDE.md` con el cambio del 2026-11-22;
  - `f0f3deb`: meta de 105 y ajustes;
  - `d748262`: manifest y `results/`;
  - y el de esta bitácora y este registro.

### Hallazgos

- **El vocabulario dejaba fuera casi una cuarta parte de la fuente (ii).** Sin ignorar ninguna palabra, (ii) tenía 85,286 llaves; con la lista original de palabras gramaticales, 111,587, y con la lista final, 111,708. Además, 4 llaves solo tienen palabras gramaticales.
- **La curación de la IA no tenía errores de formato.** Gemini entregó el CSV con sus 400 registros y ninguna fila con N traía tipo. La lectura no tuvo que tolerar nada fuera de lo previsto.

### Errores de Claude Code

- **El código pasado a Python por un heredoc de Bash perdió las barras invertidas.** Al agregar las pruebas, un `"\n"` se volvió un salto de línea real y la cadena de bytes `b"\xef\xbb\xbf"` se volvió el BOM escrito como texto. pytest falló al recolectar las pruebas, antes del commit, y Claude Code corrigió las dos líneas con la herramienta de edición. Igual que en el 6b y el 6c, el texto con barras invertidas o acentos se escribe con la herramienta de edición, no por la shell.
- **Lectura del hard set después de congelarlo.** Para comprobar que el manifest y el JSON de `results/` no traían ids, Claude Code leyó los ids de `hard.jsonl` recién congelado y los buscó en esos dos archivos.
  - No apareció ninguno, y solo se imprimieron conteos.
  - Fue un error: la regla 3 dice que un conjunto congelado no se vuelve a leer hasta la Fase 4. Además, la lectura no hacía falta, porque `congelar` ya revisa la privacidad del resultado y del manifest antes de escribir.
  - No cambió ninguna decisión ni ningún archivo.
  - El `sha256sum` de los dos archivos, justo después de escribirlos, solo calculó su hash, como ya hace `congelar` al releerlos.
  - En adelante, una revisión así se hace dentro de `congelar`, con los datos en memoria y antes de escribir.
