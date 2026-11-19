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
