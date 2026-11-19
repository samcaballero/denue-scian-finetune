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

- **"% códigos inexistentes" no tenía denominador definido.** Al redactar en el plan la regla de métricas del clasificador, Claude detectó que no estaba definido si el porcentaje se calcula sobre el total de respuestas o solo sobre las de formato válido, y lo preguntó antes de ejecutar nada. Claude Code propuso calcularlo sobre el total de respuestas y Samuel lo aprobó; el texto final de la regla (`CLAUDE.md`, sección c, regla 5) lo redactó el chat de Claude del proyecto y Samuel lo aprobó: tres categorías que suman 100%, formato válido con `^[0-9]{6}$` en dígitos ASCII, accuracy sobre el total y conteos de las tres categorías en cada `results/*.json`.
