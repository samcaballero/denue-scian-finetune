# CLAUDE.md — denue-scian-finetune

Guía para Claude Code en este repositorio. El repo será público.

## a. Proyecto y entorno

- Fine-tuning con QLoRA de `Qwen/Qwen3-0.6B` y `Qwen/Qwen3-4B` (versiones instruct, no `-Base`; **siempre con `enable_thinking=False`**) sobre el DENUE del INEGI (CDMX, edición 2026/05).
- Dos tareas: (1) nombre de negocio → clase SCIAN de 6 dígitos; (2) dirección sucia → JSON estructurado. Baseline obligatorio: TF-IDF de n-gramas de caracteres + regresión logística.
- Clasificador (tarea 1): el modelo responde solo con el código SCIAN de 6 dígitos, sin texto adicional. La descripción de la clase se obtiene del catálogo oficial del SCIAN (de la versión que use `codigo_act`), nunca del modelo.
- Proyecto para una charla técnica el 2026-11-26. El entrenamiento corre en Kaggle (GPU T4) con Unsloth y se lanza con la CLI de Kaggle desde la laptop.

Entorno local: Windows 11, Python 3.12.10, sin GPU, 16 GB de RAM. La GPU vive en Kaggle (T4; 30 h de GPU por semana).

- Claude Code ejecuta sus comandos en Git Bash (sintaxis POSIX).
- La terminal de Samuel es PowerShell 5.1: los comandos que se le den para su terminal no usan `&&` ni `||` (usar `;` o `if ($?) { ... }`).
- Todo comando de Python usa `.venv/Scripts/python` (el intérprete del entorno virtual del proyecto), sin activar el entorno. Ejemplos: `.venv/Scripts/python -m pip list`, `.venv/Scripts/python -m pytest`.

## b. Rutas y datos

- CSV original: `../datos-originales/conjunto_de_datos/denue_inegi_09_.csv`
  - Fuera del repo y **SOLO LECTURA**. Pesa ~248 MB (253,929 KB).
  - Codificación **latin-1** (ISO-8859-1), no UTF-8 (verificado el 2026-11-18 en los primeros 300 KB).
  - Leer siempre con `usecols` y `dtype=str` para cuidar la RAM:
    `pd.read_csv(RUTA, usecols=[...], dtype=str, encoding="latin-1")`
  - Columnas clave: `nom_estab` (nombre del negocio), `codigo_act` (clase SCIAN), `cod_postal`.
  - Las columnas de domicilio para la tarea 2 se definirán en la Fase 1 con el diccionario de datos.
  - Pendiente: verificar en los metadatos qué versión del SCIAN usa `codigo_act` (`metadatos_denue.txt` no la menciona).
- También de solo lectura: `../datos-originales/diccionario_de_datos/denue_diccionario_de_datos.csv` y `../datos-originales/metadatos/metadatos_denue.txt`.
- Datos derivados: `data/` (ignorado por git, salvo `data/.gitkeep` y `data/manifest.json`).
- Salidas descargadas de Kaggle: `kaggle-output/` (ignorado por git).
- Entorno de Python del proyecto: `.venv/` dentro del repo (ignorado por git), creado el 2026-11-18 con Python 3.12.10, con dependencias fijadas en `requirements.txt`. Solo para trabajo en CPU: sin torch, transformers ni unsloth.

## c. Reglas de evaluación (no negociables)

1. **Semilla fija 42** en todo: `random`, `numpy`, scikit-learn, `torch` y la generación de splits.
2. **Split por grupo:** todas las filas con el mismo nombre normalizado (mayúsculas, sin acentos, sin puntuación, espacios colapsados) van al mismo split. La normalización exacta se implementa una sola vez en `src/` y se reutiliza en todo el proyecto. Direcciones: split por código postal.
3. **Test congelado:** al crearse se guarda su hash SHA-256 en `data/manifest.json` y no se vuelve a leer, filtrar ni inspeccionar hasta la evaluación final (Fase 4). Lo mismo aplica a los conjuntos reservados solo para test: el hard set de nombres ambiguos, las 100 direcciones escritas a mano y las dos familias de ruido reservadas; todos se registran con su hash en `data/manifest.json`. Toda decisión se toma con validación.
4. **Toda métrica se escribe en `results/*.json`** con fecha, commit de git, modelo, configuración, semilla, split, hash del split usado (el de `data/manifest.json`), tamaño de muestra y, si hubo entrenamiento, horas de GPU consumidas. Ningún número se reporta si no está en `results/`.
5. **Respuestas y métricas del clasificador:** cada respuesta cae en una de tres categorías: formato inválido, código válido inexistente o código válido existente. Formato válido: tras quitar espacios y saltos de línea al inicio y al final, la respuesta coincide con `^[0-9]{6}$` (dígitos ASCII; no usar `\d`, que acepta dígitos Unicode). Código inexistente: formato válido, pero el código no está en el catálogo oficial del SCIAN en la versión fijada para el proyecto (registrada en `data/manifest.json`). Se reportan "% formato válido" y "% códigos inexistentes", ambos sobre el total de respuestas; las tres categorías suman 100%. La accuracy también se calcula sobre el total: formato inválido y código inexistente cuentan como error. Cada `results/*.json` guarda los conteos de las tres categorías, no solo los porcentajes.

## d. Seguridad

- Nunca leer, imprimir ni copiar `~/.kaggle/access_token` ni ninguna otra llave.
- Nunca agregar a git datos, llaves, `.env` ni modelos. No forzar con `git add -f` lo que `.gitignore` excluye.
- Nunca modificar `../datos-originales/`.
- Columnas sensibles del DENUE:
  - `telefono`, `correoelec`, `www` (datos de contacto): nunca se cargan (quedan fuera de `usecols`) ni se muestran en ejemplos, notebooks, resultados o diapositivas.
  - `raz_social`: no se carga salvo decisión explícita de Samuel, porque puede contener nombres de personas físicas.
- Antes de agregar capturas o textos al repo, revisar que no muestren llaves, tokens, correos, teléfonos ni contraseñas.
- `docs/sesiones/`: sus archivos solo se versionan (se agregan a git) después de una revisión de datos sensibles (correos, llaves, datos personales) aprobada por Samuel. En los commits, los archivos se agregan por nombre, nunca con `git add .` ni `git add -A`.

## e. Forma de trabajo

- Hablar con Samuel en español y explicar cada término técnico entre paréntesis la primera vez que aparezca.
- Plan numerado antes de cambios grandes; esperar aprobación.
- **CAMBIO DE DISEÑO:** cualquier cambio en modelos, tareas, splits, métricas o reglas de evaluación se marca con la etiqueta "CAMBIO DE DISEÑO", se explica (qué cambia, por qué y qué afecta) y espera la aprobación explícita de Samuel antes de ejecutarse. Reordenar tareas está permitido, avisando.
- Pedir permiso antes de instalar paquetes, borrar archivos, hacer commit o push.
- Nunca lanzar corridas en Kaggle (`kaggle kernels push`) sin aprobación de Samuel; antes de pedirla, estimar las horas de GPU.
- Commits pequeños, con mensajes en español.
- Registrar en `docs/claude-code-log.md`, con fecha, cada momento notable: un bug que Claude encontró, un bug que Claude introdujo o una decisión que Samuel corrigió.

## f. Mapa de carpetas

```text
denue-scian-finetune/
├── CLAUDE.md             # esta guía
├── README.md             # descripción pública y crédito al INEGI
├── LICENSE               # Apache-2.0
├── data/                 # datos derivados (ignorado; solo se suben .gitkeep y manifest.json)
├── src/                  # código Python del proyecto
├── notebooks/            # notebooks (entrenamiento en Kaggle)
├── evals/                # código y casos de evaluación
├── results/              # métricas en JSON: única fuente de números reportables
├── kaggle-output/        # salidas descargadas de Kaggle (ignorado; se crea al descargar)
└── docs/
    ├── bitacora.md         # bitácora del proyecto, con un encabezado por fecha
    ├── claude-code-log.md  # momentos notables del trabajo con Claude Code
    ├── evidencia/          # capturas de pantalla (revisadas antes de agregarse)
    └── sesiones/           # sesiones importantes de Claude Code guardadas con /export
```

## g. Crédito y licencia

- Datos: "Fuente: INEGI, DENUE" (Directorio Estadístico Nacional de Unidades Económicas), bajo los Términos de Libre Uso de la Información del INEGI.
- Código: licencia Apache-2.0 (`LICENSE`).
