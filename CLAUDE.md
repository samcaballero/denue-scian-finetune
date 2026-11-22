# CLAUDE.md — denue-scian-finetune

Guía para Claude Code en este repositorio. El repo será público.

## a. Proyecto y entorno

- Fine-tuning con QLoRA de `Qwen/Qwen3-0.6B` y `Qwen/Qwen3-4B` (versiones instruct, no `-Base`; **siempre con `enable_thinking=False`**) sobre el DENUE del INEGI (CDMX, edición 2026/05).
- Dos tareas: (1) nombre de negocio → clase SCIAN de 6 dígitos; (2) dirección sucia → JSON estructurado. Baselines obligatorios: TF-IDF de n-gramas de caracteres + regresión logística, y baseline de diccionario (la clase más frecuente de cada nombre normalizado visto en train; si el nombre no se vio en train, la clase más frecuente global). (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-18; registrado el 2026-11-20)
- Clasificador (tarea 1): el modelo responde solo con el código SCIAN de 6 dígitos, sin texto adicional. La descripción de la clase se obtiene del catálogo oficial del SCIAN (de la versión que use `codigo_act`), nunca del modelo.
- Proyecto para una charla técnica el 2026-11-26. El entrenamiento corre en Kaggle (GPU T4) con Unsloth y se lanza con la CLI de Kaggle desde la laptop.

Entorno local: Windows 11, Python 3.12.10, sin GPU, 16 GB de RAM. La GPU vive en Kaggle (T4; 30 h de GPU por semana).

- Claude Code ejecuta sus comandos en Git Bash (sintaxis POSIX).
- La terminal de Samuel es PowerShell 5.1: los comandos que se le den para su terminal no usan `&&` ni `||` (usar `;` o `if ($?) { ... }`).
- Todo comando de Python usa `.venv/Scripts/python` (el intérprete del entorno virtual del proyecto), sin activar el entorno. Ejemplos: `.venv/Scripts/python -m pip list`, `.venv/Scripts/python -m pytest`.

## b. Rutas y datos

- CSV original: `../datos-originales/conjunto_de_datos/denue_inegi_09_.csv`
  - Fuera del repo y **SOLO LECTURA**. Pesa ~248 MB (253,929 KB).
  - Codificación **latin-1** (ISO-8859-1), no UTF-8. Verificado el 2026-11-19 en el archivo completo: no es UTF-8 válido y no tiene bytes entre 0x80 y 0x9F, así que latin-1 y cp1252 lo leen igual.
  - Las columnas del CSV se leen solo con el cargador único `cargar_denue()` de `src/denue.py`. El cargador verifica el SHA-256 contra `data/manifest.json`, rechaza `telefono`, `correoelec`, `www` y `raz_social`, y lee solo las columnas pedidas con `dtype=str` y `keep_default_na=False`: textos como "NA", "NULL" o "None" se conservan, y un campo vacío queda como `""`.
  - Columnas clave: `nom_estab` (nombre del negocio), `codigo_act` (clase SCIAN), `cod_postal`.
  - Las columnas de domicilio para la tarea 2 se definirán en la Fase 1 con el diccionario de datos.
  - Versión del SCIAN: **SCIAN 2023**, decidida el 2026-11-20 (paso 2b) por los códigos presentes y registrada en `data/manifest.json` (`scian_version`). Evidencia: los 931 códigos distintos de `codigo_act` existen en el SCIAN 2023, y 31 de ellos (1,731 filas) no existen en el SCIAN 2018; ver `results/2026-11-20_fase1_version-scian.json`. El diccionario de datos dice SCIAN 2018; los códigos lo contradicen.
  - `nombre_act` no siempre trae el título del SCIAN 2023: en 28 de las 30 clases del DENUE que conservaron su código pero cambiaron de título en 2023, trae el título 2018 (9,469 filas), y en 14 códigos (1,567 filas) no coincide con el título de ninguna de las dos versiones; ver `results/2026-11-20_fase1_version-scian.json`. La descripción de una clase sale siempre de `data/scian_2023_clases.csv`, nunca de `nombre_act`.
- También de solo lectura: `../datos-originales/diccionario_de_datos/denue_diccionario_de_datos.csv`, `../datos-originales/metadatos/metadatos_denue.txt` y `../datos-originales/scian/` (estructuras del SCIAN 2023 y 2018 y tablas comparativas 2023-2018, PDF descargados del INEGI el 2026-11-19; URL, SHA-256 y tamaño en `data/manifest.json`).
- Catálogo SCIAN: `data/scian_2023_clases.csv` (`codigo,titulo`; UTF-8; saltos de línea `\n`; ordenado por código), extraído de la estructura oficial con `src/scian.py`. Es la única fuente de los títulos de las clases y de la lista de códigos existentes de la regla 5. El de la otra versión queda en `data/` sin versionar, con su hash en el manifest.
- Vocabulario SCIAN: `data/scian_2023_vocabulario.csv` (`palabra`; UTF-8; saltos de línea `\n`; ordenado), con las palabras de los títulos de todos los niveles (sector a clase) de la estructura oficial 2023, normalizados con `normalizar_nombre()`; extraído con `src/subgrupos.py`, con su hash en el manifest. Es la única fuente del vocabulario de la marca de genérico (regla 8).
- Finales de línea: `.gitattributes` fija LF para `*.csv`, `*.jsonl` y `*.json`, y todo archivo con hash en el manifest se escribe con `\n`, para que su SHA-256 sea igual en Windows y en Kaggle.
- Datos derivados: `data/` (ignorado por git, salvo `data/.gitkeep`, `data/manifest.json`, `data/scian_2023_clases.csv` y `data/scian_2023_vocabulario.csv`).
- Splits del clasificador: `data/splits/` (ignorado por git), con ruta, SHA-256, semilla y procedimiento en `data/manifest.json` (`splits`).
  - Generados con `src/splits.py`: `test.jsonl` y `val.jsonl` congelados y `train_candidatos.jsonl` sin congelar.
  - Generados con `src/hard_set.py`: `hard.jsonl`, el hard set del clasificador (105 filas curadas por una IA (Gemini) con ajustes del chat de Claude del proyecto, cada una con su tipo, ambiguo o indirecto; congelado), y `train_candidatos_sin_hard.jsonl`, que es train candidato sin ninguna fila de las llaves del hard set (sin congelar; de él parte el tope por clase).
  - Generado con `src/tope_clase.py`: `train_final.jsonl`, el train de los modelos del clasificador y de los baselines. Sale de `train_candidatos_sin_hard.jsonl` con un tope de 1,000 filas por clase, sorteadas con semilla 42 (187,350 filas y 929 clases). Está congelado: no se regenera ni cambia, pero a diferencia de test y hard sí se lee para entrenar.
- Datasets de Kaggle: carpetas en `../kaggle-datasets/`, fuera del repo, con sus hashes en `data/manifest.json` (`kaggle_datasets`).
  - `denue-scian-clasificador` (dataset privado `samcaballero/denue-scian-clasificador`, armado con `src/paquete_clasificador.py`): `train_final.jsonl` y `val.jsonl` completos, solo con `id`, `nom_estab` y `codigo_act`. Su candado rechaza cualquier archivo con el hash de test, de hard o de otra entrada del manifest.
  - `denue-scian-humo`: la muestra desechable de la prueba de punta a punta (`src/muestra_humo.py`).
  - La subida la hace Samuel con `kaggle datasets create -p` y la ruta absoluta de la carpeta.
- Salidas descargadas de Kaggle: `kaggle-output/` (ignorado por git).
- Entorno de Python del proyecto: `.venv/` dentro del repo (ignorado por git), creado el 2026-11-18 con Python 3.12.10, con dependencias fijadas en `requirements.txt`. Solo para trabajo en CPU: sin torch, transformers ni unsloth.

## c. Reglas de evaluación (no negociables)

1. **Semilla fija 42** en todo: `random`, `numpy`, scikit-learn, `torch` y la generación de splits.
2. **Split por grupo:** ninguna llave normalizada aparece en dos conjuntos. En el clasificador, test y validación se sacan al azar fila por fila, y las demás filas de sus llaves no entran en ningún conjunto: se excluyen de train (regla 7); el procedimiento exacto está en `data/manifest.json` (`splits`). (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-20) La normalización se implementa una sola vez, en `normalizar_nombre()` de `src/normalizacion.py`, y se reutiliza en todo el proyecto. Pasos, en este orden: forma NFKD; quitar las marcas combinantes (acentos, diéresis y la tilde de la Ñ: Ñ→N, Ü→U; º y ª quedan como O y A); mayúsculas; borrar el punto y el apóstrofo (`'` y `’`) y cambiar por espacio cualquier otro carácter que no sea letra, número ni espacio (puntuación y símbolos); colapsar y recortar los espacios, incluido el no separable. Un nombre vacío o de pura puntuación da la llave vacía. La llave sirve para agrupar (splits y hard set) y es la entrada del baseline de diccionario. Los modelos Qwen y el baseline TF-IDF + regresión logística reciben el `nom_estab` original; cualquier preprocesamiento propio del TF-IDF se documenta en su configuración (precisión aprobada por Samuel el 2026-11-20). Direcciones: split por código postal.
3. **Test congelado:** al crearse se guarda su hash SHA-256 en `data/manifest.json` y no se vuelve a leer, filtrar ni inspeccionar hasta la evaluación final (Fase 4). Lo mismo aplica a los conjuntos reservados solo para test: el hard set de nombres ambiguos, las 100 direcciones escritas a mano y las dos familias de ruido reservadas; todos se registran con su hash en `data/manifest.json`. Toda decisión se toma con validación. El hard set del clasificador se trata como el test: no se usa para elegir modelos ni hiperparámetros, no sube a Kaggle hasta la Fase 4 y se evalúa una sola vez, junto con el test. (CAMBIO DE DISEÑO propuesto por el chat de Claude del proyecto y aprobado por Samuel el 2026-11-21) La curación de ese hard set no la hizo Samuel a mano: la hizo una IA (Gemini, en Google Sheets) con los criterios que definieron Samuel y el chat de Claude del proyecto, Samuel no la revisó fila por fila y el chat ajustó 23 filas con esos mismos criterios. Queda en 105 filas en vez de 150, sin ronda 2; el tipo indirecto no llega al mínimo de 50 filas y sus resultados se reportan con su intervalo y una advertencia. Sesgo reconocido: las filas las eligió un modelo de lenguaje con su propio conocimiento, sobre todo las de tipo indirecto. (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-22)
4. **Toda métrica se escribe en `results/*.json`** con fecha, commit de git, modelo, configuración, semilla, split, hash del split usado (el de `data/manifest.json`), tamaño de muestra y, si hubo entrenamiento, horas de GPU consumidas. Ningún número se reporta si no está en `results/`.
5. **Respuestas y métricas del clasificador:** cada respuesta cae en una de tres categorías: formato inválido, código válido inexistente o código válido existente. Formato válido: tras quitar al inicio y al final solo el espacio ASCII, `\r` y `\n` (un tabulador, un espacio no separable u otro espacio Unicode en las orillas deja la respuesta en formato inválido), la respuesta coincide con `^[0-9]{6}$` (solo dígitos ASCII; no usar `\d`, que acepta dígitos Unicode); precisión aprobada por Samuel el 2026-11-21. Código inexistente: formato válido, pero el código no está en el catálogo oficial del SCIAN en la versión fijada para el proyecto (registrada en `data/manifest.json`). Se reportan "% formato válido" y "% códigos inexistentes", ambos sobre el total de respuestas; las tres categorías suman 100%. La accuracy también se calcula sobre el total: formato inválido y código inexistente cuentan como error. Cada `results/*.json` guarda los conteos de las tres categorías, no solo los porcentajes.
6. **`clee` nunca es entrada ni variable de ningún modelo**, incluidos los baselines. Es un identificador que el INEGI asigna a cada establecimiento y que un negocio nuevo no tendría al momento de clasificarlo; la única entrada del clasificador es `nom_estab`. Además contiene la etiqueta: en 400,328 de 462,732 filas (86.51 %), los caracteres 6 a 11 de `clee` son iguales a `codigo_act` (medido el 2026-11-19; ver `results/2026-11-19_fase1_diagnostico-denue.json`).
7. **Filas de train excluidas por llave compartida:** antes de congelar validación y test, se reporta cuántas filas candidatas a train se excluyen por compartir llave normalizada con validación o test (total, % y las 20 llaves que más filas excluyen). Las 20 llaves se muestran solo a Samuel, en la terminal o en un archivo dentro de `data/` (ignorado por git); en `results/` y en git solo van sus conteos y tamaños, nunca las llaves (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-20, privacidad). Si la exclusión pasa de 10%, no se congela: se detiene y decide Samuel. El experimento de split aleatorio contra split por nombre se mide en validación con ambos baselines. (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-18; registrado el 2026-11-20)
8. **Población y subgrupos del clasificador:** las filas con nombre vacío (llave vacía) se excluyen de train, validación y test. Se mantienen las filas con la frase "SIN NOMBRE", las de llave igual al título de una clase o a `nombre_act` y las de nombre genérico. La métrica principal es la del test con la distribución natural: filas sacadas al azar, sin filtrar ni balancear. En validación y test se reporta además, por separado, la accuracy de las filas con la frase "SIN NOMBRE" y la de las filas sin ella, y la de las filas genéricas y la de las no genéricas. Las tres marcas de subgrupo se definen una sola vez, en `src/subgrupos.py`: "SIN NOMBRE" (la frase como palabras completas en cualquier parte de la llave); genérico (llave no vacía cuyas palabras, sin la frase "SIN NOMBRE", están todas en `data/scian_2023_vocabulario.csv`, aceptando la palabra con o sin S o ES final, con al menos una palabra de 4 letras o más); igual a título (llave igual al título normalizado de una clase del catálogo o a `nombre_act` normalizado de la misma fila). Las marcas solo sirven para reportar: nunca son entrada ni variable de un modelo, porque la de igual a título usa `nombre_act`, que describe la etiqueta. (CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-20)

## d. Seguridad

- Nunca leer, imprimir ni copiar `~/.kaggle/access_token` ni ninguna otra llave.
- Nunca agregar a git datos, llaves, `.env` ni modelos. Excepciones de datos, ambas del SCIAN de la versión fijada y con crédito al INEGI en `README.md`: el catálogo de clases (`data/scian_2023_clases.csv`), aprobado por Samuel en el paso 2b, y el vocabulario de sus títulos (`data/scian_2023_vocabulario.csv`), aprobado en el paso 5. No forzar con `git add -f` lo que `.gitignore` excluye.
- Nunca modificar `../datos-originales/`.
- Columnas sensibles del DENUE:
  - `telefono`, `correoelec`, `www` (datos de contacto): nunca se cargan (quedan fuera de `usecols`) ni se muestran en ejemplos, notebooks, resultados o diapositivas.
  - `raz_social`: no se carga salvo decisión explícita de Samuel, porque puede contener nombres de personas físicas.
- Antes de agregar capturas o textos al repo, revisar que no muestren llaves, tokens, correos, teléfonos ni contraseñas.
- `docs/sesiones/`: sus archivos solo se versionan (se agregan a git) después de una revisión de datos sensibles (correos, llaves, datos personales) aprobada por Samuel. En los commits, los archivos se agregan por nombre, nunca con `git add .` ni `git add -A`.
- Salvo que Samuel lo pida, Claude Code no imprime filas del DENUE en la terminal: solo conteos, códigos SCIAN y nombres de columnas.
- Ningún `nom_estab` ni llave normalizada se escribe en `results/` ni en archivos versionados; los ejemplos de nombres para la charla los elige Samuel a mano. Los nombres inventados de las pruebas y los títulos del catálogo SCIAN pueden coincidir por casualidad con un `nom_estab`; eso no cuenta como copia.

## e. Forma de trabajo

- Hablar con Samuel en español y explicar cada término técnico entre paréntesis la primera vez que aparezca.
- Plan numerado antes de cambios grandes; esperar aprobación.
- **CAMBIO DE DISEÑO:** cualquier cambio en modelos, tareas, splits, métricas o reglas de evaluación se marca con la etiqueta "CAMBIO DE DISEÑO", se explica (qué cambia, por qué y qué afecta) y espera la aprobación explícita de Samuel antes de ejecutarse. Reordenar tareas está permitido, avisando.
- Pedir permiso antes de instalar paquetes, borrar archivos o hacer commit.
- El push lo hace siempre Samuel desde PowerShell, después de que el chat de Claude del proyecto revisa el reporte.
- Candados en `.claude/settings.json` (versionado): reglas `deny` de permisos de Claude Code que rechazan `git push` y `git add .`, `git add -A`, `git add --all` y `git add -f`, en Bash y en PowerShell. No se modifican sin aprobación de Samuel.
- Candado local en esta laptop: el hook `.git/hooks/pre-push` rechaza el push cuando existe la variable `CLAUDECODE`, que definen las shells de Claude Code y no la terminal de Samuel. No viaja con el repo: un clon nuevo no lo tiene. No se modifica sin aprobación de Samuel.
- Candados de Kaggle en `.claude/settings.json`: reglas `deny` que rechazan `kaggle datasets create`, `kaggle datasets version`, `kaggle datasets delete` y `kaggle kernels push`, también con los alias de la CLI (`d`, `k`, `update`) y con cualquier prefijo (ruta completa, `kaggle.exe`, `python -m kaggle`), en Bash y en PowerShell. `kaggle datasets create`, `kaggle datasets version` y `kaggle kernels push` los ejecuta Samuel desde PowerShell. No se modifican sin aprobación de Samuel.
- Claude Code no escribe ni ejecuta scripts que llamen a la API de Kaggle para crear, versionar o borrar datasets ni para lanzar kernels: las reglas `deny` solo revisan el texto del comando y no cubren esa vía.
- El modo automático solo se usa con un plan ya aprobado.
- Nunca lanzar corridas en Kaggle (`kaggle kernels push`) sin aprobación de Samuel; antes de pedirla, estimar las horas de GPU.
- Commits pequeños, con mensajes en español.
- Registrar en `docs/claude-code-log.md`, con fecha, cada momento notable: un bug que Claude encontró, un bug que Claude introdujo o una decisión que Samuel corrigió.
- En `docs/bitacora.md` y `docs/claude-code-log.md` se distingue quién propuso, quién redactó y quién aprobó cada cosa: Claude Code (la sesión en la terminal), el chat de Claude del proyecto (un chat aparte en claude.ai que a veces redacta instrucciones o el texto final de reglas) o Samuel. Fórmula: "X propuso / redactó …; Samuel lo aprobó". Por defecto, las instrucciones que Samuel pega las redactó el chat de Claude del proyecto y Samuel las aprobó; si Samuel indica otra cosa, se registra lo que él diga.

## f. Mapa de carpetas

```text
denue-scian-finetune/
├── .claude/
│   └── settings.json     # candados de permisos de Claude Code (versionado)
├── CLAUDE.md             # esta guía
├── README.md             # descripción pública y crédito al INEGI
├── LICENSE               # Apache-2.0
├── pyproject.toml        # configuración de pytest
├── data/                 # datos derivados (ignorado; solo se suben .gitkeep, manifest.json, el catálogo y el vocabulario del SCIAN)
├── src/                  # código Python del proyecto
├── tests/                # pruebas con pytest (solo datos sintéticos, nunca datos reales)
├── notebooks/            # notebooks (entrenamiento en Kaggle)
├── evals/                # código y casos de evaluación
├── results/              # métricas en JSON: única fuente de números reportables
├── kaggle-output/        # salidas descargadas de Kaggle (ignorado; se crea al descargar)
└── docs/
    ├── bitacora.md         # bitácora del proyecto, con un encabezado por fecha
    ├── claude-code-log.md  # momentos notables del trabajo con Claude Code
    ├── evidencia/          # capturas de pantalla (revisadas antes de agregarse)
    ├── figuras/            # figuras para la charla generadas por scripts (sin nombres de negocio)
    └── sesiones/           # sesiones importantes de Claude Code guardadas con /export
```

## g. Crédito y licencia

- Datos: "Fuente: INEGI, DENUE" (Directorio Estadístico Nacional de Unidades Económicas), bajo los Términos de Libre Uso de la Información del INEGI.
- Código: licencia Apache-2.0 (`LICENSE`).
