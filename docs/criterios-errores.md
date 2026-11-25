# Criterios para clasificar los errores del 4B en test (Fase 4, paso 18)

Versión 1. Se escribió y tuvo su commit antes de clasificar cualquier fila. No contiene nombres de negocios.

## Qué se clasifica

- **Las filas:** los 50 errores de clase del modelo `4b-lr2e-4-r32` en test, sorteados en el paso 17 entre 568 errores con la convención del pre-registro (`evals/fase4_preregistro.json`, `errores_para_analisis`).
- **El archivo:** `data/fase4/errores_4b-lr2e-4-r32_test.jsonl`, SHA-256 `8958bcfdcff178fe981026b8080af8856f0100a62d7b6b682d48049a4cb3d4ad`, registrado en `results/2026-11-25_fase4_evaluacion-final.json`. Git lo ignora porque lleva nombres.
- **La posición:** cada fila se identifica por su posición, de 1 a 50 en orden de `int(id)`. La posición es la llave del CSV que revisa Samuel. Los ids no salen de la laptop.

## Quién clasifica (D7)

- **El procedimiento:** CAMBIO DE DISEÑO aprobado por Samuel el 2026-11-25 (D7). Claude Code propone un tipo por fila con estos criterios, y Samuel confirma o cambia cada fila.
- **Sesgo de anclaje:** Samuel ve la propuesta antes de decidir. El resultado mide la clasificación final después de esa revisión, no una clasificación independiente de Samuel. Por eso el conteo registra también la matriz propuesto → final.

## Qué mira Claude Code

- **Solo estos datos de la fila:**
  - el nombre (`nom_estab`);
  - la clase del DENUE (`codigo_act`), con su título del catálogo SCIAN 2023;
  - la respuesta del modelo, con su categoría de la regla 5 y su título;
  - el nivel más fino acertado y si acierta el sector;
  - las tres marcas de subgrupo de la regla 8.
- **Además,** su conocimiento general del español y de los giros comerciales.
- **No busca** ningún nombre en internet ni en otras fuentes.
- **Lee** una vista sin ids con su herramienta de lectura de archivos. No imprime nombres en la terminal.

## Tipos y precedencia

Los tipos se revisan en este orden, y la fila recibe el primero que aplica. El nombre corto de cada tipo es el texto exacto que va en las columnas `tipo_propuesto` y `tipo_samuel`.

| Orden | Tipo | Se decide | Aplica si |
|---|---|---|---|
| a | `codigo_inexistente` | automático | la `categoria_regla5` de la respuesta no es `codigo_existente`: formato inválido o código válido que no está en el catálogo |
| b | `clases_hermanas` | automático | la respuesta es un código existente, distinto de la clase del DENUE, con los mismos 4 primeros dígitos (la rama del SCIAN) |
| c | `etiqueta_dudosa` | juicio | el nombre indica con claridad una actividad, la clase del DENUE no le corresponde, y la respuesta del modelo es más plausible que esa clase para esa actividad |
| d | `nombre_ambiguo` | juicio | el nombre no basta para saber la actividad |
| — | `otro` | juicio | no aplica ninguno de los anteriores; la razón dice qué pasó |

Notas por tipo:

- **a.** El nombre del tipo viene de la guía del proyecto. Incluye el formato inválido porque esa respuesta tampoco está en el catálogo.
- **b.** El script la cruza con la columna `nivel_mas_fino_acertado`, que debe ser `rama` o `subrama`. Si no coinciden, se detiene.
- **c.** Que el nombre indique una actividad "con claridad" quiere decir una de dos cosas:
  - el giro se lee en las palabras del nombre;
  - el nombre es una marca conocida en todo México y con un solo giro. En ese caso la razón lo dice, para que Samuel lo revise.

  La respuesta no tiene que ser exacta: basta que sea más plausible que la clase del DENUE.
- **d.** Situaciones típicas:
  - el nombre de una persona, sin giro;
  - una marca o un nombre de fantasía, sin giro;
  - un nombre genérico, o con la frase de la regla 8 (la marca `sin_nombre`);
  - siglas;
  - palabras que sirven a varios giros;
  - un nombre que indica varias actividades.
- **otro.** Casos típicos:
  1. El nombre es claro, la clase del DENUE le corresponde y el modelo se equivocó. La razón empieza con "Error del modelo".
  2. El nombre es claro y no le corresponden ni la clase del DENUE ni la respuesta.

Desempates:

1. Si Claude Code duda entre c y d, elige d, porque c exige claridad.
2. Si el nombre indica varias actividades y la clase del DENUE corresponde a una de ellas, es d y no c: el nombre no dice cuál es la actividad principal.
3. Si la actividad del nombre cabe razonablemente en las dos clases, la del DENUE y la de la respuesta, en ramas distintas, es d: el nombre no basta para elegir entre ellas.

## Razón

- Tiene 15 palabras como máximo, contadas por espacios.
- No copia el nombre ni partes que identifiquen al negocio. Describe la actividad con palabras genéricas y nunca menciona a una persona.
- El script rechaza una razón que contenga la llave normalizada de su fila.

## Candidatos para la charla

- La columna `candidato_charla` lleva "sí" en hasta 6 filas, cada una de un tipo distinto. Como hay 5 tipos, salen 5 como máximo.
- Solo se marcan filas con un nombre comercial, no el de una persona física.
- Es solo una sugerencia: los ejemplos de la charla los elige Samuel a mano (CLAUDE.md, sección d).

## Revisión de Samuel

- **El CSV:** `data/fase4/errores_4b_para_revision.csv`, en UTF-8 con BOM, ignorado por git.
  - Lleva `posicion` en lugar de `id`, más las demás columnas del CSV del paso 17.
  - Se le agregan `tipo_propuesto`, `razon`, `tipo_samuel` (vacía), `nota_samuel` (vacía) y `candidato_charla`.
- **`tipo_samuel`:** vacío confirma la propuesta. Si no está vacío, lleva una de las 5 etiquetas. Se aceptan mayúsculas, acentos y espacios en lugar de `_`.
- **Los tipos a y b** salen de los códigos; no son juicios. Samuel no los asigna ni los cambia. Si una fila automática merece otra lectura, Samuel la anota en `nota_samuel`. El conteo se detiene si esto no se cumple.
- **`posicion` y `tipo_propuesto` no se editan.** Si Sheets reordena las filas no pasa nada, porque el conteo usa la posición.
- **Al terminar:**
  1. Samuel exporta la hoja como CSV a `data/fase4/errores_4b_revisado.csv`.
  2. Después borra la hoja de Google Drive, porque lleva nombres.

## Conteo

- En la sesión siguiente se corre `src/errores_fase4.py --contar` con el CSV revisado.
- El tipo final de cada fila es `tipo_samuel` si no está vacío; si lo está, es `tipo_propuesto`.
- El conteo escribe `results/<fecha>_fase4_errores-por-tipo.json`, sin nombres, llaves ni ids:
  - las filas por tipo;
  - cuántas aciertan el sector;
  - cuántas cambió Samuel;
  - la matriz propuesto → final;
  - la autoría y la nota de la D7.
- `nota_samuel` nunca sale de `data/`.

## Autoría

- El chat de Claude del proyecto propuso el paso y los tipos; Samuel lo aprobó.
- Samuel aprobó la D7 y el plan del paso, con sus ajustes. Uno de ellos: la posición en lugar del id en el CSV que sube a Google Sheets.
- Claude Code redactó estos criterios y propone el tipo de cada fila. Suya es también la precisión de que Samuel no cambia los tipos automáticos.
- Samuel confirma o cambia cada fila.
