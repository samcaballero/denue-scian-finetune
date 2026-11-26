# denue-scian-finetune

Proyecto para una charla técnica (26 de noviembre de 2026): fine-tuning con QLoRA de `Qwen/Qwen3-0.6B` y `Qwen/Qwen3-4B` sobre el DENUE del INEGI (Ciudad de México, edición 05/2026) para clasificar el nombre de un negocio en una clase SCIAN de 6 dígitos.

Los modelos se comparan contra dos baselines: TF-IDF con n-gramas de caracteres y regresión logística, y un diccionario (la clase más frecuente de cada nombre normalizado visto en train). También se miden los dos Qwen3 sin afinar, en 0-shot y few-shot. El entrenamiento corrió en Kaggle (una GPU T4) con Unsloth.

## Estado

- **Clasificación** (nombre de un negocio → clase SCIAN 2023 de 6 dígitos): terminada. La evaluación final en test y en el hard set se hizo una sola vez, el 25 de noviembre de 2026.
- **Extracción de direcciones** (dirección escrita de forma desordenada → JSON estructurado): no se hizo. Quedó fuera de alcance desde el 23 de noviembre de 2026.

## Resultados en test

Copiados tal cual de [`results/2026-11-25_fase4_tabla-final.md`](results/2026-11-25_fase4_tabla-final.md), que se genera desde `results/2026-11-25_fase4_evaluacion-final.json` (commit `0c0d552`, semilla 42, IC 95 % por bootstrap de 1,000 remuestreos). Porcentajes; pareadas en puntos porcentuales.

**Test (2,000 filas, distribución natural)**

| Modelo | Clase (IC 95 %) | Sector (IC 95 %) | Macro-F1 de sector | % formato inválido | % código inexistente | Clases distintas | Tolerante (IC 95 %) | % al tope |
|---|---|---|---:|---:|---:|---:|---|---:|
| diccionario | 8.3 (7.1–9.6) | 44.1 (42.0–46.2) | 0.034 | 0.0 | 0.0 | 1 | — | — |
| tfidf-lr-C10 | 69.5 (67.5–71.5) | 84.5 (83.0–86.0) | 0.678 | 0.0 | 0.0 | 260 | — | — |
| 06b-lr2e-4-r32 | 68.8 (66.8–70.9) | 84.4 (82.8–86.0) | 0.679 | 0.0 | 0.0 | 240 | 68.8 (66.8–70.9) | 0.0 |
| 4b-lr2e-4-r32 | 71.6 (69.6–73.5) | 86.1 (84.6–87.6) | 0.724 | 0.0 | 0.0 | 250 | 71.6 (69.6–73.5) | 0.0 |
| 06b-base-0shot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.000 | 78.0 | 21.9 | 1 | 0.0 (0.0–0.0) | 0.0 |
| 4b-base-0shot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.000 | 22.0 | 78.0 | 0 | 0.0 (0.0–0.0) | 0.0 |
| 06b-base-fewshot | 0.1 (0.0–0.1) | 2.5 (1.8–3.2) | 0.005 | 0.0 | 17.4 | 5 | 0.1 (0.0–0.1) | 0.0 |
| 4b-base-fewshot | 8.9 (7.7–10.2) | 23.7 (21.8–25.6) | 0.093 | 0.3 | 38.0 | 18 | 8.9 (7.7–10.2) | 0.0 |

**Pareadas principales en test** (b − a, mismas filas)

| Grupo | a | b | Métrica | Diferencia (IC 95 %) | Solo a acierta | Solo b acierta |
|---|---|---|---|---|---:|---:|
| principales | tfidf-lr-C10 | 06b-lr2e-4-r32 | clase | -0.7 (-2.1 a +0.7) | 125 | 112 |
| principales | tfidf-lr-C10 | 4b-lr2e-4-r32 | clase | +2.1 (+0.8 a +3.6) | 86 | 128 |

**Hard set** (exploratorio: 105 filas; ambiguo 74, indirecto 31)

| Modelo | Total (IC 95 %) | Ambiguo (IC 95 %) | Indirecto (IC 95 %) |
|---|---|---|---|
| diccionario | 1.9 (0.0–4.8) | 2.7 (0.0–6.8) | 0.0 (0.0–0.0) |
| tfidf-lr-C10 | 10.5 (4.8–17.1) | 6.8 (1.4–13.5) | 19.4 (6.5–32.3) |
| 06b-lr2e-4-r32 | 11.4 (5.7–18.1) | 5.4 (1.4–10.8) | 25.8 (9.7–41.9) |
| 4b-lr2e-4-r32 | 9.5 (4.8–16.2) | 4.1 (0.0–9.5) | 22.6 (9.7–38.7) |
| 06b-base-0shot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) |
| 4b-base-0shot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) |
| 06b-base-fewshot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) |
| 4b-base-fewshot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) |

- Curación del hard set: Las 105 filas las pre-curó Gemini con criterios de Samuel y del chat; Samuel no las revisó fila por fila; el chat hizo 23 ajustes; sesgo reconocido (CAMBIO DE DISEÑO del 22 nov).
- Tipo indirecto: El tipo indirecto tiene 31 filas, menos de 50: se reporta con su intervalo y una advertencia.

La tabla completa (pareadas secundarias, validación contra test, accuracy por nivel del SCIAN y costos) está en el mismo archivo.

## Cómo se evaluó

Las reglas completas están en [`CLAUDE.md`](CLAUDE.md) (sección c) y el registro de cada paso, en [`docs/bitacora.md`](docs/bitacora.md).

- **Split por nombre normalizado:** ningún nombre normalizado aparece en dos conjuntos.
- **Test y hard set congelados:** se guardó su SHA-256 al crearlos, no se usaron para decidir nada y sus etiquetas solo se leyeron en la evaluación final, que se hizo una sola vez. Toda decisión se tomó con validación.
- **Pre-registro:** los modelos, el prompt y las métricas de la evaluación final se fijaron antes de abrir test, en [`evals/fase4_preregistro.json`](evals/fase4_preregistro.json).
- **Respuestas:** el modelo responde solo el código de 6 dígitos; un formato inválido o un código que no existe en el catálogo cuentan como error. El título de la clase sale siempre del catálogo oficial, nunca del modelo.
- **Semilla 42** en todo, y ningún número se reporta si no está en [`results/`](results/).

## Modelos

Los adaptadores LoRA del 4B y del 0.6B, y el GGUF BF16 del 0.6B que usa la demo, se publican en Hugging Face:

- Qwen3-4B con LoRA: https://huggingface.co/samcaballero/denue-scian-qwen3-4b-lora
- Qwen3-0.6B con LoRA y GGUF BF16: https://huggingface.co/samcaballero/denue-scian-qwen3-0.6b-lora-gguf

Sus model cards están en [`hf/`](hf/) y se suben con `src/subir_hf.py`, que verifica el SHA-256 de cada archivo antes de subirlo.

## Demo

`scripts/demo.ps1` corre en una laptop, sin GPU y sin red, el Qwen3-0.6B sin afinar y el afinado (GGUF BF16 con llama.cpp), y muestra la respuesta de cada uno con el título del catálogo. Necesita los GGUF y llama.cpp fuera del repo; los detalles están en `CLAUDE.md`.

## Estructura

- `src/`: código del proyecto (cargador del DENUE, splits, baselines, kernels de Kaggle, evaluación, demo).
- `tests/`: pruebas con pytest, solo con datos inventados.
- `notebooks/`: los kernels de Kaggle, generados por `src/construir_kernel.py`.
- `results/`: métricas en JSON; ningún número se reporta si no está ahí.
- `docs/`: bitácora, registro del trabajo con Claude Code, capturas y figuras.

## Datos y crédito

Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), edición 05/2026, Ciudad de México. Los datos se usan conforme a los [Términos de Libre Uso de la Información del INEGI](https://www.inegi.org.mx/inegi/terminos.html).

**Aviso:** los datos usados en este proyecto fueron transformados a partir de la fuente original: normalización de textos y división en conjuntos (splits) de entrenamiento, validación y prueba. Este proyecto no es un producto oficial del INEGI.

Los datos del DENUE no se incluyen en este repositorio. Sí se incluye `data/scian_2023_clases.csv`: el código y el título de las 1,086 clases del SCIAN 2023, extraídos de la estructura oficial publicada por el INEGI (https://www.inegi.org.mx/contenidos/app/scian/estructura2023.pdf), sin las marcas de los títulos, como la "T" de trilateral. Fuente: INEGI, Sistema de Clasificación Industrial de América del Norte, México SCIAN 2023. También se incluye `data/scian_2023_vocabulario.csv`: las palabras de los títulos de todas las categorías de esa misma estructura (de sector a clase), normalizadas sin acentos y en mayúsculas, que el proyecto usa para marcar los nombres genéricos. Misma fuente.

## Licencia

El código se publica bajo la licencia Apache-2.0 (ver [LICENSE](LICENSE)). Los pesos publicados en Hugging Face también llevan Apache-2.0, como los modelos base `Qwen/Qwen3-0.6B` y `Qwen/Qwen3-4B`.
