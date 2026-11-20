# denue-scian-finetune

Proyecto para una charla técnica (26 de noviembre de 2026): fine-tuning con QLoRA de `Qwen/Qwen3-0.6B` y `Qwen/Qwen3-4B` sobre el DENUE del INEGI (Ciudad de México, edición 05/2026), para dos tareas:

1. **Clasificación:** nombre de un negocio → clase SCIAN de 6 dígitos.
2. **Extracción:** dirección escrita de forma desordenada → JSON estructurado.

Los modelos se comparan contra un baseline de TF-IDF con n-gramas de caracteres y regresión logística. El entrenamiento corre en Kaggle (GPU T4) con Unsloth.

## Datos y crédito

Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), edición 05/2026, Ciudad de México. Los datos se usan conforme a los [Términos de Libre Uso de la Información del INEGI](https://www.inegi.org.mx/inegi/terminos.html).

**Aviso:** los datos usados en este proyecto fueron transformados a partir de la fuente original: normalización de textos, división en conjuntos (splits) de entrenamiento, validación y prueba, y ruido sintético agregado a las direcciones. Este proyecto no es un producto oficial del INEGI.

Los datos del DENUE no se incluyen en este repositorio. Sí se incluye `data/scian_2023_clases.csv`: el código y el título de las 1,086 clases del SCIAN 2023, extraídos de la estructura oficial publicada por el INEGI (https://www.inegi.org.mx/contenidos/app/scian/estructura2023.pdf), sin las marcas de los títulos, como la "T" de trilateral. Fuente: INEGI, Sistema de Clasificación Industrial de América del Norte, México SCIAN 2023. También se incluye `data/scian_2023_vocabulario.csv`: las palabras de los títulos de todas las categorías de esa misma estructura (de sector a clase), normalizadas sin acentos y en mayúsculas, que el proyecto usa para marcar los nombres genéricos. Misma fuente.

## Licencia

El código se publica bajo la licencia Apache-2.0 (ver [LICENSE](LICENSE)).
