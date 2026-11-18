# denue-scian-finetune

Proyecto para una charla técnica (26 de noviembre de 2026): fine-tuning con QLoRA de `Qwen/Qwen3-0.6B` y `Qwen/Qwen3-4B` sobre el DENUE del INEGI (Ciudad de México, edición 05/2026), para dos tareas:

1. **Clasificación:** nombre de un negocio → clase SCIAN de 6 dígitos.
2. **Extracción:** dirección escrita de forma desordenada → JSON estructurado.

Los modelos se comparan contra un baseline de TF-IDF con n-gramas de caracteres y regresión logística. El entrenamiento corre en Kaggle (GPU T4) con Unsloth.

## Datos y crédito

Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), edición 05/2026, Ciudad de México. Los datos se usan conforme a los [Términos de Libre Uso de la Información del INEGI](https://www.inegi.org.mx/inegi/terminos.html).

**Aviso:** los datos usados en este proyecto fueron transformados a partir de la fuente original: normalización de textos, división en conjuntos (splits) de entrenamiento, validación y prueba, y ruido sintético agregado a las direcciones. Este proyecto no es un producto oficial del INEGI.

Los datos no se incluyen en este repositorio.

## Licencia

El código se publica bajo la licencia Apache-2.0 (ver [LICENSE](LICENSE)).
