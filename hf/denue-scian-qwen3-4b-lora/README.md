---
license: apache-2.0
language:
- es
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- unsloth
- scian
- text-classification
---

# denue-scian-qwen3-4b-lora

Adaptador LoRA de `Qwen/Qwen3-4B` que clasifica el nombre de un negocio en una clase del SCIAN 2023 (código de 6 dígitos). Se entrenó con nombres del DENUE del INEGI (Ciudad de México, edición 05/2026) para una charla técnica del 26 de noviembre de 2026. Es un experimento de investigación y no está en producción.

**English summary.** LoRA adapter for `Qwen/Qwen3-4B` that maps a Mexican business name (in Spanish) to a 6-digit SCIAN 2023 industry class. Trained with QLoRA for one epoch on a single T4, on 187,350 business names from INEGI's DENUE (Mexico City, 05/2026). On a held-out test set of 2,000 rows split by normalized name, class accuracy is 71.6 % (95 % CI 69.6–73.5), 2.1 points above a TF-IDF + logistic regression baseline; on a small hard set of ambiguous names it is about 10 %. Research demo, not for production: the class title must come from the official SCIAN catalog, never from the model.

## Tarea

- **Entrada:** el nombre del negocio tal como se escribe (`nom_estab` del DENUE), sin normalizar.
- **Salida:** solo el código SCIAN 2023 de 6 dígitos, sin texto adicional. El título de la clase se busca en el catálogo oficial del SCIAN 2023 (en el repo de GitHub: `data/scian_2023_clases.csv`); nunca se le pide al modelo.
- **Cómo se califica una respuesta** (regla 5 del proyecto): solo se recortan en sus orillas ` ` (espacio ASCII), `\r` y `\n`; después, la respuesta debe coincidir con `^[0-9]{6}$` y estar en el catálogo. Si no, es formato inválido o código inexistente, y cuenta como error.

## Prompt exacto

Plantilla de chat de Qwen3 con `enable_thinking=False`: un solo mensaje de usuario, sin mensaje de sistema.

```text
<|im_start|>user
Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. Responde solo con el código de 6 dígitos, sin texto adicional.
Nombre: {nom_estab}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Generación greedy con 16 fichas nuevas como máximo; se corta en `<|im_end|>`.

## Uso

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ADAPTADOR = "{{usuario_hf}}/denue-scian-qwen3-4b-lora"

tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
modelo = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION, device_map="auto")
modelo = PeftModel.from_pretrained(modelo, ADAPTADOR)

instruccion = ("Clasifica el establecimiento en una clase del SCIAN 2023 según su nombre. "
               "Responde solo con el código de 6 dígitos, sin texto adicional.")
nom_estab = "..."  # el nombre del negocio
mensajes = [{"role": "user", "content": f"{instruccion}\nNombre: {nom_estab}"}]
prompt = tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True, enable_thinking=False)
entrada = tok(prompt, return_tensors="pt").to(modelo.device)
salida = modelo.generate(**entrada, max_new_tokens=16, do_sample=False)
print(tok.decode(salida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True))
```

Los resultados de abajo se midieron con la base en 4 bits (NF4 de bitsandbytes, cargada con Unsloth) y la LoRA sin fusionar. Con otra precisión de la base, algunas respuestas cambian: en el 0.6B, la base en bf16 con la LoRA fusionada repitió el 83.4 % de las respuestas de validación (`results/2026-11-25_fase4_gguf-concordancia.json`).

## Entrenamiento

- **Método:** QLoRA con Unsloth. Base en NF4; LoRA de rank 32, alpha 32 y dropout 0 en `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` y `down_proj`.
- **Hiperparámetros:** lote 32, tasa 2e-4 con programación lineal y 3 % de calentamiento, `adamw_8bit`, 1 época, largo máximo de 256 fichas, semilla 42. La tasa y el rank salen del barrido del 0.6B en validación; el 4B usó esa receta sin cambios.
- **Datos:** 187,350 filas de train (929 clases, con un tope de 1,000 filas por clase) y 1,000 de validación. Solo el nombre y la clase (`nom_estab` → `codigo_act`); nunca datos de contacto.
- **División por nombre normalizado:** ningún nombre normalizado aparece en dos conjuntos (train, validación, test y hard set).
- **Cómputo:** una GPU T4 en Kaggle, 7.209 h de GPU (`results/2026-11-24_fase2_4b-lr2e-4-r32.json`).
- **Revisión de la base:** `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`, versión instruct.

## Resultados en test

2,000 filas con la distribución natural, evaluadas una sola vez. Porcentajes; IC 95 % por bootstrap de 1,000 remuestreos. Copiados de `results/2026-11-25_fase4_tabla-final.md`.

| Modelo | Clase (IC 95 %) | Sector (IC 95 %) | Macro-F1 de sector | % formato inválido | % código inexistente |
|---|---|---|---:|---:|---:|
| 4b-lr2e-4-r32 (este adaptador) | 71.6 (69.6–73.5) | 86.1 (84.6–87.6) | 0.724 | 0.0 | 0.0 |
| tfidf-lr-C10 | 69.5 (67.5–71.5) | 84.5 (83.0–86.0) | 0.678 | 0.0 | 0.0 |
| 06b-lr2e-4-r32 | 68.8 (66.8–70.9) | 84.4 (82.8–86.0) | 0.679 | 0.0 | 0.0 |
| diccionario | 8.3 (7.1–9.6) | 44.1 (42.0–46.2) | 0.034 | 0.0 | 0.0 |
| 4b-base-0shot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.000 | 22.0 | 78.0 |
| 4b-base-fewshot | 8.9 (7.7–10.2) | 23.7 (21.8–25.6) | 0.093 | 0.3 | 38.0 |

- **Contra TF-IDF** (diferencia pareada en las mismas filas, puntos porcentuales): +2.1 (+0.8 a +3.6) en clase y +1.5 (+0.2 a +2.8) en sector. Sin corrección por comparaciones múltiples.
- **Contra el 0.6B afinado:** +2.8 (+1.6 a +3.9) en clase.
- `tfidf-lr-C10` es TF-IDF de n-gramas de caracteres con regresión logística; `diccionario` responde la clase más frecuente de cada nombre normalizado visto en train.

**Hard set** (exploratorio: 105 filas, 74 de nombres ambiguos y 31 de nombres indirectos):

| Modelo | Total (IC 95 %) | Ambiguo (IC 95 %) | Indirecto (IC 95 %) |
|---|---|---|---|
| 4b-lr2e-4-r32 | 9.5 (4.8–16.2) | 4.1 (0.0–9.5) | 22.6 (9.7–38.7) |
| tfidf-lr-C10 | 10.5 (4.8–17.1) | 6.8 (1.4–13.5) | 19.4 (6.5–32.3) |

## Límites

- **Hard set:** con nombres ambiguos o indirectos acierta alrededor del 10 %, sin diferencia clara con TF-IDF (pareada: −1.0, de −7.6 a +5.7 puntos).
- **Curación del hard set:** sus 105 filas las pre-curó un modelo de lenguaje (Gemini) con criterios del proyecto; nadie las revisó una por una. Puede tener el sesgo de ese modelo. El tipo indirecto tiene 31 filas, menos de 50, así que su intervalo es muy ancho.
- **Nombres ambiguos:** muchos nombres no dicen a qué se dedica el negocio. De 50 errores del 4B en test revisados a mano, 32 (64 %) son de nombre ambiguo (`results/2026-11-25_fase4_errores-por-tipo.json`).
- **Etiquetas del DENUE:** la clase la asigna el INEGI y puede tener errores; 4 de esos 50 errores se marcaron como etiqueta dudosa. La accuracy mide el acuerdo con el DENUE, no con una verdad revisada.
- **Cobertura:** solo Ciudad de México, edición 05/2026, SCIAN 2023. Solo 929 de las 1,086 clases tienen ejemplos en train.
- **Códigos inexistentes:** en test no respondió ninguno, pero puede pasar. Valida siempre la respuesta contra el catálogo.
- **No está en producción:** es material de una charla técnica, sin monitoreo ni mantenimiento.

## Usos fuera de alcance

- Clasificación o estadística oficial: no sustituye la asignación del INEGI.
- Decisiones sobre personas o negocios (impuestos, permisos, crédito, sanciones) sin revisión humana.
- Otros países, otras versiones del SCIAN u otros idiomas, sin evaluarlo antes.
- Direcciones u otros campos: solo se entrenó con nombres de negocios.
- Generar texto libre: está entrenado para responder solo un código.

## Licencia y crédito

- **Pesos de este adaptador:** Apache-2.0 (copia en `LICENSE`).
- **Modelo base:** `Qwen/Qwen3-4B`, con licencia Apache-2.0 según su repo en Hugging Face (revisión `1cfa9a72…`), verificada el 26 de noviembre de 2026.
- **Datos:** Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), edición 05/2026, Ciudad de México. Los datos se usan conforme a los [Términos de Libre Uso de la Información del INEGI](https://www.inegi.org.mx/inegi/terminos.html).
- **Aviso:** los datos usados en este proyecto fueron transformados a partir de la fuente original: normalización de textos y división en conjuntos (splits) de entrenamiento, validación y prueba. Este proyecto no es un producto oficial del INEGI.
- **Catálogo:** INEGI, Sistema de Clasificación Industrial de América del Norte, México SCIAN 2023.
- **Código, reglas de evaluación y bitácora:** https://github.com/samcaballero/denue-scian-finetune
