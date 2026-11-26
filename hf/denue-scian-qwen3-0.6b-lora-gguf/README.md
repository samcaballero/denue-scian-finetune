---
license: apache-2.0
language:
- es
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- unsloth
- gguf
- llama.cpp
- scian
- text-classification
---

# denue-scian-qwen3-0.6b-lora-gguf

Adaptador LoRA de `Qwen/Qwen3-0.6B` que clasifica el nombre de un negocio en una clase del SCIAN 2023 (código de 6 dígitos), y su versión fusionada en GGUF BF16 para correr en CPU con llama.cpp (la que usó la demo de la charla). Se entrenó con nombres del DENUE del INEGI (Ciudad de México, edición 05/2026) para una charla técnica del 26 de noviembre de 2026. Es un experimento de investigación y no está en producción.

**English summary.** LoRA adapter for `Qwen/Qwen3-0.6B` (plus a merged BF16 GGUF for llama.cpp on CPU) that maps a Mexican business name (in Spanish) to a 6-digit SCIAN 2023 industry class. Trained with QLoRA for one epoch on a single T4, on 187,350 business names from INEGI's DENUE (Mexico City, 05/2026). On a held-out test set of 2,000 rows split by normalized name, class accuracy is 68.8 % (95 % CI 66.8–70.9), statistically indistinguishable from a TF-IDF + logistic regression baseline (69.5 %). Research demo, not for production: the class title must come from the official SCIAN catalog, never from the model.

## Archivos

| Archivo | Qué es | Bytes | SHA-256 |
|---|---|---:|---|
| `adapter_model.safetensors` | adaptador LoRA (PEFT) | 80,792,456 | `19d9f15ecb1ef403ed1bb341dcbac27a6c56967f2f23a9907e1ab082164967b0` |
| `adapter_config.json` | configuración del adaptador | 1,307 | `d3f6f71c7dac7e35d46d59a62f0aa658486414903ace93d671805f400bbf7419` |
| `qwen3-0.6b-06b-lr2e-4-r32-BF16.gguf` | base + LoRA fusionadas, en GGUF BF16 | 1,198,182,400 | `5b97e03baf1c940909a334ea74ec4beb6d4fd9a73048f22bf2a49b8ea382fb20` |

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

## Uso con el adaptador (transformers y PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ADAPTADOR = "{{usuario_hf}}/denue-scian-qwen3-0.6b-lora-gguf"

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

## Uso con el GGUF (llama.cpp)

```text
llama-server -m qwen3-0.6b-06b-lr2e-4-r32-BF16.gguf -c 512 -np 1 -t 4 --host 127.0.0.1 --port 8080
```

Después, `POST http://127.0.0.1:8080/completion` con el prompt exacto de arriba como texto (sin BOS; los tokens especiales se reconocen dentro del texto), `"n_predict": 16`, `"temperature": 0`, `"top_k": 1`, `"top_p": 1`, `"min_p": 0` y `"repeat_penalty": 1`. La respuesta es el campo `content`.

- **Cómo se hizo:** base `Qwen/Qwen3-0.6B@c1899de2…` en bf16 + la LoRA, fusionadas con `merge_and_unload` de PEFT en CPU; conversión con `convert_hf_to_gguf.py` de llama.cpp b12359 (release v1.0.0), con la configuración y el tokenizador oficiales de Qwen. Sin cuantizar más allá de BF16.
- **Concordancia con Kaggle** (solo validación, 1,000 filas; nunca se evaluó el GGUF en test): 834 respuestas idénticas a las de Kaggle (83.4 %). Accuracy de clase del GGUF 0.677 (0.647–0.707) contra 0.687 (0.656–0.715) en Kaggle; diferencia −0.010 (−0.027 a +0.006). No se esperaba 100 %: en Kaggle la base iba en NF4 con la LoRA sin fusionar. Fuente: `results/2026-11-25_fase4_gguf-concordancia.json`.
- **Latencia en una laptop sin GPU** (Intel Core i7-1185G7, 4 hilos, contexto de 512; 200 filas de validación): p50 de 689.1 ms y p95 de 887.0 ms de punta a punta por nombre. Fuente: `results/2026-11-25_fase4_latencia.json`.

## Entrenamiento

- **Método:** QLoRA con Unsloth. Base en NF4; LoRA de rank 32, alpha 32 y dropout 0 en `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` y `down_proj`.
- **Hiperparámetros:** lote 32, tasa 2e-4 con programación lineal y 3 % de calentamiento, `adamw_8bit`, 1 época, largo máximo de 256 fichas, semilla 42.
- **Elección:** esta configuración (`06b-lr2e-4-r32`) tuvo la mayor accuracy de validación entre 5 del 0.6B: la corrida inicial (rank 16) y las 4 del barrido (tasas 1e-4 y 2e-4, rank 8 y 32). Por eso su accuracy de validación (0.687) es algo optimista; la de test no tiene ese sesgo.
- **Datos:** 187,350 filas de train (929 clases, con un tope de 1,000 filas por clase) y 1,000 de validación. Solo el nombre y la clase (`nom_estab` → `codigo_act`); nunca datos de contacto.
- **División por nombre normalizado:** ningún nombre normalizado aparece en dos conjuntos (train, validación, test y hard set).
- **Cómputo:** una GPU T4 en Kaggle, 1.192 h de GPU para esta configuración (`results/2026-11-24_fase2_06b-lr2e-4-r32.json`); el barrido completo usó 4.756 h.
- **Revisión de la base:** `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, versión instruct.

## Resultados en test

2,000 filas con la distribución natural, evaluadas una sola vez con el adaptador (base en NF4 y LoRA sin fusionar). Porcentajes; IC 95 % por bootstrap de 1,000 remuestreos. Copiados de `results/2026-11-25_fase4_tabla-final.md`.

| Modelo | Clase (IC 95 %) | Sector (IC 95 %) | Macro-F1 de sector | % formato inválido | % código inexistente |
|---|---|---|---:|---:|---:|
| 06b-lr2e-4-r32 (este adaptador) | 68.8 (66.8–70.9) | 84.4 (82.8–86.0) | 0.679 | 0.0 | 0.0 |
| tfidf-lr-C10 | 69.5 (67.5–71.5) | 84.5 (83.0–86.0) | 0.678 | 0.0 | 0.0 |
| 4b-lr2e-4-r32 | 71.6 (69.6–73.5) | 86.1 (84.6–87.6) | 0.724 | 0.0 | 0.0 |
| diccionario | 8.3 (7.1–9.6) | 44.1 (42.0–46.2) | 0.034 | 0.0 | 0.0 |
| 06b-base-0shot | 0.0 (0.0–0.0) | 0.0 (0.0–0.0) | 0.000 | 78.0 | 21.9 |
| 06b-base-fewshot | 0.1 (0.0–0.1) | 2.5 (1.8–3.2) | 0.005 | 0.0 | 17.4 |

- **Contra TF-IDF** (diferencia pareada en las mismas filas, puntos porcentuales): −0.7 (−2.1 a +0.7) en clase y −0.2 (−1.4 a +1.0) en sector. El 0.6B afinado no se distingue de TF-IDF. Sin corrección por comparaciones múltiples.
- `tfidf-lr-C10` es TF-IDF de n-gramas de caracteres con regresión logística; `diccionario` responde la clase más frecuente de cada nombre normalizado visto en train.

**Hard set** (exploratorio: 105 filas, 74 de nombres ambiguos y 31 de nombres indirectos):

| Modelo | Total (IC 95 %) | Ambiguo (IC 95 %) | Indirecto (IC 95 %) |
|---|---|---|---|
| 06b-lr2e-4-r32 | 11.4 (5.7–18.1) | 5.4 (1.4–10.8) | 25.8 (9.7–41.9) |
| tfidf-lr-C10 | 10.5 (4.8–17.1) | 6.8 (1.4–13.5) | 19.4 (6.5–32.3) |

## Límites

- **Hard set:** con nombres ambiguos o indirectos acierta alrededor del 10 %, sin diferencia clara con TF-IDF (pareada: +1.0, de −5.7 a +7.6 puntos).
- **Curación del hard set:** sus 105 filas las pre-curó un modelo de lenguaje (Gemini) con criterios del proyecto; nadie las revisó una por una. Puede tener el sesgo de ese modelo. El tipo indirecto tiene 31 filas, menos de 50, así que su intervalo es muy ancho.
- **Nombres ambiguos:** muchos nombres no dicen a qué se dedica el negocio. En la revisión a mano de 50 errores del 4B en test, 32 (64 %) fueron de nombre ambiguo (`results/2026-11-25_fase4_errores-por-tipo.json`); en el 0.6B no se hizo esa revisión.
- **Etiquetas del DENUE:** la clase la asigna el INEGI y puede tener errores. La accuracy mide el acuerdo con el DENUE, no con una verdad revisada.
- **Cobertura:** solo Ciudad de México, edición 05/2026, SCIAN 2023. Solo 929 de las 1,086 clases tienen ejemplos en train.
- **GGUF:** responde distinto que el adaptador en el 16.6 % de las filas de validación y solo se midió en validación.
- **Códigos inexistentes:** en test no respondió ninguno, pero puede pasar (el GGUF dio 2 en validación). Valida siempre la respuesta contra el catálogo.
- **No está en producción:** es material de una charla técnica, sin monitoreo ni mantenimiento.

## Usos fuera de alcance

- Clasificación o estadística oficial: no sustituye la asignación del INEGI.
- Decisiones sobre personas o negocios (impuestos, permisos, crédito, sanciones) sin revisión humana.
- Otros países, otras versiones del SCIAN u otros idiomas, sin evaluarlo antes.
- Direcciones u otros campos: solo se entrenó con nombres de negocios.
- Generar texto libre: está entrenado para responder solo un código.

## Licencia y crédito

- **Pesos de este repo (adaptador y GGUF):** Apache-2.0 (copia en `LICENSE`).
- **Modelo base:** `Qwen/Qwen3-0.6B`, con licencia Apache-2.0 según su repo en Hugging Face (revisión `c1899de2…`), verificada el 26 de noviembre de 2026. El GGUF incluye los pesos de la base fusionados con el adaptador.
- **Datos:** Fuente: INEGI, DENUE (Directorio Estadístico Nacional de Unidades Económicas), edición 05/2026, Ciudad de México. Los datos se usan conforme a los [Términos de Libre Uso de la Información del INEGI](https://www.inegi.org.mx/inegi/terminos.html).
- **Aviso:** los datos usados en este proyecto fueron transformados a partir de la fuente original: normalización de textos y división en conjuntos (splits) de entrenamiento, validación y prueba. Este proyecto no es un producto oficial del INEGI.
- **Catálogo:** INEGI, Sistema de Clasificación Industrial de América del Norte, México SCIAN 2023.
- **Código, reglas de evaluación y bitácora:** https://github.com/samcaballero/denue-scian-finetune
