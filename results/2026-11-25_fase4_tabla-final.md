# Evaluación final del clasificador (Fase 4): test y hard set

Fecha 2026-11-25 · commit `0c0d552` · pre-registro `7de9d3fc` · semilla 42 · IC 95 % por bootstrap de 1,000 remuestreos. Fuente: INEGI, DENUE (CDMX, 05/2026) y SCIAN 2023. Porcentajes; pareadas en puntos porcentuales.

## Test (2,000 filas, distribución natural)

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

## Pareadas en test (b − a, mismas filas)

| Grupo | a | b | Métrica | Diferencia (IC 95 %) | Solo a acierta | Solo b acierta |
|---|---|---|---|---|---:|---:|
| principales | tfidf-lr-C10 | 06b-lr2e-4-r32 | clase | -0.7 (-2.1 a +0.7) | 125 | 112 |
| principales | tfidf-lr-C10 | 4b-lr2e-4-r32 | clase | +2.1 (+0.8 a +3.6) | 86 | 128 |
| secundarias | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | clase | +2.8 (+1.6 a +3.9) | 38 | 93 |
| secundarias | 06b-base-0shot | 06b-lr2e-4-r32 | clase | +68.8 (+66.8 a +70.9) | 0 | 1377 |
| secundarias | 06b-base-fewshot | 06b-lr2e-4-r32 | clase | +68.8 (+66.8 a +70.8) | 1 | 1377 |
| secundarias | 4b-base-0shot | 4b-lr2e-4-r32 | clase | +71.6 (+69.6 a +73.5) | 0 | 1432 |
| secundarias | 4b-base-fewshot | 4b-lr2e-4-r32 | clase | +62.6 (+60.7 a +64.6) | 9 | 1262 |
| secundarias_sector | tfidf-lr-C10 | 06b-lr2e-4-r32 | sector | -0.2 (-1.4 a +1.0) | 83 | 79 |
| secundarias_sector | tfidf-lr-C10 | 4b-lr2e-4-r32 | sector | +1.5 (+0.2 a +2.8) | 67 | 97 |
| secundarias_sector | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | sector | +1.7 (+0.8 a +2.6) | 31 | 65 |
| secundarias_sector | 06b-base-0shot | 06b-lr2e-4-r32 | sector | +84.4 (+82.8 a +86.0) | 0 | 1687 |
| secundarias_sector | 06b-base-fewshot | 06b-lr2e-4-r32 | sector | +81.8 (+80.0 a +83.6) | 18 | 1655 |
| secundarias_sector | 4b-base-0shot | 4b-lr2e-4-r32 | sector | +86.1 (+84.6 a +87.6) | 0 | 1721 |
| secundarias_sector | 4b-base-fewshot | 4b-lr2e-4-r32 | sector | +62.4 (+60.0 a +64.6) | 31 | 1278 |

## Hard set (exploratorio: 105 filas; ambiguo 74, indirecto 31)

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

### Pareadas en el hard set (b − a)

| Subconjunto | Grupo | a | b | Métrica | Diferencia (IC 95 %) |
|---|---|---|---|---|---|
| total | principales | tfidf-lr-C10 | 06b-lr2e-4-r32 | clase | +1.0 (-5.7 a +7.6) |
| total | principales | tfidf-lr-C10 | 4b-lr2e-4-r32 | clase | -1.0 (-7.6 a +5.7) |
| total | secundarias | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | clase | -1.9 (-4.8 a +0.0) |
| total | secundarias | 06b-base-0shot | 06b-lr2e-4-r32 | clase | +11.4 (+5.7 a +18.1) |
| total | secundarias | 06b-base-fewshot | 06b-lr2e-4-r32 | clase | +11.4 (+5.7 a +18.1) |
| total | secundarias | 4b-base-0shot | 4b-lr2e-4-r32 | clase | +9.5 (+4.8 a +16.2) |
| total | secundarias | 4b-base-fewshot | 4b-lr2e-4-r32 | clase | +9.5 (+4.8 a +16.2) |
| total | secundarias_sector | tfidf-lr-C10 | 06b-lr2e-4-r32 | sector | +7.6 (-1.9 a +17.1) |
| total | secundarias_sector | tfidf-lr-C10 | 4b-lr2e-4-r32 | sector | +3.8 (-4.8 a +12.4) |
| total | secundarias_sector | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | sector | -3.8 (-8.6 a +1.0) |
| total | secundarias_sector | 06b-base-0shot | 06b-lr2e-4-r32 | sector | +42.9 (+34.3 a +52.4) |
| total | secundarias_sector | 06b-base-fewshot | 06b-lr2e-4-r32 | sector | +34.3 (+23.8 a +44.8) |
| total | secundarias_sector | 4b-base-0shot | 4b-lr2e-4-r32 | sector | +39.0 (+30.5 a +48.6) |
| total | secundarias_sector | 4b-base-fewshot | 4b-lr2e-4-r32 | sector | +30.5 (+20.9 a +40.0) |
| ambiguo | principales | tfidf-lr-C10 | 06b-lr2e-4-r32 | clase | -1.4 (-8.1 a +5.4) |
| ambiguo | principales | tfidf-lr-C10 | 4b-lr2e-4-r32 | clase | -2.7 (-9.5 a +4.1) |
| ambiguo | secundarias | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | clase | -1.4 (-4.1 a +0.0) |
| ambiguo | secundarias | 06b-base-0shot | 06b-lr2e-4-r32 | clase | +5.4 (+1.4 a +10.8) |
| ambiguo | secundarias | 06b-base-fewshot | 06b-lr2e-4-r32 | clase | +5.4 (+1.4 a +10.8) |
| ambiguo | secundarias | 4b-base-0shot | 4b-lr2e-4-r32 | clase | +4.1 (+0.0 a +9.5) |
| ambiguo | secundarias | 4b-base-fewshot | 4b-lr2e-4-r32 | clase | +4.1 (+0.0 a +9.5) |
| ambiguo | secundarias_sector | tfidf-lr-C10 | 06b-lr2e-4-r32 | sector | +5.4 (-5.4 a +16.2) |
| ambiguo | secundarias_sector | tfidf-lr-C10 | 4b-lr2e-4-r32 | sector | +2.7 (-8.1 a +13.5) |
| ambiguo | secundarias_sector | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | sector | -2.7 (-8.1 a +2.7) |
| ambiguo | secundarias_sector | 06b-base-0shot | 06b-lr2e-4-r32 | sector | +44.6 (+33.8 a +56.8) |
| ambiguo | secundarias_sector | 06b-base-fewshot | 06b-lr2e-4-r32 | sector | +36.5 (+24.3 a +48.6) |
| ambiguo | secundarias_sector | 4b-base-0shot | 4b-lr2e-4-r32 | sector | +41.9 (+31.1 a +52.7) |
| ambiguo | secundarias_sector | 4b-base-fewshot | 4b-lr2e-4-r32 | sector | +31.1 (+20.3 a +41.9) |
| indirecto | principales | tfidf-lr-C10 | 06b-lr2e-4-r32 | clase | +6.5 (-9.7 a +22.6) |
| indirecto | principales | tfidf-lr-C10 | 4b-lr2e-4-r32 | clase | +3.2 (-9.7 a +16.1) |
| indirecto | secundarias | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | clase | -3.2 (-9.7 a +0.0) |
| indirecto | secundarias | 06b-base-0shot | 06b-lr2e-4-r32 | clase | +25.8 (+9.7 a +41.9) |
| indirecto | secundarias | 06b-base-fewshot | 06b-lr2e-4-r32 | clase | +25.8 (+9.7 a +41.9) |
| indirecto | secundarias | 4b-base-0shot | 4b-lr2e-4-r32 | clase | +22.6 (+9.7 a +38.7) |
| indirecto | secundarias | 4b-base-fewshot | 4b-lr2e-4-r32 | clase | +22.6 (+9.7 a +38.7) |
| indirecto | secundarias_sector | tfidf-lr-C10 | 06b-lr2e-4-r32 | sector | +12.9 (-3.2 a +29.0) |
| indirecto | secundarias_sector | tfidf-lr-C10 | 4b-lr2e-4-r32 | sector | +6.5 (-9.7 a +22.6) |
| indirecto | secundarias_sector | 06b-lr2e-4-r32 | 4b-lr2e-4-r32 | sector | -6.5 (-16.1 a +0.0) |
| indirecto | secundarias_sector | 06b-base-0shot | 06b-lr2e-4-r32 | sector | +38.7 (+22.6 a +54.8) |
| indirecto | secundarias_sector | 06b-base-fewshot | 06b-lr2e-4-r32 | sector | +29.0 (+9.7 a +45.2) |
| indirecto | secundarias_sector | 4b-base-0shot | 4b-lr2e-4-r32 | sector | +32.3 (+16.1 a +48.4) |
| indirecto | secundarias_sector | 4b-base-fewshot | 4b-lr2e-4-r32 | sector | +29.0 (+9.7 a +45.2) |

## Validación contra test (accuracy de clase)

| Modelo | Validación (IC 95 %) | Test (IC 95 %) | Test dentro del IC de validación |
|---|---|---|---|
| diccionario | 8.3 (6.6–9.9) | 8.3 (7.1–9.6) | sí |
| tfidf-lr-C10 | 68.2 (65.2–71.1) | 69.5 (67.5–71.5) | sí |
| 06b-lr2e-4-r32 | 68.7 (65.6–71.5) | 68.8 (66.8–70.9) | sí |
| 4b-lr2e-4-r32 | 70.7 (67.6–73.5) | 71.6 (69.6–73.5) | sí |

### Accuracy por nivel del SCIAN

| Modelo | Conjunto | Sector | Subsector | Rama | Subrama | Clase |
|---|---|---:|---:|---:|---:|---:|
| tfidf-lr-C10 | validacion | 84.5 | 78.6 | 75.3 | 73.4 | 68.2 |
| tfidf-lr-C10 | test | 84.5 | 80.7 | 76.7 | 74.9 | 69.5 |
| 4b-lr2e-4-r32 | validacion | 86.2 | 80.2 | 77.5 | 75.3 | 70.7 |
| 4b-lr2e-4-r32 | test | 86.1 | 81.8 | 78.4 | 76.6 | 71.6 |

## Costos

Kernel de inferencia en una T4: 1.35 h de GPU en total (161 s fuera de las etapas). Baselines: CPU de la laptop, 0 h de GPU (ajuste de TF-IDF + LR: 68 min).

| Modelo | Horas de GPU (sus etapas) | s/fila en test | % al tope en test | % al tope en hard |
|---|---:|---:|---:|---:|
| 06b-lr2e-4-r32 | 0.220 | 0.223 | 0.0 | 0.0 |
| 4b-lr2e-4-r32 | 0.359 | 0.381 | 0.0 | 0.0 |
| 06b-base-0shot | 0.087 | 0.126 | 0.0 | 0.0 |
| 4b-base-0shot | 0.205 | 0.322 | 0.0 | 0.0 |
| 06b-base-fewshot | 0.113 | 0.168 | 0.0 | 0.0 |
| 4b-base-fewshot | 0.325 | 0.523 | 0.0 | 0.0 |

## Notas

- Hard set: Exploratorio. Se trata como el test (regla 3): no elige modelos ni hiperparámetros y se evalúa una sola vez, junto con el test.
- Curación del hard set: Las 105 filas las pre-curó Gemini con criterios de Samuel y del chat; Samuel no las revisó fila por fila; el chat hizo 23 ajustes; sesgo reconocido (CAMBIO DE DISEÑO del 22 nov).
- Tipo indirecto: El tipo indirecto tiene 31 filas, menos de 50: se reporta con su intervalo y una advertencia.
- Comparaciones múltiples: Sin corrección por comparaciones múltiples; el reporte lo dice.
- Test contra validación: Si test contradice validación, se reporta tal cual.
- Diccionario: Por construcción, ninguna llave de test ni de hard está en train final (comprobado al calcular este archivo: 0 y 0). Por eso responde la clase global en todas sus filas, igual que en validación.
- 06b-lr2e-4-r32: Ganadora del barrido del 0.6B; su accuracy de validación es algo optimista, porque se eligió con la misma validación.
- Métrica tolerante: solo secundaria; la principal es la accuracy de clase de la regla 5.
