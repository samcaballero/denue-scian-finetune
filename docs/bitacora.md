# Bitácora del proyecto

> Las secciones del 2026-11-17 son una copia con formato de `../evidencia/bitacora.txt` (fuera del repo), hecha el 2026-11-18. Su texto es el original, salvo dos cambios pedidos por Samuel:
> (1) la fecha, el título y la evidencia de la parte 1 los indicó Samuel, porque esas líneas no tenían fecha en el archivo;
> (2) en la parte 2 se restauraron los acentos de: último, falló, código, desinstaló, también, dejó, resolvió y técnicas.
> Lo demás solo es formato Markdown: títulos, viñetas, rutas y comandos entre comillas invertidas y enlaces a las capturas.
> La sección del 2026-11-18 la dictó Samuel durante la sesión de Claude Code de ese día.

## 2026-11-17 - Fase 0, parte 1

- Horas de GPU disponibles en Kaggle por semana: 30 horas.
- Usuarios — GitHub: samcaballero | Kaggle: samcaballero | Hugging Face: samcaballero
- https://github.com/samcaballero/denue-scian-finetune
- Edición del DENUE: 2026/05 (Periodo 05/2026)
- Archivo CSV: denue_inegi_09_.csv
- Tamaño: 253,929 KB (~248 MB)
- Evidencia: [2026-11-17_fase0_gpu-kaggle.png](evidencia/2026-11-17_fase0_gpu-kaggle.png)

## 2026-11-17 - Fase 0, parte 2: entorno local

- Versiones: Git 2.56.0.windows.4 | Python 3.12.10 | Claude Code 2.1.327 (npm, canal latest) | Kaggle CLI 2.2.5 | Node v24.13.0 | pip 26.3
- Python se mantiene en 3.12.10: es el último instalador de 3.12 para Windows; el entrenamiento corre en Kaggle con su propio Python.
- Incidente Git: `winget upgrade` falló (código 4, "Acceso denegado" al crear `C:\Program Files\Git\dev`) y el rollback desinstaló también la 2.53. Tras reiniciar, `winget install` dejó la 2.56 en `AppData\Local\Programs\Git`.
- Incidente Kaggle: la CLI 2.2.5 no arrancaba (`ModuleNotFoundError: six`). Se resolvió con `pip install six`.
- Llave de Kaggle en `C:\Users\Invitado\.kaggle\access_token`, fuera del repo. Verificada con: `kaggle kernels list --mine`
- Git: user.name "Sam Caballero", correo privado noreply de GitHub, bloqueo de pushes que expongan el correo, rama por defecto main.
- Repo clonado en `C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune\`
- Evidencia: [2026-11-17_fase0_kaggle-cli.png](evidencia/2026-11-17_fase0_kaggle-cli.png) (la laptop controla Kaggle desde la terminal; diapositiva "Decisiones técnicas", flujo Claude Code -> Kaggle)
- Evidencia: [2026-11-17_fase0_herramientas.png](evidencia/2026-11-17_fase0_herramientas.png) (versiones del entorno; respaldo para preguntas sobre el setup)

## 2026-11-18 - Fase 0, cierre: estructura del repo

- Modelo de Claude Code: de Fable 5.2 (guardado por error como predeterminado) a Opus 5.5 con esfuerzo xhigh (subido a propósito para obtener más calidad).
- Primera sesión de Claude Code en el repo: CLAUDE.md, carpetas, .gitignore, evidencia, bitácora y README.
- Hallazgos de Claude Code: ruta real del CSV, tamaño real de unos 248 MB, codificación latin-1, columnas de contacto excluidas y correo personal en el commit inicial.
- Correo del commit inicial corregido: se reescribió el autor con el noreply de GitHub (commit 012e43a, mismo contenido). Pendiente: borrar y recrear el repo en GitHub para eliminar el commit viejo, y subir el historial limpio.
- Evidencia: [2026-11-18_fase0_plan-claude-code.png](evidencia/2026-11-18_fase0_plan-claude-code.png) y [2026-11-18_fase0_hallazgos-claude-code.png](evidencia/2026-11-18_fase0_hallazgos-claude-code.png), para la diapositiva "Decisiones técnicas: qué propuso Claude Code y qué revisé yo".
