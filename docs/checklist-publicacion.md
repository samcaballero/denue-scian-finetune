# Checklist para publicar el repo y los modelos (paso 20)

Preparado el 26 de noviembre de 2026 por Claude Code. Esa noche no salió nada de la laptop: sin push, sin subida a Hugging Face y sin cambios de visibilidad. Lo marcado con [x] ya está comprobado; lo marcado con [ ] lo hace Samuel.

## 1. Repo de GitHub: lo que ya está comprobado

- [x] **README** con el estado del proyecto, los resultados en test copiados de `results/2026-11-25_fase4_tabla-final.md`, el crédito al INEGI con el texto que ya usaba el repo y la tarea 2 declarada como no hecha.
- [x] **LICENSE:** Apache-2.0, el texto que puso GitHub al crear el repo, sin cambios.
- [x] **Nada de datos versionado:** en ningún commit hay archivos bajo `data/` fuera de `data/.gitkeep`, `data/manifest.json`, `data/scian_2023_clases.csv` y `data/scian_2023_vocabulario.csv`, ni bajo `kaggle-output/`, ni con extensión de datos o de modelo (auditoría de todo el historial, `src/auditoria_publicacion.py`).
- [x] **Sin nombres, ids ni secretos del DENUE:** la auditoría no encontró ningún hallazgo que detenga; los conteos están en `results/2026-11-26_paso20_auditoria-privacidad.json`.
- [x] **Autor de los commits:** el noreply de GitHub de `samcaballero` en todos los commits.
- [x] **`.claude/settings.local.json`** no está versionado. Desde el cierre del paso 20 también lo ignora el `.gitignore` del repo, así que un clon nuevo tampoco lo sube.
- [x] **`docs/sesiones/2026-11-18_sesion01_estructura-repo.md`** no se versiona ni se publica (decisión de Samuel); se queda en la laptop sin borrar.
- [x] **El transcript de la sesión del paso 20** no se exporta ni se versiona: en ella se leyeron datos y capturas.
- [x] **Commit viejo `1f60028`:** solo existe en el reflog local de esta laptop, fuera de las ramas. Un push solo manda lo alcanzable desde las ramas, así que no se publica; GitHub ya lo da 404 desde que se recreó el repo el 18 de noviembre. No se toca.

## 2. Subida a Hugging Face (empezar temprano)

Total a subir: 9 archivos, 1,543,328,915 bytes (unos 1.5 GB), casi todo el GGUF BF16 del 0.6B (1.2 GB). A 10 Mb/s de subida tarda alrededor de 20 minutos.

- [ ] **Token de escritura:** en https://huggingface.co/settings/tokens, crea un token de tipo *Write* (o uno *fine-grained* con permiso de escritura en tus repos). No lo pegues en ningún archivo ni en la línea de comandos: el script lo pide con `getpass`, que no lo muestra al pegarlo ni lo guarda en el historial de PowerShell.
- [ ] **Ir a la carpeta del repo** (en PowerShell):

```powershell
Set-Location "C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune"
```

- [ ] **Dry-run con tu usuario:** debe terminar con "Todos los SHA-256 verificados" y "Dry-run: nada se subió."

```powershell
& "C:\sca\fine-tuning\proyecto-denue\venv-tokenizador\Scripts\python.exe" -m src.subir_hf --usuario samcaballero
```

- [ ] **Subida real:** pega el token cuando lo pida. Crea los dos repos como privados, sube cada uno en un solo commit y después compara bytes y SHA-256 de cada archivo en Hugging Face contra lo local.

```powershell
& "C:\sca\fine-tuning\proyecto-denue\venv-tokenizador\Scripts\python.exe" -m src.subir_hf --usuario samcaballero --subir
```

- [ ] **Revisar las dos páginas** (card, archivos y tamaños):
  - https://huggingface.co/samcaballero/denue-scian-qwen3-4b-lora
  - https://huggingface.co/samcaballero/denue-scian-qwen3-0.6b-lora-gguf
- [ ] **Hacerlos públicos** desde la web: Settings del repo → Change repository visibility.
- [ ] **Opcional:** borrar el token en https://huggingface.co/settings/tokens al terminar.

Si algo falla a medio camino, el script se detiene y dice en qué archivo. Se puede volver a correr: los repos ya creados se reutilizan, y se niega a subir si el repo tiene archivos que no son los suyos.

## 3. Enlaces de Hugging Face en el README

- [x] Los dos enlaces ya están en la sección "Modelos" del README (cierre del paso 20). Mientras los repos sigan privados, quien no tenga sesión verá un error al abrirlos.

## 4. Antes del push

- [ ] **Repetir la auditoría** sobre el historial completo, con todos los commits nuevos. Desde la carpeta del repo:

```powershell
& "C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune\.venv\Scripts\python.exe" -m src.auditoria_publicacion
```

  Debe terminar con "Hallazgos que detienen: 0". Tarda unos 5 minutos y necesita el CSV del DENUE y los paquetes de Kaggle en la laptop.

- [ ] **Push** (después de que el chat de Claude del proyecto revise el reporte):

```powershell
git -C "C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune" push origin main
```

## 5. Hacer público el repo de GitHub

- [ ] En https://github.com/samcaballero/denue-scian-finetune: Settings → General → Danger Zone → Change repository visibility → Public.
- [ ] Revisar que el README se vea bien y que sus enlaces funcionen.

## Pendientes de la auditoría para revisar con calma

- **Pasada extra contra todo el DENUE:** 645 coincidencias parciales quedaron sin categoría: 481 llaves de 2 o más palabras dentro de una frase más larga y 164 nombres dentro de una palabra o de menos de 5 caracteres. Por el criterio de Samuel no detienen. Su detalle, con las llaves, está en `data/auditoria/` (ignorado por git).
- **Aviso del crédito:** una frase del aviso contiene dos palabras seguidas que son la llave de un negocio. Se conservó porque es el texto actual del crédito, que Samuel pidió mantener; está en el README y en las dos cards.
