# Demo de la charla (Fase 4, paso 19, Tema B): Qwen3-0.6B base y afinado en esta laptop, sin GPU y sin red.
#
# Uso, desde cualquier carpeta de PowerShell 5.1:
#   powershell -ExecutionPolicy Bypass -File "C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune\scripts\demo.ps1"
#
# Los nombres salen de data\demo\nombres.txt (uno por linea; ignorado por git) y despues se teclean.
# Argumentos extra pasan tal cual a src.demo, por ejemplo: --ancho 48, --pausa o --nombres "C:\ruta\otros.txt".
# Este archivo va solo en ASCII: PowerShell 5.1 lee como ANSI los .ps1 sin BOM.

$Repo = "C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune"
$Python = "C:\sca\fine-tuning\proyecto-denue\denue-scian-finetune\.venv\Scripts\python.exe"
$LlamaServer = "C:\sca\fine-tuning\proyecto-denue\herramientas\llama.cpp-b12359\bin-win-cpu-x64\llama-server.exe"
$DirGguf = "C:\sca\fine-tuning\proyecto-denue\modelos\gguf"

Push-Location $Repo
try {
    & $Python -m src.demo --llama-server $LlamaServer --dir-gguf $DirGguf @args
    $Codigo = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $Codigo
