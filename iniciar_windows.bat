@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo   RAG LOCAL PARA OBSIDIAN CON OLLAMA
echo ==============================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo No se ha encontrado Python.
    echo Instala Python 3.11 o 3.12 y marca "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    py -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"

echo Usando las dependencias ya instaladas en el entorno virtual...

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo Se abrira el archivo .env.
    echo Revisa OBSIDIAN_VAULT_PATH y los modelos locales.
    echo Guarda y cierra el Bloc de notas para continuar.
    notepad ".env"
)

set "OLLAMA_BASE_URL=http://127.0.0.1:11434"
set "OLLAMA_CHAT_MODEL=qwen3.5:0.8b"
set "OLLAMA_EMBEDDING_MODEL=nomic-embed-text"
for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
    if /I "%%A"=="OLLAMA_BASE_URL" set "OLLAMA_BASE_URL=%%B"
    if /I "%%A"=="OLLAMA_CHAT_MODEL" set "OLLAMA_CHAT_MODEL=%%B"
    if /I "%%A"=="OLLAMA_EMBEDDING_MODEL" set "OLLAMA_EMBEDDING_MODEL=%%B"
)

python -c "from app.config import settings" >nul
if errorlevel 1 (
    echo La configuracion de Ollama no es local o no es valida.
    echo Revisa OLLAMA_BASE_URL y los modelos configurados en .env.
    goto :error
)

echo.
echo Comprobando Ollama local...
where ollama >nul 2>nul
if errorlevel 1 (
    echo AVISO: no se encuentra el comando ollama.
    echo Instala Ollama para Windows desde https://ollama.com/download/windows
    echo Despues ejecuta:
    echo   ollama pull %OLLAMA_CHAT_MODEL%
    echo   ollama pull %OLLAMA_EMBEDDING_MODEL%
    goto :launch
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { Invoke-RestMethod -Uri ($env:OLLAMA_BASE_URL.TrimEnd('/') + '/api/version') -TimeoutSec 5 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo AVISO: Ollama no responde en %OLLAMA_BASE_URL%.
    echo Abre la aplicacion Ollama o ejecuta en otra terminal:
    echo   ollama serve
    goto :launch
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { $body = @{model=$env:OLLAMA_CHAT_MODEL} | ConvertTo-Json; Invoke-RestMethod -Uri ($env:OLLAMA_BASE_URL.TrimEnd('/') + '/api/show') -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo AVISO: falta el modelo de chat %OLLAMA_CHAT_MODEL%.
    echo Ejecuta:
    echo   ollama pull %OLLAMA_CHAT_MODEL%
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { $body = @{model=$env:OLLAMA_EMBEDDING_MODEL} | ConvertTo-Json; Invoke-RestMethod -Uri ($env:OLLAMA_BASE_URL.TrimEnd('/') + '/api/show') -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 10 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo AVISO: falta el modelo de embeddings %OLLAMA_EMBEDDING_MODEL%.
    echo Ejecuta:
    echo   ollama pull %OLLAMA_EMBEDDING_MODEL%
)

:launch
echo.
echo Comprobando si ya hay una instancia del RAG en el puerto 8000...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "try { $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/status' -TimeoutSec 3; if ($status.capabilities.project_agents -eq $true) { exit 0 }; exit 1 } catch { exit 2 }"
if errorlevel 2 goto :start_server
if errorlevel 1 goto :old_server
goto :open_existing

:start_server
echo.
echo Abriendo http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
python -m app.main
goto :eof

:open_existing
echo Ya hay una instancia actual ejecutandose en el puerto 8000.
echo Abriendo la interfaz existente...
start "" "http://127.0.0.1:8000"
goto :eof

:old_server
echo.
echo AVISO: el puerto 8000 esta ocupado por una version anterior del RAG.
echo Cierra la ventana o el proceso antiguo y vuelve a ejecutar este script.
echo La version nueva necesita las capacidades project_agents y project_memory.
pause
goto :eof

:error
echo.
echo Se produjo un error durante el arranque.
pause
exit /b 1
