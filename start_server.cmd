@echo off
setlocal enabledelayedexpansion
title Tendou Launcher v0.16.3-stable — Studio Web UI & Local AI Server
cd /d "%~dp0"
set "ROOT_DIR=%CD%"

:: Prepend v0.16.3-stable & fallback Windows runtime DLL directories to PATH
set "EXTRA_PATH=%ROOT_DIR%\releases\tendou-v0.16.3-stable\bin;%ROOT_DIR%\llama.cpp\build\bin\Release;%ROOT_DIR%\releases\tendou-v0.16.2-stable\bin;%ROOT_DIR%\releases\tendou-v0.16.1-stable\bin;%ROOT_DIR%\releases\tendou-v0.16.0-stable\bin;%ROOT_DIR%\releases\tendou-v0.15.0-stable\bin;%ROOT_DIR%\releases\tendou-v0.14.0-stable\bin"
if defined CUDA_PATH if exist "%CUDA_PATH%\bin" set "EXTRA_PATH=%EXTRA_PATH%;%CUDA_PATH%\bin"
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin" set "EXTRA_PATH=%EXTRA_PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin" set "EXTRA_PATH=%EXTRA_PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
set "PATH=%EXTRA_PATH%;%PATH%"

:: Runtime threading & CUDA host registration defaults (1:1 parity with start_server.sh)
if not defined GGML_CUDA_REGISTER_HOST set "GGML_CUDA_REGISTER_HOST=1"
if not defined OMP_PROC_BIND set "OMP_PROC_BIND=close"
if not defined OMP_PLACES set "OMP_PLACES=cores"

:: Resolve Python interpreter (.venv-audit -> .venv -> system python)
set "PYTHON_BIN="
if exist "%ROOT_DIR%\.venv-audit\Scripts\python.exe" (
    set "PYTHON_BIN=%ROOT_DIR%\.venv-audit\Scripts\python.exe"
) else if exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
) else (
    set "PYTHON_BIN=python"
)

echo ============================================================================
echo   TENDOU LAUNCHER v0.16.3-stable — TENDOU STUDIO + OPENAI/ANTHROPIC API
echo   Web UI : http://127.0.0.1:8000
echo   API    : http://127.0.0.1:8000/v1  ^|  /v1/messages
echo   Python : %PYTHON_BIN%
echo ============================================================================
echo.

"%PYTHON_BIN%" "%ROOT_DIR%\atlas_server.py" %*
exit /b %errorlevel%
