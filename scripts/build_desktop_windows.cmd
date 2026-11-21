@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo ============================================================================
echo   TENDOU STUDIO v0.16.3 — WINDOWS 11 TAURI v2 DESKTOP BUILD (WebView2)
echo ============================================================================
echo.

where npm >nul 2>&1 || (echo [ERROR] Node.js / npm not found in PATH. & exit /b 1)
where cargo >nul 2>&1 || (echo [ERROR] Rust / cargo not found in PATH. Install via https://rustup.rs & exit /b 1)

echo [1/3] Building Tendou Studio React+TS frontend (ui\dist)...
call npm --prefix ui install --no-audit --no-fund || exit /b 1
call npm --prefix ui run build || exit /b 1

echo [2/3] Building Windows Tauri v2 desktop application (WebView2)...
cargo tauri --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing tauri-cli via cargo (one-time)...
    cargo install tauri-cli --version "^2" --locked || exit /b 1
)
pushd desktop\src-tauri
cargo tauri build --bundles msi,nsis
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" exit /b %RC%

echo [3/3] Windows Tauri v2 Desktop build complete!
echo       Installers: desktop\src-tauri\target\release\bundle\msi\ and nsis\
exit /b 0
