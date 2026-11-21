@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

:: Locate MSVC vcvars64.bat via vswhere or standard paths
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
    )
)
if not defined VCVARS (
    for %%p in (
        "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if exist %%~p if not defined VCVARS set "VCVARS=%%~p"
    )
)
if not defined VCVARS (
    echo [ERROR] vcvars64.bat not found. Run scripts\build_windows_all.cmd from a VS Developer Command Prompt.
    exit /b 1
)
call "%VCVARS%" || exit /b 1

if not exist "llama.cpp\examples\atlas-engine" mkdir "llama.cpp\examples\atlas-engine"
copy /y src\atlas\atlas-engine.cpp    llama.cpp\examples\atlas-engine\atlas-engine.cpp >nul || exit /b 1
copy /y src\atlas\atlas-engine.h      llama.cpp\examples\atlas-engine\atlas-engine.h >nul || exit /b 1
copy /y src\atlas\atlas-moe-cache.h   llama.cpp\examples\atlas-engine\atlas-moe-cache.h >nul || exit /b 1
copy /y src\atlas\atlas-iouring-dma.h llama.cpp\examples\atlas-engine\atlas-iouring-dma.h >nul || exit /b 1
copy /y src\atlas\CMakeLists.txt      llama.cpp\examples\atlas-engine\CMakeLists.txt >nul || exit /b 1

if exist "llama.cpp\build-win\CMakeCache.txt" (
    cmake --build llama.cpp\build-win --target llama-atlas-engine llama-server -j %NUMBER_OF_PROCESSORS% || exit /b 1
    set "SRC_BIN=llama.cpp\build-win\bin"
) else if exist "llama.cpp\build\CMakeCache.txt" (
    cmake --build llama.cpp\build --config Release --target llama-atlas-engine llama-server -j %NUMBER_OF_PROCESSORS% || exit /b 1
    set "SRC_BIN=llama.cpp\build\bin\Release"
) else (
    echo [INFO] No CMake build tree found; delegating to scripts\build_windows_all.cmd...
    call scripts\build_windows_all.cmd
    exit /b %errorlevel%
)

if not exist "llama.cpp\build\bin\Release" mkdir "llama.cpp\build\bin\Release"
if not exist "releases\tendou-v0.16.3-stable\bin" mkdir "releases\tendou-v0.16.3-stable\bin"
copy /y "!SRC_BIN!\llama-atlas-engine.exe" llama.cpp\build\bin\Release\tendou-engine.exe >nul || exit /b 1
copy /y "!SRC_BIN!\llama-atlas-engine.exe" releases\tendou-v0.16.3-stable\bin\tendou-engine.exe >nul || exit /b 1
copy /y "!SRC_BIN!\llama-atlas-engine.exe" releases\tendou-v0.16.3-stable\bin\llama-atlas-engine.exe >nul || exit /b 1
for %%f in ("!SRC_BIN!\*.dll") do (
    copy /y "%%f" releases\tendou-v0.16.3-stable\bin\ >nul
)
echo [BUILD_RUNTIME] Success: Windows tendou-engine.exe synced to releases\tendou-v0.16.3-stable\bin\
exit /b 0
