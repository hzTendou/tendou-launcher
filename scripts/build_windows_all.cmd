@echo off
setlocal enabledelayedexpansion

title Tendou Launcher v0.16.3-stable — Windows Full Build (MSVC + CUDA)
cd /d "%~dp0.."
set "ROOT_DIR=%CD%"

echo ============================================================================
echo   TENDOU LAUNCHER v0.16.3-stable — WINDOWS FULL NATIVE BUILD
echo   Targets: ggml-cuda.dll, ggml-cpu.dll (AVX-512 IQ4_XS_x4 + IQ3_XXS_x4 + IQ3_S_x4),
echo            llama.dll, tendou-engine.exe, atlas-native-tests.exe (Gates 1-48)
echo ============================================================================
echo.

:: 1. Locate Visual Studio x64 Build Environment (vswhere -> fallback paths)
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
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if exist %%~p if not defined VCVARS set "VCVARS=%%~p"
    )
)
if not defined VCVARS (
    echo [ERROR] Visual Studio C++ x64 Build Tools (vcvars64.bat) not found!
    echo         Install "Desktop development with C++" via Visual Studio Installer.
    exit /b 1
)
echo [1/6] Initializing MSVC x64 environment: %VCVARS%
call "%VCVARS%"
if errorlevel 1 exit /b 1

:: Verify CMake & CUDA nvcc
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cmake not found in PATH. Install CMake 3.26+ or VS CMake component.
    exit /b 1
)
where nvcc >nul 2>&1
if errorlevel 1 (
    if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" (
        set "PATH=%CUDA_PATH%\bin;%PATH%"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
        set "PATH=!CUDA_PATH!\bin;!PATH!"
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
        set "PATH=!CUDA_PATH!\bin;!PATH!"
    )
)
where nvcc >nul 2>&1
if errorlevel 1 (
    echo [WARN] nvcc.exe not found in PATH. CMake will try default CUDA toolkit detection.
) else (
    for /f "tokens=*" %%v in ('nvcc --version ^| findstr /i "release"') do echo [CUDA] %%v
)

:: 2. Sync Tendou Atlas Engine sources into llama.cpp/examples/atlas-engine
echo [2/6] Syncing src\atlas\* into llama.cpp\examples\atlas-engine\...
if not exist "llama.cpp\examples\atlas-engine" mkdir "llama.cpp\examples\atlas-engine"
copy /y "src\atlas\atlas-engine.cpp"    "llama.cpp\examples\atlas-engine\atlas-engine.cpp" >nul || exit /b 1
copy /y "src\atlas\atlas-engine.h"      "llama.cpp\examples\atlas-engine\atlas-engine.h" >nul || exit /b 1
copy /y "src\atlas\atlas-moe-cache.h"   "llama.cpp\examples\atlas-engine\atlas-moe-cache.h" >nul || exit /b 1
copy /y "src\atlas\atlas-iouring-dma.h" "llama.cpp\examples\atlas-engine\atlas-iouring-dma.h" >nul || exit /b 1
copy /y "src\atlas\CMakeLists.txt"      "llama.cpp\examples\atlas-engine\CMakeLists.txt" >nul || exit /b 1

:: 3. Configure CMake (Clean Ninja if available, otherwise default VS multi-config)
set "USE_NINJA=0"
where ninja >nul 2>&1 && set "USE_NINJA=1"

echo [3/6] Configuring CMake (GGML_CUDA=ON, GGML_NATIVE=ON, sm_120;89;86)...
if "%USE_NINJA%"=="1" (
    set "BUILD_DIR=llama.cpp\build-win"
    set "BIN_OUT=llama.cpp\build-win\bin"
    set "LIB_BASE=llama.cpp\build-win"
    cmake -S llama.cpp -B "!BUILD_DIR!" -G Ninja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DGGML_CUDA=ON ^
        -DGGML_NATIVE=ON ^
        -DBUILD_SHARED_LIBS=ON ^
        -DCMAKE_CUDA_ARCHITECTURES="120;89;86"
    if errorlevel 1 exit /b 1
    echo [4/6] Building ggml-cuda, ggml-cpu, llama, llama-atlas-engine, llama-server...
    cmake --build "!BUILD_DIR!" --target ggml-cuda ggml-cpu llama llama-atlas-engine llama-server -j %NUMBER_OF_PROCESSORS%
    if errorlevel 1 exit /b 1
) else (
    set "BUILD_DIR=llama.cpp\build"
    set "BIN_OUT=llama.cpp\build\bin\Release"
    set "LIB_BASE=llama.cpp\build"
    cmake -S llama.cpp -B "!BUILD_DIR!" ^
        -DGGML_CUDA=ON ^
        -DGGML_NATIVE=ON ^
        -DBUILD_SHARED_LIBS=ON ^
        -DCMAKE_CUDA_ARCHITECTURES="120;89;86"
    if errorlevel 1 exit /b 1
    echo [4/6] Building Release targets via MSBuild...
    cmake --build "!BUILD_DIR!" --config Release --target ggml-cuda ggml-cpu llama llama-atlas-engine llama-server -j %NUMBER_OF_PROCESSORS%
    if errorlevel 1 exit /b 1
)

:: 4. Sync compiled .exe/.dll into llama.cpp/build/bin/Release and releases/tendou-v0.16.3-stable/bin
echo [5/6] Syncing v0.16.3-stable Windows binaries...
if not exist "llama.cpp\build\bin\Release" mkdir "llama.cpp\build\bin\Release"
if not exist "releases\tendou-v0.16.3-stable\bin" mkdir "releases\tendou-v0.16.3-stable\bin"

copy /y "!BIN_OUT!\llama-atlas-engine.exe" "llama.cpp\build\bin\Release\tendou-engine.exe" >nul || exit /b 1
copy /y "!BIN_OUT!\llama-atlas-engine.exe" "releases\tendou-v0.16.3-stable\bin\tendou-engine.exe" >nul || exit /b 1
copy /y "!BIN_OUT!\llama-atlas-engine.exe" "releases\tendou-v0.16.3-stable\bin\llama-atlas-engine.exe" >nul || exit /b 1
if /i not "!BIN_OUT!"=="llama.cpp\build\bin\Release" (
    copy /y "!BIN_OUT!\llama-atlas-engine.exe" "llama.cpp\build\bin\Release\llama-atlas-engine.exe" >nul || exit /b 1
)
if exist "!BIN_OUT!\llama-server.exe" (
    if /i not "!BIN_OUT!"=="llama.cpp\build\bin\Release" (
        copy /y "!BIN_OUT!\llama-server.exe" "llama.cpp\build\bin\Release\llama-server.exe" >nul
    )
    copy /y "!BIN_OUT!\llama-server.exe" "releases\tendou-v0.16.3-stable\bin\llama-server.exe" >nul
)
for %%f in ("!BIN_OUT!\*.dll") do (
    if /i not "!BIN_OUT!"=="llama.cpp\build\bin\Release" (
        copy /y "%%f" "llama.cpp\build\bin\Release\" >nul
    )
    copy /y "%%f" "releases\tendou-v0.16.3-stable\bin\" >nul
)

:: 5. Compile & Run Native Tests (Gates 1-48 + Partition Tests)
echo [6/6] Compiling and running native tests (Gates 1-48)...
if "%USE_NINJA%"=="1" (
    set "L_COMMON=!LIB_BASE!\common\llama-common.lib"
    set "L_LLAMA=!LIB_BASE!\src\llama.lib"
    set "L_GGML=!LIB_BASE!\ggml\src\ggml.lib"
    set "L_CPU=!LIB_BASE!\ggml\src\ggml-cpu.lib"
    set "L_BASE=!LIB_BASE!\ggml\src\ggml-base.lib"
    set "L_CUDA=!LIB_BASE!\ggml\src\ggml-cuda\ggml-cuda.lib"
) else (
    set "L_COMMON=!LIB_BASE!\common\Release\llama-common.lib"
    set "L_LLAMA=!LIB_BASE!\src\Release\llama.lib"
    set "L_GGML=!LIB_BASE!\ggml\src\Release\ggml.lib"
    set "L_CPU=!LIB_BASE!\ggml\src\Release\ggml-cpu.lib"
    set "L_BASE=!LIB_BASE!\ggml\src\Release\ggml-base.lib"
    set "L_CUDA=!LIB_BASE!\ggml\src\ggml-cuda\Release\ggml-cuda.lib"
)
set "L_COMMON_BASE="
if exist "!LIB_BASE!\common\Release\llama-common-base.lib" set "L_COMMON_BASE=!LIB_BASE!\common\Release\llama-common-base.lib"
if exist "!LIB_BASE!\common\llama-common-base.lib" set "L_COMMON_BASE=!LIB_BASE!\common\llama-common-base.lib"

set "TEST_INC=/Isrc/atlas /Illama.cpp /Illama.cpp/examples/atlas-engine /Illama.cpp/common /Illama.cpp/include /Illama.cpp/src /Illama.cpp/ggml/include /Illama.cpp/ggml/src /Illama.cpp/ggml/src/ggml-cpu /Illama.cpp/ggml/src/ggml-cpu/arch/x86 /Illama.cpp/vendor /Illama.cpp/vendor/sheredom"
set "TEST_DEFS=/DNDEBUG /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN /DGGML_USE_CUDA /DGGML_USE_CPU /DLLAMA_SHARED /DGGML_SHARED /DGGML_BACKEND_SHARED"

cl /nologo /O2 /MD /EHsc /std:c++17 /arch:AVX2 /utf-8 %TEST_DEFS% %TEST_INC% ^
    tests/native/test_async_prefetch.cpp ^
    /Fo:"!BUILD_DIR!\atlas-native-tests.obj" ^
    /Fe:"llama.cpp\build\bin\Release\atlas-native-tests.exe" ^
    /link "!L_COMMON!" !L_COMMON_BASE! "!L_LLAMA!" "!L_GGML!" "!L_CPU!" "!L_BASE!" "!L_CUDA!" kernel32.lib user32.lib advapi32.lib ws2_32.lib
if errorlevel 1 exit /b 1

cl /nologo /O2 /MD /EHsc /std:c++17 /arch:AVX2 /utf-8 %TEST_DEFS% %TEST_INC% ^
    tests/native/test_expert_partition.cpp ^
    /Fo:"!BUILD_DIR!\atlas-partition-tests.obj" ^
    /Fe:"llama.cpp\build\bin\Release\atlas-partition-tests.exe" ^
    /link "!L_COMMON!" !L_COMMON_BASE! "!L_LLAMA!" "!L_GGML!" "!L_CPU!" "!L_BASE!" "!L_CUDA!" kernel32.lib user32.lib advapi32.lib ws2_32.lib
if errorlevel 1 exit /b 1

set "PATH=%ROOT_DIR%\releases\tendou-v0.16.3-stable\bin;%PATH%"
"llama.cpp\build\bin\Release\atlas-native-tests.exe"
if errorlevel 1 exit /b 1
"llama.cpp\build\bin\Release\atlas-partition-tests.exe"
if errorlevel 1 exit /b 1

echo.
echo ============================================================================
echo   [SUCCESS] tendou-v0.16.3-stable Windows build + Gates 1-48 PASSED!
echo   Binaries: releases\tendou-v0.16.3-stable\bin\tendou-engine.exe
echo   Next:     Run start_server.cmd --port 8000 and open http://127.0.0.1:8000
echo ============================================================================
exit /b 0
