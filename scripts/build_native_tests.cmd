@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

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
    echo [ERROR] vcvars64.bat not found.
    exit /b 1
)
call "%VCVARS%" || exit /b 1

if exist "llama.cpp\build-win\src\llama.lib" (
    set "BUILD_DIR=llama.cpp\build-win"
    set "L_COMMON=llama.cpp\build-win\common\llama-common.lib"
    set "L_LLAMA=llama.cpp\build-win\src\llama.lib"
    set "L_GGML=llama.cpp\build-win\ggml\src\ggml.lib"
    set "L_CPU=llama.cpp\build-win\ggml\src\ggml-cpu.lib"
    set "L_BASE=llama.cpp\build-win\ggml\src\ggml-base.lib"
    set "L_CUDA=llama.cpp\build-win\ggml\src\ggml-cuda\ggml-cuda.lib"
) else (
    set "BUILD_DIR=llama.cpp\build"
    set "L_COMMON=llama.cpp\build\common\Release\llama-common.lib"
    set "L_LLAMA=llama.cpp\build\src\Release\llama.lib"
    set "L_GGML=llama.cpp\build\ggml\src\Release\ggml.lib"
    set "L_CPU=llama.cpp\build\ggml\src\Release\ggml-cpu.lib"
    set "L_BASE=llama.cpp\build\ggml\src\Release\ggml-base.lib"
    set "L_CUDA=llama.cpp\build\ggml\src\ggml-cuda\Release\ggml-cuda.lib"
)
set "L_COMMON_BASE="
if exist "llama.cpp\build\common\Release\llama-common-base.lib" set "L_COMMON_BASE=llama.cpp\build\common\Release\llama-common-base.lib"
if exist "llama.cpp\build-win\common\llama-common-base.lib" set "L_COMMON_BASE=llama.cpp\build-win\common\llama-common-base.lib"

if not exist "llama.cpp\build\bin\Release" mkdir "llama.cpp\build\bin\Release"
set "TEST_INC=/Isrc/atlas /Illama.cpp /Illama.cpp/examples/atlas-engine /Illama.cpp/common /Illama.cpp/include /Illama.cpp/src /Illama.cpp/ggml/include /Illama.cpp/ggml/src /Illama.cpp/ggml/src/ggml-cpu /Illama.cpp/ggml/src/ggml-cpu/arch/x86 /Illama.cpp/vendor /Illama.cpp/vendor/sheredom"
set "TEST_DEFS=/DNDEBUG /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX /DWIN32_LEAN_AND_MEAN /DGGML_USE_CUDA /DGGML_USE_CPU /DLLAMA_SHARED /DGGML_SHARED /DGGML_BACKEND_SHARED"

cl /nologo /O2 /MD /EHsc /std:c++17 /arch:AVX2 /utf-8 %TEST_DEFS% %TEST_INC% ^
    tests/native/test_async_prefetch.cpp ^
    /Fo:"!BUILD_DIR!\atlas-native-tests.obj" ^
    /Fe:"llama.cpp\build\bin\Release\atlas-native-tests.exe" ^
    /link "!L_COMMON!" !L_COMMON_BASE! "!L_LLAMA!" "!L_GGML!" "!L_CPU!" "!L_BASE!" "!L_CUDA!" kernel32.lib user32.lib advapi32.lib ws2_32.lib || exit /b 1

cl /nologo /O2 /MD /EHsc /std:c++17 /arch:AVX2 /utf-8 %TEST_DEFS% %TEST_INC% ^
    tests/native/test_expert_partition.cpp ^
    /Fo:"!BUILD_DIR!\atlas-partition-tests.obj" ^
    /Fe:"llama.cpp\build\bin\Release\atlas-partition-tests.exe" ^
    /link "!L_COMMON!" !L_COMMON_BASE! "!L_LLAMA!" "!L_GGML!" "!L_CPU!" "!L_BASE!" "!L_CUDA!" kernel32.lib user32.lib advapi32.lib ws2_32.lib || exit /b 1

echo [BUILD_NATIVE_TESTS] Success: Windows atlas-native-tests.exe (Gates 1-46) and atlas-partition-tests.exe built.
exit /b 0
