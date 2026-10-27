@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

if not exist build mkdir build

cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /arch:AVX2 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS ^
   /Illama.cpp/include /Illama.cpp/ggml/include /Illama.cpp/ggml/src ^
   scripts/bench_q5k_isolation.cpp ^
   /Fe:build/bench_q5k_isolation.exe ^
   /link kernel32.lib
if errorlevel 1 exit /b 1

build\bench_q5k_isolation.exe
exit /b %errorlevel%
