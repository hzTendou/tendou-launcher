@echo off
setlocal
cd /d "%~dp0.."
if not exist build mkdir build
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /arch:AVX2 /EHsc /std:c++17 scripts\mem_bench.cpp /Fo:build\mem_bench.obj /Fe:build\mem_bench.exe
build\mem_bench.exe
exit /b %errorlevel%
