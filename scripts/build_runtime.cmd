@echo off
setlocal
cd /d "%~dp0.."
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
copy /y src\atlas\atlas-engine.cpp llama.cpp\examples\atlas-engine\atlas-engine.cpp >nul
if errorlevel 1 exit /b 1
copy /y src\atlas\atlas-engine.h llama.cpp\examples\atlas-engine\atlas-engine.h >nul
if errorlevel 1 exit /b 1
rem Relink Atlas against the local Release libraries. MTP DLL changes have separate build scripts.
cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /DGGML_USE_CUDA /DGGML_USE_CPU /DLLAMA_SHARED /DGGML_SHARED /DGGML_BACKEND_SHARED /DLLAMA_SUBPROCESS /Illama.cpp/common /Illama.cpp/include /Illama.cpp/ggml/include /Illama.cpp/vendor /Illama.cpp/vendor/sheredom llama.cpp/examples/atlas-engine/atlas-engine.cpp /Fo:llama.cpp/build/examples/atlas-engine/atlas-runtime-audit.obj /Fe:llama.cpp/build/bin/Release/llama-atlas-engine.exe /link llama.cpp/build/common/Release/llama-common.lib llama.cpp/build/common/Release/llama-common-base.lib llama.cpp/build/src/Release/llama.lib llama.cpp/build/ggml/src/Release/ggml.lib llama.cpp/build/ggml/src/Release/ggml-cpu.lib llama.cpp/build/ggml/src/Release/ggml-base.lib llama.cpp/build/ggml/src/ggml-cuda/Release/ggml-cuda.lib kernel32.lib user32.lib advapi32.lib ws2_32.lib
exit /b %errorlevel%
