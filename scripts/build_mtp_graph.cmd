@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0..\llama.cpp\build\src"
cl /c /I"C:\USERS\ALI\WEBPROJECTS\ATLAS-ENGINE-V1\LLAMA.CPP\SRC\." /I"C:\USERS\ALI\WEBPROJECTS\ATLAS-ENGINE-V1\LLAMA.CPP\SRC\..\INCLUDE" /I"C:\USERS\ALI\WEBPROJECTS\ATLAS-ENGINE-V1\LLAMA.CPP\GGML\SRC\..\INCLUDE" /I"C:\PROGRAM FILES\NVIDIA GPU COMPUTING TOOLKIT\CUDA\V13.3\INCLUDE" /nologo /W1 /WX- /diagnostics:column /O2 /Ob2 /D _WINDLL /D _MBCS /D WIN32 /D _WINDOWS /D NDEBUG /D "LLAMA_VERSION=\"0.3.0-dev\"" /D "LLAMA_COMMIT=\"5d1f1b03e\"" /D LLAMA_BUILD /D LLAMA_SHARED /D _CRT_SECURE_NO_WARNINGS /D GGML_USE_CPU /D GGML_USE_CUDA /D GGML_SHARED /D GGML_BACKEND_SHARED /D "CMAKE_INTDIR=\"Release\"" /D llama_EXPORTS /EHsc /MD /std:c++17 /Fo"LLAMA.DIR\RELEASE\\" /Fd"LLAMA.DIR\RELEASE\VC145.PDB" /external:W1 /TP  /utf-8 /bigobj C:\USERS\ALI\WEBPROJECTS\ATLAS-ENGINE-V1\LLAMA.CPP\SRC\MODELS\QWEN4EXP.CPP
if errorlevel 1 exit /b 1
link @"%~dp0..\experiments\runtime_audit\mtp_sources\link.rsp"
exit /b %errorlevel%
