@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0..\llama.cpp\build\ggml\src"
rem Recorded original Release flags, with paths relocated for this audited checkout.
cl @"%~dp0..\experiments\runtime_audit\2026-09-10\cpu-compile.rsp"
if errorlevel 1 exit /b 1
link @"%~dp0..\experiments\runtime_audit\2026-09-10\cpu-link.rsp"
exit /b %errorlevel%
