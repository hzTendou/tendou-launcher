@echo off
set ATLAS_DEEP_TELEMETRY=0
llama.cpp\build\bin\Release\llama-cli.exe -m C:\Users\Ali\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf -p "<|im_start|>user\n17 ile 19 sayilarinin carpimini hesapla. Yalnizca sayiyi yaz. Sonrasinda bosluk birakarak ayni islemi uzunca acikla. Bunu yaparken mumkun oldugunca fazla detay ver ki token sayisi 64'u gecsin.<|im_end|>\n<|im_start|>assistant\n" -n 128 -t 16 --atlas-physical-map config\atlas_physical_map_35b.json
