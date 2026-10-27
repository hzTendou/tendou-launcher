# Atlas Engine — Qwen3.8 Flash-Next strict trace prototype

Bu paket əvvəlki Atlas V7 kodlarını saxlayır və üstünə `atlas_qwen38_trace.py`
əlavə edir.

## Nəyi düzəldir?

Qwen metadata:
- architecture: `qwen4exp`
- 48 block
- 512 expert
- token başına 10 selected expert
- context: 262144
- quant: Q5_K_M

`atlas_qwen38_trace.py` trace-i **yalnız bütün session tam doğrulanandan sonra**
atomik şəkildə publish edir. Aşağıdakılardan biri pozularsa trace uğursuz sayılır:

1. `ATLAS_DONE` yoxdur
2. prompt call=0 yoxdur
3. hər token üçün 48 layer-in hamısı yoxdur
4. top-k 10 deyil
5. expert ID 0..511 xaricindədir
6. top-k daxilində duplicate expert var
7. prompt/decode call ardıcıllığı qırılıb
8. routed prompt token sayı binary-nin bildirdiyi prompt token sayına uyğun deyil
9. decode call sayı `ATLAS_DONE.n_decoded` ilə uyğun deyil

Beləliklə yarımçıq və ya səssizcə korlanmış trace `--out` faylına yazılmır.

## Vacib həqiqət

Bu collector **real router trace** toplamaq üçün modelin routing graph-ını həqiqətən
icra edən `atlas-trace.cpp` ilə build olunmuş llama.cpp binary-si tələb edir.
Metadata və architecture şəkli təkbaşına real expert seçimlərini verə bilməz.
Ona görə heç bir prototip modelin router/forward pass-ını icra etmədən "dəqiq real
trace" yarada bilməz.

## Bir dəfəlik build

Mövcud `atlas-trace.cpp`-ni uyğun llama.cpp checkout-dakı
`examples/eval-callback/eval-callback.cpp` ilə əvəz et və həmin llama.cpp versiyası
üçün build et. Sonra:

```powershell
python atlas_qwen38_trace.py `
  --binary C:\llama.cpp\build\bin\Release\llama-eval-callback.exe `
  --model D:\models\Qwen3.8-Flash-Next.Q5_K_M-00001-of-00003.gguf `
  --out trace_qwen38.jsonl `
  --max-new-tokens 64 `
  --ngl 99 `
  --ctx-size 4096
```

Split GGUF-dursa `--model` birinci shard-a işarə etməlidir və digər shard-lar eyni
qovluqda düzgün adlarla mövcud olmalıdır.

Çıxış:
- `trace_qwen38.jsonl`
- `trace_qwen38.jsonl.manifest.json`

`PASS — atomic trace written` görmədən trace-i benchmark üçün istifadə etmə.

## Növbəti addım

```powershell
python analyze_trace.py --trace trace_qwen38.jsonl --expert-size-mb 1.2 --cache-sizes 8,16,32,64,128,256 --nvme-gbps 7.0
```

Fiziki ölçüləri real Qwen GGUF-dan almaq üçün:

```powershell
python build_physical_map.py --gguf D:\models\Qwen3.8-Flash-Next.Q5_K_M-00001-of-00003.gguf --out atlas_physical_map_qwen38.json
```
