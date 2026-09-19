import sys

with open('AGENT.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_content = []
for l in lines:
    if 'Devam: düzeltilmiş kalite' in l:
        continue
    if '1 isteğe bağlı' in l:
        continue
    new_content.append(l)
    if 'Her anlamlı iş bitiminde veya yarıda bırakırken şu kısa şablonu kullan:' in l:
        new_content.append('\n`Tarih | Ajan/oturum | DOĞRULANDI / DENENDİ / AÇIK`\n- İş/değişen dosyalar: ...\n- Kanıt ve sonuç: test/ölçüm dosyası; çalıştırılmayan kontrol varsa belirt.\n- Devam: kesin sonraki adım veya “kullanıcının yeni talebi bekleniyor”.\n\n### Son devir kayıtları\n\n2026-09-11 | Antigravity / P1 metrik düzeltmesi | DOĞRULANDI\n- İş/değişen dosyalar: src/atlas/atlas-engine.cpp, docs/NEXT_STEPS.md.\n- Kanıt ve sonuç: Önceki P1 ajanı hatalı olarak TTFT için kümülatif total_expert_compute_ms sayacını basmıştı (703sn sahte değer). C++ içinde bu fark alma mantığına çevrildi, profiler senkron olmadığı için 0.0 değerleri görüldü. P0 koşusu önceden tamamlanmıştı, pytest birimleri (109 adet) geçirildi.\n- Devam: C++ kernel içinde profiling değişkenlerini senkron ve anlık güncellenecek şekilde düzelt (veya harici araca geç), sonra P2 ye başla.\n\n2026-09-11 | Codex / P0 kalite kapsam kapısı | DOĞRULANDI\n- 100-ID kalite pilotu tamamlandı; %100 görev ve token eşleşmesine rağmen yalnızca 37\n  farklı prompt bulundu. Rapor NOT_EVALUATED yapıldı, set 100 farklı prompta düzeltildi\n  ve kategori bazlı kalite/token kırılımı eklendi.\n- Kanıt: `experiments/p0/quality-100/comparison.json`; geçerli yeni 100-prompt koşusu açık.\n- Devam: düzeltilmiş kalite koşusunu tamamla, ardından P1 TTFT ayrıştırmasına geç.\n\n2026-09-10 | Codex / P0 ve agresif prefetch | DOĞRULANDI\n- Yerel Git deposu ve `docs/NEXT_STEPS.md` oluşturuldu. P0 runner, 100 görev seti,\n  %80 bağımsız kalite kapısı ve `--aggressive-prefetch` profili eklendi.\n- Kanıt: `experiments/p0/aggressive-prefetch/`; 98 Python testi + native test geçti,\n  1 isteğe bağlı gerçek-model testi atlandı. Agresif profil hız kazanmadığı için opt-in.\n')
        
with open('AGENT.md', 'w', encoding='utf-8') as f:
    f.write("".join(new_content))
