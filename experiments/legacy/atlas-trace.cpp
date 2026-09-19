// Atlas Engine — GGUF MoE Router Trace Collector
//
// examples/eval-callback/eval-callback.cpp faylının yerinə qoyulur (eyni CMake
// target-i "llama-eval-callback" istifadə edir, əlavə CMake dəyişikliyi lazım deyil).
//
// Modelin router-inin hər layer/token üçün seçdiyi expert-ləri ("ffn_moe_topk"
// tensoru) stdout-a maşın-oxunaqlı sətirlər kimi çap edir:
//
//   ATLAS_MOE call=<N> layer=<L> n_used=<K> n_tok=<T> vals=e0,e1,e2,...
//
// vals sırası: token0-un K expert-i, sonra token1-in K expert-i, və s.
// (yəni row-major, [n_tok][n_used])
//
// Prompt emalı (bir decode() çağırışı, bütün prompt token-ləri birlikdə)
// VƏ sonrakı greedy generasiya (hər addım bir token) — hər ikisi trace olunur,
// beləliklə orijinal collect_trace.py-nin (prompt + max-new-tokens) məntiqi
// GGUF üçün də qorunur.
//
// İSTİFADƏ:
//   llama-eval-callback -m model.gguf -p "Sual mətni" -n 64 --ctx-size 2048 -ngl 99

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct atlas_cb_data {
    long long call_idx = 0;
    int32_t expected_top_k = -1;  // GGUF metadata-dan oxunan gözlənilən top-k
    bool topk_mismatch_warned = false;
    const char * phase = "prompt";
};

// ggml_backend_sched_eval_callback:
//   ask=true  -> "bu tensor lazımdırmı?" sualı, YALNIZ true/false qaytarır,
//                qrafın hesablanmasına TƏSİR ETMİR (ggml-backend.cpp-də
//                doğrulanıb) — sadəcə lazımsız data-copy-ni azaldır.
//   ask=false -> data artıq host-da oxunmaq üçün hazırdır. BURADA false
//                qaytarmaq qalan qrafın hesablanmasını EFEKTİV EDİR
//                (ggml-backend.cpp: "if (need && !cb(...)) break;"),
//                ona görə bu budaqda HƏMİŞƏ true qaytarılır.
static bool atlas_moe_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (atlas_cb_data *) user_data;

    static const char * prefix = "ffn_moe_topk-";
    const size_t prefix_len = strlen(prefix);
    const bool matches = strncmp(t->name, prefix, prefix_len) == 0;

    if (ask) {
        return matches;
    }

    if (!matches) {
        return true;
    }

    int il = -1;
    sscanf(t->name + prefix_len, "%d", &il);

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    std::vector<uint8_t> tmp;
    const uint8_t * data_ptr;
    if (!is_host) {
        const size_t n_bytes = ggml_nbytes(t);
        tmp.resize(n_bytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, n_bytes);
        data_ptr = tmp.data();
    } else {
        data_ptr = (const uint8_t *) t->data;
    }

    // t: I32, shape ne[0]=n_expert_used (top-k), ne[1]=n_tokens bu decode çağırışında
    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];

    // Sanity-check: bu tensor-dakı n_used, GGUF metadata-dakı expert_used_count
    // ilə üst-üstə düşməlidir. Düşmürsə, bu tensor bizim güman etdiyimiz
    // "final selected top-k experts" DEYİL (bəlkə intermediate/fərqli bir
    // tensor-dur) — bir dəfə XƏBƏRDARLIQ çap edib davam edirik.
    if (cb_data->expected_top_k > 0 && n_used != cb_data->expected_top_k
        && !cb_data->topk_mismatch_warned) {
        fprintf(stderr,
                "[atlas-trace] XƏBƏRDARLIQ: layer=%d-də n_used=%lld, gözlənilən "
                "expert_used_count=%d ilə UYĞUN GƏLMİR. Bu, ffn_moe_topk tensor-unun "
                "güman edilən router-selected-expert semantikasını daşımaya biləcəyini "
                "göstərir — nəticələrə şübhə ilə yanaş, bu barədə bizə bildir.\n",
                il, (long long) n_used, cb_data->expected_top_k);
        cb_data->topk_mismatch_warned = true;
    }

    printf("ATLAS_MOE phase=%s call=%lld layer=%d n_used=%lld n_tok=%lld vals=",
           cb_data->phase, cb_data->call_idx, il, (long long) n_used, (long long) n_tok);

    for (int64_t i1 = 0; i1 < n_tok; ++i1) {
        for (int64_t i0 = 0; i0 < n_used; ++i0) {
            const size_t off = (size_t) i1 * t->nb[1] + (size_t) i0 * t->nb[0];
            const int32_t v = *(const int32_t *) (data_ptr + off);
            printf("%d", v);
            if (!(i1 == n_tok - 1 && i0 == n_used - 1)) {
                printf(",");
            }
        }
    }
    printf("\n");
    fflush(stdout);

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    atlas_cb_data cb_data;

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.prompt.empty()) {
        fprintf(stderr, "[atlas-trace] XƏTA: -p \"prompt mətni\" tələb olunur\n");
        return 1;
    }
    if (params.n_predict < 0) {
        params.n_predict = 32; // default, -n ilə override edilə bilər
    }

    params.cb_eval = atlas_moe_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "[atlas-trace] XƏTA: model/context yüklənmədi\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Sanity-check: modelin öz GGUF metadata-sındakı konfiqurasiya edilmiş
    // top-k (expert_used_count) dəyərini oxuyub, sonradan trace-də gördüyümüz
    // n_used ilə tutuşduracayıq. Bu, ffn_moe_topk-in həqiqətən router-in son
    // seçdiyi top-k expert ID-ləri olduğunu (intermediate tensor yox) əlavə
    // təsdiqləyir — uyğunsuzluq varsa, XƏBƏRDARLIQ çap olunur.
    char arch_buf[128] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf)) > 0) {
        char key_buf[160];
        snprintf(key_buf, sizeof(key_buf), "%s.expert_used_count", arch_buf);
        char val_buf[32] = {0};
        if (llama_model_meta_val_str(model, key_buf, val_buf, sizeof(val_buf)) > 0) {
            cb_data.expected_top_k = atoi(val_buf);
        }
    }
    fprintf(stderr, "[atlas-trace] model arch=%s, gguf-dəki expert_used_count=%d "
            "(bu, ilk ATLAS_MOE sətrindəki n_used ilə eyni olmalıdır)\n",
            arch_buf[0] ? arch_buf : "naməlum", cb_data.expected_top_k);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "[atlas-trace] XƏTA: prompt tokenləşmədi\n");
        return 1;
    }
    fprintf(stderr, "[atlas-trace] prompt tokens=%zu, n_predict=%d\n",
            prompt_tokens.size(), params.n_predict);

    // sadə greedy sampler (orijinal collect_trace.py-dəki argmax ilə eyni,
    // deterministik router davranışı üçün)
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const auto t_start = ggml_time_us();

    // 1) PROMPT keçidi — bütün prompt tokenləri BİR decode() çağırışında,
    //    hər layer üçün ffn_moe_topk bir dəfə (n_tok=prompt uzunluğu) trace olunur
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "[atlas-trace] XƏTA: prompt decode uğursuz oldu\n");
        return 1;
    }
    cb_data.call_idx++;
    cb_data.phase = "decode";

    // 2) GENERASİYA — hər addım bir yeni token, hər addımda cb yenidən işə düşür
    int n_decoded = 0;
    for (int step = 0; step < params.n_predict; ++step) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, next_batch)) {
            fprintf(stderr, "[atlas-trace] XƏTA: generasiya decode uğursuz oldu (step=%d)\n", step);
            break;
        }
        cb_data.call_idx++;
        n_decoded++;
    }

    const auto t_end = ggml_time_us();
    const double secs = (t_end - t_start) / 1e6;
    const double tps = n_decoded > 0 ? n_decoded / secs : 0.0;

    printf("ATLAS_DONE prompt_tokens=%zu n_decoded=%d seconds=%.3f tok_per_sec=%.3f\n",
           prompt_tokens.size(), n_decoded, secs, tps);

    fprintf(stderr, "[atlas-trace] bitdi: %d token, %.2f s, %.2f tok/s\n", n_decoded, secs, tps);

    llama_sampler_free(smpl);
    llama_backend_free();

    return 0;
}
