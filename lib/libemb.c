/**
 * libemb.c - Vector Embedding Library Implementation
 * Summary: Core logic for generating embeddings using ggml.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#include <unistd.h>
#endif

#include "emb.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifndef _WIN32
#include <pthread.h>
#else
#include <windows.h>
#endif

#if defined(_WIN32) && !defined(_WIN64)
extern const unsigned char binary_model_gguf_start[] __asm__("_binary_model_gguf_start");
extern const unsigned char binary_model_gguf_end[]   __asm__("_binary_model_gguf_end");
#define model_gguf     ((unsigned char *)binary_model_gguf_start)
#define model_gguf_len ((unsigned int)(binary_model_gguf_end - binary_model_gguf_start))
#else
extern const unsigned char _binary_model_gguf_start[];
extern const unsigned char _binary_model_gguf_end[];
#define model_gguf     ((unsigned char *)_binary_model_gguf_start)
#define model_gguf_len ((unsigned int)(_binary_model_gguf_end - _binary_model_gguf_start))
#endif

typedef struct {
    char *str;
    int id;
} hash_entry;

typedef struct {
    struct ggml_tensor *attn_q_w;
    struct ggml_tensor *attn_q_b;
    struct ggml_tensor *attn_k_w;
    struct ggml_tensor *attn_k_b;
    struct ggml_tensor *attn_v_w;
    struct ggml_tensor *attn_v_b;
    struct ggml_tensor *attn_out_w;
    struct ggml_tensor *attn_out_b;
    struct ggml_tensor *attn_norm_w;
    struct ggml_tensor *attn_norm_b;
    struct ggml_tensor *ffn_up_w;
    struct ggml_tensor *ffn_up_b;
    struct ggml_tensor *ffn_down_w;
    struct ggml_tensor *ffn_down_b;
    struct ggml_tensor *layer_norm_w;
    struct ggml_tensor *layer_norm_b;
} kc_emb_layer_t;

typedef struct {
    struct ggml_context *ctx;
    struct gguf_context *gguf;

    int n_vocab;
    int n_embd;
    int n_layer;
    int n_head;
    int n_ctx;
    float layer_norm_eps;

    int cls_token_id;
    int sep_token_id;
    int pad_token_id;
    int unk_token_id;

    char **vocab;
    hash_entry *vocab_hash;
    int vocab_hash_size;

    struct ggml_tensor *token_embd;
    struct ggml_tensor *pos_embd;
    struct ggml_tensor *type_embd;
    struct ggml_tensor *token_embd_norm_w;
    struct ggml_tensor *token_embd_norm_b;

    kc_emb_layer_t *layers;

    void *compute_buf;
    size_t compute_buf_size;
    ggml_backend_t backend;
    ggml_gallocr_t galloc;

    int n_threads;
    int *tokens;
    int *pos_data;
    int *type_data;
    char *norm_input;
    size_t norm_input_size;
} kc_emb_ctx_t;

typedef struct kc_emb_worker kc_emb_worker_t;

struct kc_emb_worker {
    kc_emb_ctx_t *ectx;

#ifndef _WIN32
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_req;
    pthread_cond_t cond_res;
#else
    HANDLE thread;
    CRITICAL_SECTION mutex;
    CONDITION_VARIABLE cond_req;
    CONDITION_VARIABLE cond_res;
#endif

    const char *input;
    float *out;
    int result;
    int has_req;
    int done;
    int shutdown;
};

struct kc_emb {
    kc_emb_worker_t *workers;
    int n_workers;
    int n_embd;

#ifndef _WIN32
    pthread_mutex_t pool_mutex;
    pthread_cond_t pool_cond;
#else
    CRITICAL_SECTION pool_mutex;
    CONDITION_VARIABLE pool_cond;
#endif
};

/**
 * Check if a character is ASCII whitespace.
 * @param c Input character.
 * @return 1 if whitespace, 0 otherwise.
 */
static int kc_isspace(int c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

/**
 * Convert an ASCII character to lowercase.
 * @param c Input character.
 * @return Lowercase character.
 */
static int kc_tolower(int c) {
    return (c >= 'A' && c <= 'Z') ? (c + 32) : c;
}

/**
 * Count available CPU threads for one inference worker.
 * @return Positive CPU count.
 */
static int kc_emb_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;

    GetSystemInfo(&sysinfo);
    if (sysinfo.dwNumberOfProcessors == 0) {
        return 1;
    }
    return sysinfo.dwNumberOfProcessors > 4 ? 4 :
        (int)sysinfo.dwNumberOfProcessors;
#else
    long nproc;

    nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc <= 0) {
        return 1;
    }
    return nproc > 4 ? 4 : (int)nproc;
#endif
}

/**
 * Compute a djb2 hash for a string.
 * @param str Input string.
 * @return Hash value.
 */
static uint32_t hash_str(const char *str) {
    uint32_t h = 5381;
    int c;
    while ((c = *str++)) h = ((h << 5) + h) + (uint8_t)c;
    return h;
}

/**
 * Build the vocabulary hash table from the loaded vocab array.
 * @param ectx Worker context pointer.
 * @return 0 on success, -1 on allocation failure.
 */
static int build_vocab_hash(kc_emb_ctx_t *ectx) {
    ectx->vocab_hash_size = ectx->n_vocab * 2 + 1;
    ectx->vocab_hash = (hash_entry *)calloc(ectx->vocab_hash_size, sizeof(hash_entry));

    if (!ectx->vocab_hash) {
        return -1;
    }

    for (int i = 0; i < ectx->n_vocab; i++) {
        uint32_t h = hash_str(ectx->vocab[i]) % ectx->vocab_hash_size;
        while (ectx->vocab_hash[h].str != NULL) {
            h = (h + 1) % ectx->vocab_hash_size;
        }
        ectx->vocab_hash[h].str = ectx->vocab[i];
        ectx->vocab_hash[h].id = i;
    }
    return 0;
}

/**
 * Portable fmemopen for platforms that lack it.
 * @param buf Data buffer.
 * @param size Buffer size.
 * @param mode Open mode.
 * @return FILE pointer or NULL on failure.
 */
static FILE *kc_fmemopen(const void *buf, size_t size, const char *mode) {
#if defined(_WIN32) || defined(__ANDROID__)
    (void)mode;
    FILE *f = tmpfile();
    if (!f) return NULL;
    if (fwrite(buf, 1, size, f) != size) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    return f;
#else
    return fmemopen((void *)buf, size, mode);
#endif
}

/**
 * Null log callback to silence GGML diagnostics.
 * @param level Log level.
 * @param text Log text.
 * @param user_data User data pointer.
 * @return No return value.
 */
static void kc_ggml_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

/**
 * Look up a token string in the vocabulary hash table.
 * @param ectx Worker context pointer.
 * @param str Token string.
 * @return Token ID or -1 if not found.
 */
static int find_token(kc_emb_ctx_t *ectx, const char *str) {
    if (!ectx->vocab_hash || ectx->vocab_hash_size == 0) {
        return -1;
    }

    uint32_t h = hash_str(str) % ectx->vocab_hash_size;
    while (ectx->vocab_hash[h].str != NULL) {
        if (strcmp(ectx->vocab_hash[h].str, str) == 0) {
            return ectx->vocab_hash[h].id;
        }
        h = (h + 1) % ectx->vocab_hash_size;
    }
    return -1;
}

/**
 * Read a uint32 metadata value from a GGUF context with type flexibility.
 * @param ctx GGUF context pointer.
 * @param key Metadata key.
 * @param def Default value if key is absent.
 * @return Metadata value or default.
 */
static uint32_t get_kv_u32(const struct gguf_context *ctx, const char *key, uint32_t def) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) return def;
    enum gguf_type type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_UINT32) return gguf_get_val_u32(ctx, id);
    if (type == GGUF_TYPE_INT32) return (uint32_t)gguf_get_val_i32(ctx, id);
    if (type == GGUF_TYPE_UINT64) return (uint32_t)gguf_get_val_u64(ctx, id);
    if (type == GGUF_TYPE_INT64) return (uint32_t)gguf_get_val_i64(ctx, id);
    return def;
}

/**
 * Release all resources held by a worker context.
 * @param ectx Worker context pointer.
 * @return No return value.
 */
static void kc_emb_ctx_free(kc_emb_ctx_t *ectx) {
    if (!ectx) return;

    if (ectx->vocab) {
        free(ectx->vocab);
        ectx->vocab = NULL;
    }
    if (ectx->vocab_hash) {
        free(ectx->vocab_hash);
        ectx->vocab_hash = NULL;
    }
    if (ectx->layers) {
        free(ectx->layers);
        ectx->layers = NULL;
    }
    if (ectx->compute_buf) {
        free(ectx->compute_buf);
        ectx->compute_buf = NULL;
    }
    if (ectx->tokens) {
        free(ectx->tokens);
        ectx->tokens = NULL;
    }
    if (ectx->pos_data) {
        free(ectx->pos_data);
        ectx->pos_data = NULL;
    }
    if (ectx->type_data) {
        free(ectx->type_data);
        ectx->type_data = NULL;
    }
    if (ectx->norm_input) {
        free(ectx->norm_input);
        ectx->norm_input = NULL;
    }
    if (ectx->galloc) {
        ggml_gallocr_free(ectx->galloc);
        ectx->galloc = NULL;
    }
    if (ectx->backend) {
        ggml_backend_free(ectx->backend);
        ectx->backend = NULL;
    }
    if (ectx->gguf) {
        gguf_free(ectx->gguf);
        ectx->gguf = NULL;
    }
    if (ectx->ctx) {
        ggml_free(ectx->ctx);
        ectx->ctx = NULL;
    }
    free(ectx);
}

/**
 * Allocate and initialize one worker inference context.
 * @return Worker context pointer or NULL on failure.
 */
static kc_emb_ctx_t *kc_emb_ctx_open(void) {
    kc_emb_ctx_t *ectx = (kc_emb_ctx_t *)calloc(1, sizeof(kc_emb_ctx_t));
    if (!ectx) return NULL;

    FILE *f = kc_fmemopen(model_gguf, model_gguf_len, "rb");
    if (!f) {
        free(ectx);
        return NULL;
    }

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &ectx->ctx,
    };

    ectx->gguf = gguf_init_from_file_ptr(f, params);
    fclose(f);

    if (!ectx->gguf) goto failure;

    {
        size_t data_offset = gguf_get_data_offset(ectx->gguf);
        int n_tensors = gguf_get_n_tensors(ectx->gguf);
        for (int i = 0; i < n_tensors; i++) {
            const char *name = gguf_get_tensor_name(ectx->gguf, i);
            struct ggml_tensor *t = ggml_get_tensor(ectx->ctx, name);
            if (t) {
                t->data = (char *)model_gguf + data_offset + gguf_get_tensor_offset(ectx->gguf, i);
            }
        }
    }

    ectx->n_embd  = (int)get_kv_u32(ectx->gguf, "bert.embedding_length", 0);
    ectx->n_layer = (int)get_kv_u32(ectx->gguf, "bert.block_count", 0);
    ectx->n_head  = (int)get_kv_u32(ectx->gguf, "bert.attention.head_count", 0);
    ectx->n_ctx   = (int)get_kv_u32(ectx->gguf, "bert.context_length", 512);

    if (ectx->n_embd <= 0 || ectx->n_layer <= 0 || ectx->n_head <= 0 || ectx->n_ctx <= 0
        || (ectx->n_embd % ectx->n_head) != 0) {
        goto failure;
    }

    {
        int64_t kid;
        kid = gguf_find_key(ectx->gguf, "bert.attention.layer_norm_epsilon");
        ectx->layer_norm_eps = (kid >= 0) ? gguf_get_val_f32(ectx->gguf, kid) : 1e-12f;

        kid = gguf_find_key(ectx->gguf, "tokenizer.ggml.tokens");
        if (kid < 0) goto failure;

        ectx->n_vocab = (int)gguf_get_arr_n(ectx->gguf, kid);
        if (ectx->n_vocab <= 0) goto failure;

        ectx->vocab = (char **)malloc(ectx->n_vocab * sizeof(char *));
        if (!ectx->vocab) goto failure;

        for (int i = 0; i < ectx->n_vocab; i++) {
            ectx->vocab[i] = (char *)gguf_get_arr_str(ectx->gguf, kid, i);
            if (i < 0) {
            }
        }

        if (build_vocab_hash(ectx) != 0) {
            if (ectx->vocab_hash) { free(ectx->vocab_hash); ectx->vocab_hash = NULL; }
            free(ectx->vocab); ectx->vocab = NULL;
            goto failure;
        }

        ectx->cls_token_id = (int)get_kv_u32(ectx->gguf, "tokenizer.ggml.cls_token_id", 101);
        ectx->sep_token_id = (int)get_kv_u32(ectx->gguf, "tokenizer.ggml.sep_token_id", 102);
        ectx->pad_token_id = (int)get_kv_u32(ectx->gguf, "tokenizer.ggml.pad_token_id", 0);
        ectx->unk_token_id = (int)get_kv_u32(ectx->gguf, "tokenizer.ggml.unknown_token_id", 100);
    }

    ectx->token_embd       = ggml_get_tensor(ectx->ctx, "token_embd.weight");
    ectx->pos_embd         = ggml_get_tensor(ectx->ctx, "position_embd.weight");
    ectx->type_embd        = ggml_get_tensor(ectx->ctx, "token_types.weight");
    ectx->token_embd_norm_w = ggml_get_tensor(ectx->ctx, "token_embd_norm.weight");
    ectx->token_embd_norm_b = ggml_get_tensor(ectx->ctx, "token_embd_norm.bias");

    if (!ectx->token_embd || !ectx->pos_embd || !ectx->type_embd
        || !ectx->token_embd_norm_w || !ectx->token_embd_norm_b) {
        goto failure;
    }

    ectx->layers = (kc_emb_layer_t *)calloc(ectx->n_layer, sizeof(kc_emb_layer_t));
    if (!ectx->layers) goto failure;

    for (int i = 0; i < ectx->n_layer; i++) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);       ectx->layers[i].attn_q_w   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", i);         ectx->layers[i].attn_q_b   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);       ectx->layers[i].attn_k_w   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", i);         ectx->layers[i].attn_k_b   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);       ectx->layers[i].attn_v_w   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", i);         ectx->layers[i].attn_v_b   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);  ectx->layers[i].attn_out_w  = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output.bias", i);    ectx->layers[i].attn_out_b  = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output_norm.weight", i); ectx->layers[i].attn_norm_w = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output_norm.bias", i);   ectx->layers[i].attn_norm_b = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);       ectx->layers[i].ffn_up_w   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.bias", i);         ectx->layers[i].ffn_up_b   = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);     ectx->layers[i].ffn_down_w = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.bias", i);       ectx->layers[i].ffn_down_b = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.layer_output_norm.weight", i); ectx->layers[i].layer_norm_w = ggml_get_tensor(ectx->ctx, name);
        snprintf(name, sizeof(name), "blk.%d.layer_output_norm.bias", i);   ectx->layers[i].layer_norm_b = ggml_get_tensor(ectx->ctx, name);

        if (!ectx->layers[i].attn_q_w || !ectx->layers[i].attn_q_b
            || !ectx->layers[i].attn_k_w || !ectx->layers[i].attn_k_b
            || !ectx->layers[i].attn_v_w || !ectx->layers[i].attn_v_b
            || !ectx->layers[i].attn_out_w || !ectx->layers[i].attn_out_b
            || !ectx->layers[i].attn_norm_w || !ectx->layers[i].attn_norm_b
            || !ectx->layers[i].ffn_up_w || !ectx->layers[i].ffn_up_b
            || !ectx->layers[i].ffn_down_w || !ectx->layers[i].ffn_down_b
            || !ectx->layers[i].layer_norm_w || !ectx->layers[i].layer_norm_b) {
            goto failure;
        }
    }

    ectx->compute_buf_size = 64 * 1024 * 1024;
    ectx->compute_buf = malloc(ectx->compute_buf_size);
    if (!ectx->compute_buf) goto failure;

    ectx->tokens = (int *)calloc((size_t)ectx->n_ctx, sizeof(int));
    ectx->pos_data = (int *)calloc((size_t)ectx->n_ctx, sizeof(int));
    ectx->type_data = (int *)calloc((size_t)ectx->n_ctx, sizeof(int));
    if (!ectx->tokens || !ectx->pos_data || !ectx->type_data) goto failure;

    ectx->backend = ggml_backend_cpu_init();
    if (!ectx->backend) goto failure;
    ectx->n_threads = kc_emb_cpu_count();
    ggml_backend_cpu_set_n_threads(ectx->backend, ectx->n_threads);

    ectx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ectx->backend));
    if (!ectx->galloc) goto failure;

    ggml_log_set(kc_ggml_log_callback, NULL);
    return ectx;

failure:
    kc_emb_ctx_free(ectx);
    return NULL;
}

/**
 * Tokenize input text using WordPiece segmentation.
 * @param ectx Worker context pointer.
 * @param input Input text.
 * @param tokens Output token ID array.
 * @param n_tokens Output token count.
 * @return No return value.
 */
static void wordpiece_tokenize(kc_emb_ctx_t *ectx, const char *input, int *tokens, int *n_tokens) {
    size_t i = 0;
    size_t len = strlen(input);
    const char *space_prefix = "\xe2\x96\x81";

    *n_tokens = 0;
    if (*n_tokens < ectx->n_ctx) {
        tokens[(*n_tokens)++] = ectx->cls_token_id;
    }

    while (i < len && *n_tokens < ectx->n_ctx - 1) {
        while (i < len && kc_isspace((uint8_t)input[i])) i++;
        if (i == len) break;

        size_t word_start = i;
        while (i < len && !kc_isspace((uint8_t)input[i])) i++;
        size_t word_len = i - word_start;

        size_t start = 0;
        while (start < word_len && *n_tokens < ectx->n_ctx - 1) {
            int best_id = -1;
            size_t best_len = 0;

            for (size_t end = word_len; end > start; end--) {
                char buf[128];
                size_t blen = 0;
                int found = 0;

                if (start == 0) {
                    memcpy(buf, space_prefix, 3);
                    blen = 3;
                    if (blen + (end - start) < sizeof(buf)) {
                        for (size_t k = start; k < end; k++) {
                            buf[blen++] = (char)kc_tolower((uint8_t)input[word_start + k]);
                        }
                        buf[blen] = '\0';
                        best_id = find_token(ectx, buf);
                        if (best_id >= 0) {
                            found = 1;
                        }
                    }
                }

                if (!found && start > 0) {
                    memcpy(buf, "##", 2);
                    blen = 2;
                    if (blen + (end - start) < sizeof(buf)) {
                        for (size_t k = start; k < end; k++) {
                            buf[blen++] = (char)kc_tolower((uint8_t)input[word_start + k]);
                        }
                        buf[blen] = '\0';
                        best_id = find_token(ectx, buf);
                        if (best_id >= 0) {
                            found = 1;
                        }
                    }
                }

                if (!found) {
                    blen = 0;
                    if (blen + (end - start) < sizeof(buf)) {
                        for (size_t k = start; k < end; k++) {
                            buf[blen++] = (char)kc_tolower((uint8_t)input[word_start + k]);
                        }
                        buf[blen] = '\0';
                        best_id = find_token(ectx, buf);
                        if (best_id >= 0) {
                            found = 1;
                        }
                    }
                }

                if (found) {
                    best_len = end - start;
                    break;
                }
            }

            if (best_id != -1) {
                tokens[(*n_tokens)++] = best_id;
                start += best_len;
            } else {
                if (*n_tokens < ectx->n_ctx) tokens[(*n_tokens)++] = ectx->unk_token_id;
                start++; 
            }
        }
    }

    if (*n_tokens < ectx->n_ctx) tokens[(*n_tokens)++] = ectx->sep_token_id;
}

/**
 * Perform proper LayerNorm: (x - mean) / std * weight + bias.
 * @param ctx GGML context.
 * @param a   Input tensor.
 * @param w   Weight tensor.
 * @param b   Bias tensor.
 * @param eps Epsilon for numerical stability.
 * @return Normalized tensor.
 */
static struct ggml_tensor * kc_ggml_layer_norm(struct ggml_context * ctx,
struct ggml_tensor * a, struct ggml_tensor * w, struct ggml_tensor * b,
float eps) {
    struct ggml_tensor * mean = ggml_mean(ctx, a);
    struct ggml_tensor * sub  = ggml_sub(ctx, a, mean);
    struct ggml_tensor * norm = ggml_norm(ctx, sub, eps);
    return ggml_add(ctx, ggml_mul(ctx, norm, w), b);
}

/**
 * Run inference on one worker context and write the result to out.
 * @param ectx Worker context pointer.
 * @param input Input text.
 * @param out Caller-supplied output buffer of n_embd floats.
 * @return KC_EMB_OK on success, KC_EMB_ERROR on failure.
 */
static int kc_emb_ctx_exec(kc_emb_ctx_t *ectx, const char *input, float *out) {
    int n_tokens = 0;
    struct ggml_context *ctx0 = NULL;
    struct ggml_tensor *inp_tokens = NULL;
    struct ggml_tensor *inp_pos = NULL;
    struct ggml_tensor *inp_type = NULL;
    struct ggml_cgraph *gf = NULL;
    struct ggml_tensor *cur = NULL;
    struct ggml_tensor *res = NULL;
    int d_head = 0;
    wordpiece_tokenize(ectx, input, ectx->tokens, &n_tokens);

    if (n_tokens < 2 || n_tokens > ectx->n_ctx) goto failure;

    for (int i = 0; i < n_tokens; i++) ectx->pos_data[i] = i;

    {
        struct ggml_init_params params;
        params.mem_size   = ectx->compute_buf_size;
        params.mem_buffer = ectx->compute_buf;
        params.no_alloc   = true;
        ctx0 = ggml_init(params);
    }
    if (!ctx0) goto failure;

    inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp_pos    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    inp_type   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    if (!inp_tokens || !inp_pos || !inp_type) goto failure;

    inp_tokens->data = ectx->tokens;
    inp_pos->data    = ectx->pos_data;
    inp_type->data   = ectx->type_data;

    gf = ggml_new_graph(ctx0);
    if (!gf) goto failure;

    cur = ggml_get_rows(ctx0, ectx->token_embd, inp_tokens);
    {
        struct ggml_tensor *pos = ggml_get_rows(ctx0, ectx->pos_embd, inp_pos);
        struct ggml_tensor *typ = ggml_get_rows(ctx0, ectx->type_embd, inp_type);
        if (!cur || !pos || !typ) goto failure;
        cur = ggml_add(ctx0, cur, pos);
        cur = ggml_add(ctx0, cur, typ);
    }
    cur = kc_ggml_layer_norm(ctx0, cur, ectx->token_embd_norm_w, ectx->token_embd_norm_b, ectx->layer_norm_eps);

    d_head = ectx->n_embd / ectx->n_head;

    for (int il = 0; il < ectx->n_layer; il++) {
        struct ggml_tensor *inp_L = cur;
        struct ggml_tensor *q = ggml_add(ctx0, ggml_mul_mat(ctx0, ectx->layers[il].attn_q_w, cur), ectx->layers[il].attn_q_b);
        struct ggml_tensor *k = ggml_add(ctx0, ggml_mul_mat(ctx0, ectx->layers[il].attn_k_w, cur), ectx->layers[il].attn_k_b);
        struct ggml_tensor *v = ggml_add(ctx0, ggml_mul_mat(ctx0, ectx->layers[il].attn_v_w, cur), ectx->layers[il].attn_v_b);

        q = ggml_reshape_3d(ctx0, q, d_head, ectx->n_head, n_tokens);
        q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));

        k = ggml_reshape_3d(ctx0, k, d_head, ectx->n_head, n_tokens);
        k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));

        struct ggml_tensor *kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_scale_inplace(ctx0, kq, 1.0f / sqrtf((float)d_head));
        kq = ggml_soft_max_inplace(ctx0, kq);

        v = ggml_reshape_3d(ctx0, v, d_head, ectx->n_head, n_tokens);
        v = ggml_cont(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3));

        struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v)), kq);
        kqv = ggml_cont(ctx0, ggml_permute(ctx0, kqv, 0, 2, 1, 3));
        kqv = ggml_reshape_2d(ctx0, kqv, ectx->n_embd, n_tokens);

        struct ggml_tensor *attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, ectx->layers[il].attn_out_w, kqv), ectx->layers[il].attn_out_b);
        cur = ggml_add(ctx0, inp_L, attn_out);
        cur = kc_ggml_layer_norm(ctx0, cur, ectx->layers[il].attn_norm_w, ectx->layers[il].attn_norm_b, ectx->layer_norm_eps);

        struct ggml_tensor *inp_F = cur;
        struct ggml_tensor *ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, ectx->layers[il].ffn_up_w, cur), ectx->layers[il].ffn_up_b);
        ffn = ggml_gelu(ctx0, ffn);
        ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, ectx->layers[il].ffn_down_w, ffn), ectx->layers[il].ffn_down_b);

        cur = ggml_add(ctx0, inp_F, ffn);
        cur = kc_ggml_layer_norm(ctx0, cur, ectx->layers[il].layer_norm_w, ectx->layers[il].layer_norm_b, ectx->layer_norm_eps);
    }

    if (cur->ne[0] != ectx->n_embd) goto failure;

    res = ggml_view_2d(ctx0, cur, ectx->n_embd, 1, cur->nb[1], 0);
    res = ggml_cont(ctx0, res);
    ggml_build_forward_expand(gf, res);

    if (!ggml_gallocr_alloc_graph(ectx->galloc, gf)) goto failure;

    memcpy(inp_tokens->data, ectx->tokens, n_tokens * sizeof(int));
    memcpy(inp_pos->data, ectx->pos_data, n_tokens * sizeof(int));
    memset(inp_type->data,   0,         n_tokens * sizeof(int));

    ggml_backend_graph_compute(ectx->backend, gf);

    if (!res->data || res->ne[0] != ectx->n_embd) goto failure;

    {
        float *p = (float *)res->data;
        double sum = 0;
        for (int i = 0; i < ectx->n_embd; i++) sum += (double)p[i] * p[i];
        float norm = (float)sqrt(sum);
        if (norm > 1e-12f) {
            for (int i = 0; i < ectx->n_embd; i++) p[i] /= norm;
        }
    }

    memcpy(out, res->data, ectx->n_embd * sizeof(float));

    ggml_free(ctx0);
    return KC_EMB_OK;

failure:
    if (ctx0) ggml_free(ctx0);
    return KC_EMB_ERROR;
}

#ifndef _WIN32
/**
 * Worker thread entry point. Waits for requests and executes inference.
 * @param arg Pointer to the worker struct.
 * @return NULL on completion.
 */
static void *kc_emb_worker_thread(void *arg) {
#else
/**
 * Worker thread entry point. Waits for requests and executes inference.
 * @param arg Pointer to the worker struct.
 * @return 0 on completion.
 */
static DWORD WINAPI kc_emb_worker_thread(LPVOID arg) {
#endif
    kc_emb_worker_t *w = (kc_emb_worker_t *)arg;

#ifndef _WIN32
    pthread_mutex_lock(&w->mutex);
    while (!w->shutdown) {
        while (!w->has_req && !w->shutdown) {
            pthread_cond_wait(&w->cond_req, &w->mutex);
        }
        if (w->shutdown) break;
        w->result = kc_emb_ctx_exec(w->ectx, w->input, w->out);
        w->has_req = 0;
        w->done = 1;
        pthread_cond_signal(&w->cond_res);
    }
    pthread_mutex_unlock(&w->mutex);
    return NULL;
#else
    EnterCriticalSection(&w->mutex);
    while (!w->shutdown) {
        while (!w->has_req && !w->shutdown) {
            SleepConditionVariableCS(&w->cond_req, &w->mutex, INFINITE);
        }
        if (w->shutdown) break;
        w->result = kc_emb_ctx_exec(w->ectx, w->input, w->out);
        w->has_req = 0;
        w->done = 1;
        WakeConditionVariable(&w->cond_res);
    }
    LeaveCriticalSection(&w->mutex);
    return 0;
#endif
}

/**
 * Initialize a worker: allocate its context and start its thread.
 * @param w Worker pointer.
 * @return 0 on success, -1 on failure.
 */
static int kc_emb_worker_init(kc_emb_worker_t *w) {
    w->ectx = kc_emb_ctx_open();
    if (!w->ectx) return -1;

    w->input    = NULL;
    w->out      = NULL;
    w->result   = 0;
    w->has_req  = 0;
    w->done     = 0;
    w->shutdown = 0;

#ifndef _WIN32
    if (pthread_mutex_init(&w->mutex, NULL) != 0) {
        kc_emb_ctx_free(w->ectx); w->ectx = NULL;
        return -1;
    }
    if (pthread_cond_init(&w->cond_req, NULL) != 0) {
        pthread_mutex_destroy(&w->mutex);
        kc_emb_ctx_free(w->ectx); w->ectx = NULL;
        return -1;
    }
    if (pthread_cond_init(&w->cond_res, NULL) != 0) {
        pthread_cond_destroy(&w->cond_req);
        pthread_mutex_destroy(&w->mutex);
        kc_emb_ctx_free(w->ectx); w->ectx = NULL;
        return -1;
    }
    if (pthread_create(&w->thread, NULL, kc_emb_worker_thread, w) != 0) {
        pthread_cond_destroy(&w->cond_res);
        pthread_cond_destroy(&w->cond_req);
        pthread_mutex_destroy(&w->mutex);
        kc_emb_ctx_free(w->ectx); w->ectx = NULL;
        return -1;
    }
#else
    InitializeCriticalSection(&w->mutex);
    InitializeConditionVariable(&w->cond_req);
    InitializeConditionVariable(&w->cond_res);
    w->thread = CreateThread(NULL, 0, kc_emb_worker_thread, w, 0, NULL);
    if (!w->thread) {
        DeleteCriticalSection(&w->mutex);
        kc_emb_ctx_free(w->ectx); w->ectx = NULL;
        return -1;
    }
#endif
    return 0;
}

/**
 * Signal a worker to shut down, join its thread, and free its context.
 * @param w Worker pointer.
 * @return No return value.
 */
static void kc_emb_worker_destroy(kc_emb_worker_t *w) {
#ifndef _WIN32
    pthread_mutex_lock(&w->mutex);
    w->shutdown = 1;
    pthread_cond_signal(&w->cond_req);
    pthread_mutex_unlock(&w->mutex);
    pthread_join(w->thread, NULL);
    pthread_cond_destroy(&w->cond_res);
    pthread_cond_destroy(&w->cond_req);
    pthread_mutex_destroy(&w->mutex);
#else
    EnterCriticalSection(&w->mutex);
    w->shutdown = 1;
    WakeConditionVariable(&w->cond_req);
    LeaveCriticalSection(&w->mutex);
    WaitForSingleObject(w->thread, INFINITE);
    CloseHandle(w->thread);
    DeleteCriticalSection(&w->mutex);
#endif
    kc_emb_ctx_free(w->ectx);
    w->ectx = NULL;
}

/**
 * Initialize a new emb pool.
 * @return Pool pointer or NULL on failure.
 */
kc_emb_t *kc_emb_open(void) {
    int n_workers;

    n_workers = 1;

    kc_emb_t *ctx = (kc_emb_t *)calloc(1, sizeof(kc_emb_t));
    if (!ctx) return NULL;

    ctx->workers = (kc_emb_worker_t *)calloc(n_workers, sizeof(kc_emb_worker_t));
    if (!ctx->workers) { free(ctx); return NULL; }

#ifndef _WIN32
    if (pthread_mutex_init(&ctx->pool_mutex, NULL) != 0) {
        free(ctx->workers); free(ctx); return NULL;
    }
    if (pthread_cond_init(&ctx->pool_cond, NULL) != 0) {
        pthread_mutex_destroy(&ctx->pool_mutex);
        free(ctx->workers); free(ctx); return NULL;
    }
#else
    InitializeCriticalSection(&ctx->pool_mutex);
    InitializeConditionVariable(&ctx->pool_cond);
#endif

    int started = 0;
    for (int i = 0; i < n_workers; i++) {
        if (kc_emb_worker_init(&ctx->workers[i]) != 0) break;
        started++;
    }

    if (started == 0) {
        for (int i = 0; i < started; i++) kc_emb_worker_destroy(&ctx->workers[i]);
#ifndef _WIN32
        pthread_cond_destroy(&ctx->pool_cond);
        pthread_mutex_destroy(&ctx->pool_mutex);
#else
        DeleteCriticalSection(&ctx->pool_mutex);
#endif
        free(ctx->workers); free(ctx); return NULL;
    }

    ctx->n_workers = started;
    ctx->n_embd    = ctx->workers[0].ectx->n_embd;
    return ctx;
}

/**
 * Release a emb pool and shut down all workers.
 * @param ctx Pool pointer.
 * @return No return value.
 */
void kc_emb_close(kc_emb_t *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_workers; i++) kc_emb_worker_destroy(&ctx->workers[i]);
#ifndef _WIN32
    pthread_cond_destroy(&ctx->pool_cond);
    pthread_mutex_destroy(&ctx->pool_mutex);
#else
    DeleteCriticalSection(&ctx->pool_mutex);
#endif
    free(ctx->workers);
    free(ctx);
}

/**
 * Retrieve the embedding dimension.
 * @param ctx Pool pointer.
 * @return Dimension size, or 0 on invalid input.
 */
int kc_emb_dim(kc_emb_t *ctx) {
    return ctx ? ctx->n_embd : 0;
}

/**
 * Generate an embedding for the given input text.
 * @param ctx Pool pointer.
 * @param input Null-terminated input text.
 * @param out Caller-supplied buffer of at least kc_emb_dim(ctx) floats.
 * @return KC_EMB_OK on success, KC_EMB_ERROR on failure.
 */
int kc_emb_exec(kc_emb_t *ctx, const char *input, float *out) {
    kc_emb_worker_t *w = NULL;

    if (!ctx || !input || !out) return KC_EMB_ERROR;

#ifndef _WIN32
    pthread_mutex_lock(&ctx->pool_mutex);
    while (1) {
        for (int i = 0; i < ctx->n_workers; i++) {
            kc_emb_worker_t *cand = &ctx->workers[i];
            pthread_mutex_lock(&cand->mutex);
            if (!cand->has_req) {
                w = cand;
                break;
            }
            pthread_mutex_unlock(&cand->mutex);
        }
        if (w) break;
        pthread_cond_wait(&ctx->pool_cond, &ctx->pool_mutex);
    }
    pthread_mutex_unlock(&ctx->pool_mutex);

    w->input  = input;
    w->out    = out;
    w->done   = 0;
    w->has_req = 1;
    pthread_cond_signal(&w->cond_req);

    while (!w->done) {
        pthread_cond_wait(&w->cond_res, &w->mutex);
    }
    int result = w->result;
    pthread_mutex_unlock(&w->mutex);

    pthread_mutex_lock(&ctx->pool_mutex);
    pthread_cond_signal(&ctx->pool_cond);
    pthread_mutex_unlock(&ctx->pool_mutex);

    return result;
#else
    EnterCriticalSection(&ctx->pool_mutex);
    while (1) {
        for (int i = 0; i < ctx->n_workers; i++) {
            kc_emb_worker_t *cand = &ctx->workers[i];
            EnterCriticalSection(&cand->mutex);
            if (!cand->has_req) {
                w = cand;
                break;
            }
            LeaveCriticalSection(&cand->mutex);
        }
        if (w) break;
        SleepConditionVariableCS(&ctx->pool_cond, &ctx->pool_mutex, INFINITE);
    }
    LeaveCriticalSection(&ctx->pool_mutex);

    w->input  = input;
    w->out    = out;
    w->done   = 0;
    w->has_req = 1;
    WakeConditionVariable(&w->cond_req);

    while (!w->done) {
        SleepConditionVariableCS(&w->cond_res, &w->mutex, INFINITE);
    }
    int result = w->result;
    LeaveCriticalSection(&w->mutex);

    EnterCriticalSection(&ctx->pool_mutex);
    WakeConditionVariable(&ctx->pool_cond);
    LeaveCriticalSection(&ctx->pool_mutex);

    return result;
#endif
}
