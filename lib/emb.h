/**
 * emb.h - Vector Embedding Library Public API
 * Summary: Public interface for the ggml-based embedding library.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_EMB_H
#define KC_EMB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_emb kc_emb_t;

#define KC_EMB_OK      0
#define KC_EMB_ERROR  -1

/**
 * Initialize a new emb context.
 * Prepares one GGML inference context backed by the embedded model. The
 * worker uses a bounded CPU thread set for each embedding request.
 * @return Context pointer or NULL on failure.
 */
kc_emb_t *kc_emb_open(void);

/**
 * Release a emb context.
 * Shuts down the worker thread and frees all resources.
 * Must not be called while kc_emb_exec() is active on any thread.
 * @param ctx Context pointer.
 * @return None.
 */
void kc_emb_close(kc_emb_t *ctx);

/**
 * Retrieve the embedding dimension.
 * @param ctx Context pointer.
 * @return Dimension size, or 0 on invalid input.
 */
int kc_emb_dim(kc_emb_t *ctx);

/**
 * Generate an embedding for the given input text.
 * Dispatches to the prepared worker and blocks until the result is ready.
 * Multiple callers on the same context are serialized. The result is written
 * into the caller-supplied buffer.
 * @param ctx Context pointer.
 * @param input Null-terminated input text.
 * @param out Caller-supplied buffer of at least kc_emb_dim(ctx) floats.
 * @return KC_EMB_OK on success, KC_EMB_ERROR on failure.
 */
int kc_emb_exec(kc_emb_t *ctx, const char *input, float *out);

#ifdef __cplusplus
}
#endif

#endif
