/**
 * ipa.h - Island Parser Library
 * Summary: Public API for the ipa schema compiler and runtime library.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_IPA_H
#define KC_IPA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_ipa_schema kc_ipa_schema_t;

#define KC_IPA_OK       0
#define KC_IPA_ERROR   -1
#define KC_IPA_EFORMAT -2

/**
 * One emitted key-value pair from a matched node.
 * @param key   Emit key string (owned).
 * @param value Emit value string (owned). "$raw" resolves to the matched span.
 */
typedef struct {
    char *key;
    char *value;
} kc_ipa_emit_t;

/**
 * One semantic match produced by the parser.
 * Matches form a tree: a top-level match may have children from recursive node
 * refinement. Children are allocated in the same result and freed together.
 * @param id         Node identifier string (owned).
 * @param span       Raw text of the matched span (owned).
 * @param score      Match confidence in [0, 1].
 * @param emits      Array of emitted key-value pairs (owned).
 * @param emit_count Number of entries in emits.
 * @param children   Array of child matches (owned).
 * @param child_count Number of child matches.
 */
typedef struct kc_ipa_match {
    char              *id;
    char              *span;
    float              score;
    kc_ipa_emit_t     *emits;
    int                emit_count;
    struct kc_ipa_match *children;
    int                child_count;
} kc_ipa_match_t;

/**
 * Top-level parse result.
 * @param ok          Non-zero when at least one match was found.
 * @param matches     Array of top-level matches (owned).
 * @param match_count Number of top-level matches.
 */
typedef struct {
    int             ok;
    kc_ipa_match_t *matches;
    int             match_count;
} kc_ipa_result_t;

typedef void (*kc_ipa_signal_callback_t)(kc_ipa_schema_t *ctx);

typedef struct {
    char *schema_path;
} kc_ipa_options_t;

kc_ipa_options_t kc_ipa_options_default(void);
void kc_ipa_options_load_env(kc_ipa_options_t *opts);
void kc_ipa_options_free(kc_ipa_options_t *opts);

int kc_ipa_on_signal(kc_ipa_schema_t *ctx, int sig, kc_ipa_signal_callback_t cb);
int kc_ipa_raise_signal(kc_ipa_schema_t *ctx, int sig);
int kc_ipa_listen_signals(kc_ipa_schema_t *ctx);
int kc_ipa_listen_signal(kc_ipa_schema_t *ctx, int sig_id);
void kc_ipa_signal_listener(int sig);

/**
 * Open a compiled schema resource.
 * Maps the .ipa file into memory and builds runtime indexes.
 * The caller owns the returned handle and must call kc_ipa_close().
 * @param ctx  Pointer to destination schema handle.
 * @param opts Configuration options.
 * @return KC_IPA_OK on success, KC_IPA_ERROR on failure.
 */
int kc_ipa_open(kc_ipa_schema_t **ctx, kc_ipa_options_t *opts);

/**
 * Close a compiled schema resource and release all resources.
 * @param schema Schema handle.
 * @return KC_IPA_OK.
 */
int kc_ipa_close(kc_ipa_schema_t *schema);

/**
 * Parse input text against a compiled schema.
 * Scans for semantic islands, scores them against schema nodes, and recurses
 * into children only where the node definition requests it.
 * Concurrent calls on the same schema serialize through the embedding context.
 * @param schema Schema handle.
 * @param input  Null-terminated input text.
 * @param out    Destination result. Caller must call kc_ipa_result_free().
 * @return KC_IPA_OK on success, KC_IPA_ERROR on failure.
 */
int kc_ipa_parse(kc_ipa_schema_t *schema, const char *input, kc_ipa_result_t *out);

/**
 * Release resources owned by a result structure.
 * Does not free the result struct itself.
 * @param result Result to free.
 * @return None.
 */
void kc_ipa_result_free(kc_ipa_result_t *result);

/**
 * Compile a JSON schema definition into a .ipa binary resource.
 * Reads json_path, validates the schema, generates embeddings, and writes
 * the compiled binary to output_path.
 * @param json_path   Path to the source schema.json file.
 * @param output_path Destination path for the compiled .ipa file.
 * @return KC_IPA_OK on success, KC_IPA_EFORMAT on schema error,
 *         KC_IPA_ERROR on I/O or system error.
 */
int kc_ipa_build(const char *json_path, const char *output_path);

#ifdef __cplusplus
}
#endif

#endif
