/**
 * libipa.c - Island Parser Library
 * Summary: Core implementation for the ipa schema compiler and runtime library.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#endif

#include "ipa.h"

#include "emb.h"
#include "hnsw.h"
#include "ngram.h"
#include "mmap.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#ifndef _WIN32
#include <pthread.h>
#include <signal.h>
#else
#include <windows.h>
#define strncasecmp _strnicmp
#define strcasecmp  _stricmp
#endif

#define KC_IPA_MAX_MATCHES    128
#define KC_IPA_MAX_DEPTH      8
#define KC_IPA_STRTAB_INIT    65536
#define KC_IPA_BUF_INIT       131072

typedef enum {
    KC_ENV_TYPE_INT,
    KC_ENV_TYPE_FLOAT,
    KC_ENV_TYPE_STR,
} kc_env_type_t;

typedef struct {
    const char *env_var;
    size_t offset;
    kc_env_type_t type;
} kc_env_map_t;

static const kc_env_map_t env_config_table[] = {
    { "KC_IPA_SCHEMA", offsetof(kc_ipa_options_t, schema_path), KC_ENV_TYPE_STR },
};
static const int env_config_table_n = sizeof(env_config_table) / sizeof(env_config_table[0]);

typedef struct {
    int sig;
    kc_ipa_signal_callback_t cb;
} kc_ipa_signal_entry_t;

static kc_ipa_schema_t *g_signal_ctx = NULL;

#define KC_IPA_MAGIC_0        'K'
#define KC_IPA_MAGIC_1        'I'
#define KC_IPA_MAGIC_2        'P'
#define KC_IPA_MAGIC_3        'A'
#define KC_IPA_FMT_VERSION    ((uint32_t)1)

typedef struct {
    uint8_t  magic[4];
    uint32_t version;
    uint32_t file_size;
    uint32_t strtab_off;
    uint32_t strtab_size;
    uint32_t id_off;
    uint32_t id_len;
    uint32_t vendor_off;
    uint32_t vendor_len;
    uint32_t schemaver_off;
    uint32_t schemaver_len;
    float    node_threshold;
    uint32_t min_ngram;
    uint32_t max_ngram;
    uint32_t emb_dim;
    uint32_t node_emb_off;
    uint32_t node_emb_count;
    uint32_t child_emb_off;
    uint32_t child_emb_count;
    uint32_t node_meta_off;
    uint32_t node_meta_count;
    uint32_t child_meta_off;
    uint32_t child_meta_count;
} kc_ipa_file_hdr_t;

typedef struct {
    uint32_t label_off;
    uint32_t label_len;
} kc_ipa_emb_hdr_t;

typedef struct {
    uint32_t node_id_off;
    uint32_t node_id_len;
    uint32_t emit_key_off;
    uint32_t emit_key_len;
    uint32_t emit_val_off;
    uint32_t emit_val_len;
    uint32_t has_children;
} kc_ipa_node_meta_t;

typedef struct {
    uint32_t parent_id_off;
    uint32_t parent_id_len;
    uint32_t node_id_off;
    uint32_t node_id_len;
    uint32_t emit_key_off;
    uint32_t emit_key_len;
    uint32_t emit_val_off;
    uint32_t emit_val_len;
} kc_ipa_child_meta_t;

typedef struct {
    char  *key;
    char  *value;
} kc_ipa_emit_def_t;

typedef struct kc_ipa_node_def kc_ipa_node_def_t;

struct kc_ipa_node_def {
    char              *id;
    char             **descriptions;
    int                desc_count;
    kc_ipa_emit_def_t *emits;
    int                emit_count;
    kc_ipa_node_def_t *children;
    int                child_count;
};

typedef struct {
    kc_ipa_node_def_t *nodes;
    int                node_count;
} kc_ipa_schema_def_t;

typedef struct {
    char              *schema;
    char              *id;
    char              *vendor;
    char              *version;
    float              node_threshold;
    int                min_ngram;
    int                max_ngram;
    kc_ipa_schema_def_t map;
} kc_ipa_manifest_t;

typedef struct {
    kc_hnsw_t               *node_hnsw;
    kc_hnsw_t               *child_hnsw;
    const kc_ipa_node_meta_t  *node_meta;
    uint32_t                  node_meta_count;
    const kc_ipa_child_meta_t *child_meta;
    uint32_t                  child_meta_count;
} kc_ipa_runtime_index_t;

struct kc_ipa_schema {
    kc_ipa_options_t opts;
    kc_ipa_signal_entry_t *signal_handlers;
    int n_signal_handlers;
    int signal_handlers_capacity;

    kc_mmap_t            mmap;
    kc_emb_t            *emb;
    int                  emb_dim;

    float                node_threshold;
    int                  min_ngram;
    int                  max_ngram;

    const char          *strtab;

    kc_ipa_runtime_index_t index;

#ifndef _WIN32
    pthread_mutex_t lock;
#else
    CRITICAL_SECTION lock;
#endif
};

typedef struct {
    char  *data;
    size_t size;
    size_t capacity;
} kc_ipa_strtab_t;

/**
 * Initializes a string table.
 * @param st String table to initialize.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_strtab_init(kc_ipa_strtab_t *st) {
    st->data = (char *)malloc(KC_IPA_STRTAB_INIT);
    if (!st->data) return -1;
    st->size = 0;
    st->capacity = KC_IPA_STRTAB_INIT;
    return 0;
}

/**
 * Releases resources owned by a string table.
 * @param st String table to free.
 * @return None.
 */
static void kc_ipa_strtab_free(kc_ipa_strtab_t *st) {
    free(st->data);
    st->data = NULL;
    st->size = 0;
    st->capacity = 0;
}

/**
 * Appends a string to the string table and records its offset and length.
 * @param st      String table to append to.
 * @param str     String data to append.
 * @param len     Byte length of str.
 * @param out_off Destination for strtab offset.
 * @param out_len Destination for string length.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_strtab_add(
    kc_ipa_strtab_t *st, const char *str, size_t len,
    uint32_t *out_off, uint32_t *out_len)
{
    size_t needed = st->size + len + 1;
    if (needed > st->capacity) {
        size_t next = st->capacity * 2;
        while (next < needed) next *= 2;
        char *p = (char *)realloc(st->data, next);
        if (!p) return -1;
        st->data = p;
        st->capacity = next;
    }
    *out_off = (uint32_t)st->size;
    *out_len = (uint32_t)len;
    memcpy(st->data + st->size, str, len);
    st->data[st->size + len] = '\0';
    st->size += len + 1;
    return 0;
}

typedef struct {
    uint8_t *data;
    size_t   size;
    size_t   capacity;
} kc_ipa_wbuf_t;

/**
 * Initializes a write buffer.
 * @param w Write buffer to initialize.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_wbuf_init(kc_ipa_wbuf_t *w) {
    w->data = (uint8_t *)malloc(KC_IPA_BUF_INIT);
    if (!w->data) return -1;
    w->size = 0;
    w->capacity = KC_IPA_BUF_INIT;
    return 0;
}

/**
 * Releases resources owned by a write buffer.
 * @param w Write buffer to free.
 * @return None.
 */
static void kc_ipa_wbuf_free(kc_ipa_wbuf_t *w) {
    free(w->data);
    w->data = NULL;
    w->size = 0;
    w->capacity = 0;
}

/**
 * Ensures the write buffer has room for at least needed additional bytes.
 * @param w      Write buffer.
 * @param needed Additional bytes required.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_wbuf_reserve(kc_ipa_wbuf_t *w, size_t needed) {
    if (w->size + needed <= w->capacity) return 0;
    size_t next = w->capacity * 2;
    while (next < w->size + needed) next *= 2;
    uint8_t *p = (uint8_t *)realloc(w->data, next);
    if (!p) return -1;
    w->data = p;
    w->capacity = next;
    return 0;
}

/**
 * Appends raw bytes to a write buffer.
 * @param w   Write buffer.
 * @param src Source data.
 * @param len Byte count.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_wbuf_append(kc_ipa_wbuf_t *w, const void *src, size_t len) {
    if (kc_ipa_wbuf_reserve(w, len) != 0) return -1;
    memcpy(w->data + w->size, src, len);
    w->size += len;
    return 0;
}

typedef struct {
    const char *src;
    size_t      pos;
    size_t      len;
    char        errbuf[256];
} kc_ipa_json_t;

/**
 * Sets a human-readable error message on a JSON tokenizer.
 * @param j   JSON tokenizer.
 * @param msg Error description.
 * @return None.
 */
static void kc_ipa_json_set_error(kc_ipa_json_t *j, const char *msg) {
    snprintf(j->errbuf, sizeof(j->errbuf), "%s (pos %zu)", msg, j->pos);
}

/**
 * Advances the JSON tokenizer past any whitespace.
 * @param j JSON tokenizer.
 * @return None.
 */
static void kc_ipa_json_skip_ws(kc_ipa_json_t *j) {
    while (j->pos < j->len && isspace((unsigned char)j->src[j->pos]))
        j->pos++;
}

/**
 * Consumes an expected character from the JSON token stream.
 * @param j JSON tokenizer.
 * @param c Character to consume.
 * @return 0 on success, -1 if not found.
 */
static int kc_ipa_json_expect(kc_ipa_json_t *j, char c) {
    kc_ipa_json_skip_ws(j);
    if (j->pos >= j->len || j->src[j->pos] != c) {
        char msg[64];
        snprintf(msg, sizeof(msg), "expected '%c'", c);
        kc_ipa_json_set_error(j, msg);
        return -1;
    }
    j->pos++;
    return 0;
}

/**
 * Parses a JSON string value into a newly allocated buffer.
 * @param j   JSON tokenizer.
 * @param out Destination for the allocated string.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_json_parse_string(kc_ipa_json_t *j, char **out) {
    kc_ipa_json_skip_ws(j);
    if (j->pos >= j->len || j->src[j->pos] != '"') {
        kc_ipa_json_set_error(j, "expected string");
        return -1;
    }
    j->pos++;
    size_t cap = 256;
    char *buf = (char *)malloc(cap);
    if (!buf) return -1;
    size_t n = 0;
    while (j->pos < j->len && j->src[j->pos] != '"') {
        char c = j->src[j->pos++];
        if (c == '\\') {
            if (j->pos >= j->len) { free(buf); return -1; }
            char e = j->src[j->pos++];
            switch (e) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                default:   c = e;    break;
            }
        }
        if (n + 2 >= cap) {
            cap *= 2;
            char *p = (char *)realloc(buf, cap);
            if (!p) { free(buf); return -1; }
            buf = p;
        }
        buf[n++] = c;
    }
    if (j->pos >= j->len) { free(buf); kc_ipa_json_set_error(j, "unterminated string"); return -1; }
    j->pos++;
    buf[n] = '\0';
    *out = buf;
    return 0;
}

/**
 * Parses a JSON numeric value.
 * @param j   JSON tokenizer.
 * @param out Destination for the parsed value.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_json_parse_double(kc_ipa_json_t *j, double *out) {
    kc_ipa_json_skip_ws(j);
    char *end;
    *out = strtod(j->src + j->pos, &end);
    if (end == j->src + j->pos) { kc_ipa_json_set_error(j, "expected number"); return -1; }
    j->pos = (size_t)(end - j->src);
    return 0;
}

static int kc_ipa_json_skip_value(kc_ipa_json_t *j);

/**
 * Skips a complete JSON object.
 * @param j JSON tokenizer.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_json_skip_object(kc_ipa_json_t *j) {
    if (kc_ipa_json_expect(j, '{') != 0) return -1;
    kc_ipa_json_skip_ws(j);
    if (j->pos < j->len && j->src[j->pos] == '}') { j->pos++; return 0; }
    for (;;) {
        char *key = NULL;
        if (kc_ipa_json_parse_string(j, &key) != 0) return -1;
        free(key);
        if (kc_ipa_json_expect(j, ':') != 0) return -1;
        if (kc_ipa_json_skip_value(j) != 0) return -1;
        kc_ipa_json_skip_ws(j);
        if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
        break;
    }
    return kc_ipa_json_expect(j, '}');
}

/**
 * Skips a complete JSON array.
 * @param j JSON tokenizer.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_json_skip_array(kc_ipa_json_t *j) {
    if (kc_ipa_json_expect(j, '[') != 0) return -1;
    kc_ipa_json_skip_ws(j);
    if (j->pos < j->len && j->src[j->pos] == ']') { j->pos++; return 0; }
    for (;;) {
        if (kc_ipa_json_skip_value(j) != 0) return -1;
        kc_ipa_json_skip_ws(j);
        if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
        break;
    }
    return kc_ipa_json_expect(j, ']');
}

/**
 * Skips a complete JSON value of any type.
 * @param j JSON tokenizer.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_json_skip_value(kc_ipa_json_t *j) {
    kc_ipa_json_skip_ws(j);
    if (j->pos >= j->len) { kc_ipa_json_set_error(j, "unexpected end of input"); return -1; }
    char c = j->src[j->pos];
    if (c == '"') { char *s = NULL; int rc = kc_ipa_json_parse_string(j, &s); free(s); return rc; }
    if (c == '{') return kc_ipa_json_skip_object(j);
    if (c == '[') return kc_ipa_json_skip_array(j);
    if (c == 't') {
        if (j->pos + 4 <= j->len && strncmp(j->src + j->pos, "true", 4) == 0) { j->pos += 4; return 0; }
        kc_ipa_json_set_error(j, "invalid token"); return -1;
    }
    if (c == 'f') {
        if (j->pos + 5 <= j->len && strncmp(j->src + j->pos, "false", 5) == 0) { j->pos += 5; return 0; }
        kc_ipa_json_set_error(j, "invalid token"); return -1;
    }
    if (c == 'n') {
        if (j->pos + 4 <= j->len && strncmp(j->src + j->pos, "null", 4) == 0) { j->pos += 4; return 0; }
        kc_ipa_json_set_error(j, "invalid token"); return -1;
    }
    {
        char *end;
        strtod(j->src + j->pos, &end);
        if (end == j->src + j->pos) { kc_ipa_json_set_error(j, "invalid value"); return -1; }
        j->pos = (size_t)(end - j->src);
        return 0;
    }
}

/**
 * Frees a dynamically allocated string array.
 * @param arr   Array of string pointers.
 * @param count Number of elements.
 * @return None.
 */
static void kc_ipa_free_str_array(char **arr, int count) {
    int i;
    for (i = 0; i < count; i++) free(arr[i]);
    free(arr);
}

/**
 * Parses a JSON string array into a newly allocated array.
 * @param j     JSON tokenizer.
 * @param out   Destination for the allocated array.
 * @param count Destination for array length.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_parse_string_array(kc_ipa_json_t *j, char ***out, int *count) {
    if (kc_ipa_json_expect(j, '[') != 0) return -1;
    char **arr = NULL;
    int n = 0, cap = 0;
    kc_ipa_json_skip_ws(j);
    if (j->pos < j->len && j->src[j->pos] == ']') { j->pos++; *out = arr; *count = 0; return 0; }
    for (;;) {
        kc_ipa_json_skip_ws(j);
        char *s = NULL;
        if (kc_ipa_json_parse_string(j, &s) != 0) { kc_ipa_free_str_array(arr, n); return -1; }
        if (n >= cap) {
            int nc = cap ? cap * 2 : 8;
            char **p = (char **)realloc(arr, (size_t)nc * sizeof(char *));
            if (!p) { free(s); kc_ipa_free_str_array(arr, n); return -1; }
            arr = p; cap = nc;
        }
        arr[n++] = s;
        kc_ipa_json_skip_ws(j);
        if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
        break;
    }
    if (kc_ipa_json_expect(j, ']') != 0) { kc_ipa_free_str_array(arr, n); return -1; }
    *out = arr; *count = n;
    return 0;
}

/**
 * Frees a dynamically allocated emit definition array.
 * @param arr   Array of emit definitions.
 * @param count Number of elements.
 * @return None.
 */
static void kc_ipa_free_emit_array(kc_ipa_emit_def_t *arr, int count) {
    int i;
    for (i = 0; i < count; i++) { free(arr[i].key); free(arr[i].value); }
    free(arr);
}

/**
 * Parses a JSON emit object into a newly allocated array of emit definitions.
 * @param j     JSON tokenizer.
 * @param out   Destination for the allocated array.
 * @param count Destination for array length.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_parse_emit_object(kc_ipa_json_t *j, kc_ipa_emit_def_t **out, int *count) {
    *out = NULL; *count = 0;
    if (kc_ipa_json_expect(j, '{') != 0) return -1;
    kc_ipa_json_skip_ws(j);
    if (j->pos < j->len && j->src[j->pos] == '}') { j->pos++; return 0; }
    kc_ipa_emit_def_t *arr = NULL;
    int n = 0, cap = 0;
    for (;;) {
        char *key = NULL, *val = NULL;
        if (kc_ipa_json_parse_string(j, &key) != 0) { kc_ipa_free_emit_array(arr, n); return -1; }
        if (kc_ipa_json_expect(j, ':') != 0) { free(key); kc_ipa_free_emit_array(arr, n); return -1; }
        if (kc_ipa_json_parse_string(j, &val) != 0) { free(key); kc_ipa_free_emit_array(arr, n); return -1; }
        if (n >= cap) {
            int nc = cap ? cap * 2 : 4;
            kc_ipa_emit_def_t *p = (kc_ipa_emit_def_t *)realloc(arr, (size_t)nc * sizeof(*arr));
            if (!p) { free(key); free(val); kc_ipa_free_emit_array(arr, n); return -1; }
            arr = p; cap = nc;
        }
        arr[n].key = key; arr[n].value = val; n++;
        kc_ipa_json_skip_ws(j);
        if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
        break;
    }
    if (kc_ipa_json_expect(j, '}') != 0) { kc_ipa_free_emit_array(arr, n); return -1; }
    *out = arr; *count = n;
    return 0;
}

static void kc_ipa_free_node_def(kc_ipa_node_def_t *nd);

/**
 * Frees a dynamically allocated node definition array.
 * @param arr   Array of node definitions.
 * @param count Number of elements.
 * @return None.
 */
static void kc_ipa_free_node_array(kc_ipa_node_def_t *arr, int count) {
    int i;
    for (i = 0; i < count; i++) kc_ipa_free_node_def(&arr[i]);
    free(arr);
}

/**
 * Frees resources owned by a node definition.
 * @param nd Node definition to free.
 * @return None.
 */
static void kc_ipa_free_node_def(kc_ipa_node_def_t *nd) {
    free(nd->id);
    kc_ipa_free_str_array(nd->descriptions, nd->desc_count);
    kc_ipa_free_emit_array(nd->emits, nd->emit_count);
    kc_ipa_free_node_array(nd->children, nd->child_count);
}

static int kc_ipa_parse_node(kc_ipa_json_t *j, kc_ipa_node_def_t *nd);

/**
 * Parses a JSON node array into a newly allocated array of node definitions.
 * @param j     JSON tokenizer.
 * @param out   Destination for the allocated array.
 * @param count Destination for array length.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_parse_node_array(kc_ipa_json_t *j, kc_ipa_node_def_t **out, int *count) {
    *out = NULL; *count = 0;
    if (kc_ipa_json_expect(j, '[') != 0) return -1;
    kc_ipa_json_skip_ws(j);
    if (j->pos < j->len && j->src[j->pos] == ']') { j->pos++; return 0; }
    kc_ipa_node_def_t *arr = NULL;
    int n = 0, cap = 0;
    for (;;) {
        if (n >= cap) {
            int nc = cap ? cap * 2 : 8;
            kc_ipa_node_def_t *p = (kc_ipa_node_def_t *)realloc(arr, (size_t)nc * sizeof(*arr));
            if (!p) { kc_ipa_free_node_array(arr, n); return -1; }
            arr = p; cap = nc;
        }
        kc_ipa_json_skip_ws(j);
        memset(&arr[n], 0, sizeof(arr[n]));
        if (kc_ipa_parse_node(j, &arr[n]) != 0) { kc_ipa_free_node_array(arr, n); return -1; }
        n++;
        kc_ipa_json_skip_ws(j);
        if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
        break;
    }
    if (kc_ipa_json_expect(j, ']') != 0) { kc_ipa_free_node_array(arr, n); return -1; }
    *out = arr; *count = n;
    return 0;
}

/**
 * Parses a JSON node object into a node definition.
 * @param j  JSON tokenizer.
 * @param nd Destination node definition.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_parse_node(kc_ipa_json_t *j, kc_ipa_node_def_t *nd) {
    memset(nd, 0, sizeof(*nd));
    if (kc_ipa_json_expect(j, '{') != 0) return -1;
    kc_ipa_json_skip_ws(j);
    if (j->pos < j->len && j->src[j->pos] == '}') { j->pos++; return 0; }
    for (;;) {
        char *key = NULL;
        if (kc_ipa_json_parse_string(j, &key) != 0) return -1;
        if (kc_ipa_json_expect(j, ':') != 0) { free(key); return -1; }
        kc_ipa_json_skip_ws(j);
        if (strcmp(key, "id") == 0) {
            free(key);
            if (kc_ipa_json_parse_string(j, &nd->id) != 0) return -1;
        } else if (strcmp(key, "descriptions") == 0) {
            free(key);
            if (kc_ipa_parse_string_array(j, &nd->descriptions, &nd->desc_count) != 0) return -1;
        } else if (strcmp(key, "emit") == 0) {
            free(key);
            if (kc_ipa_parse_emit_object(j, &nd->emits, &nd->emit_count) != 0) return -1;
        } else if (strcmp(key, "children") == 0) {
            free(key);
            if (kc_ipa_parse_node_array(j, &nd->children, &nd->child_count) != 0) return -1;
        } else {
            free(key);
            if (kc_ipa_json_skip_value(j) != 0) return -1;
        }
        kc_ipa_json_skip_ws(j);
        if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
        break;
    }
    return kc_ipa_json_expect(j, '}');
}

/**
 * Frees resources owned by a manifest.
 * @param m Manifest to free.
 * @return None.
 */
static void kc_ipa_free_manifest(kc_ipa_manifest_t *m) {
    free(m->schema);
    free(m->id);
    free(m->vendor);
    free(m->version);
    kc_ipa_free_node_array(m->map.nodes, m->map.node_count);
}

/**
 * Parses a JSON manifest object.
 * @param j JSON tokenizer.
 * @param m Destination manifest.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_parse_manifest(kc_ipa_json_t *j, kc_ipa_manifest_t *m) {
    memset(m, 0, sizeof(*m));
    m->node_threshold = 0.72f;
    m->min_ngram      = 1;
    m->max_ngram      = 5;

    if (kc_ipa_json_expect(j, '{') != 0) return -1;
    kc_ipa_json_skip_ws(j);
    if (j->pos < j->len && j->src[j->pos] == '}') { j->pos++; return 0; }

    for (;;) {
        char *key = NULL;
        if (kc_ipa_json_parse_string(j, &key) != 0) return -1;
        if (kc_ipa_json_expect(j, ':') != 0) { free(key); return -1; }
        kc_ipa_json_skip_ws(j);

        if (strcmp(key, "schema") == 0) {
            free(key);
            if (kc_ipa_json_parse_string(j, &m->schema) != 0) return -1;
        } else if (strcmp(key, "id") == 0) {
            free(key);
            if (kc_ipa_json_parse_string(j, &m->id) != 0) return -1;
        } else if (strcmp(key, "vendor") == 0) {
            free(key);
            if (kc_ipa_json_parse_string(j, &m->vendor) != 0) return -1;
        } else if (strcmp(key, "version") == 0) {
            free(key);
            if (kc_ipa_json_parse_string(j, &m->version) != 0) return -1;
        } else if (strcmp(key, "defaults") == 0) {
            free(key);
            if (kc_ipa_json_expect(j, '{') != 0) return -1;
            kc_ipa_json_skip_ws(j);
            if (j->pos < j->len && j->src[j->pos] == '}') { j->pos++; goto next_top_key; }
            for (;;) {
                char *dk = NULL;
                if (kc_ipa_json_parse_string(j, &dk) != 0) return -1;
                if (kc_ipa_json_expect(j, ':') != 0) { free(dk); return -1; }
                kc_ipa_json_skip_ws(j);
                if (strcmp(dk, "node_threshold") == 0) {
                    double v; free(dk);
                    if (kc_ipa_json_parse_double(j, &v) != 0) return -1;
                    m->node_threshold = (float)v;
                } else if (strcmp(dk, "min_ngram") == 0) {
                    double v; free(dk);
                    if (kc_ipa_json_parse_double(j, &v) != 0) return -1;
                    m->min_ngram = (int)v;
                } else if (strcmp(dk, "max_ngram") == 0) {
                    double v; free(dk);
                    if (kc_ipa_json_parse_double(j, &v) != 0) return -1;
                    m->max_ngram = (int)v;
                } else {
                    free(dk);
                    if (kc_ipa_json_skip_value(j) != 0) return -1;
                }
                kc_ipa_json_skip_ws(j);
                if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
                break;
            }
            if (kc_ipa_json_expect(j, '}') != 0) return -1;
        } else if (strcmp(key, "nodes") == 0) {
            free(key);
            if (kc_ipa_parse_node_array(j, &m->map.nodes, &m->map.node_count) != 0) return -1;
        } else {
            free(key);
            if (kc_ipa_json_skip_value(j) != 0) return -1;
        }
next_top_key:
        kc_ipa_json_skip_ws(j);
        if (j->pos < j->len && j->src[j->pos] == ',') { j->pos++; continue; }
        break;
    }
    return kc_ipa_json_expect(j, '}');
}

/**
 * Validates a parsed manifest for required fields and constraints.
 * @param m      Manifest to validate.
 * @param errbuf Buffer to receive a human-readable error message.
 * @param errsz  Byte capacity of errbuf.
 * @return 0 on success, -1 on validation failure.
 */
static int kc_ipa_validate_manifest(const kc_ipa_manifest_t *m, char *errbuf, size_t errsz) {
    if (!m->schema || strcmp(m->schema, "ipa.map.v1") != 0) {
        snprintf(errbuf, errsz, "schema must be ipa.map.v1");
        return -1;
    }
    if (!m->id || !m->id[0]) {
        snprintf(errbuf, errsz, "id is required");
        return -1;
    }
    if (!m->vendor || !m->vendor[0]) {
        snprintf(errbuf, errsz, "vendor is required");
        return -1;
    }
    if (!m->version || !m->version[0]) {
        snprintf(errbuf, errsz, "version is required");
        return -1;
    }
    if (m->map.node_count < 1) {
        snprintf(errbuf, errsz, "at least one node is required");
        return -1;
    }
    if (m->node_threshold < 0.0f || m->node_threshold > 1.0f) {
        snprintf(errbuf, errsz, "node_threshold must be in [0, 1]");
        return -1;
    }
    if (m->min_ngram < 1 || m->max_ngram < m->min_ngram) {
        snprintf(errbuf, errsz, "min_ngram and max_ngram must be positive, max >= min");
        return -1;
    }
    int k;
    for (k = 0; k < m->map.node_count; k++) {
        const kc_ipa_node_def_t *nd = &m->map.nodes[k];
        if (!nd->id || !nd->id[0]) {
            snprintf(errbuf, errsz, "node id is required");
            return -1;
        }
        if (nd->desc_count < 1) {
            snprintf(errbuf, errsz, "node %s must have at least one description", nd->id);
            return -1;
        }
    }
    return 0;
}

typedef struct { uint32_t off, len; } kc_ipa_str_ref_t;

/**
 * Compiles a JSON schema file into a binary .ipa resource file.
 * @param json_path   Path to the input schema.json.
 * @param output_path Path to write the compiled .ipa file.
 * @return KC_IPA_OK on success, KC_IPA_EFORMAT on bad schema, or KC_IPA_ERROR.
 */
int kc_ipa_build(const char *json_path, const char *output_path) {
    int rc = KC_IPA_ERROR;
    char *json_text = NULL;
    kc_ipa_manifest_t manifest;
    memset(&manifest, 0, sizeof(manifest));
    kc_emb_t *emb = NULL;
    kc_ipa_strtab_t strtab;
    kc_ipa_wbuf_t buf;
    memset(&strtab, 0, sizeof(strtab));
    memset(&buf, 0, sizeof(buf));
    float *emb_vec = NULL;

    FILE *fin = fopen(json_path, "rb");
    if (!fin) { fprintf(stderr, "ipa: cannot open %s\n", json_path); return KC_IPA_ERROR; }
    if (fseek(fin, 0, SEEK_END) != 0) { fclose(fin); return KC_IPA_ERROR; }
    long fsize = ftell(fin);
    rewind(fin);
    if (fsize <= 0) { fclose(fin); return KC_IPA_EFORMAT; }
    json_text = (char *)malloc((size_t)fsize + 1);
    if (!json_text) { fclose(fin); return KC_IPA_ERROR; }
    if (fread(json_text, 1, (size_t)fsize, fin) != (size_t)fsize) {
        fclose(fin); free(json_text);
        fprintf(stderr, "ipa: read error %s\n", json_path);
        return KC_IPA_ERROR;
    }
    fclose(fin);
    json_text[fsize] = '\0';

    kc_ipa_json_t j;
    j.src = json_text; j.pos = 0; j.len = (size_t)fsize; j.errbuf[0] = '\0';
    if (kc_ipa_parse_manifest(&j, &manifest) != 0) {
        fprintf(stderr, "ipa: JSON parse error: %s\n", j.errbuf);
        free(json_text);
        return KC_IPA_EFORMAT;
    }
    free(json_text);
    json_text = NULL;

    char errbuf[256];
    if (kc_ipa_validate_manifest(&manifest, errbuf, sizeof(errbuf)) != 0) {
        fprintf(stderr, "ipa: manifest error: %s\n", errbuf);
        kc_ipa_free_manifest(&manifest);
        return KC_IPA_EFORMAT;
    }

    emb = kc_emb_open();
    if (!emb) { fprintf(stderr, "ipa: failed to open embedding context\n"); kc_ipa_free_manifest(&manifest); return KC_IPA_ERROR; }
    int emb_dim = kc_emb_dim(emb);
    if (emb_dim <= 0) { fprintf(stderr, "ipa: invalid embedding dimension %d\n", emb_dim); kc_emb_close(emb); kc_ipa_free_manifest(&manifest); return KC_IPA_ERROR; }
    emb_vec = (float *)malloc((size_t)emb_dim * sizeof(float));
    if (!emb_vec) { kc_emb_close(emb); kc_ipa_free_manifest(&manifest); return KC_IPA_ERROR; }

    if (kc_ipa_strtab_init(&strtab) != 0 || kc_ipa_wbuf_init(&buf) != 0) goto cleanup;

    {
        kc_ipa_file_hdr_t ph;
        memset(&ph, 0, sizeof(ph));
        if (kc_ipa_wbuf_append(&buf, &ph, sizeof(ph)) != 0) goto cleanup;
    }

    int ki;
    int total_node_emb  = 0;
    int total_child_emb = 0;
    for (ki = 0; ki < manifest.map.node_count; ki++) {
        const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
        total_node_emb += nd->desc_count;
        int ci;
        for (ci = 0; ci < nd->child_count; ci++)
            total_child_emb += nd->children[ci].desc_count;
    }

    kc_ipa_str_ref_t *node_emb_label  = (kc_ipa_str_ref_t *)calloc(
        (size_t)(total_node_emb  + 1), sizeof(*node_emb_label));
    kc_ipa_str_ref_t *child_emb_label = (kc_ipa_str_ref_t *)calloc(
        (size_t)(total_child_emb + 1), sizeof(*child_emb_label));
    kc_ipa_str_ref_t *child_parent_id = (kc_ipa_str_ref_t *)calloc(
        (size_t)(total_child_emb + 1), sizeof(*child_parent_id));

    if (!node_emb_label || !child_emb_label || !child_parent_id) {
        free(node_emb_label); free(child_emb_label); free(child_parent_id);
        goto cleanup;
    }

    uint32_t id_off, id_len, vendor_off, vendor_len, schemaver_off, schemaver_len;
    {
        const char *id  = manifest.id      ? manifest.id      : "";
        const char *vnd = manifest.vendor  ? manifest.vendor  : "";
        const char *ver = manifest.version ? manifest.version : "";
        if (kc_ipa_strtab_add(&strtab, id,  strlen(id),  &id_off,        &id_len)        != 0 ||
            kc_ipa_strtab_add(&strtab, vnd, strlen(vnd), &vendor_off,    &vendor_len)    != 0 ||
            kc_ipa_strtab_add(&strtab, ver, strlen(ver), &schemaver_off, &schemaver_len) != 0) {
            free(node_emb_label); free(child_emb_label); free(child_parent_id);
            goto cleanup;
        }
    }

    {
        int ok = 1;
        int ie = 0, ce = 0;
        for (ki = 0; ki < manifest.map.node_count && ok; ki++) {
            const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
            int di;
            for (di = 0; di < nd->desc_count && ok; di++) {
                if (kc_ipa_strtab_add(&strtab, nd->id, strlen(nd->id),
                    &node_emb_label[ie].off, &node_emb_label[ie].len) != 0)
                    ok = 0;
                ie++;
            }
            int ci;
            for (ci = 0; ci < nd->child_count && ok; ci++) {
                const kc_ipa_node_def_t *ch = &nd->children[ci];
                int di2;
                for (di2 = 0; di2 < ch->desc_count && ok; di2++) {
                    if (kc_ipa_strtab_add(&strtab, ch->id, strlen(ch->id),
                        &child_emb_label[ce].off, &child_emb_label[ce].len) != 0 ||
                        kc_ipa_strtab_add(&strtab, nd->id, strlen(nd->id),
                        &child_parent_id[ce].off, &child_parent_id[ce].len) != 0)
                        ok = 0;
                    ce++;
                }
            }
        }
        if (!ok) { free(node_emb_label); free(child_emb_label); free(child_parent_id); goto cleanup; }
    }

    typedef struct {
        uint32_t nid_off, nid_len, ekey_off, ekey_len, eval_off, eval_len;
        uint32_t has_children;
    } kc_ipa_node_meta_strtab_t;

    typedef struct {
        uint32_t pid_off, pid_len, cid_off, cid_len;
        uint32_t ekey_off, ekey_len, eval_off, eval_len;
    } kc_ipa_child_meta_strtab_t;

    int total_node_meta  = 0;
    int total_child_meta = 0;
    for (ki = 0; ki < manifest.map.node_count; ki++) {
        const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
        total_node_meta += nd->emit_count > 0 ? nd->emit_count : 1;
        int ci;
        for (ci = 0; ci < nd->child_count; ci++) {
            const kc_ipa_node_def_t *ch = &nd->children[ci];
            total_child_meta += ch->emit_count > 0 ? ch->emit_count : 1;
        }
    }

    kc_ipa_node_meta_strtab_t *nmeta = (kc_ipa_node_meta_strtab_t *)calloc(
        (size_t)(total_node_meta + 1), sizeof(*nmeta));
    kc_ipa_child_meta_strtab_t *cmeta = (kc_ipa_child_meta_strtab_t *)calloc(
        (size_t)(total_child_meta + 1), sizeof(*cmeta));
    if (!nmeta || !cmeta) {
        free(nmeta); free(cmeta);
        free(node_emb_label); free(child_emb_label); free(child_parent_id);
        goto cleanup;
    }

    {
        int ok = 1;
        int ni = 0, ci2 = 0;
        for (ki = 0; ki < manifest.map.node_count && ok; ki++) {
            const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
            int n_emits = nd->emit_count > 0 ? nd->emit_count : 1;
            int ei;
            for (ei = 0; ei < n_emits && ok; ei++) {
                uint32_t nid_off, nid_len, ekey_off, ekey_len, eval_off, eval_len;
                if (kc_ipa_strtab_add(&strtab, nd->id, strlen(nd->id), &nid_off, &nid_len) != 0)
                    { ok = 0; break; }
                if (nd->emit_count > 0) {
                    const char *ek = nd->emits[ei].key;
                    const char *ev = nd->emits[ei].value;
                    if (kc_ipa_strtab_add(&strtab, ek, strlen(ek), &ekey_off, &ekey_len) != 0 ||
                        kc_ipa_strtab_add(&strtab, ev, strlen(ev), &eval_off, &eval_len) != 0)
                        { ok = 0; break; }
                } else {
                    ekey_off = 0; ekey_len = 0; eval_off = 0; eval_len = 0;
                }
                nmeta[ni].nid_off      = nid_off;
                nmeta[ni].nid_len      = nid_len;
                nmeta[ni].ekey_off     = ekey_off;
                nmeta[ni].ekey_len     = ekey_len;
                nmeta[ni].eval_off     = eval_off;
                nmeta[ni].eval_len     = eval_len;
                nmeta[ni].has_children = (uint32_t)(nd->child_count > 0 ? 1 : 0);
                ni++;
            }
            int ci;
            for (ci = 0; ci < nd->child_count && ok; ci++) {
                const kc_ipa_node_def_t *ch = &nd->children[ci];
                int n_cemits = ch->emit_count > 0 ? ch->emit_count : 1;
                int ei2;
                for (ei2 = 0; ei2 < n_cemits && ok; ei2++) {
                    uint32_t pid_off, pid_len, cid_off, cid_len;
                    uint32_t ekey_off, ekey_len, eval_off, eval_len;
                    if (kc_ipa_strtab_add(&strtab, nd->id, strlen(nd->id), &pid_off, &pid_len) != 0 ||
                        kc_ipa_strtab_add(&strtab, ch->id, strlen(ch->id), &cid_off, &cid_len) != 0)
                        { ok = 0; break; }
                    if (ch->emit_count > 0) {
                        const char *ek = ch->emits[ei2].key;
                        const char *ev = ch->emits[ei2].value;
                        if (kc_ipa_strtab_add(&strtab, ek, strlen(ek), &ekey_off, &ekey_len) != 0 ||
                            kc_ipa_strtab_add(&strtab, ev, strlen(ev), &eval_off, &eval_len) != 0)
                            { ok = 0; break; }
                    } else {
                        ekey_off = 0; ekey_len = 0; eval_off = 0; eval_len = 0;
                    }
                    cmeta[ci2].pid_off  = pid_off;
                    cmeta[ci2].pid_len  = pid_len;
                    cmeta[ci2].cid_off  = cid_off;
                    cmeta[ci2].cid_len  = cid_len;
                    cmeta[ci2].ekey_off = ekey_off;
                    cmeta[ci2].ekey_len = ekey_len;
                    cmeta[ci2].eval_off = eval_off;
                    cmeta[ci2].eval_len = eval_len;
                    ci2++;
                }
            }
        }
        if (!ok) {
            free(nmeta); free(cmeta);
            free(node_emb_label); free(child_emb_label); free(child_parent_id);
            goto cleanup;
        }
    }

    uint32_t strtab_file_off = (uint32_t)buf.size;
    if (kc_ipa_wbuf_append(&buf, strtab.data, strtab.size) != 0) {
        free(nmeta); free(cmeta);
        free(node_emb_label); free(child_emb_label); free(child_parent_id);
        goto cleanup;
    }

    uint32_t node_emb_off   = (uint32_t)buf.size;
    uint32_t node_emb_cnt   = (uint32_t)total_node_emb;
    uint32_t child_emb_off  = 0;
    uint32_t child_emb_cnt  = 0;
    uint32_t node_meta_off  = 0;
    uint32_t node_meta_cnt  = 0;
    uint32_t child_meta_off = 0;
    uint32_t child_meta_cnt = 0;

    {
        int ok = 1;
        int ie = 0;
        for (ki = 0; ki < manifest.map.node_count && ok; ki++) {
            const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
            int di;
            for (di = 0; di < nd->desc_count && ok; di++) {
                if (kc_emb_exec(emb, nd->descriptions[di], emb_vec) != KC_EMB_OK) {
                    fprintf(stderr, "ipa: embedding failed: %s\n", nd->descriptions[di]);
                    ok = 0; break;
                }
                kc_ipa_emb_hdr_t eh;
                eh.label_off = node_emb_label[ie].off;
                eh.label_len = node_emb_label[ie].len;
                if (kc_ipa_wbuf_append(&buf, &eh, sizeof(eh)) != 0 ||
                    kc_ipa_wbuf_append(&buf, emb_vec, (size_t)emb_dim * sizeof(float)) != 0)
                    ok = 0;
                ie++;
            }
        }
        if (!ok) {
            free(nmeta); free(cmeta);
            free(node_emb_label); free(child_emb_label); free(child_parent_id);
            goto cleanup;
        }
    }

    child_emb_off = (uint32_t)buf.size;
    child_emb_cnt = (uint32_t)total_child_emb;
    {
        int ok = 1;
        int ce = 0;
        for (ki = 0; ki < manifest.map.node_count && ok; ki++) {
            const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
            int ci;
            for (ci = 0; ci < nd->child_count && ok; ci++) {
                const kc_ipa_node_def_t *ch = &nd->children[ci];
                int di;
                for (di = 0; di < ch->desc_count && ok; di++) {
                    if (kc_emb_exec(emb, ch->descriptions[di], emb_vec) != KC_EMB_OK) {
                        fprintf(stderr, "ipa: embedding failed: %s\n", ch->descriptions[di]);
                        ok = 0; break;
                    }
                    kc_ipa_emb_hdr_t eh;
                    eh.label_off = child_emb_label[ce].off;
                    eh.label_len = child_emb_label[ce].len;
                    if (kc_ipa_wbuf_append(&buf, &eh, sizeof(eh)) != 0 ||
                        kc_ipa_wbuf_append(&buf, emb_vec, (size_t)emb_dim * sizeof(float)) != 0)
                        ok = 0;
                    ce++;
                }
            }
        }
        if (!ok) {
            free(nmeta); free(cmeta);
            free(node_emb_label); free(child_emb_label); free(child_parent_id);
            goto cleanup;
        }
    }

    node_meta_off = (uint32_t)buf.size;
    {
        int ok = 1;
        int ni = 0;
        for (ki = 0; ki < manifest.map.node_count && ok; ki++) {
            const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
            int n_emits = nd->emit_count > 0 ? nd->emit_count : 1;
            int ei;
            for (ei = 0; ei < n_emits && ok; ei++) {
                kc_ipa_node_meta_t m;
                m.node_id_off  = nmeta[ni].nid_off;
                m.node_id_len  = nmeta[ni].nid_len;
                m.emit_key_off = nmeta[ni].ekey_off;
                m.emit_key_len = nmeta[ni].ekey_len;
                m.emit_val_off = nmeta[ni].eval_off;
                m.emit_val_len = nmeta[ni].eval_len;
                m.has_children = nmeta[ni].has_children;
                if (kc_ipa_wbuf_append(&buf, &m, sizeof(m)) != 0) ok = 0;
                node_meta_cnt++;
                ni++;
            }
        }
        if (!ok) {
            free(nmeta); free(cmeta);
            free(node_emb_label); free(child_emb_label); free(child_parent_id);
            goto cleanup;
        }
    }

    child_meta_off = (uint32_t)buf.size;
    {
        int ok = 1;
        int ci2 = 0;
        for (ki = 0; ki < manifest.map.node_count && ok; ki++) {
            const kc_ipa_node_def_t *nd = &manifest.map.nodes[ki];
            int ci;
            for (ci = 0; ci < nd->child_count && ok; ci++) {
                const kc_ipa_node_def_t *ch = &nd->children[ci];
                int n_cemits = ch->emit_count > 0 ? ch->emit_count : 1;
                int ei;
                for (ei = 0; ei < n_cemits && ok; ei++) {
                    kc_ipa_child_meta_t m;
                    m.parent_id_off = cmeta[ci2].pid_off;
                    m.parent_id_len = cmeta[ci2].pid_len;
                    m.node_id_off   = cmeta[ci2].cid_off;
                    m.node_id_len   = cmeta[ci2].cid_len;
                    m.emit_key_off  = cmeta[ci2].ekey_off;
                    m.emit_key_len  = cmeta[ci2].ekey_len;
                    m.emit_val_off  = cmeta[ci2].eval_off;
                    m.emit_val_len  = cmeta[ci2].eval_len;
                    if (kc_ipa_wbuf_append(&buf, &m, sizeof(m)) != 0) ok = 0;
                    child_meta_cnt++;
                    ci2++;
                }
            }
        }
        if (!ok) {
            free(nmeta); free(cmeta);
            free(node_emb_label); free(child_emb_label); free(child_parent_id);
            goto cleanup;
        }
    }

    free(nmeta); free(cmeta);
    free(node_emb_label); free(child_emb_label); free(child_parent_id);

    {
        kc_ipa_file_hdr_t hdr;
        hdr.magic[0]          = KC_IPA_MAGIC_0;
        hdr.magic[1]          = KC_IPA_MAGIC_1;
        hdr.magic[2]          = KC_IPA_MAGIC_2;
        hdr.magic[3]          = KC_IPA_MAGIC_3;
        hdr.version           = KC_IPA_FMT_VERSION;
        hdr.file_size         = (uint32_t)buf.size;
        hdr.strtab_off        = strtab_file_off;
        hdr.strtab_size       = (uint32_t)strtab.size;
        hdr.id_off            = id_off;
        hdr.id_len            = id_len;
        hdr.vendor_off        = vendor_off;
        hdr.vendor_len        = vendor_len;
        hdr.schemaver_off     = schemaver_off;
        hdr.schemaver_len     = schemaver_len;
        hdr.node_threshold    = manifest.node_threshold;
        hdr.min_ngram         = (uint32_t)manifest.min_ngram;
        hdr.max_ngram         = (uint32_t)manifest.max_ngram;
        hdr.emb_dim           = (uint32_t)emb_dim;
        hdr.node_emb_off      = node_emb_off;
        hdr.node_emb_count    = node_emb_cnt;
        hdr.child_emb_off     = child_emb_off;
        hdr.child_emb_count   = child_emb_cnt;
        hdr.node_meta_off     = node_meta_off;
        hdr.node_meta_count   = node_meta_cnt;
        hdr.child_meta_off    = child_meta_off;
        hdr.child_meta_count  = child_meta_cnt;
        memcpy(buf.data, &hdr, sizeof(hdr));
    }

    {
        FILE *fout = fopen(output_path, "wb");
        if (!fout) {
            fprintf(stderr, "ipa: cannot write %s\n", output_path);
        } else {
            if (fwrite(buf.data, 1, buf.size, fout) == buf.size)
                rc = KC_IPA_OK;
            else
                fprintf(stderr, "ipa: write error %s\n", output_path);
            fclose(fout);
        }
    }

cleanup:
    free(emb_vec);
    kc_emb_close(emb);
    kc_ipa_strtab_free(&strtab);
    kc_ipa_wbuf_free(&buf);
    kc_ipa_free_manifest(&manifest);
    return rc;
}

/**
 * Checks that a byte range is within a mapped buffer.
 * @param base      Pointer to the mapped buffer.
 * @param base_size Total buffer size in bytes.
 * @param off       Byte offset of the range.
 * @param len       Byte length of the range.
 * @return Non-zero if the range is within bounds, zero otherwise.
 */
static int kc_ipa_bounds_ok(const void *base, size_t base_size, uint32_t off, size_t len) {
    (void)base;
    return (uint64_t)off + (uint64_t)len <= (uint64_t)base_size;
}

/**
 * Builds an HNSW index from an embedding block in the mapped file.
 * @param base       Pointer to the mapped file base.
 * @param base_size  Total mapped file size.
 * @param emb_off    Offset of the embedding block.
 * @param emb_count  Number of embedding entries.
 * @param emb_dim    Embedding vector dimension.
 * @param strtab     Pointer to the string table.
 * @return Allocated HNSW index, or NULL on failure.
 */
static kc_hnsw_t *kc_ipa_build_hnsw(
    const uint8_t *base, size_t base_size,
    uint32_t emb_off, uint32_t emb_count, uint32_t emb_dim,
    const char *strtab)
{
    kc_hnsw_t *hnsw = kc_hnsw_open((size_t)emb_dim, KC_HNSW_METRIC_COSINE);
    if (!hnsw) return NULL;
    if (emb_count == 0) return hnsw;
    if (kc_hnsw_reserve(hnsw, (size_t)emb_count) != KC_HNSW_OK) {
        kc_hnsw_close(hnsw); return NULL;
    }
    size_t stride = sizeof(kc_ipa_emb_hdr_t) + (size_t)emb_dim * sizeof(float);
    uint32_t i;
    for (i = 0; i < emb_count; i++) {
        size_t entry_off = (size_t)emb_off + i * stride;
        if (!kc_ipa_bounds_ok(base, base_size, (uint32_t)entry_off, stride)) {
            kc_hnsw_close(hnsw); return NULL;
        }
        const kc_ipa_emb_hdr_t *eh = (const kc_ipa_emb_hdr_t *)(base + entry_off);
        if (!kc_ipa_bounds_ok(base, base_size, eh->label_off, (size_t)eh->label_len + 1)) {
            kc_hnsw_close(hnsw); return NULL;
        }
        const char  *label = strtab + eh->label_off;
        const float *vec   = (const float *)(base + entry_off + sizeof(kc_ipa_emb_hdr_t));
        if (kc_hnsw_add(hnsw, label, vec) != KC_HNSW_OK) {
            kc_hnsw_close(hnsw); return NULL;
        }
    }
    if (kc_hnsw_build(hnsw) != KC_HNSW_OK) {
        kc_hnsw_close(hnsw); return NULL;
    }
    return hnsw;
}

/**
 * Return default options for the library.
 * @return Default options struct.
 */
kc_ipa_options_t kc_ipa_options_default(void) {
    kc_ipa_options_t opts;
    memset(&opts, 0, sizeof(opts));
    return opts;
}

/**
 * Load configuration overrides from environment variables.
 * @param opts Options to override.
 * @return None.
 */
void kc_ipa_options_load_env(kc_ipa_options_t *opts) {
    if (!opts) return;
    int i;
    for (i = 0; i < env_config_table_n; i++) {
        const char *val = getenv(env_config_table[i].env_var);
        if (!val) continue;
        switch (env_config_table[i].type) {
            case KC_ENV_TYPE_INT: {
                char *end;
                long v = strtol(val, &end, 10);
                if (end != val && *end == '\0')
                    *(int *)((char *)opts + env_config_table[i].offset) = (int)v;
                break;
            }
            case KC_ENV_TYPE_FLOAT: {
                char *end;
                float v = strtof(val, &end);
                if (end != val && *end == '\0')
                    *(float *)((char *)opts + env_config_table[i].offset) = v;
                break;
            }
            case KC_ENV_TYPE_STR: {
                char **p = (char **)((char *)opts + env_config_table[i].offset);
                free(*p);
                *p = strdup(val);
                break;
            }
        }
    }
}

/**
 * Release resources owned by options struct.
 * @param opts Options to free.
 * @return None.
 */
void kc_ipa_options_free(kc_ipa_options_t *opts) {
    if (!opts) return;
    free(opts->schema_path);
    opts->schema_path = NULL;
}

/**
 * Register or remove a signal handler.
 * @param ctx Context handle.
 * @param sig Signal ID.
 * @param cb  Callback or NULL to remove.
 * @return KC_IPA_OK on success, KC_IPA_ERROR on failure.
 */
int kc_ipa_on_signal(kc_ipa_schema_t *ctx, int sig, kc_ipa_signal_callback_t cb) {
    int i;
    for (i = 0; i < ctx->n_signal_handlers; i++) {
        if (ctx->signal_handlers[i].sig == sig) {
            if (cb) {
                ctx->signal_handlers[i].cb = cb;
            } else {
                int tail = ctx->n_signal_handlers - i - 1;
                if (tail > 0)
                    memmove(&ctx->signal_handlers[i],
                            &ctx->signal_handlers[i + 1],
                            (size_t)tail * sizeof(kc_ipa_signal_entry_t));
                ctx->n_signal_handlers--;
            }
            return KC_IPA_OK;
        }
    }
    if (!cb) return KC_IPA_OK;
    if (ctx->n_signal_handlers >= ctx->signal_handlers_capacity) {
        int new_cap = ctx->signal_handlers_capacity ? ctx->signal_handlers_capacity * 2 : 4;
        kc_ipa_signal_entry_t *p = (kc_ipa_signal_entry_t *)realloc(ctx->signal_handlers,
            (size_t)new_cap * sizeof(kc_ipa_signal_entry_t));
        if (!p) return KC_IPA_ERROR;
        ctx->signal_handlers = p;
        ctx->signal_handlers_capacity = new_cap;
    }
    ctx->signal_handlers[ctx->n_signal_handlers].sig = sig;
    ctx->signal_handlers[ctx->n_signal_handlers].cb = cb;
    ctx->n_signal_handlers++;
    return KC_IPA_OK;
}

/**
 * Raise a signal, calling the registered callback if any.
 * @param ctx Context handle.
 * @param sig Signal ID.
 * @return KC_IPA_OK if handled, KC_IPA_ERROR if not.
 */
int kc_ipa_raise_signal(kc_ipa_schema_t *ctx, int sig) {
    int i;
    for (i = 0; i < ctx->n_signal_handlers; i++) {
        if (ctx->signal_handlers[i].sig == sig) {
            ctx->signal_handlers[i].cb(ctx);
            return KC_IPA_OK;
        }
    }
    return KC_IPA_ERROR;
}

/**
 * Store context internally for use by the static signal listener.
 * @param ctx Context handle.
 * @return KC_IPA_OK on success, KC_IPA_ERROR on failure.
 */
int kc_ipa_listen_signals(kc_ipa_schema_t *ctx) {
    if (!ctx) return KC_IPA_ERROR;
    g_signal_ctx = ctx;
    return KC_IPA_OK;
}

/**
 * Wire a specific OS signal to the library listener.
 * @param ctx    Context handle.
 * @param sig_id OS signal ID.
 * @return KC_IPA_OK on success, KC_IPA_ERROR on failure.
 */
int kc_ipa_listen_signal(kc_ipa_schema_t *ctx, int sig_id) {
    if (!ctx) return KC_IPA_ERROR;
    g_signal_ctx = ctx;
#ifdef _WIN32
    (void)sig_id;
#else
    signal(sig_id, kc_ipa_signal_listener);
#endif
    return KC_IPA_OK;
}

/**
 * Generic OS-signal handler.
 * @param sig OS signal ID.
 * @return None.
 */
void kc_ipa_signal_listener(int sig) {
    if (g_signal_ctx)
        kc_ipa_raise_signal(g_signal_ctx, sig);
}

/**
 * Opens and validates a compiled .ipa schema file, returning a runtime schema.
 * @param ctx_out Destination schema handle pointer.
 * @param opts    Configuration options.
 * @return KC_IPA_OK on success, KC_IPA_ERROR on failure.
 */
int kc_ipa_open(kc_ipa_schema_t **ctx_out, kc_ipa_options_t *opts) {
    if (!ctx_out || !opts || !opts->schema_path) return KC_IPA_ERROR;
    *ctx_out = NULL;

    kc_ipa_schema_t *s = (kc_ipa_schema_t *)calloc(1, sizeof(*s));
    if (!s) return KC_IPA_ERROR;

    s->opts = *opts;
    s->opts.schema_path = strdup(opts->schema_path);
    if (!s->opts.schema_path) { free(s); return KC_IPA_ERROR; }
    s->signal_handlers = NULL;
    s->n_signal_handlers = 0;
    s->signal_handlers_capacity = 0;

#ifndef _WIN32
    if (pthread_mutex_init(&s->lock, NULL) != 0) { free(s->opts.schema_path); free(s); return KC_IPA_ERROR; }
#else
    InitializeCriticalSection(&s->lock);
#endif

    if (kc_mmap_open(&s->mmap, s->opts.schema_path) != KC_MMAP_OK) {
        fprintf(stderr, "ipa: cannot open %s\n", s->opts.schema_path);
        goto fail;
    }

    const uint8_t *base  = (const uint8_t *)kc_mmap_data(&s->mmap);
    size_t         fsz   = kc_mmap_size(&s->mmap);

    if (fsz < sizeof(kc_ipa_file_hdr_t)) { fprintf(stderr, "ipa: file too small: %s\n", s->opts.schema_path); goto fail; }

    const kc_ipa_file_hdr_t *hdr = (const kc_ipa_file_hdr_t *)base;

    if (hdr->magic[0] != KC_IPA_MAGIC_0 || hdr->magic[1] != KC_IPA_MAGIC_1 ||
        hdr->magic[2] != KC_IPA_MAGIC_2 || hdr->magic[3] != KC_IPA_MAGIC_3) {
        fprintf(stderr, "ipa: invalid magic: %s\n", s->opts.schema_path);
        goto fail;
    }
    if (hdr->version != KC_IPA_FMT_VERSION) {
        fprintf(stderr, "ipa: unsupported version %u: %s\n", hdr->version, s->opts.schema_path);
        goto fail;
    }
    if (hdr->file_size != (uint32_t)fsz) {
        fprintf(stderr, "ipa: file size mismatch: %s\n", s->opts.schema_path);
        goto fail;
    }
    if (!kc_ipa_bounds_ok(base, fsz, hdr->strtab_off, hdr->strtab_size)) {
        fprintf(stderr, "ipa: corrupt strtab: %s\n", s->opts.schema_path);
        goto fail;
    }

    s->strtab         = (const char *)(base + hdr->strtab_off);
    s->node_threshold = hdr->node_threshold;
    s->min_ngram      = (int)hdr->min_ngram;
    s->max_ngram      = (int)hdr->max_ngram;

    s->emb = kc_emb_open();
    if (!s->emb) { fprintf(stderr, "ipa: cannot open emb\n"); goto fail; }
    s->emb_dim = kc_emb_dim(s->emb);
    if (s->emb_dim <= 0 || (uint32_t)s->emb_dim != hdr->emb_dim) {
        fprintf(stderr, "ipa: emb_dim mismatch\n"); goto fail;
    }

    kc_ipa_runtime_index_t *idx = &s->index;

    idx->node_hnsw = kc_ipa_build_hnsw(
        base, fsz, hdr->node_emb_off, hdr->node_emb_count, hdr->emb_dim, s->strtab);
    if (!idx->node_hnsw) { fprintf(stderr, "ipa: cannot build node index\n"); goto fail; }

    if (hdr->child_emb_count > 0) {
        idx->child_hnsw = kc_ipa_build_hnsw(
            base, fsz, hdr->child_emb_off, hdr->child_emb_count, hdr->emb_dim, s->strtab);
        if (!idx->child_hnsw) { fprintf(stderr, "ipa: cannot build child index\n"); goto fail; }
    }

    if (hdr->node_meta_count > 0) {
        size_t nm_sz = (size_t)hdr->node_meta_count * sizeof(kc_ipa_node_meta_t);
        if (!kc_ipa_bounds_ok(base, fsz, hdr->node_meta_off, nm_sz)) { fprintf(stderr, "ipa: corrupt node meta\n"); goto fail; }
        idx->node_meta       = (const kc_ipa_node_meta_t *)(base + hdr->node_meta_off);
        idx->node_meta_count = hdr->node_meta_count;
    }

    if (hdr->child_meta_count > 0) {
        size_t cm_sz = (size_t)hdr->child_meta_count * sizeof(kc_ipa_child_meta_t);
        if (!kc_ipa_bounds_ok(base, fsz, hdr->child_meta_off, cm_sz)) { fprintf(stderr, "ipa: corrupt child meta\n"); goto fail; }
        idx->child_meta       = (const kc_ipa_child_meta_t *)(base + hdr->child_meta_off);
        idx->child_meta_count = hdr->child_meta_count;
    }

    *ctx_out = s;
    return KC_IPA_OK;

fail:
    kc_ipa_close(s);
    return KC_IPA_ERROR;
}

/**
 * Closes a schema handle and releases all associated resources.
 * @param s Schema handle to close. Safe to call with NULL.
 * @return KC_IPA_OK.
 */
int kc_ipa_close(kc_ipa_schema_t *s) {
    if (!s) return KC_IPA_OK;
    kc_hnsw_close(s->index.node_hnsw);
    kc_hnsw_close(s->index.child_hnsw);
    kc_emb_close(s->emb);
    kc_mmap_close(&s->mmap);
#ifndef _WIN32
    pthread_mutex_destroy(&s->lock);
#else
    DeleteCriticalSection(&s->lock);
#endif
    kc_ipa_options_free(&s->opts);
    free(s->signal_handlers);
    free(s);
    return KC_IPA_OK;
}

/**
 * Resolves an emit value template against a matched span.
 * @param tpl  Emit value template string. "$raw" is replaced with span.
 * @param span Raw text of the matched span.
 * @return Allocated resolved string, or NULL on failure.
 */
static char *kc_ipa_resolve_emit(const char *tpl, const char *span) {
    if (strcmp(tpl, "$raw") == 0) return strdup(span);
    return strdup(tpl);
}

/**
 * Populates the emit array of a match from node metadata.
 * @param idx     Runtime index.
 * @param strtab  String table pointer.
 * @param node_id Node identifier to look up.
 * @param span    Raw matched span text used for $raw resolution.
 * @param m       Match to populate emits on.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_build_match_emits(
    const kc_ipa_runtime_index_t *idx, const char *strtab,
    const char *node_id, const char *span,
    kc_ipa_match_t *m)
{
    int count = 0;
    uint32_t i;
    for (i = 0; i < idx->node_meta_count; i++) {
        const kc_ipa_node_meta_t *nm = &idx->node_meta[i];
        if (nm->emit_key_len == 0) continue;
        if (strcmp(strtab + nm->node_id_off, node_id) == 0)
            count++;
    }
    if (count == 0) { m->emits = NULL; m->emit_count = 0; return 0; }
    m->emits = (kc_ipa_emit_t *)calloc((size_t)count, sizeof(kc_ipa_emit_t));
    if (!m->emits) return -1;
    int k = 0;
    for (i = 0; i < idx->node_meta_count; i++) {
        const kc_ipa_node_meta_t *nm = &idx->node_meta[i];
        if (nm->emit_key_len == 0) continue;
        if (strcmp(strtab + nm->node_id_off, node_id) != 0) continue;
        m->emits[k].key   = strdup(strtab + nm->emit_key_off);
        m->emits[k].value = kc_ipa_resolve_emit(strtab + nm->emit_val_off, span);
        if (!m->emits[k].key || !m->emits[k].value) return -1;
        k++;
    }
    m->emit_count = k;
    return 0;
}

/**
 * Returns non-zero if any node meta entry for node_id has children.
 * @param idx     Runtime index.
 * @param strtab  String table pointer.
 * @param node_id Node identifier to check.
 * @return 1 if node has children, 0 otherwise.
 */
static int kc_ipa_node_has_children(
    const kc_ipa_runtime_index_t *idx, const char *strtab, const char *node_id)
{
    uint32_t i;
    for (i = 0; i < idx->node_meta_count; i++) {
        const kc_ipa_node_meta_t *nm = &idx->node_meta[i];
        if (strcmp(strtab + nm->node_id_off, node_id) == 0 && nm->has_children)
            return 1;
    }
    return 0;
}

typedef struct {
    kc_emb_t                   *emb;
    int                         emb_dim;
    float                      *emb_buf;
    const kc_ipa_runtime_index_t *idx;
    const char                 *strtab;
    float                       threshold;
    const char                 *parent_id;
    kc_ipa_match_t             *matches;
    int                         count;
    int                         cap;
} kc_ipa_scan_ctx_t;

/**
 * Appends a match to the scan context's dynamic match array.
 * @param ctx Scan context owning the match array.
 * @param m   Match to append (copied by value).
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_match_push(kc_ipa_scan_ctx_t *ctx, kc_ipa_match_t *m) {
    if (ctx->count >= ctx->cap) {
        int nc = ctx->cap ? ctx->cap * 2 : 8;
        kc_ipa_match_t *p = (kc_ipa_match_t *)realloc(
            ctx->matches, (size_t)nc * sizeof(kc_ipa_match_t));
        if (!p) return -1;
        ctx->matches = p;
        ctx->cap = nc;
    }
    ctx->matches[ctx->count++] = *m;
    return 0;
}

static void kc_ipa_free_match(kc_ipa_match_t *m);

/**
 * Frees a dynamically allocated match array and all owned resources.
 * @param arr   Array of match structs.
 * @param count Number of elements.
 * @return None.
 */
static void kc_ipa_free_match_array(kc_ipa_match_t *arr, int count) {
    int i;
    for (i = 0; i < count; i++) kc_ipa_free_match(&arr[i]);
    free(arr);
}

/**
 * Frees all resources owned by a single match struct.
 * @param m Match to free.
 * @return None.
 */
static void kc_ipa_free_match(kc_ipa_match_t *m) {
    int i;
    free(m->id);
    free(m->span);
    for (i = 0; i < m->emit_count; i++) { free(m->emits[i].key); free(m->emits[i].value); }
    free(m->emits);
    kc_ipa_free_match_array(m->children, m->child_count);
}

/**
 * Scans a span for child node matches belonging to a given parent.
 * @param emb          Embedding context.
 * @param emb_dim      Embedding vector dimension.
 * @param emb_buf      Scratch buffer for embedding output.
 * @param idx          Runtime index.
 * @param strtab       String table pointer.
 * @param threshold    Minimum cosine similarity to accept a child.
 * @param min_ngram    Minimum ngram token count.
 * @param max_ngram    Maximum ngram token count.
 * @param parent_id    Node id of the parent whose children to scan.
 * @param span         Text span to scan.
 * @param out_children Destination for the allocated child match array.
 * @param out_count    Destination for child match count.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_scan_children(
    kc_emb_t *emb, int emb_dim, float *emb_buf,
    const kc_ipa_runtime_index_t *idx, const char *strtab,
    float threshold, int min_ngram, int max_ngram,
    const char *parent_id, const char *span,
    kc_ipa_match_t **out_children, int *out_count);

/**
 * Ngram chunk visitor called during top-level or child span scanning.
 * @param chunk   Current ngram chunk with span byte offsets.
 * @param context Pointer to a kc_ipa_scan_ctx_t scan context.
 * @return 1 to close the span on match, 0 to skip, -1 on fatal error.
 */
static int kc_ipa_ngram_visit(const kc_ngram_chunk_t *chunk, void *context) {
    kc_ipa_scan_ctx_t *ctx = (kc_ipa_scan_ctx_t *)context;
    if (!chunk->input || chunk->byte_start >= chunk->byte_end) return 0;

    size_t span_len = chunk->byte_end - chunk->byte_start;
    char *span = (char *)malloc(span_len + 1);
    if (!span) return 0;
    memcpy(span, chunk->input + chunk->byte_start, span_len);
    span[span_len] = '\0';

    if (kc_emb_exec(ctx->emb, span, ctx->emb_buf) != KC_EMB_OK) { free(span); return 0; }

    kc_hnsw_t *hnsw = ctx->parent_id ? ctx->idx->child_hnsw : ctx->idx->node_hnsw;
    if (!hnsw) { free(span); return 0; }

    kc_hnsw_result_t hr;
    int found = kc_hnsw_search(hnsw, ctx->emb_buf, 1, 0.0, &hr);
    if (found <= 0) { free(span); return 0; }

    float score = (float)hr.score;
    if (score < ctx->threshold) { free(span); return 0; }

    if (ctx->parent_id) {
        uint32_t i;
        int matched_parent = 0;
        for (i = 0; i < ctx->idx->child_meta_count; i++) {
            const kc_ipa_child_meta_t *cm = &ctx->idx->child_meta[i];
            if (strcmp(ctx->strtab + cm->node_id_off, hr.id) == 0 &&
                strcmp(ctx->strtab + cm->parent_id_off, ctx->parent_id) == 0) {
                matched_parent = 1;
                break;
            }
        }
        if (!matched_parent) { free(span); return 0; }
    }

    kc_ipa_match_t m;
    memset(&m, 0, sizeof(m));
    m.id    = strdup(hr.id);
    m.span  = span;
    m.score = score;

    if (!m.id) { free(span); return 0; }

    if (ctx->parent_id) {
        int k;
        for (k = 0; k < ctx->count; k++) {
            if (strcmp(ctx->matches[k].id, hr.id) == 0) {
                if (score > ctx->matches[k].score) {
                    kc_ipa_free_match(&ctx->matches[k]);
                    ctx->matches[k] = m;
                } else {
                    kc_ipa_free_match(&m);
                }
                return 1;
            }
        }
        if (kc_ipa_match_push(ctx, &m) != 0) { kc_ipa_free_match(&m); return -1; }
        return 1;
    }

    if (kc_ipa_build_match_emits(ctx->idx, ctx->strtab, hr.id, span, &m) != 0) {
        kc_ipa_free_match(&m); return 0;
    }

    if (kc_ipa_node_has_children(ctx->idx, ctx->strtab, hr.id) && ctx->idx->child_hnsw) {
        kc_ipa_scan_children(
            ctx->emb, ctx->emb_dim, ctx->emb_buf,
            ctx->idx, ctx->strtab,
            ctx->threshold, 1, chunk->size,
            hr.id, span,
            &m.children, &m.child_count);
    }

    if (kc_ipa_match_push(ctx, &m) != 0) { kc_ipa_free_match(&m); return -1; }
    return 1;
}

/**
 * Scans a span for child node matches belonging to a given parent.
 * @param emb          Embedding context.
 * @param emb_dim      Embedding vector dimension.
 * @param emb_buf      Scratch buffer for embedding output.
 * @param idx          Runtime index.
 * @param strtab       String table pointer.
 * @param threshold    Minimum cosine similarity to accept a child.
 * @param min_ngram    Minimum ngram token count.
 * @param max_ngram    Maximum ngram token count.
 * @param parent_id    Node id of the parent whose children to scan.
 * @param span         Text span to scan.
 * @param out_children Destination for the allocated child match array.
 * @param out_count    Destination for child match count.
 * @return 0 on success, -1 on failure.
 */
static int kc_ipa_scan_children(
    kc_emb_t *emb, int emb_dim, float *emb_buf,
    const kc_ipa_runtime_index_t *idx, const char *strtab,
    float threshold, int min_ngram, int max_ngram,
    const char *parent_id, const char *span,
    kc_ipa_match_t **out_children, int *out_count)
{
    *out_children = NULL;
    *out_count = 0;

    kc_ipa_scan_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.emb       = emb;
    ctx.emb_dim   = emb_dim;
    ctx.emb_buf   = emb_buf;
    ctx.idx       = idx;
    ctx.strtab    = strtab;
    ctx.threshold = threshold;
    ctx.parent_id = parent_id;

    kc_ngram_options_t opts;
    kc_ngram_options_default(&opts);
    opts.min_tokens = min_ngram;
    opts.max_tokens = max_ngram;

    kc_ngram_execute(span, &opts, kc_ipa_ngram_visit, &ctx);

    int i;
    for (i = 0; i < ctx.count; i++) {
        kc_ipa_match_t *ch = &ctx.matches[i];
        if (ch->emit_count == 0) {
            uint32_t k;
            int n_emits = 0;
            for (k = 0; k < idx->child_meta_count; k++) {
                const kc_ipa_child_meta_t *cm = &idx->child_meta[k];
                if (cm->emit_key_len == 0) continue;
                if (strcmp(strtab + cm->node_id_off, ch->id) == 0 &&
                    strcmp(strtab + cm->parent_id_off, parent_id) == 0)
                    n_emits++;
            }
            if (n_emits > 0) {
                ch->emits = (kc_ipa_emit_t *)calloc((size_t)n_emits, sizeof(kc_ipa_emit_t));
                if (ch->emits) {
                    int ei = 0;
                    for (k = 0; k < idx->child_meta_count; k++) {
                        const kc_ipa_child_meta_t *cm = &idx->child_meta[k];
                        if (cm->emit_key_len == 0) continue;
                        if (strcmp(strtab + cm->node_id_off, ch->id) != 0) continue;
                        if (strcmp(strtab + cm->parent_id_off, parent_id) != 0) continue;
                        ch->emits[ei].key   = strdup(strtab + cm->emit_key_off);
                        ch->emits[ei].value = kc_ipa_resolve_emit(strtab + cm->emit_val_off, ch->span);
                        ei++;
                    }
                    ch->emit_count = ei;
                }
            }
        }
    }

    *out_children = ctx.matches;
    *out_count    = ctx.count;
    return 0;
}

/**
 * Parses an input string against a compiled schema and returns matches.
 * @param schema Opened schema handle.
 * @param input  Null-terminated input string to parse.
 * @param out    Destination for the parse result.
 * @return KC_IPA_OK on success, KC_IPA_ERROR otherwise.
 */
int kc_ipa_parse(kc_ipa_schema_t *schema, const char *input, kc_ipa_result_t *out) {
    if (!schema || !input || !out) return KC_IPA_ERROR;
    memset(out, 0, sizeof(*out));

#ifndef _WIN32
    pthread_mutex_lock(&schema->lock);
#else
    EnterCriticalSection(&schema->lock);
#endif

    int rc = KC_IPA_ERROR;

    float *emb_buf = (float *)malloc((size_t)schema->emb_dim * sizeof(float));
    if (!emb_buf) goto done;

    kc_ipa_scan_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.emb       = schema->emb;
    ctx.emb_dim   = schema->emb_dim;
    ctx.emb_buf   = emb_buf;
    ctx.idx       = &schema->index;
    ctx.strtab    = schema->strtab;
    ctx.threshold = schema->node_threshold;
    ctx.parent_id = NULL;

    kc_ngram_options_t opts;
    kc_ngram_options_default(&opts);
    opts.min_tokens = schema->min_ngram;
    opts.max_tokens = schema->max_ngram;

    kc_ngram_execute(input, &opts, kc_ipa_ngram_visit, &ctx);

    out->matches     = ctx.matches;
    out->match_count = ctx.count;
    out->ok          = ctx.count > 0;
    rc = KC_IPA_OK;

done:
    free(emb_buf);

#ifndef _WIN32
    pthread_mutex_unlock(&schema->lock);
#else
    LeaveCriticalSection(&schema->lock);
#endif

    if (rc != KC_IPA_OK) {
        if (ctx.matches) kc_ipa_free_match_array(ctx.matches, ctx.count);
    }
    return rc;
}

/**
 * Frees all resources owned by a parse result.
 * @param result Parse result to free. Safe to call with NULL.
 * @return None.
 */
void kc_ipa_result_free(kc_ipa_result_t *result) {
    if (!result) return;
    kc_ipa_free_match_array(result->matches, result->match_count);
    result->matches     = NULL;
    result->match_count = 0;
    result->ok          = 0;
}
