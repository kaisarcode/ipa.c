/**
 * ipa.c - Island Parser CLI
 * Summary: Command line interface for the ipa schema compiler and runtime.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#include "ipa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KC_IPA_VERSION "0.1.0"

/**
 * Reads all of standard input into a dynamically allocated buffer.
 * Strips trailing newlines from the result.
 * @param out_text Destination pointer for the allocated text.
 * @return 0 on success, -1 on failure.
 */
static int read_stdin(char **out_text) {
    char *data = NULL;
    size_t len = 0, cap = 0;
    char chunk[4096];
    size_t n;

    if (!out_text) return -1;

    while ((n = fread(chunk, 1, sizeof(chunk), stdin)) > 0) {
        if (len + n + 1 > cap) {
            size_t next = cap ? cap * 2 : 4096;
            while (next < len + n + 1) next *= 2;
            char *p = (char *)realloc(data, next);
            if (!p) { free(data); return -1; }
            data = p; cap = next;
        }
        memcpy(data + len, chunk, n);
        len += n;
    }

    if (ferror(stdin)) { free(data); return -1; }
    if (len == 0) { *out_text = NULL; return 0; }
    while (len > 0 && (data[len - 1] == '\n' || data[len - 1] == '\r')) len--;
    data[len] = '\0';
    *out_text = data;
    return 0;
}

/**
 * Writes a JSON-escaped string to stdout, including surrounding quotes.
 * Writes null (unquoted) when s is NULL.
 * @param s String to escape and print.
 * @return None.
 */
static void json_print_string(const char *s) {
    if (!s) { printf("null"); return; }
    putchar('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if      (c == '"')  printf("\\\"");
        else if (c == '\\') printf("\\\\");
        else if (c == '\n') printf("\\n");
        else if (c == '\r') printf("\\r");
        else if (c == '\t') printf("\\t");
        else if (c < 0x20)  printf("\\u%04x", c);
        else                putchar(c);
    }
    putchar('"');
}

/**
 * Prints a JSON emit object to stdout.
 * @param emits  Array of emit pairs.
 * @param count  Number of pairs.
 * @param indent Leading spaces for formatting.
 * @return None.
 */
static void print_emits(const kc_ipa_emit_t *emits, int count, int indent) {
    int i;
    char pad[32];
    int p = indent < 30 ? indent : 30;
    for (i = 0; i < p; i++) pad[i] = ' ';
    pad[p] = '\0';

    printf("%s\"emit\":{", pad);
    for (i = 0; i < count; i++) {
        if (i > 0) putchar(',');
        json_print_string(emits[i].key);
        putchar(':');
        json_print_string(emits[i].value);
    }
    putchar('}');
}

/**
 * Prints a single match as a JSON object to stdout.
 * @param m      Match to print.
 * @param indent Leading spaces for formatting.
 * @return None.
 */
static void print_match(const kc_ipa_match_t *m, int indent) {
    int i;
    char pad[32];
    int p = indent < 30 ? indent : 30;
    for (i = 0; i < p; i++) pad[i] = ' ';
    pad[p] = '\0';

    printf("%s{", pad);
    printf("\"id\":");
    json_print_string(m->id);
    printf(",\"span\":");
    json_print_string(m->span);
    printf(",\"score\":%.4f", (double)m->score);
    if (m->emit_count > 0) {
        putchar(',');
        print_emits(m->emits, m->emit_count, 0);
    }
    if (m->child_count > 0) {
        printf(",\"children\":[");
        for (i = 0; i < m->child_count; i++) {
            if (i > 0) putchar(',');
            print_match(&m->children[i], 0);
        }
        putchar(']');
    }
    putchar('}');
}

/**
 * Prints a parse result as a JSON object to stdout.
 * @param r Result to print.
 * @return None.
 */
static void print_result(const kc_ipa_result_t *r) {
    int i;
    printf("{\"ok\":%s,\"matches\":[", r->ok ? "true" : "false");
    for (i = 0; i < r->match_count; i++) {
        if (i > 0) putchar(',');
        print_match(&r->matches[i], 0);
    }
    printf("]}\n");
}

/**
 * Print command usage information.
 * @param name Program executable name.
 * @return None.
 */
static void print_help(const char *name) {
    printf("Usage: %s <schema.ipa> [text]          Parse mode\n", name);
    printf("       %s <schema.json> -b              Build mode\n", name);
    printf("\n");
    printf("Options:\n");
    printf("    -b, --build         Compile schema.json into schema.ipa\n");
    printf("    -h, --help          Show this help message\n");
    printf("    -v, --version       Show version\n");
}

/**
 * Print version information.
 * @return None.
 */
static void print_version(void) {
    printf("ipa %s\n", KC_IPA_VERSION);
}

/**
 * Execute the command line interface.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Process status code.
 */
int main(int argc, char **argv) {
    const char *schema_path = NULL;
    const char *text_arg    = NULL;
    int build_mode = 0;
    int i = 1;

    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            print_version();
            return 0;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--build") == 0) {
            build_mode = 1;
        } else if (argv[i][0] != '-') {
            if (!schema_path)  schema_path = argv[i];
            else if (!text_arg) text_arg   = argv[i];
        } else {
            fprintf(stderr, "ipa: unknown option '%s'\n", argv[i]);
            return 1;
        }
        i++;
    }

    if (!schema_path) { print_help(argv[0]); return 1; }

    if (build_mode) {
        size_t plen = strlen(schema_path);
        char *out_path;
        if (plen > 5 && strcmp(schema_path + plen - 5, ".json") == 0) {
            out_path = (char *)malloc(plen + 1);
            if (!out_path) { fprintf(stderr, "ipa: out of memory\n"); return 1; }
            memcpy(out_path, schema_path, plen - 5);
            memcpy(out_path + plen - 5, ".ipa", 5);
        } else {
            out_path = (char *)malloc(plen + 5);
            if (!out_path) { fprintf(stderr, "ipa: out of memory\n"); return 1; }
            memcpy(out_path, schema_path, plen);
            memcpy(out_path + plen, ".ipa", 5);
        }

        int rc = kc_ipa_build(schema_path, out_path);
        free(out_path);
        if (rc == KC_IPA_EFORMAT) { fprintf(stderr, "ipa: invalid schema: %s\n", schema_path); return 2; }
        if (rc != KC_IPA_OK)      { fprintf(stderr, "ipa: build failed: %s\n", schema_path);   return 2; }
        return 0;
    }

    char *stdin_text = NULL;
    const char *text = text_arg;

    if (!text) {
        if (read_stdin(&stdin_text) != 0) { fprintf(stderr, "ipa: failed to read stdin\n"); return 1; }
        text = stdin_text;
    }

    if (!text || !*text) { free(stdin_text); return 0; }

    kc_ipa_schema_t *schema = kc_ipa_open(schema_path);
    if (!schema) { fprintf(stderr, "ipa: failed to open schema: %s\n", schema_path); free(stdin_text); return 1; }

    kc_ipa_result_t result;
    int rc = kc_ipa_parse(schema, text, &result);
    if (rc != KC_IPA_OK) {
        fprintf(stderr, "ipa: parse failed (rc=%d)\n", rc);
        kc_ipa_close(schema);
        free(stdin_text);
        return 1;
    }

    print_result(&result);
    kc_ipa_result_free(&result);
    kc_ipa_close(schema);
    free(stdin_text);
    return 0;
}
