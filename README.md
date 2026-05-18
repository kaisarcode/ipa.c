# ipa.c - Schema-Driven Semantic Island Parser 

`ipa.c` is a compact, stand-alone island parser library and CLI. Given a schema of named nodes and example descriptions, it finds which nodes match spans of natural language text, what they emit, and how they break down into child nodes.

---

## CLI

### Build mode

Compile a schema manifest into a binary resource:

```bash
./bin/x86_64/linux/ipa path/to/schema.json -b
```

The compiled resource is written to the same directory with the `.json` extension replaced by `.ipa`.

### Parse mode

Match input text against a compiled schema:

```bash
./bin/x86_64/linux/ipa path/to/schema.ipa "install firefox"
```

Pipe input through standard input:

```bash
echo "install firefox" | ./bin/x86_64/linux/ipa path/to/schema.ipa
```

### Output

Parse results are written to stdout as a JSON object:

```json
{
  "ok": true,
  "matches": [
    {
      "id": "install_command",
      "span": "install firefox",
      "score": 0.9821,
      "emit": {"action": "install"},
      "children": [
        {"id": "software", "span": "firefox", "score": 0.9934, "emit": {"software": "firefox"}}
      ]
    }
  ]
}
```

When no node matches above the threshold, `matches` is empty and `ok` is `false`.

---

### Parameters

| Flag | Description |
| :--- | :--- |
| `-b`, `--build` | Compile `schema.json` into `schema.ipa` |
| `-h`, `--help` | Show help and usage |
| `-v`, `--version` | Show version |

### Exit codes

| Code | Meaning |
| :--- | :--- |
| `0` | Success |
| `1` | Runtime or parse error |
| `2` | Build or schema error |

---

## Schema manifest

Schemas are defined in JSON with schema `ipa.map.v1`:

```json
{
  "schema": "ipa.map.v1",
  "id": "myschema",
  "vendor": "myvendor",
  "version": "0.1.0",
  "defaults": {
    "node_threshold": 0.80,
    "min_ngram": 1,
    "max_ngram": 5
  },
  "nodes": [
    {
      "id": "install_command",
      "descriptions": [
        "install firefox",
        "install chrome",
        "setup software",
        "add application"
      ],
      "emit": {
        "action": "install"
      },
      "children": [
        {
          "id": "software",
          "descriptions": ["firefox", "chrome", "brave", "vlc"],
          "emit": {
            "software": "$raw"
          }
        }
      ]
    }
  ]
}
```

### Nodes

Each node has an `id`, a list of `descriptions` used to generate its embedding, an optional `emit` map of key-value pairs, and an optional `children` array of child nodes.

The parser scans the input using descending n-gram windows. When a span scores above the threshold, the span is closed (not re-subdivided) and the match is recorded. If the matched node has `children`, the same span is scanned recursively against the child index to find refinements.

### Emit values

Emit values are literal strings. The special value `"$raw"` is resolved at runtime to the raw text of the matched span.

---

## Public API

```c
#include "ipa.h"

/* Open a compiled schema resource */
kc_ipa_schema_t *schema = kc_ipa_open("path/to/schema.ipa");

/* Parse input */
kc_ipa_result_t result;
kc_ipa_parse(schema, "install firefox", &result);

/* Use result */
if (result.ok) {
    printf("id: %s\n", result.matches[0].id);
}

/* Free result and schema */
kc_ipa_result_free(&result);
kc_ipa_close(schema);
```

---

## Lifecycle

- `kc_ipa_open()` — maps the `.ipa` resource into memory and builds runtime HNSW indexes.
- `kc_ipa_parse()` — scans input for semantic islands and scores them against the schema. Thread-safe.
- `kc_ipa_result_free()` — releases all strings and arrays owned by a result.
- `kc_ipa_close()` — releases all resources owned by the schema handle.
- `kc_ipa_build()` — compiles a `schema.json` manifest into a `.ipa` binary resource.

---

## Build

Compiled artifacts are generated under `bin/{arch}/{platform}/` for the host architecture running the build. The build compiles the vendored source snapshots from `lib/`, including `lib/ggml` and `lib/model.gguf`, and does not require external kclib archives.

```bash
make clean && make
```

### Multiarch Builds

The project is prepared to build artifacts for multiple architectures under `bin/{arch}/{platform}/`. A plain `make` builds only the current host architecture, while the targets below build the full matrix or a specific target.

```bash
make all
make x86_64/linux
make x86_64/windows
make i686/linux
make i686/windows
make aarch64/linux
make aarch64/android
make armv7/linux
make armv7/android
make armv7hf/linux
make riscv64/linux
make powerpc64le/linux
make mips/linux
make mipsel/linux
make mips64el/linux
make s390x/linux
make loongarch64/linux
```

## License

[![GPLv3](https://www.gnu.org/graphics/gplv3-127x51.png)](https://www.gnu.org/licenses/gpl-3.0.html)

This project is distributed under the **GNU General Public License version 3 (GPLv3)**. 
