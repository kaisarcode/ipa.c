# ipa.c — Schema-Driven Semantic Island Parser

## Overview
ipa.c is a compact, standalone island parser library and CLI. Given a JSON schema manifest of named nodes with example descriptions and emit definitions, it compiles the schema into a binary `.ipa` resource, then matches natural language text spans against the compiled schema using GGML-based embedding similarity and HNSW approximate nearest-neighbor search. Architecture: node embeddings generated via an embedded BERT-style model (`lib/model.gguf`), stored in a flat binary format with string table, and indexed at load time with HNSW for runtime ANN queries. Real-world use: intent classification, slot-filling, semantic command parsing.

## Architecture
The pipeline has two phases: **Build** reads a JSON manifest, validates it, generates embeddings for each description via `libemb.c` (GGML BERT), serializes everything into a `.ipa` binary (header, string table, embedding blocks, node/chlid metadata). **Runtime** memory-maps the `.ipa`, rebuilds in-memory HNSW indexes from the embedding blocks, then scans input text via descending n-gram windows (`libngram.c`), embedding each span and querying the HNSW index for matches above a configurable threshold. Matched nodes resolve emit templates (including `$raw` span substitution) and optionally recurse into child indexes for refinement.

## Directory Layout
| Path | Contents |
|------|----------|
| `src/ipa.c` | CLI entry point, arg parsing, JSON stdout output |
| `src/ipa.h` | Public API: `kc_ipa_schema_t`, result/match/emit types, parse/build lifecycle |
| `src/libipa.c` | Core implementation: JSON manifest parser, binary compiler, HNSW index builder, ngram-based scan with embedding scoring |
| `lib/emb.h` `lib/libemb.c` | GGML BERT embedding engine (WordPiece tokenizer, worker thread pool, LayerNorm, GELU) |
| `lib/hnsw.h` `lib/libhnsw.c` | HNSW vector index (cosine/inner/L2, M=16, ef=64, rwlock, xorshift64 RNG) |
| `lib/mmap.h` `lib/libmmap.c` | Portable read-only memory mapping (POSIX mmap / Win32 MapViewOfFile) |
| `lib/ngram.h` `lib/libngram.c` | Descending sliding-window n-gram traversal with visitor callback |
| `lib/ggml/` | Vendored GGML tensor library (architecture dispatch: x86, ARM, RISC-V, etc.) |
| `lib/model.gguf` | Embedded BERT embedding model (compiled into binary via objcopy) |
| `Makefile` `CMakeLists.txt` | Cross-compilation build system (16+ target arch/platform combos) |
| `test.sh` | Shell test suite: build valid/invalid schemas, parse with argv/stdin, emit/child/threshold/malformed tests |

## Data Model
### Internal Structures
| Symbol | Type | Role |
|--------|------|------|
| `kc_ipa_file_hdr_t` | struct | Binary `.ipa` file header: 4-byte magic `KIPA`, version, strtab offset/size, schema identity offsets, node thresholds, min/max ngram, emb_dim, embedding and metadata block pointers |
| `kc_ipa_emb_hdr_t` | struct | Embedding entry header: string table offset+length for the node/child label |
| `kc_ipa_node_meta_t` | struct | Node metadata: string table refs for node ID, emit key, emit value; `has_children` flag |
| `kc_ipa_child_meta_t` | struct | Child metadata: string table refs for parent ID, child ID, emit key, emit value |
| `kc_ipa_emit_def_t` | struct | Parsed emit key-value definition (build-time only) |
| `kc_ipa_node_def_t` | struct | Parsed node definition: id, descriptions array, emits array, children array (build-time tree) |
| `kc_ipa_schema_def_t` | struct | Parsed schema definition: nodes array+count |
| `kc_ipa_manifest_t` | struct | Full parsed manifest: schema string, id, vendor, version, default threshold/ngram bounds, map |
| `kc_ipa_runtime_index_t` | struct | Runtime indexes: node HNSW, child HNSW, node_meta array+count, child_meta array+count |
| `kc_ipa_schema_t` | struct (opaque) | Schema handle: mmap region, embedding context, emb_dim, threshold, ngram bounds, strtab pointer, runtime index, mutex |
| `kc_ipa_strtab_t` | struct | Build-time string table builder (dynamic byte buffer with append) |
| `kc_ipa_wbuf_t` | struct | Build-time binary write buffer (dynamic byte buffer with reserve/append) |
| `kc_ipa_json_t` | struct | JSON tokenizer: source pointer, position, length, error buffer |
| `kc_ipa_str_ref_t` | struct | String table reference (offset+length) |
| `kc_ipa_scan_ctx_t` | struct | Scan context: embedding handle, dimension, scratch buffer, runtime index, strtab, threshold, parent_id (NULL for top-level), dynamic match array |
| `kc_ipa_node_meta_strtab_t` | struct | Build-time intermediate: strtab refs for node_meta fields |
| `kc_ipa_child_meta_strtab_t` | struct | Build-time intermediate: strtab refs for child_meta fields |
| `kc_ipa_match_t` | struct (public) | One match: id, span, score, emits array, children array (tree) |
| `kc_ipa_emit_t` | struct (public) | One emit pair: key, value (owned strings) |
| `kc_ipa_result_t` | struct (public) | Parse result: ok flag, matches array, match_count |
| `kc_emb_t` | opaque | Embedding pool: worker thread pool with BERT inference contexts |
| `kc_emb_worker_t` | struct | Embedding worker: own BERT ctx, request/response condvars, shutdown flag |
| `kc_emb_layer_t` | struct | BERT layer tensor handles (attn QKV, output, norms, FFN up/down) |
| `kc_emb_ctx_t` | struct | BERT inference context: tensors, vocab hash, compute buffers, backend/galloc |
| `kc_hnsw_t` | struct | HNSW index: items array, dimension, metric, M/ef params, RW lock, RNG |
| `kc_hnsw_item_t` | struct | HNSW item: id string, float* values, precomputed norm, level, neighbor lists |
| `kc_hnsw_edge_t` | struct | HNSW edge: target index |
| `kc_hnsw_neighbor_list_t` | struct | HNSW neighbor list: edges dynamic array |
| `kc_hnsw_heap_t` | struct | HNSW priority heap: node-score array with metric-aware ordering |
| `kc_hnsw_node_score_t` | struct | Search result node: index + score |
| `kc_hnsw_result_t` | struct (public) | Public search result: id string + score |
| `kc_ngram_chunk_t` | struct | Emitted n-gram chunk: input pointer, byte offsets, token indices, size |
| `kc_ngram_token_t` | struct | Internal token: byte_start, byte_end |
| `kc_ngram_token_list_t` | struct | Dynamic token list |
| `kc_ngram_span_t` | struct | Closed span for dedup: start/end token index |

### Hard Limits
| Limit | Value | Symbol |
|-------|-------|--------|
| Max matches per scan | 128 | `KC_IPA_MAX_MATCHES` |
| Max recursion depth | 8 | `KC_IPA_MAX_DEPTH` |
| Initial string table size | 65536 | `KC_IPA_STRTAB_INIT` |
| Initial write buffer size | 131072 | `KC_IPA_BUF_INIT` |
| Magic bytes | `KIPA` | `KC_IPA_MAGIC_0..3` |
| Format version | 1 | `KC_IPA_FMT_VERSION` |
| HNSW M | 16 | `KC_HNSW_HNSW_M` |
| HNSW ef_construction | 64 | `KC_HNSW_HNSW_EF_CONSTRUCTION` |
| HNSW ef_search | 64 (env `HNSW_EF_SEARCH` overrides) | `KC_HNSW_HNSW_EF_SEARCH` |
| HNSW level cap | 16 | (hardcoded in random_level) |
| Embedding worker threads | 1 | (hardcoded in `kc_emb_open`) |
| BERT compute buffer | 64 MiB | (hardcoded) |

## Input Format / Schema
**Build mode**: Reads a JSON manifest (schema `ipa.map.v1`) containing `id`, `vendor`, `version`, optional `defaults` (node_threshold, min_ngram, max_ngram), and a `nodes` array. Each node has `id`, `descriptions` (string array used for embedding generation), optional `emit` (key-value map), and optional `children` (recursive node array). Emit value `"$raw"` resolves at runtime to the matched span text. The build validates all required fields, generates embeddings for every description via the embedded BERT model, and writes a binary `.ipa` file: header → string table → node embedding block (headers + float vectors) → child embedding block → node metadata → child metadata. The `-b`/`--build` flag triggers compilation.

## Execution Pipeline / Algorithm
1. **kc_ipa_open**: memory-map `.ipa`, validate header (magic, version, size), extract string table pointer and configuration (threshold, ngram bounds). Open embedding context. Build two HNSW indexes: one for top-level node descriptions, one for child descriptions (if present). Walk embedding blocks, insert each embedding with its label into the appropriate HNSW, then call `kc_hnsw_build` to construct the multi-level graph. 2. **kc_ipa_parse (thread-safe via mutex)**: Initialize an n-gram traversal over the input text using descending window sizes from `max_ngram` down to `min_ngram`. For each unique span (tracking closed spans to avoid re-subdivision), embed the span via `kc_emb_exec`, query the top-level HNSW for the nearest neighbor above threshold, and if matched: build emit array from node metadata, recurse into child HNSW (if node has children) using `kc_ipa_scan_children` with the same span and a tighter window. Collect all matches in a dynamic array. 3. **kc_ipa_result_free**: recursively free all match resources. **CLI output**: JSON object with `ok`, `matches` array (id, span, score, emit object, optional children).

## Public API
| Function | Description |
|----------|-------------|
| `kc_ipa_open(path)` | Open compiled `.ipa`, return opaque `kc_ipa_schema_t*` or NULL |
| `kc_ipa_close(schema)` | Release schema: close HNSW, mmap, embedding; destroy mutex; free |
| `kc_ipa_parse(schema, input, out)` | Parse input text: scans via ngram + embedding HNSW, fills `kc_ipa_result_t`. Thread-safe via mutex. Returns `KC_IPA_OK` or `KC_IPA_ERROR` |
| `kc_ipa_result_free(result)` | Free owned strings/arrays in result (not the struct itself) |
| `kc_ipa_build(json_path, output_path)` | Compile JSON schema to `.ipa` binary. Returns `KC_IPA_OK`, `KC_IPA_EFORMAT`, or `KC_IPA_ERROR` |

## CLI
| Flag | Description |
|------|-------------|
| `-b`, `--build` | Compile `schema.json` → `schema.ipa` |
| `-h`, `--help` | Show help and usage |
| `-v`, `--version` | Show version |

Usage: `ipa <schema.ipa> [text]` (parse mode, text from argv or stdin); `ipa <schema.json> -b` (build mode).

| Exit | Meaning |
|------|---------|
| 0 | Success |
| 1 | Runtime/parse error (bad file, parse failure, OOM) |
| 2 | Build or schema error (invalid JSON, validation failure) |

## Build
`make` (native), `make all` (multiarch matrix), `make x86_64/linux` etc. Output: `bin/{arch}/{platform}/ipa` (and `libipa.a`/`libipa.dll`/`libipa.so`). Vendored: `lib/ggml/` (GGML tensor library), `lib/model.gguf` (embedded BERT model). Build requires CMake, Ninja, cross-compilers for non-native targets. CUDA not applicable (CPU-only BERT embedding).

## Error Handling
| Code / Message | Scenario |
|----------------|----------|
| `KC_IPA_OK` (0) | Success |
| `KC_IPA_ERROR` (-1) | Generic runtime error (OOM, I/O, embedding failure) |
| `KC_IPA_EFORMAT` (-2) | Schema validation or format error |
| `ipa: cannot open %s` | File not found or unreadable |
| `ipa: JSON parse error: %s` | Malformed JSON |
| `ipa: manifest error: %s` | Missing/invalid schema fields (id, vendor, version, nodes, threshold range) |
| `ipa: invalid magic: %s` | Bad `.ipa` magic bytes |
| `ipa: unsupported version %u` | `.ipa` format version mismatch |
| `ipa: corrupt strtab/node meta` | Out-of-bounds file offset |
| `ipa: embedding failed: %s` | BERT inference failure on a description string |
| `ipa: out of memory` | Allocation failure |

## Constraints
CPU-only (no GPU support). Single worker thread for embeddings (sequential per request). Requires `lib/model.gguf` during build (embedded BERT, not a large LLM). HNSW uses cosine similarity by default. Node threshold must be in [0,1]. Max 128 matches per parse, max 8 levels of child recursion. Supports any UTF-8 text; embedding quality depends on the BERT model. Cross-compilation matrix covers 16+ target triples.
