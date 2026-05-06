/**
 * hnsw.h - HNSW Vector Search
 * Summary: HNSW-based approximate nearest neighbor search library.
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef HNSW_H
#define HNSW_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_hnsw kc_hnsw_t;

#define KC_HNSW_OK 0
#define KC_HNSW_EINVAL -1
#define KC_HNSW_ENOMEM -2
#define KC_HNSW_ESTATE -3

#define KC_HNSW_METRIC_COSINE 1
#define KC_HNSW_METRIC_INNER_PRODUCT 2
#define KC_HNSW_METRIC_L2 3

/**
 * One ranked search result.
 * @param id User-provided vector identifier.
 * @param score Similarity score or distance.
 * @return No return value.
 */
typedef struct {
    const char *id;
    double score;
} kc_hnsw_result_t;

/**
 * Creates one vector index instance.
 * @param dimension Fixed vector dimension for all entries.
 * @param metric Configured similarity metric.
 * @return Index pointer or NULL on allocation failure.
 */
kc_hnsw_t *kc_hnsw_open(size_t dimension, int metric);

/**
 * Releases one vector index instance.
 * Must not be called while any other thread holds the index.
 * @param hnsw Index pointer.
 * @return No return value.
 */
void kc_hnsw_close(kc_hnsw_t *hnsw);

/**
 * Reserves capacity for a target number of vectors.
 * Acquires an exclusive write lock internally.
 * @param hnsw Index pointer.
 * @param capacity Target vector capacity.
 * @return Status code.
 */
int kc_hnsw_reserve(kc_hnsw_t *hnsw, size_t capacity);

/**
 * Inserts one vector and its identifier into the index.
 * Acquires an exclusive write lock internally.
 * @param hnsw Index pointer.
 * @param id User-defined identifier string.
 * @param values Vector values with the configured dimension.
 * @return Status code.
 */
int kc_hnsw_add(kc_hnsw_t *hnsw, const char *id, const float *values);

/**
 * Executes one top-K nearest-neighbor search.
 * Acquires a shared read lock internally. Concurrent calls from multiple
 * threads are safe after kc_hnsw_build() completes. Each caller must
 * supply its own output buffer.
 * @param hnsw Index pointer.
 * @param query Query vector.
 * @param limit Maximum number of results to write.
 * @param threshold Minimum score (cosine/inner) or maximum distance (L2)
 *                  to accept.
 * @param out Caller-provided output buffer.
 * @return Number of results written, or a negative status code on failure.
 */
int kc_hnsw_search(
    const kc_hnsw_t *hnsw,
    const float *query,
    size_t limit,
    double threshold,
    kc_hnsw_result_t *out
);

/**
 * Constructs the HNSW graph from previously added vectors.
 * Acquires an exclusive write lock internally. After a successful build
 * the index is ready for concurrent searches.
 * @param hnsw Index pointer.
 * @return Status code.
 */
int kc_hnsw_build(kc_hnsw_t *hnsw);

/**
 * Returns the configured vector dimension.
 * @param hnsw Index pointer.
 * @return Dimension value, or 0 on invalid input.
 */
size_t kc_hnsw_dimension(const kc_hnsw_t *hnsw);

/**
 * Returns the configured similarity metric.
 * @param hnsw Index pointer.
 * @return Metric constant, or 0 on invalid input.
 */
int kc_hnsw_metric(const kc_hnsw_t *hnsw);

/**
 * Returns the number of inserted vectors.
 * @param hnsw Index pointer.
 * @return Vector count, or 0 on invalid input.
 */
size_t kc_hnsw_count(const kc_hnsw_t *hnsw);

/**
 * Resolves one metric name into a metric constant.
 * @param name Metric text name.
 * @return Metric constant, or 0 on invalid input.
 */
int kc_hnsw_metric_from_string(const char *name);

/**
 * Resolves one metric constant into a metric name.
 * @param metric Metric constant.
 * @return Static metric name, or NULL on invalid input.
 */
const char *kc_hnsw_metric_to_string(int metric);

/**
 * Resolves one status code into text.
 * @param rc Status code.
 * @return Static string.
 */
const char *kc_hnsw_strerror(int rc);

#ifdef __cplusplus
}
#endif

#endif
