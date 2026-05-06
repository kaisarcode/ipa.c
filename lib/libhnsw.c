/**
 * libhnsw.c - HNSW Vector Search
 * Summary: HNSW-based approximate nearest neighbor search implementation.
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "hnsw.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _WIN32
#include <pthread.h>
#else
#include <windows.h>
#endif

#define KC_HNSW_HNSW_M 16
#define KC_HNSW_HNSW_EF_CONSTRUCTION 64
#define KC_HNSW_HNSW_EF_SEARCH 64
#define KC_HNSW_HNSW_MULT 0.42

typedef struct {
    size_t target_idx;
} kc_hnsw_edge_t;

typedef struct {
    kc_hnsw_edge_t *edges;
    size_t count;
    size_t capacity;
} kc_hnsw_neighbor_list_t;

typedef struct {
    char *id;
    float *values;
    double norm;
    int level;
    kc_hnsw_neighbor_list_t *neighbors;
} kc_hnsw_item_t;

struct kc_hnsw {
    size_t dimension;
    int metric;
    kc_hnsw_item_t *items;
    size_t count;
    size_t capacity;

    int max_level;
    size_t entry_point_idx;
    int entry_point_set;

    int M;
    int ef_construction;
    int ef_search;

#ifndef _WIN32
    pthread_rwlock_t rwlock;
#else
    SRWLOCK rwlock;
#endif
    uint64_t rng_state;
};

typedef struct {
    size_t idx;
    double score;
} kc_hnsw_node_score_t;

typedef struct {
    kc_hnsw_node_score_t *data;
    size_t size;
    size_t capacity;
    int metric;
    int worst_first;
} kc_hnsw_heap_t;

static char *kc_hnsw_strdup(const char *text);
static int kc_hnsw_metric_valid(int metric);
static double kc_hnsw_vector_norm(const float *values, size_t dimension);
static double kc_hnsw_inner_product(const float *left, const float *right, size_t dimension);
static double kc_hnsw_dist(const kc_hnsw_t *hnsw, const float *v1, double n1, const float *v2, double n2);

static int kc_hnsw_random_level(kc_hnsw_t *hnsw);
static uint64_t kc_hnsw_rand64(kc_hnsw_t *hnsw);
static double kc_hnsw_rand_double(kc_hnsw_t *hnsw);
static kc_hnsw_heap_t *kc_hnsw_heap_create(size_t capacity, int metric, int worst_first);
static void kc_hnsw_heap_destroy(kc_hnsw_heap_t *heap);
static int kc_hnsw_heap_push(kc_hnsw_heap_t *heap, size_t idx, double score);
static kc_hnsw_node_score_t kc_hnsw_heap_pop(kc_hnsw_heap_t *heap);
static int score_better(int metric, double a, double b);
static int score_worse(int metric, double a, double b);
static int kc_hnsw_heap_has_priority(int metric, double s1, double s2, int worst_first);

static int kc_hnsw_search_level(const kc_hnsw_t *hnsw, const float *query, double query_norm, size_t entry_idx, int level, int ef, kc_hnsw_heap_t *results);
static int kc_hnsw_add_edge(kc_hnsw_t *hnsw, size_t src_idx, size_t dst_idx, int level);
static void kc_hnsw_neighbor_list_init(kc_hnsw_neighbor_list_t *list);
static void kc_hnsw_neighbor_list_free(kc_hnsw_neighbor_list_t *list);

/**
 * Acquire a shared read lock on the index.
 * @param hnsw Index pointer.
 * @return KC_HNSW_OK on success, KC_HNSW_EINVAL on failure.
 */
static int kc_hnsw_rlock(kc_hnsw_t *hnsw) {
#ifndef _WIN32
    return pthread_rwlock_rdlock(&hnsw->rwlock) == 0 ? KC_HNSW_OK : KC_HNSW_EINVAL;
#else
    AcquireSRWLockShared(&hnsw->rwlock);
    return KC_HNSW_OK;
#endif
}

/**
 * Acquire an exclusive write lock on the index.
 * @param hnsw Index pointer.
 * @return KC_HNSW_OK on success, KC_HNSW_EINVAL on failure.
 */
static int kc_hnsw_wlock(kc_hnsw_t *hnsw) {
#ifndef _WIN32
    return pthread_rwlock_wrlock(&hnsw->rwlock) == 0 ? KC_HNSW_OK : KC_HNSW_EINVAL;
#else
    AcquireSRWLockExclusive(&hnsw->rwlock);
    return KC_HNSW_OK;
#endif
}

/**
 * Release a shared read lock on the index.
 * @param hnsw Index pointer.
 * @return No return value.
 */
static void kc_hnsw_runlock(kc_hnsw_t *hnsw) {
#ifndef _WIN32
    pthread_rwlock_unlock(&hnsw->rwlock);
#else
    ReleaseSRWLockShared(&hnsw->rwlock);
#endif
}

/**
 * Release an exclusive write lock on the index.
 * @param hnsw Index pointer.
 * @return No return value.
 */
static void kc_hnsw_wunlock(kc_hnsw_t *hnsw) {
#ifndef _WIN32
    pthread_rwlock_unlock(&hnsw->rwlock);
#else
    ReleaseSRWLockExclusive(&hnsw->rwlock);
#endif
}

/**
 * Creates one vector index instance.
 * @param dimension Fixed vector dimension for all entries.
 * @param metric Configured similarity metric.
 * @return Index pointer or NULL on allocation failure.
 */
kc_hnsw_t *kc_hnsw_open(size_t dimension, int metric) {
    kc_hnsw_t *hnsw;

    if (dimension == 0 || !kc_hnsw_metric_valid(metric)) {
        return NULL;
    }

    hnsw = (kc_hnsw_t *)calloc(1, sizeof(*hnsw));
    if (hnsw == NULL) {
        return NULL;
    }

    hnsw->dimension = dimension;
    hnsw->metric = metric;
    hnsw->max_level = -1;
    hnsw->entry_point_set = 0;
    hnsw->M = KC_HNSW_HNSW_M;
    hnsw->ef_construction = KC_HNSW_HNSW_EF_CONSTRUCTION;
    hnsw->ef_search = KC_HNSW_HNSW_EF_SEARCH;

    const char *env_ef = getenv("HNSW_EF_SEARCH");
    if (env_ef != NULL) {
        int ef = atoi(env_ef);
        if (ef > 0) {
            hnsw->ef_search = ef;
        }
    }

#ifndef _WIN32
    if (pthread_rwlock_init(&hnsw->rwlock, NULL) != 0) {
        free(hnsw);
        return NULL;
    }
#else
    InitializeSRWLock(&hnsw->rwlock);
#endif

    hnsw->rng_state = (uint64_t)time(NULL) ^ (uint64_t)(uintptr_t)hnsw;
    if (hnsw->rng_state == 0) hnsw->rng_state = 1;

    return hnsw;
}

/**
 * Releases one vector index instance.
 * @param hnsw Index pointer.
 * @return No return value.
 */
void kc_hnsw_close(kc_hnsw_t *hnsw) {
    size_t i;
    int j;

    if (hnsw == NULL) {
        return;
    }

    for (i = 0; i < hnsw->count; i++) {
        free(hnsw->items[i].id);
        free(hnsw->items[i].values);
        if (hnsw->items[i].neighbors) {
            for (j = 0; j <= hnsw->items[i].level; j++) {
                kc_hnsw_neighbor_list_free(&hnsw->items[i].neighbors[j]);
            }
            free(hnsw->items[i].neighbors);
        }
    }

    free(hnsw->items);
#ifndef _WIN32
    pthread_rwlock_destroy(&hnsw->rwlock);
#endif
    free(hnsw);
}

/**
 * Reserves capacity without acquiring the lock.
 * @param hnsw Index pointer.
 * @param capacity Target vector capacity.
 * @return Status code.
 */
static int kc_hnsw_reserve_locked(kc_hnsw_t *hnsw, size_t capacity) {
    kc_hnsw_item_t *items;

    if (capacity <= hnsw->capacity) {
        return KC_HNSW_OK;
    }
    items = (kc_hnsw_item_t *)realloc(hnsw->items, capacity * sizeof(*items));
    if (items == NULL) {
        return KC_HNSW_ENOMEM;
    }
    hnsw->items = items;
    hnsw->capacity = capacity;
    return KC_HNSW_OK;
}

/**
 * Reserves capacity for a target number of vectors.
 * Acquires an exclusive write lock internally.
 * @param hnsw Index pointer.
 * @param capacity Target vector capacity.
 * @return Status code.
 */
int kc_hnsw_reserve(kc_hnsw_t *hnsw, size_t capacity) {
    int rc;

    if (hnsw == NULL) {
        return KC_HNSW_EINVAL;
    }
    if (kc_hnsw_wlock(hnsw) != KC_HNSW_OK) {
        return KC_HNSW_EINVAL;
    }
    rc = kc_hnsw_reserve_locked(hnsw, capacity);
    kc_hnsw_wunlock(hnsw);
    return rc;
}

/**
 * Inserts one vector and its identifier into the index.
 * @param hnsw Index pointer.
 * @param id User-defined identifier string.
 * @param values Vector values with the configured dimension.
 * @return Status code.
 */
int kc_hnsw_add(kc_hnsw_t *hnsw, const char *id, const float *values) {
    float *copy;
    char *id_copy;
    size_t next_capacity;

    if (hnsw == NULL || id == NULL || id[0] == '\0' || values == NULL) {
        return KC_HNSW_EINVAL;
    }
    if (kc_hnsw_wlock(hnsw) != KC_HNSW_OK) {
        return KC_HNSW_EINVAL;
    }

    if (hnsw->count == hnsw->capacity) {
        next_capacity = hnsw->capacity == 0 ? 8 : hnsw->capacity * 2;
        if (kc_hnsw_reserve_locked(hnsw, next_capacity) != KC_HNSW_OK) {
            kc_hnsw_wunlock(hnsw);
            return KC_HNSW_ENOMEM;
        }
    }

    id_copy = kc_hnsw_strdup(id);
    if (id_copy == NULL) {
        kc_hnsw_wunlock(hnsw);
        return KC_HNSW_ENOMEM;
    }

    copy = (float *)malloc(hnsw->dimension * sizeof(*copy));
    if (copy == NULL) {
        free(id_copy);
        kc_hnsw_wunlock(hnsw);
        return KC_HNSW_ENOMEM;
    }

    memcpy(copy, values, hnsw->dimension * sizeof(*copy));

    hnsw->items[hnsw->count].id = id_copy;
    hnsw->items[hnsw->count].values = copy;
    hnsw->items[hnsw->count].norm = kc_hnsw_vector_norm(copy, hnsw->dimension);
    hnsw->items[hnsw->count].level = -1;
    hnsw->items[hnsw->count].neighbors = NULL;
    hnsw->count++;
    hnsw->entry_point_set = 0;
    hnsw->max_level = -1;

    kc_hnsw_wunlock(hnsw);
    return KC_HNSW_OK;
}

/**
 * HNSW Build Phase.
 * @param hnsw Index pointer.
 * @return Status code.
 */
int kc_hnsw_build(kc_hnsw_t *hnsw) {
    if (hnsw == NULL) return KC_HNSW_EINVAL;
    if (kc_hnsw_wlock(hnsw) != KC_HNSW_OK) return KC_HNSW_EINVAL;

    if (hnsw->count == 0) {
        kc_hnsw_wunlock(hnsw);
        return KC_HNSW_OK;
    }

    kc_hnsw_item_t *tmp_items = (kc_hnsw_item_t *)malloc(hnsw->count * sizeof(kc_hnsw_item_t));
    if (!tmp_items) {
        kc_hnsw_wunlock(hnsw);
        return KC_HNSW_ENOMEM;
    }

    memcpy(tmp_items, hnsw->items, hnsw->count * sizeof(kc_hnsw_item_t));
    for (size_t i = 0; i < hnsw->count; i++) {
        tmp_items[i].neighbors = NULL;
        tmp_items[i].level = -1;
    }

    kc_hnsw_t tmp_hnsw = *hnsw;
    tmp_hnsw.items = tmp_items;
    tmp_hnsw.max_level = -1;
    tmp_hnsw.entry_point_set = 0;

    for (size_t i = 0; i < tmp_hnsw.count; i++) {
        size_t j = i + (size_t)(kc_hnsw_rand64(&tmp_hnsw) % (tmp_hnsw.count - i));
        kc_hnsw_item_t temp = tmp_items[i];
        tmp_items[i] = tmp_items[j];
        tmp_items[j] = temp;
    }

    int rc = KC_HNSW_OK;

    for (size_t i = 0; i < tmp_hnsw.count; i++) {
        int level = kc_hnsw_random_level(&tmp_hnsw);
        tmp_items[i].level = level;
        tmp_items[i].neighbors = (kc_hnsw_neighbor_list_t *)calloc(level + 1, sizeof(kc_hnsw_neighbor_list_t));
        if (!tmp_items[i].neighbors) {
            rc = KC_HNSW_ENOMEM;
            goto fail;
        }
        for (int j = 0; j <= level; j++) {
            kc_hnsw_neighbor_list_init(&tmp_items[i].neighbors[j]);
        }

        if (!tmp_hnsw.entry_point_set) {
            tmp_hnsw.entry_point_idx = i;
            tmp_hnsw.max_level = level;
            tmp_hnsw.entry_point_set = 1;
            continue;
        }

        size_t curr_idx = tmp_hnsw.entry_point_idx;
        double curr_dist = kc_hnsw_dist(&tmp_hnsw, tmp_items[i].values, tmp_items[i].norm,
                tmp_items[curr_idx].values, tmp_items[curr_idx].norm);

        for (int l = tmp_hnsw.max_level; l > level; l--) {
            int changed = 1;
            while (changed) {
                changed = 0;
                kc_hnsw_neighbor_list_t *neighbors = &tmp_items[curr_idx].neighbors[l];
                for (size_t n = 0; n < neighbors->count; n++) {
                    size_t neighbor_idx = neighbors->edges[n].target_idx;
                    double d = kc_hnsw_dist(&tmp_hnsw, tmp_items[i].values, tmp_items[i].norm,
                            tmp_items[neighbor_idx].values, tmp_items[neighbor_idx].norm);
                    if (score_better(tmp_hnsw.metric, d, curr_dist)) {
                        curr_dist = d;
                        curr_idx = neighbor_idx;
                        changed = 1;
                    }
                }
            }
        }

        for (int l = (level < tmp_hnsw.max_level ? level : tmp_hnsw.max_level); l >= 0; l--) {
            kc_hnsw_heap_t *candidates = kc_hnsw_heap_create(tmp_hnsw.ef_construction, tmp_hnsw.metric, 1);
            if (!candidates) {
                rc = KC_HNSW_ENOMEM;
                goto fail;
            }
            if (kc_hnsw_search_level(&tmp_hnsw, tmp_items[i].values, tmp_items[i].norm, curr_idx, l,
                    tmp_hnsw.ef_construction, candidates) != KC_HNSW_OK) {
                kc_hnsw_heap_destroy(candidates);
                rc = KC_HNSW_ENOMEM;
                goto fail;
            }
            
            size_t c_size = candidates->size;
            kc_hnsw_node_score_t *sorted = (kc_hnsw_node_score_t *)malloc(c_size * sizeof(kc_hnsw_node_score_t));
            if (!sorted && c_size > 0) {
                kc_hnsw_heap_destroy(candidates);
                rc = KC_HNSW_ENOMEM;
                goto fail;
            }
            for (size_t k = 0; k < c_size; k++) {
                sorted[k] = candidates->data[k];
            }
            
            for (size_t x = 0; x < c_size; x++) {
                for (size_t y = x + 1; y < c_size; y++) {
                    if (score_worse(tmp_hnsw.metric, sorted[x].score, sorted[y].score)) {
                        kc_hnsw_node_score_t tmp_node = sorted[x];
                        sorted[x] = sorted[y];
                        sorted[y] = tmp_node;
                    }
                }
            }

            int connected = 0;
            for (size_t k = 0; k < c_size && connected < tmp_hnsw.M; k++) {
                kc_hnsw_node_score_t best = sorted[k];
                if (kc_hnsw_add_edge(&tmp_hnsw, i, best.idx, l) != KC_HNSW_OK ||
                    kc_hnsw_add_edge(&tmp_hnsw, best.idx, i, l) != KC_HNSW_OK) {
                    free(sorted);
                    kc_hnsw_heap_destroy(candidates);
                    rc = KC_HNSW_ENOMEM;
                    goto fail;
                }
                
                if (connected == 0) {
                    curr_idx = best.idx;
                    curr_dist = best.score;
                }
                connected++;
            }
            free(sorted);
            kc_hnsw_heap_destroy(candidates);
        }

        if (level > tmp_hnsw.max_level) {
            tmp_hnsw.max_level = level;
            tmp_hnsw.entry_point_idx = i;
        }
    }

    for (size_t i = 0; i < hnsw->count; i++) {
        if (hnsw->items[i].neighbors) {
            for (int j = 0; j <= hnsw->items[i].level; j++) {
                kc_hnsw_neighbor_list_free(&hnsw->items[i].neighbors[j]);
            }
            free(hnsw->items[i].neighbors);
        }
    }
    free(hnsw->items);

    hnsw->items = tmp_items;
    hnsw->max_level = tmp_hnsw.max_level;
    hnsw->entry_point_idx = tmp_hnsw.entry_point_idx;
    hnsw->entry_point_set = tmp_hnsw.entry_point_set;
    hnsw->rng_state = tmp_hnsw.rng_state;

    kc_hnsw_wunlock(hnsw);
    return KC_HNSW_OK;

fail:
    for (size_t i = 0; i < tmp_hnsw.count; i++) {
        if (tmp_items[i].neighbors) {
            for (int j = 0; j <= tmp_items[i].level; j++) {
                kc_hnsw_neighbor_list_free(&tmp_items[i].neighbors[j]);
            }
            free(tmp_items[i].neighbors);
        }
    }
    free(tmp_items);
    kc_hnsw_wunlock(hnsw);
    return rc;
}

/**
 * HNSW Search logic.
 * @param hnsw Index pointer.
 * @param query Query vector.
 * @param limit Maximum number of results to write.
 * @param threshold Minimum score or maximum distance to accept.
 * @param out Caller-provided output buffer.
 * @return Number of results written, or a negative status code on failure.
 */
int kc_hnsw_search(const kc_hnsw_t *hnsw, const float *query, size_t limit, double threshold, kc_hnsw_result_t *out) {
    kc_hnsw_t *mhnsw = (kc_hnsw_t *)(uintptr_t)hnsw;
    if (hnsw == NULL || query == NULL || (limit > 0 && out == NULL)) return KC_HNSW_EINVAL;
    if (kc_hnsw_rlock(mhnsw) != KC_HNSW_OK) return KC_HNSW_EINVAL;
    if (limit == 0 || hnsw->count == 0) {
        kc_hnsw_runlock(mhnsw);
        return 0;
    }
    if (hnsw->count > 0 && !hnsw->entry_point_set) {
        kc_hnsw_runlock(mhnsw);
        return KC_HNSW_ESTATE;
    }

    int ef = hnsw->ef_search;
    if ((size_t)ef < limit) ef = (int)limit;
    if (ef < 64) ef = 64;

    int use_brute_force = (hnsw->count <= (size_t)ef || hnsw->count <= 1024);
    double q_norm = kc_hnsw_vector_norm(query, hnsw->dimension);
    size_t candidates_count = 0;
    kc_hnsw_node_score_t *results = NULL;

    if (use_brute_force) {
        results = (kc_hnsw_node_score_t *)malloc(hnsw->count * sizeof(kc_hnsw_node_score_t));
        if (!results) {
            kc_hnsw_runlock(mhnsw);
            return KC_HNSW_ENOMEM;
        }
        for (size_t i = 0; i < hnsw->count; i++) {
            if (hnsw->items[i].id == NULL) continue;
            double d = kc_hnsw_dist(hnsw, query, q_norm, hnsw->items[i].values, hnsw->items[i].norm);
            results[candidates_count].idx = i;
            results[candidates_count].score = d;
            candidates_count++;
        }
    } else {
        size_t curr_idx = hnsw->entry_point_idx;
        double curr_dist = kc_hnsw_dist(hnsw, query, q_norm, hnsw->items[curr_idx].values, hnsw->items[curr_idx].norm);
        
        for (int l = hnsw->max_level; l > 0; l--) {
            int changed = 1;
            while (changed) {
                changed = 0;
                kc_hnsw_neighbor_list_t *neighbors = &hnsw->items[curr_idx].neighbors[l];
                for (size_t n = 0; n < neighbors->count; n++) {
                    size_t neighbor_idx = neighbors->edges[n].target_idx;
                    double d = kc_hnsw_dist(hnsw, query, q_norm, hnsw->items[neighbor_idx].values, hnsw->items[neighbor_idx].norm);
                    if (score_better(hnsw->metric, d, curr_dist)) {
                        curr_dist = d;
                        curr_idx = neighbor_idx;
                        changed = 1;
                    }
                }
            }
        }
        
        kc_hnsw_heap_t *top_k = kc_hnsw_heap_create(ef, hnsw->metric, 1);
        if (!top_k) {
            kc_hnsw_runlock(mhnsw);
            return KC_HNSW_ENOMEM;
        }
        if (kc_hnsw_search_level(hnsw, query, q_norm, curr_idx, 0, ef, top_k) != KC_HNSW_OK) {
            kc_hnsw_heap_destroy(top_k);
            kc_hnsw_runlock(mhnsw);
            return KC_HNSW_ENOMEM;
        }
        candidates_count = top_k->size;
        results = (kc_hnsw_node_score_t *)malloc(candidates_count * sizeof(kc_hnsw_node_score_t));
        if (!results && candidates_count > 0) {
            kc_hnsw_heap_destroy(top_k);
            kc_hnsw_runlock(mhnsw);
            return KC_HNSW_ENOMEM;
        }
        for (size_t i = 0; i < candidates_count; i++) {
            results[i] = kc_hnsw_heap_pop(top_k);
            if (hnsw->items[results[i].idx].id != NULL) {
                results[i].score = kc_hnsw_dist(hnsw, query, q_norm, hnsw->items[results[i].idx].values, hnsw->items[results[i].idx].norm);
            }
        }
        kc_hnsw_heap_destroy(top_k);
    }

    for (size_t x = 0; x < candidates_count; x++) {
        for (size_t y = x + 1; y < candidates_count; y++) {
            if (score_worse(hnsw->metric, results[x].score, results[y].score)) {
                kc_hnsw_node_score_t tmp = results[x];
                results[x] = results[y];
                results[y] = tmp;
            }
        }
    }

    size_t written = 0;
    for (size_t i = 0; i < candidates_count && written < limit; i++) {
        if (hnsw->items[results[i].idx].id == NULL) continue;
        int match = 0;
        if (hnsw->metric == KC_HNSW_METRIC_L2) {
            if (results[i].score <= threshold) match = 1;
        } else {
            if (results[i].score >= threshold) match = 1;
        }
        if (match) {
            out[written].id = hnsw->items[results[i].idx].id;
            out[written].score = results[i].score;
            written++;
        }
    }

    free(results);
    kc_hnsw_runlock(mhnsw);
    return (int)written;
}

/**
 * Performs a search at a specific HNSW level.
 * @param hnsw Index pointer.
 * @param query Query vector.
 * @param query_norm Precomputed query norm.
 * @param entry_idx Starting node index.
 * @param level Graph level to search.
 * @param ef Search budget (ef).
 * @param results Output heap to store found nodes.
 * @return Status code.
 */
static int kc_hnsw_search_level(const kc_hnsw_t *hnsw, const float *query, double query_norm, size_t entry_idx, int level, int ef, kc_hnsw_heap_t *results) {
    kc_hnsw_heap_t *candidates = kc_hnsw_heap_create(ef * 2, hnsw->metric, 0);
    if (!candidates) return KC_HNSW_ENOMEM;
    
    char *visited = (char *)calloc(hnsw->count, 1);
    if (!visited) {
        kc_hnsw_heap_destroy(candidates);
        return KC_HNSW_ENOMEM;
    }

    double d = kc_hnsw_dist(hnsw, query, query_norm, hnsw->items[entry_idx].values, hnsw->items[entry_idx].norm);
    if (kc_hnsw_heap_push(candidates, entry_idx, d) != KC_HNSW_OK ||
        kc_hnsw_heap_push(results, entry_idx, d) != KC_HNSW_OK) {
        free(visited);
        kc_hnsw_heap_destroy(candidates);
        return KC_HNSW_ENOMEM;
    }
    visited[entry_idx] = 1;

    while (candidates->size > 0) {
        kc_hnsw_node_score_t c = kc_hnsw_heap_pop(candidates);
        
        kc_hnsw_node_score_t worst_res = results->data[0];
        if (results->size >= (size_t)ef &&
            score_worse(hnsw->metric, c.score, worst_res.score)) {
            break;
        }

        kc_hnsw_neighbor_list_t *neighbors = &hnsw->items[c.idx].neighbors[level];
        for (size_t n = 0; n < neighbors->count; n++) {
            size_t v_idx = neighbors->edges[n].target_idx;
            if (!visited[v_idx]) {
                visited[v_idx] = 1;
                double v_dist = kc_hnsw_dist(hnsw, query, query_norm, hnsw->items[v_idx].values, hnsw->items[v_idx].norm);
                
                worst_res = results->data[0];
                if (results->size < (size_t)ef || score_better(hnsw->metric, v_dist, worst_res.score)) {
                    if (kc_hnsw_heap_push(candidates, v_idx, v_dist) != KC_HNSW_OK ||
                        kc_hnsw_heap_push(results, v_idx, v_dist) != KC_HNSW_OK) {
                        free(visited);
                        kc_hnsw_heap_destroy(candidates);
                        return KC_HNSW_ENOMEM;
                    }
                    if (results->size > (size_t)ef) {
                        kc_hnsw_heap_pop(results);
                    }
                }
            }
        }
    }

    free(visited);
    kc_hnsw_heap_destroy(candidates);
    return KC_HNSW_OK;
}

/**
 * Adds a directed edge between two nodes in the graph.
 * @param hnsw Index pointer.
 * @param src_idx Source node index.
 * @param dst_idx Destination node index.
 * @param level Graph level.
 * @return Status code.
 */
static int kc_hnsw_add_edge(kc_hnsw_t *hnsw, size_t src_idx, size_t dst_idx, int level) {
    kc_hnsw_neighbor_list_t *list = &hnsw->items[src_idx].neighbors[level];
    
    for (size_t i = 0; i < list->count; i++) {
        if (list->edges[i].target_idx == dst_idx) return KC_HNSW_OK;
    }

    if (list->count == list->capacity) {
        size_t next_cap = list->capacity == 0 ? 4 : list->capacity * 2;
        kc_hnsw_edge_t *tmp = (kc_hnsw_edge_t *)realloc(list->edges, next_cap * sizeof(kc_hnsw_edge_t));
        if (!tmp) return KC_HNSW_ENOMEM;
        list->edges = tmp;
        list->capacity = next_cap;
    }
    list->edges[list->count++].target_idx = dst_idx;

    size_t M_max = (level == 0) ? (size_t)hnsw->M * 2 : (size_t)hnsw->M;
    if (list->count > M_max) {
        int worst_idx = -1;
        double worst_score = 0;

        for (size_t i = 0; i < list->count; i++) {
            size_t n_idx = list->edges[i].target_idx;
            double s = kc_hnsw_dist(hnsw, hnsw->items[src_idx].values, hnsw->items[src_idx].norm,
                    hnsw->items[n_idx].values, hnsw->items[n_idx].norm);
            if (worst_idx == -1 || score_worse(hnsw->metric, s, worst_score)) {
                worst_score = s;
                worst_idx = (int)i;
            }
        }

        if (worst_idx != -1) {
            list->edges[worst_idx] = list->edges[list->count - 1];
            list->count--;
        }
    }
    return KC_HNSW_OK;
}

/**
 * Xorshift64* PRNG.
 * @param hnsw Index pointer.
 * @return Random 64-bit value.
 */
static uint64_t kc_hnsw_rand64(kc_hnsw_t *hnsw) {
    uint64_t x = hnsw->rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    hnsw->rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/**
 * Returns a random double in [0, 1).
 * @param hnsw Index pointer.
 * @return Random double.
 */
static double kc_hnsw_rand_double(kc_hnsw_t *hnsw) {
    return (double)(kc_hnsw_rand64(hnsw) & 0x1FFFFFFFFFFFFFULL) / (double)0x20000000000000ULL;
}

/**
 * Generates a random level for a new node.
 * @param hnsw Index pointer.
 * @return Randomized level index.
 */
static int kc_hnsw_random_level(kc_hnsw_t *hnsw) {
    double r = kc_hnsw_rand_double(hnsw);
    int level = 0;
    while (r < 0.1 && level < 16) {
        level++;
        r = kc_hnsw_rand_double(hnsw);
    }
    return level;
}

/**
 * Initializes a neighbor list structure.
 * @param list List pointer.
 * @return No return value.
 */
static void kc_hnsw_neighbor_list_init(kc_hnsw_neighbor_list_t *list) {
    list->edges = NULL;
    list->count = 0;
    list->capacity = 0;
}

/**
 * Releases a neighbor list structure.
 * @param list List pointer.
 * @return No return value.
 */
static void kc_hnsw_neighbor_list_free(kc_hnsw_neighbor_list_t *list) {
    free(list->edges);
}

/**
 * Creates a new priority heap.
 * @param capacity Initial capacity.
 * @param metric Metric constant.
 * @param worst_first Flag to keep worst element at top (1) or best at top (0).
 * @return Heap pointer.
 */
static kc_hnsw_heap_t *kc_hnsw_heap_create(size_t capacity, int metric, int worst_first) {
    kc_hnsw_heap_t *heap = (kc_hnsw_heap_t *)malloc(sizeof(kc_hnsw_heap_t));
    if (!heap) return NULL;
    heap->data = (kc_hnsw_node_score_t *)malloc(capacity * sizeof(kc_hnsw_node_score_t));
    if (!heap->data) {
        free(heap);
        return NULL;
    }
    heap->size = 0;
    heap->capacity = capacity;
    heap->metric = metric;
    heap->worst_first = worst_first;
    return heap;
}

/**
 * Releases a priority heap.
 * @param heap Heap pointer.
 * @return No return value.
 */
static void kc_hnsw_heap_destroy(kc_hnsw_heap_t *heap) {
    free(heap->data);
    free(heap);
}

/**
 * Checks if one score is better than another.
 * @param metric Metric constant.
 * @param a First score.
 * @param b Second score.
 * @return 1 if a is better than b, 0 otherwise.
 */
static int score_better(int metric, double a, double b) {
    if (metric == KC_HNSW_METRIC_L2) return a < b;
    return a > b;
}

/**
 * Checks if one score is worse than another.
 * @param metric Metric constant.
 * @param a First score.
 * @param b Second score.
 * @return 1 if a is worse than b, 0 otherwise.
 */
static int score_worse(int metric, double a, double b) {
    if (metric == KC_HNSW_METRIC_L2) return a > b;
    return a < b;
}

/**
 * Checks if a score has priority to move up the heap.
 * @param metric Metric constant.
 * @param s1 First score.
 * @param s2 Second score.
 * @param worst_first Flag to keep worst element at top.
 * @return 1 if s1 has priority over s2, 0 otherwise.
 */
static int kc_hnsw_heap_has_priority(int metric, double s1, double s2, int worst_first) {
    if (worst_first) return score_worse(metric, s1, s2);
    return score_better(metric, s1, s2);
}

/**
 * Pushes a new node into the heap.
 * @param heap Heap pointer.
 * @param idx Node index.
 * @param score Similarity score or distance.
 * @return Status code.
 */
static int kc_hnsw_heap_push(kc_hnsw_heap_t *heap, size_t idx, double score) {
    if (heap->size == heap->capacity) {
        size_t next_cap = heap->capacity * 2;
        kc_hnsw_node_score_t *tmp = (kc_hnsw_node_score_t *)realloc(heap->data, next_cap * sizeof(kc_hnsw_node_score_t));
        if (!tmp) return KC_HNSW_ENOMEM;
        heap->data = tmp;
        heap->capacity = next_cap;
    }
    size_t i = heap->size++;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!kc_hnsw_heap_has_priority(heap->metric, score, heap->data[p].score, heap->worst_first)) break;
        heap->data[i] = heap->data[p];
        i = p;
    }
    heap->data[i].idx = idx;
    heap->data[i].score = score;
    return KC_HNSW_OK;
}

/**
 * Pops and returns the heap root.
 * The root is best-first when worst_first is 0.
 * The root is worst-first when worst_first is 1.
 * @param heap Heap pointer.
 * @return Root node score structure.
 */
static kc_hnsw_node_score_t kc_hnsw_heap_pop(kc_hnsw_heap_t *heap) {
    kc_hnsw_node_score_t top = heap->data[0];
    kc_hnsw_node_score_t last = heap->data[--heap->size];
    size_t i = 0;
    while (i * 2 + 1 < heap->size) {
        size_t c = i * 2 + 1;
        if (c + 1 < heap->size && kc_hnsw_heap_has_priority(heap->metric, heap->data[c+1].score, heap->data[c].score, heap->worst_first)) c++;
        if (!kc_hnsw_heap_has_priority(heap->metric, heap->data[c].score, last.score, heap->worst_first)) break;
        heap->data[i] = heap->data[c];
        i = c;
    }
    heap->data[i] = last;
    return top;
}

/**
 * Duplicates one string into heap memory.
 * @param text Source string.
 * @return Heap copy or NULL on allocation failure.
 */
static char *kc_hnsw_strdup(const char *text) {
    size_t size = strlen(text) + 1;
    char *copy = (char *)malloc(size);
    if (copy) memcpy(copy, text, size);
    return copy;
}

/**
 * Checks whether one metric constant is supported.
 * @param metric Metric constant.
 * @return 1 when valid, or 0 when invalid.
 */
static int kc_hnsw_metric_valid(int metric) {
    return (metric == KC_HNSW_METRIC_COSINE || metric == KC_HNSW_METRIC_INNER_PRODUCT || metric == KC_HNSW_METRIC_L2);
}

/**
 * Computes one vector Euclidean norm.
 * @param values Vector values.
 * @param dimension Vector dimension.
 * @return Non-negative norm.
 */
static double kc_hnsw_vector_norm(const float *values, size_t dimension) {
    double sum = 0.0;
    for (size_t i = 0; i < dimension; i++) sum += (double)values[i] * (double)values[i];
    return sqrt(sum);
}

/**
 * Computes one vector inner product.
 * @param left Left vector.
 * @param right Right vector.
 * @param dimension Vector dimension.
 * @return Dot-product value.
 */
static double kc_hnsw_inner_product(const float *left, const float *right, size_t dimension) {
    double sum = 0.0;
    for (size_t i = 0; i < dimension; i++) sum += (double)left[i] * (double)right[i];
    return sum;
}

/**
 * Computes the distance/similarity between two vectors.
 * @param hnsw Index pointer.
 * @param v1 First vector.
 * @param n1 Precomputed norm of v1.
 * @param v2 Second vector.
 * @param n2 Precomputed norm of v2.
 * @return Distance or similarity score.
 */
static double kc_hnsw_dist(const kc_hnsw_t *hnsw, const float *v1, double n1, const float *v2, double n2) {
    double ip = kc_hnsw_inner_product(v1, v2, hnsw->dimension);
    if (hnsw->metric == KC_HNSW_METRIC_INNER_PRODUCT) return ip;
    if (hnsw->metric == KC_HNSW_METRIC_COSINE) return (n1 == 0.0 || n2 == 0.0) ? 0.0 : ip / (n1 * n2);
    return n1 * n1 + n2 * n2 - 2.0 * ip;
}

/**
 * Returns the configured vector dimension.
 * @param hnsw Index pointer.
 * @return Dimension value, or 0 on invalid input.
 */
size_t kc_hnsw_dimension(const kc_hnsw_t *hnsw) { return hnsw ? hnsw->dimension : 0; }

/**
 * Returns the configured similarity metric.
 * @param hnsw Index pointer.
 * @return Metric constant, or 0 on invalid input.
 */
int kc_hnsw_metric(const kc_hnsw_t *hnsw) { return hnsw ? hnsw->metric : 0; }

/**
 * Returns the number of inserted vectors.
 * @param hnsw Index pointer.
 * @return Vector count, or 0 on invalid input.
 */
size_t kc_hnsw_count(const kc_hnsw_t *hnsw) {
    if (!hnsw) {
        return 0;
    }

    kc_hnsw_t *mhnsw = (kc_hnsw_t *)(uintptr_t)hnsw;
    if (kc_hnsw_rlock(mhnsw) != KC_HNSW_OK) {
        return 0;
    }

    size_t count = hnsw->count;
    kc_hnsw_runlock(mhnsw);

    return count;
}

/**
 * Resolves one metric name into a metric constant.
 * @param name Metric text name.
 * @return Metric constant, or 0 on invalid input.
 */
int kc_hnsw_metric_from_string(const char *name) {
    if (!name) return 0;
    if (strcmp(name, "cosine") == 0) return KC_HNSW_METRIC_COSINE;
    if (strcmp(name, "inner") == 0 || strcmp(name, "inner_product") == 0) return KC_HNSW_METRIC_INNER_PRODUCT;
    if (strcmp(name, "l2") == 0 || strcmp(name, "euclidean") == 0) return KC_HNSW_METRIC_L2;
    return 0;
}

/**
 * Resolves one metric constant into a metric name.
 * @param metric Metric constant.
 * @return Static metric name, or NULL on invalid input.
 */
const char *kc_hnsw_metric_to_string(int metric) {
    switch (metric) {
        case KC_HNSW_METRIC_COSINE: return "cosine";
        case KC_HNSW_METRIC_INNER_PRODUCT: return "inner";
        case KC_HNSW_METRIC_L2: return "l2";
        default: return NULL;
    }
}

/**
 * Resolves one status code into text.
 * @param rc Status code.
 * @return Static string.
 */
const char *kc_hnsw_strerror(int rc) {
    switch (rc) {
        case KC_HNSW_OK: return "ok";
        case KC_HNSW_EINVAL: return "invalid argument";
        case KC_HNSW_ENOMEM: return "out of memory";
        case KC_HNSW_ESTATE: return "invalid state";
        default: return "unknown error";
    }
}
