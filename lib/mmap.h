/**
 * mmap.h - Readonly memory mapping.
 * Summary: Public API for the mmap library.
 *
 * Author:  KaisarCode
 * Website: https://kaisarcode.com
 * License: https://www.gnu.org/licenses/gpl-3.0.html
 */

#ifndef KC_MMAP_H
#define KC_MMAP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kc_mmap kc_mmap_t;

struct kc_mmap {
    void *data;
    size_t size;
    int fd;
    void *priv1;
    void *priv2;
};

#define KC_MMAP_OK      0
#define KC_MMAP_ERROR  -1

/**
 * Initialize a new mmap context and map a file.
 * @param map Map context pointer.
 * @param path File path.
 * @return Status code.
 */
int kc_mmap_open(kc_mmap_t *map, const char *path);

/**
 * Get pointer to mapped data.
 * @param map Map context pointer.
 * @return Pointer to data or NULL.
 */
const void *kc_mmap_data(const kc_mmap_t *map);

/**
 * Get mapped data size.
 * @param map Map context pointer.
 * @return Size in bytes.
 */
size_t kc_mmap_size(const kc_mmap_t *map);

/**
 * Release a mmap context.
 * @param map Map context pointer.
 * @return None.
 */
void kc_mmap_close(kc_mmap_t *map);

#ifdef __cplusplus
}
#endif

#endif
