#ifndef COARSE_SEARCH_MODULE_H
#define COARSE_SEARCH_MODULE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct coarse_search_module coarse_search_module_t;

int coarse_search_module_init(coarse_search_module_t **out_module,
                              const float *centroids,
                              uint32_t dim,
                              uint32_t nlist);

void coarse_search_module_destroy(coarse_search_module_t *module);

int coarse_search_module_search(const coarse_search_module_t *module,
                                const float *query,
                                uint32_t k,
                                uint32_t *labels_out,
                                float *distances_out);

#ifdef __cplusplus
}
#endif

#endif
