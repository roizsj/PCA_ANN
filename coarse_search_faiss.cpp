#include "coarse_search_module.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <omp.h>

namespace {

constexpr int kHnswM = 32;
constexpr int kEfConstruction = 80;
constexpr int kEfSearch = 128;

struct omp_thread_guard {
    int prev_threads;

    omp_thread_guard()
        : prev_threads(omp_get_max_threads()) {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }

    ~omp_thread_guard() {
        omp_set_num_threads(prev_threads);
    }
};

} // namespace

struct coarse_search_module {
    std::unique_ptr<faiss::IndexHNSWFlat> index;
    uint32_t dim;
    uint32_t nlist;
};

extern "C" int coarse_search_module_init(coarse_search_module_t **out_module,
                                          const float *centroids,
                                          uint32_t dim,
                                          uint32_t nlist)
{
    if (!out_module || !centroids || dim == 0 || nlist == 0) {
        return -1;
    }

    *out_module = nullptr;

    try {
        std::unique_ptr<coarse_search_module_t> module(new coarse_search_module_t());
        module->dim = dim;
        module->nlist = nlist;
        module->index = std::make_unique<faiss::IndexHNSWFlat>(
            static_cast<int>(dim),
            kHnswM,
            faiss::METRIC_L2);
        module->index->hnsw.efConstruction = kEfConstruction;
        module->index->hnsw.efSearch = kEfSearch;
        {
            // HNSW build uses OpenMP internally; forcing single-threaded build
            // avoids the SPDK/DPDK process interaction that was corrupting the heap.
            omp_thread_guard omp_guard;
            module->index->add(static_cast<faiss::idx_t>(nlist), centroids);
        }
        *out_module = module.release();
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void coarse_search_module_destroy(coarse_search_module_t *module)
{
    delete module;
}

extern "C" int coarse_search_module_search(const coarse_search_module_t *module,
                                            const float *query,
                                            uint32_t k,
                                            uint32_t *labels_out,
                                            float *distances_out)
{
    if (!module || !module->index || !query || !labels_out || !distances_out ||
        k == 0 || k > module->nlist) {
        return -1;
    }

    try {
        std::vector<faiss::idx_t> labels(k, -1);
        module->index->search(1, query, static_cast<faiss::idx_t>(k), distances_out, labels.data());

        for (uint32_t i = 0; i < k; i++) {
            if (labels[i] < 0) {
                labels_out[i] = UINT32_MAX;
                distances_out[i] = std::numeric_limits<float>::infinity();
            } else {
                labels_out[i] = static_cast<uint32_t>(labels[i]);
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
