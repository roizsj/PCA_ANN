#define _GNU_SOURCE
#include <spdk/env.h>
#include <spdk/nvme.h>

#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "query_loader.h"

#define NUM_DISKS 4
#define TOPK 10
#define META_MAGIC 0x4751504du

static const char *DEFAULT_META_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/ivfpq_disk_meta.bin";
static const char *DEFAULT_SORTED_IDS_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/sorted_vec_ids_ivfpq.bin";
static const char *DEFAULT_QUERY_FVECS_PATH = "/home/zhangshujie/ann_nic/gist/gist_query.fvecs";
static const char *DEFAULT_GT_IVECS_PATH = "/home/zhangshujie/ann_nic/gist/gist_groundtruth.ivecs";

typedef struct {
    uint32_t cluster_id;
    uint32_t disk_id;
    uint64_t pq_start_lba;
    uint32_t pq_num_lbas;
    uint64_t raw_start_lba;
    uint32_t raw_num_lbas;
    uint32_t num_vectors;
    uint32_t sorted_id_base;
} ClusterMeta;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t nlist;
    uint32_t num_vectors;
    uint32_t num_disks;
    uint32_t sector_size;
    uint32_t pq_m;
    uint32_t pq_ksub;
    uint32_t pq_nbits;
    uint32_t pq_subdim;
    uint32_t pq_code_bytes;
    uint32_t pq_codes_per_lba;
    uint32_t raw_vectors_per_lba;
    uint64_t base_lba;
    uint64_t raw_region_lba[NUM_DISKS];
} MetaHeader;

typedef struct {
    MetaHeader hdr;
    ClusterMeta *clusters;
    float *centroids;
    float *pq_codebooks;
} IvfPqMeta;

typedef struct {
    uint32_t vec_id;
    uint32_t cluster_id;
    uint32_t local_idx;
    uint32_t disk_id;
    uint32_t lane;
    uint64_t raw_lba;
    float dist;
} Candidate;

typedef struct {
    Candidate *items;
    uint32_t size;
    uint32_t cap;
} CandidateTop;

typedef struct {
    uint32_t vec_id;
    float dist;
} TopkItem;

typedef struct {
    TopkItem items[TOPK];
    uint32_t size;
} TopkState;

typedef struct {
    uint32_t cluster_id;
    float dist;
} CoarseHit;

typedef struct {
    uint32_t n_queries;
    uint32_t k;
    int32_t *ids;
} GtSet;

typedef struct {
    volatile bool done;
    int ok;
} IoWaiter;

typedef struct {
    uint32_t disk_id;
    uint64_t lba;
    uint32_t begin;
    uint32_t end;
} RawGroup;

typedef struct {
    IoWaiter waiter;
    uint8_t *buf;
    uint32_t group_idx;
    bool active;
} RawReadCtx;

typedef struct Worker Worker;

typedef struct {
    Worker *owner;
    pthread_t tid;
    bool created;
    bool ready;
    int init_rc;
    uint32_t shard_idx;
    uint32_t h_begin;
    uint32_t h_end;
    struct spdk_nvme_qpair *qpair[NUM_DISKS];
    uint8_t *pq_buf;
    CandidateTop approx_top;
    uint64_t io_us;
    uint64_t scan_us;
    uint64_t pq_candidates;
    uint32_t generation_seen;
} ScanShard;

typedef struct {
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    const char *traddr;
    uint32_t sector_size;
} DiskCtx;

typedef struct {
    const char *meta_path;
    const char *sorted_ids_path;
    const char *query_fvecs_path;
    const char *gt_ivecs_path;
    const char *disk_traddr[NUM_DISKS];
    const char *cores_spec;
    uint32_t nprobe;
    uint32_t rerank_k;
    uint32_t io_depth;
    uint32_t pq_read_lbas;
    uint32_t cluster_threads;
    uint32_t max_queries;
    uint32_t threads;
    bool threads_explicit;
    int base_core;
} Config;

typedef struct {
    int rc;
    uint64_t latency_us;
    uint64_t coarse_us;
    uint64_t table_us;
    uint64_t pq_io_us;
    uint64_t pq_scan_us;
    uint64_t rerank_io_us;
    uint64_t rerank_compute_us;
    uint64_t pq_candidates;
    uint64_t rerank_candidates;
    uint64_t done_ts_us;
    double recall10;
} QueryResult;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint32_t ready_workers;
    uint32_t next_query_idx;
    int fatal_rc;
    bool start;
} QueryDispatch;

struct Worker {
    pthread_t tid;
    int init_rc;
    int core_id;
    struct spdk_nvme_ctrlr *ctrlr[NUM_DISKS];
    struct spdk_nvme_ns *ns[NUM_DISKS];
    struct spdk_nvme_qpair *qpair[NUM_DISKS];
    uint8_t *pq_buf;
    uint8_t *raw_buf;
    RawReadCtx *raw_reads;
    float *dist_table;
    CoarseHit *hits;
    CandidateTop approx_top;
    const IvfPqMeta *meta;
    const uint32_t *sorted_ids;
    const float *queries;
    uint32_t query_dim;
    const GtSet *gt;
    QueryResult *results;
    QueryDispatch *dispatch;
    uint32_t nprobe;
    uint32_t rerank_k;
    uint32_t io_depth;
    uint32_t pq_read_lbas;
    uint32_t cluster_threads;
    uint32_t max_queries;

    pthread_mutex_t scan_mu;
    pthread_cond_t scan_cv;
    bool scan_sync_init;
    bool scan_shutdown;
    uint32_t scan_generation;
    uint32_t scan_ready_helpers;
    uint32_t scan_completed_helpers;
    int scan_rc;
    const float *scan_query;
    ScanShard *scan_shards;
};

static inline uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static void bind_to_core(int core_id)
{
    if (core_id < 0) {
        return;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (rc != 0) {
        errno = rc;
        perror("pthread_setaffinity_np");
        exit(EXIT_FAILURE);
    }
}

static float l2(const float *a, const float *b, uint32_t dim)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

static void topk_insert(TopkState *st, uint32_t vec_id, float dist)
{
    if (st->size < TOPK) {
        st->items[st->size++] = (TopkItem){.vec_id = vec_id, .dist = dist};
        return;
    }
    uint32_t worst = 0;
    for (uint32_t i = 1; i < st->size; i++) {
        if (st->items[i].dist > st->items[worst].dist) {
            worst = i;
        }
    }
    if (dist < st->items[worst].dist) {
        st->items[worst] = (TopkItem){.vec_id = vec_id, .dist = dist};
    }
}

static int cmp_topk(const void *a, const void *b)
{
    const TopkItem *x = (const TopkItem *)a;
    const TopkItem *y = (const TopkItem *)b;
    if (x->dist < y->dist) return -1;
    if (x->dist > y->dist) return 1;
    return (x->vec_id > y->vec_id) - (x->vec_id < y->vec_id);
}

static void topk_finalize(TopkState *st)
{
    if (st->size > 1) {
        qsort(st->items, st->size, sizeof(st->items[0]), cmp_topk);
    }
}

static void cand_top_reset(CandidateTop *top)
{
    top->size = 0;
}

static int cand_top_init(CandidateTop *top, uint32_t cap)
{
    memset(top, 0, sizeof(*top));
    top->cap = cap;
    top->items = (Candidate *)calloc(cap ? cap : 1u, sizeof(top->items[0]));
    return top->items ? 0 : -1;
}

static void cand_top_free(CandidateTop *top)
{
    free(top->items);
    memset(top, 0, sizeof(*top));
}

static void cand_top_insert(CandidateTop *top,
                            uint32_t vec_id,
                            uint32_t cluster_id,
                            uint32_t local_idx,
                            uint32_t disk_id,
                            uint64_t raw_lba,
                            uint32_t lane,
                            float dist)
{
    if (top->cap == 0) {
        return;
    }
    Candidate cand = {
        .vec_id = vec_id,
        .cluster_id = cluster_id,
        .local_idx = local_idx,
        .disk_id = disk_id,
        .lane = lane,
        .raw_lba = raw_lba,
        .dist = dist
    };
    if (top->size < top->cap) {
        top->items[top->size++] = cand;
        return;
    }
    uint32_t worst = 0;
    for (uint32_t i = 1; i < top->size; i++) {
        if (top->items[i].dist > top->items[worst].dist) {
            worst = i;
        }
    }
    if (dist < top->items[worst].dist) {
        top->items[worst] = cand;
    }
}

static int cmp_candidate_raw_lba(const void *a, const void *b)
{
    const Candidate *x = (const Candidate *)a;
    const Candidate *y = (const Candidate *)b;
    if (x->disk_id != y->disk_id) {
        return (x->disk_id > y->disk_id) - (x->disk_id < y->disk_id);
    }
    if (x->raw_lba != y->raw_lba) {
        return (x->raw_lba > y->raw_lba) - (x->raw_lba < y->raw_lba);
    }
    if (x->lane != y->lane) {
        return (x->lane > y->lane) - (x->lane < y->lane);
    }
    return (x->vec_id > y->vec_id) - (x->vec_id < y->vec_id);
}

static double recall_at_10(const TopkState *pred, const GtSet *gt, uint32_t query_idx)
{
    if (!gt || !gt->ids || query_idx >= gt->n_queries || gt->k == 0) {
        return -1.0;
    }
    uint32_t eval_k = TOPK;
    if (eval_k > pred->size) eval_k = pred->size;
    if (eval_k > gt->k) eval_k = gt->k;
    if (eval_k == 0) return 0.0;

    bool used[TOPK] = {false};
    uint32_t hits = 0;
    const int32_t *row = gt->ids + (size_t)query_idx * gt->k;
    for (uint32_t i = 0; i < eval_k; i++) {
        int32_t pred_id = (int32_t)pred->items[i].vec_id;
        for (uint32_t j = 0; j < eval_k; j++) {
            if (!used[j] && pred_id == row[j]) {
                used[j] = true;
                hits++;
                break;
            }
        }
    }
    return (double)hits / (double)eval_k;
}

static int load_ivecs_topk(const char *path, GtSet *gt)
{
    memset(gt, 0, sizeof(*gt));
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen groundtruth");
        return -1;
    }
    int dim = 0;
    if (fread(&dim, sizeof(dim), 1, fp) != 1 || dim <= 0) {
        fclose(fp);
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }
    long size = ftell(fp);
    rewind(fp);
    long rec_size = (long)sizeof(int32_t) + (long)dim * sizeof(int32_t);
    if (size < 0 || size % rec_size != 0) {
        fclose(fp);
        return -1;
    }
    uint32_t nq = (uint32_t)(size / rec_size);
    int32_t *ids = (int32_t *)malloc((size_t)nq * (size_t)dim * sizeof(int32_t));
    if (!ids) {
        fclose(fp);
        return -1;
    }
    for (uint32_t i = 0; i < nq; i++) {
        int row_dim = 0;
        if (fread(&row_dim, sizeof(row_dim), 1, fp) != 1 || row_dim != dim ||
            fread(ids + (size_t)i * dim, sizeof(int32_t), (size_t)dim, fp) != (size_t)dim) {
            free(ids);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    gt->n_queries = nq;
    gt->k = (uint32_t)dim;
    gt->ids = ids;
    return 0;
}

static void free_gt(GtSet *gt)
{
    free(gt->ids);
    memset(gt, 0, sizeof(*gt));
}

static int load_meta(const char *path, IvfPqMeta *meta)
{
    memset(meta, 0, sizeof(*meta));
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen meta");
        return -1;
    }
    if (fread(&meta->hdr, sizeof(meta->hdr), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    if (meta->hdr.magic != META_MAGIC || meta->hdr.num_disks != NUM_DISKS ||
        meta->hdr.dim == 0 || meta->hdr.nlist == 0 || meta->hdr.pq_m == 0 ||
        meta->hdr.pq_ksub == 0 || meta->hdr.pq_subdim == 0 ||
        meta->hdr.pq_m * meta->hdr.pq_subdim != meta->hdr.dim ||
        meta->hdr.pq_code_bytes != meta->hdr.pq_m) {
        fprintf(stderr, "invalid ivfpq metadata layout\n");
        fclose(fp);
        return -1;
    }
    meta->clusters = (ClusterMeta *)calloc(meta->hdr.nlist, sizeof(meta->clusters[0]));
    meta->centroids = (float *)calloc((size_t)meta->hdr.nlist * meta->hdr.dim, sizeof(float));
    meta->pq_codebooks = (float *)calloc((size_t)meta->hdr.pq_m * meta->hdr.pq_ksub * meta->hdr.pq_subdim, sizeof(float));
    if (!meta->clusters || !meta->centroids || !meta->pq_codebooks) {
        fclose(fp);
        return -1;
    }
    if (fread(meta->clusters, sizeof(ClusterMeta), meta->hdr.nlist, fp) != meta->hdr.nlist ||
        fread(meta->centroids, sizeof(float), (size_t)meta->hdr.nlist * meta->hdr.dim, fp) != (size_t)meta->hdr.nlist * meta->hdr.dim ||
        fread(meta->pq_codebooks, sizeof(float), (size_t)meta->hdr.pq_m * meta->hdr.pq_ksub * meta->hdr.pq_subdim, fp) != (size_t)meta->hdr.pq_m * meta->hdr.pq_ksub * meta->hdr.pq_subdim) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static void free_meta(IvfPqMeta *meta)
{
    free(meta->clusters);
    free(meta->centroids);
    free(meta->pq_codebooks);
    memset(meta, 0, sizeof(*meta));
}

static int load_sorted_ids(const char *path, uint32_t expected_n, uint32_t **ids_out)
{
    *ids_out = NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen sorted ids");
        return -1;
    }
    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, fp) != 1 || n != expected_n) {
        fclose(fp);
        return -1;
    }
    uint32_t *ids = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    if (!ids) {
        fclose(fp);
        return -1;
    }
    if (fread(ids, sizeof(uint32_t), n, fp) != n) {
        free(ids);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *ids_out = ids;
    return 0;
}

static int coarse_search(const float *query, const IvfPqMeta *meta, uint32_t nprobe, CoarseHit *hits)
{
    for (uint32_t i = 0; i < nprobe; i++) {
        hits[i] = (CoarseHit){.cluster_id = UINT32_MAX, .dist = INFINITY};
    }
    for (uint32_t cid = 0; cid < meta->hdr.nlist; cid++) {
        float dist = l2(query, meta->centroids + (size_t)cid * meta->hdr.dim, meta->hdr.dim);
        int pos = -1;
        for (uint32_t i = 0; i < nprobe; i++) {
            if (dist < hits[i].dist) {
                pos = (int)i;
                break;
            }
        }
        if (pos >= 0) {
            for (int j = (int)nprobe - 1; j > pos; j--) {
                hits[j] = hits[j - 1];
            }
            hits[pos] = (CoarseHit){.cluster_id = cid, .dist = dist};
        }
    }
    return 0;
}

static void build_dist_table(const float *query, const IvfPqMeta *meta, float *table)
{
    uint32_t m = meta->hdr.pq_m;
    uint32_t ksub = meta->hdr.pq_ksub;
    uint32_t subdim = meta->hdr.pq_subdim;
    for (uint32_t part = 0; part < m; part++) {
        const float *qsub = query + (size_t)part * subdim;
        for (uint32_t k = 0; k < ksub; k++) {
            const float *centroid = meta->pq_codebooks + ((size_t)part * ksub + k) * subdim;
            table[(size_t)part * ksub + k] = l2(qsub, centroid, subdim);
        }
    }
}

static void io_done(void *arg, const struct spdk_nvme_cpl *cpl)
{
    IoWaiter *waiter = (IoWaiter *)arg;
    waiter->ok = !spdk_nvme_cpl_is_error(cpl);
    waiter->done = true;
}

static int read_lba(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair, void *buf, uint64_t lba, uint32_t count)
{
    IoWaiter waiter = {.done = false, .ok = 0};
    int rc = spdk_nvme_ns_cmd_read(ns, qpair, buf, lba, count, io_done, &waiter, 0);
    if (rc != 0) {
        return -1;
    }
    while (!waiter.done) {
        int cpl_rc = spdk_nvme_qpair_process_completions(qpair, 0);
        if (cpl_rc < 0) {
            return -1;
        }
    }
    return waiter.ok ? 0 : -1;
}

static float adc_dist(const uint8_t *code, const float *table, uint32_t m, uint32_t ksub)
{
    float s = 0.0f;
    for (uint32_t part = 0; part < m; part++) {
        s += table[(size_t)part * ksub + code[part]];
    }
    return s;
}

static int scan_pq_range(Worker *w,
                         struct spdk_nvme_qpair *qpair[NUM_DISKS],
                         uint8_t *pq_buf,
                         CandidateTop *top,
                         uint32_t h_begin,
                         uint32_t h_end,
                         uint64_t *io_us,
                         uint64_t *scan_us,
                         uint64_t *pq_candidates)
{
    const IvfPqMeta *meta = w->meta;
    cand_top_reset(top);
    *io_us = 0;
    *scan_us = 0;
    *pq_candidates = 0;

    if (h_end > w->nprobe) {
        h_end = w->nprobe;
    }
    for (uint32_t h = h_begin; h < h_end; h++) {
        uint32_t cid = w->hits[h].cluster_id;
        if (cid == UINT32_MAX) {
            continue;
        }
        const ClusterMeta *cm = &meta->clusters[cid];
        uint32_t disk = cm->disk_id;

        for (uint32_t b = 0; b < cm->pq_num_lbas;) {
            uint32_t chunk_lbas = cm->pq_num_lbas - b;
            if (chunk_lbas > w->pq_read_lbas) {
                chunk_lbas = w->pq_read_lbas;
            }

            uint64_t t0 = now_us();
            if (read_lba(w->ns[disk], qpair[disk], pq_buf, cm->pq_start_lba + b, chunk_lbas) != 0) {
                fprintf(stderr, "failed to read pq cluster=%u disk=%u lba=%" PRIu64 " count=%u\n",
                        cid, disk, cm->pq_start_lba + b, chunk_lbas);
                return -1;
            }
            *io_us += now_us() - t0;

            t0 = now_us();
            for (uint32_t cb = 0; cb < chunk_lbas; cb++) {
                uint32_t base_local = (b + cb) * meta->hdr.pq_codes_per_lba;
                const uint8_t *lba_base = pq_buf + (size_t)cb * meta->hdr.sector_size;
                for (uint32_t lane = 0; lane < meta->hdr.pq_codes_per_lba; lane++) {
                    uint32_t local_idx = base_local + lane;
                    if (local_idx >= cm->num_vectors) {
                        break;
                    }
                    uint32_t sorted_pos = cm->sorted_id_base + local_idx;
                    uint32_t vec_id = w->sorted_ids[sorted_pos];
                    uint32_t raw_lba_idx = local_idx / meta->hdr.raw_vectors_per_lba;
                    uint32_t raw_lane = local_idx % meta->hdr.raw_vectors_per_lba;
                    uint64_t raw_lba = cm->raw_start_lba + raw_lba_idx;
                    const uint8_t *code = lba_base + (size_t)lane * meta->hdr.pq_code_bytes;
                    float dist = adc_dist(code, w->dist_table, meta->hdr.pq_m, meta->hdr.pq_ksub);
                    cand_top_insert(top,
                                    vec_id,
                                    cid,
                                    local_idx,
                                    disk,
                                    raw_lba,
                                    raw_lane,
                                    dist);
                    (*pq_candidates)++;
                }
            }
            *scan_us += now_us() - t0;
            b += chunk_lbas;
        }
    }
    return 0;
}

static void cand_top_merge(CandidateTop *dst, const CandidateTop *src)
{
    for (uint32_t i = 0; i < src->size; i++) {
        const Candidate *cand = &src->items[i];
        cand_top_insert(dst,
                        cand->vec_id,
                        cand->cluster_id,
                        cand->local_idx,
                        cand->disk_id,
                        cand->raw_lba,
                        cand->lane,
                        cand->dist);
    }
}

static void assign_scan_ranges(Worker *w)
{
    uint32_t shards = w->cluster_threads;
    uint32_t base = w->nprobe / shards;
    uint32_t rem = w->nprobe % shards;
    uint32_t begin = 0;
    for (uint32_t i = 0; i < shards; i++) {
        uint32_t count = base + (i < rem ? 1u : 0u);
        w->scan_shards[i].h_begin = begin;
        w->scan_shards[i].h_end = begin + count;
        begin += count;
    }
}

static void *scan_helper_main(void *arg)
{
    ScanShard *shard = (ScanShard *)arg;
    Worker *w = shard->owner;

    shard->init_rc = 0;
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        shard->qpair[d] = spdk_nvme_ctrlr_alloc_io_qpair(w->ctrlr[d], NULL, 0);
        if (!shard->qpair[d]) {
            shard->init_rc = -1;
            break;
        }
    }
    if (shard->init_rc == 0) {
        shard->pq_buf = (uint8_t *)spdk_zmalloc((size_t)w->pq_read_lbas * w->meta->hdr.sector_size,
                                                w->meta->hdr.sector_size,
                                                NULL,
                                                SPDK_ENV_NUMA_ID_ANY,
                                                SPDK_MALLOC_DMA);
        if (!shard->pq_buf || cand_top_init(&shard->approx_top, w->rerank_k) != 0) {
            shard->init_rc = -1;
        }
    }

    pthread_mutex_lock(&w->scan_mu);
    shard->ready = true;
    w->scan_ready_helpers++;
    if (shard->init_rc != 0 && w->scan_rc == 0) {
        w->scan_rc = shard->init_rc;
    }
    pthread_cond_broadcast(&w->scan_cv);

    while (!w->scan_shutdown && shard->init_rc == 0) {
        while (!w->scan_shutdown && shard->generation_seen == w->scan_generation) {
            pthread_cond_wait(&w->scan_cv, &w->scan_mu);
        }
        if (w->scan_shutdown) {
            break;
        }
        shard->generation_seen = w->scan_generation;
        pthread_mutex_unlock(&w->scan_mu);

        int rc = scan_pq_range(w,
                               shard->qpair,
                               shard->pq_buf,
                               &shard->approx_top,
                               shard->h_begin,
                               shard->h_end,
                               &shard->io_us,
                               &shard->scan_us,
                               &shard->pq_candidates);

        pthread_mutex_lock(&w->scan_mu);
        if (rc != 0 && w->scan_rc == 0) {
            w->scan_rc = rc;
        }
        w->scan_completed_helpers++;
        pthread_cond_broadcast(&w->scan_cv);
    }
    pthread_mutex_unlock(&w->scan_mu);

    cand_top_free(&shard->approx_top);
    if (shard->pq_buf) spdk_free(shard->pq_buf);
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        if (shard->qpair[d]) {
            spdk_nvme_ctrlr_free_io_qpair(shard->qpair[d]);
        }
    }
    return NULL;
}

static int init_scan_helpers(Worker *w)
{
    w->scan_shards = (ScanShard *)calloc(w->cluster_threads, sizeof(w->scan_shards[0]));
    if (!w->scan_shards) {
        return -1;
    }
    if (pthread_mutex_init(&w->scan_mu, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&w->scan_cv, NULL) != 0) {
        pthread_mutex_destroy(&w->scan_mu);
        return -1;
    }
    w->scan_sync_init = true;
    for (uint32_t i = 0; i < w->cluster_threads; i++) {
        w->scan_shards[i].owner = w;
        w->scan_shards[i].shard_idx = i;
    }
    assign_scan_ranges(w);

    for (uint32_t i = 1; i < w->cluster_threads; i++) {
        ScanShard *shard = &w->scan_shards[i];
        if (pthread_create(&shard->tid, NULL, scan_helper_main, shard) != 0) {
            shard->init_rc = -1;
            return -1;
        }
        shard->created = true;
    }

    pthread_mutex_lock(&w->scan_mu);
    while (w->scan_ready_helpers < w->cluster_threads - 1 && w->scan_rc == 0) {
        pthread_cond_wait(&w->scan_cv, &w->scan_mu);
    }
    int rc = w->scan_rc;
    pthread_mutex_unlock(&w->scan_mu);
    return rc == 0 ? 0 : -1;
}

static void shutdown_scan_helpers(Worker *w)
{
    if (!w->scan_shards) {
        return;
    }
    if (w->scan_sync_init) {
        pthread_mutex_lock(&w->scan_mu);
        w->scan_shutdown = true;
        pthread_cond_broadcast(&w->scan_cv);
        pthread_mutex_unlock(&w->scan_mu);
    }
    for (uint32_t i = 1; i < w->cluster_threads; i++) {
        if (w->scan_shards[i].created) {
            pthread_join(w->scan_shards[i].tid, NULL);
        }
    }
    if (w->scan_sync_init) {
        pthread_cond_destroy(&w->scan_cv);
        pthread_mutex_destroy(&w->scan_mu);
        w->scan_sync_init = false;
    }
    free(w->scan_shards);
    w->scan_shards = NULL;
}

static int scan_pq_parallel(Worker *w,
                            uint64_t *io_us,
                            uint64_t *scan_us,
                            uint64_t *pq_candidates)
{
    uint64_t main_io = 0;
    uint64_t main_scan = 0;
    uint64_t main_candidates = 0;
    *io_us = 0;
    *scan_us = 0;
    *pq_candidates = 0;

    if (w->cluster_threads <= 1) {
        return scan_pq_range(w,
                             w->qpair,
                             w->pq_buf,
                             &w->approx_top,
                             0,
                             w->nprobe,
                             io_us,
                             scan_us,
                             pq_candidates);
    }

    assign_scan_ranges(w);
    pthread_mutex_lock(&w->scan_mu);
    w->scan_rc = 0;
    w->scan_completed_helpers = 0;
    w->scan_generation++;
    pthread_cond_broadcast(&w->scan_cv);
    pthread_mutex_unlock(&w->scan_mu);

    int rc = scan_pq_range(w,
                           w->qpair,
                           w->pq_buf,
                           &w->approx_top,
                           w->scan_shards[0].h_begin,
                           w->scan_shards[0].h_end,
                           &main_io,
                           &main_scan,
                           &main_candidates);

    pthread_mutex_lock(&w->scan_mu);
    if (rc != 0 && w->scan_rc == 0) {
        w->scan_rc = rc;
    }
    while (w->scan_completed_helpers < w->cluster_threads - 1 && w->scan_rc == 0) {
        pthread_cond_wait(&w->scan_cv, &w->scan_mu);
    }
    rc = w->scan_rc;
    pthread_mutex_unlock(&w->scan_mu);

    if (rc != 0) {
        return -1;
    }

    *io_us = main_io;
    *scan_us = main_scan;
    *pq_candidates = main_candidates;
    for (uint32_t i = 1; i < w->cluster_threads; i++) {
        ScanShard *shard = &w->scan_shards[i];
        *io_us += shard->io_us;
        *scan_us += shard->scan_us;
        *pq_candidates += shard->pq_candidates;
        cand_top_merge(&w->approx_top, &shard->approx_top);
    }
    return 0;
}

static int submit_raw_read(Worker *w, RawReadCtx *ctx, const RawGroup *group)
{
    memset(&ctx->waiter, 0, sizeof(ctx->waiter));
    ctx->active = true;

    int rc = spdk_nvme_ns_cmd_read(w->ns[group->disk_id],
                                   w->qpair[group->disk_id],
                                   ctx->buf,
                                   group->lba,
                                   1,
                                   io_done,
                                   &ctx->waiter,
                                   0);
    if (rc != 0) {
        ctx->active = false;
        return -1;
    }
    return 0;
}

static void process_raw_group(Worker *w,
                              const float *query,
                              TopkState *topk,
                              const RawGroup *group,
                              const uint8_t *buf,
                              uint64_t *compute_us)
{
    const IvfPqMeta *meta = w->meta;
    uint64_t t0 = now_us();
    for (uint32_t i = group->begin; i < group->end; i++) {
        const Candidate *cand = &w->approx_top.items[i];
        const float *vec = (const float *)(buf + (size_t)cand->lane * meta->hdr.dim * sizeof(float));
        float dist = l2(vec, query, meta->hdr.dim);
        topk_insert(topk, cand->vec_id, dist);
    }
    *compute_us += now_us() - t0;
}

static int build_raw_groups(Worker *w, RawGroup **groups_out, uint32_t *group_count_out)
{
    *groups_out = NULL;
    *group_count_out = 0;
    if (w->approx_top.size == 0) {
        return 0;
    }

    qsort(w->approx_top.items, w->approx_top.size, sizeof(w->approx_top.items[0]), cmp_candidate_raw_lba);

    RawGroup *groups = (RawGroup *)calloc(w->approx_top.size, sizeof(groups[0]));
    if (!groups) {
        return -1;
    }

    uint32_t group_count = 0;
    uint32_t begin = 0;
    while (begin < w->approx_top.size) {
        const Candidate *first = &w->approx_top.items[begin];
        uint32_t end = begin + 1;
        while (end < w->approx_top.size &&
               w->approx_top.items[end].disk_id == first->disk_id &&
               w->approx_top.items[end].raw_lba == first->raw_lba) {
            end++;
        }
        groups[group_count++] = (RawGroup){
            .disk_id = first->disk_id,
            .lba = first->raw_lba,
            .begin = begin,
            .end = end
        };
        begin = end;
    }

    *groups_out = groups;
    *group_count_out = group_count;
    return 0;
}

static int rerank(Worker *w, const float *query, TopkState *topk, uint64_t *io_us, uint64_t *compute_us)
{
    memset(topk, 0, sizeof(*topk));
    *io_us = 0;
    *compute_us = 0;

    RawGroup *groups = NULL;
    uint32_t group_count = 0;
    if (build_raw_groups(w, &groups, &group_count) != 0) {
        return -1;
    }

    uint32_t next_group = 0;
    uint32_t inflight = 0;
    uint64_t io_begin = now_us();

    while (next_group < group_count || inflight > 0) {
        for (uint32_t slot = 0; slot < w->io_depth && next_group < group_count; slot++) {
            RawReadCtx *ctx = &w->raw_reads[slot];
            if (ctx->active) {
                continue;
            }
            ctx->group_idx = next_group;
            if (submit_raw_read(w, ctx, &groups[next_group]) != 0) {
                fprintf(stderr, "failed to submit raw read disk=%u lba=%" PRIu64 "\n",
                        groups[next_group].disk_id, groups[next_group].lba);
                free(groups);
                return -1;
            }
            next_group++;
            inflight++;
        }

        bool progressed = false;
        for (uint32_t slot = 0; slot < w->io_depth; slot++) {
            RawReadCtx *ctx = &w->raw_reads[slot];
            if (!ctx->active || !ctx->waiter.done) {
                continue;
            }
            if (!ctx->waiter.ok) {
                fprintf(stderr, "raw read completion error group=%u\n", ctx->group_idx);
                free(groups);
                return -1;
            }
            process_raw_group(w, query, topk, &groups[ctx->group_idx], ctx->buf, compute_us);
            ctx->active = false;
            inflight--;
            progressed = true;
        }

        if (!progressed && inflight > 0) {
            for (uint32_t d = 0; d < NUM_DISKS; d++) {
                int cpl_rc = spdk_nvme_qpair_process_completions(w->qpair[d], 0);
                if (cpl_rc < 0) {
                    free(groups);
                    return -1;
                }
            }
        }
    }

    {
        uint64_t rerank_wall = now_us() - io_begin;
        *io_us = rerank_wall > *compute_us ? rerank_wall - *compute_us : 0;
    }
    topk_finalize(topk);
    free(groups);
    return 0;
}

static int run_query(Worker *w, uint32_t query_idx)
{
    QueryResult *res = &w->results[query_idx];
    memset(res, 0, sizeof(*res));
    res->rc = -1;
    const float *query = w->queries + (size_t)query_idx * w->query_dim;
    TopkState topk = {0};

    uint64_t total0 = now_us();
    uint64_t t0 = now_us();
    if (coarse_search(query, w->meta, w->nprobe, w->hits) != 0) {
        return -1;
    }
    res->coarse_us = now_us() - t0;

    t0 = now_us();
    build_dist_table(query, w->meta, w->dist_table);
    res->table_us = now_us() - t0;

    if (scan_pq_parallel(w, &res->pq_io_us, &res->pq_scan_us, &res->pq_candidates) != 0) {
        return -1;
    }
    res->rerank_candidates = w->approx_top.size;
    if (rerank(w, query, &topk, &res->rerank_io_us, &res->rerank_compute_us) != 0) {
        return -1;
    }

    res->latency_us = now_us() - total0;
    res->done_ts_us = now_us();
    res->recall10 = recall_at_10(&topk, w->gt, query_idx);
    res->rc = 0;
    return 0;
}

static void *worker_main(void *arg)
{
    Worker *w = (Worker *)arg;
    bind_to_core(w->core_id);
    w->init_rc = 0;

    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        w->qpair[d] = spdk_nvme_ctrlr_alloc_io_qpair(w->ctrlr[d], NULL, 0);
        if (!w->qpair[d]) {
            w->init_rc = -1;
            break;
        }
    }
    if (w->init_rc == 0) {
        w->pq_buf = (uint8_t *)spdk_zmalloc((size_t)w->pq_read_lbas * w->meta->hdr.sector_size,
                                            w->meta->hdr.sector_size,
                                            NULL,
                                            SPDK_ENV_NUMA_ID_ANY, SPDK_MALLOC_DMA);
        w->raw_reads = (RawReadCtx *)calloc(w->io_depth, sizeof(w->raw_reads[0]));
        w->dist_table = (float *)calloc((size_t)w->meta->hdr.pq_m * w->meta->hdr.pq_ksub, sizeof(float));
        w->hits = (CoarseHit *)calloc(w->nprobe, sizeof(CoarseHit));
        if (w->raw_reads) {
            for (uint32_t i = 0; i < w->io_depth; i++) {
                w->raw_reads[i].buf = (uint8_t *)spdk_zmalloc(w->meta->hdr.sector_size,
                                                              w->meta->hdr.sector_size,
                                                              NULL,
                                                              SPDK_ENV_NUMA_ID_ANY,
                                                              SPDK_MALLOC_DMA);
                if (!w->raw_reads[i].buf) {
                    w->init_rc = -1;
                    break;
                }
            }
        }
        if (!w->pq_buf || !w->raw_reads || !w->dist_table || !w->hits ||
            cand_top_init(&w->approx_top, w->rerank_k) != 0) {
            w->init_rc = -1;
        }
        if (w->init_rc == 0 && w->cluster_threads > 1 && init_scan_helpers(w) != 0) {
            w->init_rc = -1;
        }
    }

    pthread_mutex_lock(&w->dispatch->mu);
    if (w->init_rc != 0 && w->dispatch->fatal_rc == 0) {
        w->dispatch->fatal_rc = w->init_rc;
    }
    w->dispatch->ready_workers++;
    pthread_cond_broadcast(&w->dispatch->cv);
    while (!w->dispatch->start && w->dispatch->fatal_rc == 0) {
        pthread_cond_wait(&w->dispatch->cv, &w->dispatch->mu);
    }
    pthread_mutex_unlock(&w->dispatch->mu);

    if (w->init_rc == 0) {
        for (;;) {
            uint32_t query_idx = UINT32_MAX;
            pthread_mutex_lock(&w->dispatch->mu);
            if (w->dispatch->fatal_rc != 0 || w->dispatch->next_query_idx >= w->max_queries) {
                pthread_mutex_unlock(&w->dispatch->mu);
                break;
            }
            query_idx = w->dispatch->next_query_idx++;
            pthread_mutex_unlock(&w->dispatch->mu);

            if (run_query(w, query_idx) != 0) {
                pthread_mutex_lock(&w->dispatch->mu);
                if (w->dispatch->fatal_rc == 0) {
                    w->dispatch->fatal_rc = -1;
                }
                pthread_mutex_unlock(&w->dispatch->mu);
                break;
            }
        }
    }

    cand_top_free(&w->approx_top);
    shutdown_scan_helpers(w);
    free(w->hits);
    free(w->dist_table);
    if (w->pq_buf) spdk_free(w->pq_buf);
    if (w->raw_reads) {
        for (uint32_t i = 0; i < w->io_depth; i++) {
            if (w->raw_reads[i].buf) {
                spdk_free(w->raw_reads[i].buf);
            }
        }
        free(w->raw_reads);
    }
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        if (w->qpair[d]) {
            spdk_nvme_ctrlr_free_io_qpair(w->qpair[d]);
        }
    }
    return NULL;
}

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    DiskCtx *disks = (DiskCtx *)cb_ctx;
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        if (strcmp(trid->traddr, disks[d].traddr) == 0) {
            return true;
        }
    }
    return false;
}

static void attach_cb(void *cb_ctx,
                      const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    DiskCtx *disks = (DiskCtx *)cb_ctx;
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        if (strcmp(trid->traddr, disks[d].traddr) == 0) {
            disks[d].ctrlr = ctrlr;
            disks[d].ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
            if (!disks[d].ns || !spdk_nvme_ns_is_active(disks[d].ns)) {
                fprintf(stderr, "inactive ns for %s\n", trid->traddr);
                exit(EXIT_FAILURE);
            }
            disks[d].sector_size = spdk_nvme_ns_get_sector_size(disks[d].ns);
            fprintf(stderr, "[attach] disk%u=%s sector=%u\n", d, trid->traddr, disks[d].sector_size);
        }
    }
}

static int probe_disks(DiskCtx disks[NUM_DISKS])
{
    if (spdk_nvme_probe(NULL, disks, probe_cb, attach_cb, NULL) != 0) {
        return -1;
    }
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        if (!disks[d].ctrlr || !disks[d].ns) {
            fprintf(stderr, "failed to attach disk%u=%s\n", d, disks[d].traddr);
            return -1;
        }
    }
    return 0;
}

static void cleanup_disks(DiskCtx disks[NUM_DISKS])
{
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        if (disks[d].ctrlr) {
            spdk_nvme_detach(disks[d].ctrlr);
        }
    }
}

static int parse_core_list(const char *spec, int **cores_out, uint32_t *count_out)
{
    *cores_out = NULL;
    *count_out = 0;
    char *tmp = strdup(spec);
    if (!tmp) return -1;
    uint32_t count = 0;
    char *save = NULL;
    for (char *tok = strtok_r(tmp, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        count++;
    }
    free(tmp);
    int *cores = (int *)malloc((size_t)count * sizeof(int));
    if (!cores) return -1;
    tmp = strdup(spec);
    if (!tmp) {
        free(cores);
        return -1;
    }
    save = NULL;
    uint32_t idx = 0;
    for (char *tok = strtok_r(tmp, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        cores[idx++] = (int)strtol(tok, NULL, 10);
    }
    free(tmp);
    *cores_out = cores;
    *count_out = count;
    return 0;
}

static int resolve_cores(const Config *cfg, uint32_t *workers_io, int **cores_out)
{
    *cores_out = NULL;
    if (cfg->cores_spec) {
        uint32_t core_count = 0;
        if (parse_core_list(cfg->cores_spec, cores_out, &core_count) != 0) {
            return -1;
        }
        if (cfg->threads_explicit && *workers_io != core_count) {
            fprintf(stderr, "--threads=%u mismatches --cores count=%u\n", *workers_io, core_count);
            free(*cores_out);
            *cores_out = NULL;
            return -1;
        }
        *workers_io = core_count;
        return 0;
    }
    if (cfg->base_core >= 0) {
        int *cores = (int *)malloc((size_t)*workers_io * sizeof(int));
        if (!cores) return -1;
        for (uint32_t i = 0; i < *workers_io; i++) {
            cores[i] = cfg->base_core + (int)i;
        }
        *cores_out = cores;
    }
    return 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --disk0 0000:65:00.0 --disk1 0000:66:00.0 --disk2 0000:67:00.0 --disk3 0000:68:00.0 \\\n"
            "     --nprobe 32 [--rerank-k 100] [--io-depth 8] [--pq-read-lbas 16] [--cluster-threads 1] \\\n"
            "     [--max-queries 1000] [--threads 8] [--base-core 0] [--cores 0,1,2,3]\n",
            prog);
}

static int parse_args(int argc, char **argv, Config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->meta_path = DEFAULT_META_PATH;
    cfg->sorted_ids_path = DEFAULT_SORTED_IDS_PATH;
    cfg->query_fvecs_path = DEFAULT_QUERY_FVECS_PATH;
    cfg->gt_ivecs_path = DEFAULT_GT_IVECS_PATH;
    cfg->nprobe = 32;
    cfg->rerank_k = 100;
    cfg->io_depth = 8;
    cfg->pq_read_lbas = 16;
    cfg->cluster_threads = 1;
    cfg->max_queries = 0;
    cfg->threads = 1;
    cfg->base_core = -1;

    static struct option opts[] = {
        {"meta", required_argument, 0, 'm'},
        {"sorted-ids", required_argument, 0, 's'},
        {"queries", required_argument, 0, 'q'},
        {"gt", required_argument, 0, 'g'},
        {"disk0", required_argument, 0, 1000},
        {"disk1", required_argument, 0, 1001},
        {"disk2", required_argument, 0, 1002},
        {"disk3", required_argument, 0, 1003},
        {"nprobe", required_argument, 0, 'n'},
        {"rerank-k", required_argument, 0, 1004},
        {"max-queries", required_argument, 0, 1005},
        {"threads", required_argument, 0, 1006},
        {"base-core", required_argument, 0, 1007},
        {"cores", required_argument, 0, 1008},
        {"io-depth", required_argument, 0, 1009},
        {"pq-read-lbas", required_argument, 0, 1010},
        {"cluster-threads", required_argument, 0, 1011},
        {0, 0, 0, 0}
    };

    int opt = 0;
    int idx = 0;
    while ((opt = getopt_long(argc, argv, "m:s:q:g:n:", opts, &idx)) != -1) {
        switch (opt) {
            case 'm': cfg->meta_path = optarg; break;
            case 's': cfg->sorted_ids_path = optarg; break;
            case 'q': cfg->query_fvecs_path = optarg; break;
            case 'g': cfg->gt_ivecs_path = optarg; break;
            case 'n': cfg->nprobe = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1000: cfg->disk_traddr[0] = optarg; break;
            case 1001: cfg->disk_traddr[1] = optarg; break;
            case 1002: cfg->disk_traddr[2] = optarg; break;
            case 1003: cfg->disk_traddr[3] = optarg; break;
            case 1004: cfg->rerank_k = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1005: cfg->max_queries = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1006:
                cfg->threads = (uint32_t)strtoul(optarg, NULL, 10);
                cfg->threads_explicit = true;
                break;
            case 1007: cfg->base_core = (int)strtol(optarg, NULL, 10); break;
            case 1008: cfg->cores_spec = optarg; break;
            case 1009: cfg->io_depth = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1010: cfg->pq_read_lbas = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1011: cfg->cluster_threads = (uint32_t)strtoul(optarg, NULL, 10); break;
            default:
                usage(argv[0]);
                return -1;
        }
    }
    if (!cfg->disk_traddr[0] || !cfg->disk_traddr[1] || !cfg->disk_traddr[2] ||
        !cfg->disk_traddr[3] || cfg->nprobe == 0 || cfg->rerank_k == 0 ||
        cfg->threads == 0 || cfg->io_depth == 0 || cfg->pq_read_lbas == 0 ||
        cfg->cluster_threads == 0) {
        usage(argv[0]);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    Config cfg;
    IvfPqMeta meta;
    query_set_t qs;
    GtSet gt;
    DiskCtx disks[NUM_DISKS];
    uint32_t *sorted_ids = NULL;
    Worker *workers = NULL;
    QueryResult *results = NULL;
    QueryDispatch dispatch;
    int *worker_cores = NULL;
    bool dispatch_init = false;
    uint64_t run_begin = 0;
    uint32_t worker_count = 0;
    uint32_t created_workers = 0;
    uint32_t max_queries = 0;
    int rc = 1;

    memset(&meta, 0, sizeof(meta));
    memset(&qs, 0, sizeof(qs));
    memset(&gt, 0, sizeof(gt));
    memset(disks, 0, sizeof(disks));
    memset(&dispatch, 0, sizeof(dispatch));

    if (parse_args(argc, argv, &cfg) != 0) return 1;

    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "gist_ivfpq_baseline";
    opts.mem_size = 1024;
    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "spdk_env_init failed\n");
        return 1;
    }

    if (load_meta(cfg.meta_path, &meta) != 0) goto cleanup;
    if (cfg.nprobe > meta.hdr.nlist) {
        fprintf(stderr, "nprobe=%u > nlist=%u\n", cfg.nprobe, meta.hdr.nlist);
        goto cleanup;
    }
    if (cfg.cluster_threads > cfg.nprobe) {
        cfg.cluster_threads = cfg.nprobe;
    }
    if (load_sorted_ids(cfg.sorted_ids_path, meta.hdr.num_vectors, &sorted_ids) != 0) goto cleanup;
    if (load_fvecs(cfg.query_fvecs_path, &qs.data, &qs.n_queries, &qs.dim) != 0) goto cleanup;
    if (qs.dim != meta.hdr.dim) {
        fprintf(stderr, "query dim mismatch got=%u expected=%u\n", qs.dim, meta.hdr.dim);
        goto cleanup;
    }
    if (load_ivecs_topk(cfg.gt_ivecs_path, &gt) != 0) goto cleanup;

    max_queries = cfg.max_queries;
    if (max_queries == 0 || max_queries > qs.n_queries) max_queries = qs.n_queries;
    if (max_queries > gt.n_queries) max_queries = gt.n_queries;

    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        disks[d].traddr = cfg.disk_traddr[d];
    }
    if (probe_disks(disks) != 0) goto cleanup;
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        if (disks[d].sector_size != meta.hdr.sector_size) {
            fprintf(stderr, "sector mismatch disk%u=%u meta=%u\n", d, disks[d].sector_size, meta.hdr.sector_size);
            goto cleanup;
        }
    }
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        uint32_t max_xfer = spdk_nvme_ns_get_max_io_xfer_size(disks[d].ns);
        uint32_t max_lbas = max_xfer / disks[d].sector_size;
        if (max_lbas == 0) {
            max_lbas = 1;
        }
        if (cfg.pq_read_lbas > max_lbas) {
            cfg.pq_read_lbas = max_lbas;
        }
    }

    worker_count = cfg.threads;
    if (resolve_cores(&cfg, &worker_count, &worker_cores) != 0) goto cleanup;
    if (worker_count == 0) worker_count = 1;

    results = (QueryResult *)calloc(max_queries ? max_queries : 1u, sizeof(results[0]));
    workers = (Worker *)calloc(worker_count, sizeof(workers[0]));
    if (!results || !workers) goto cleanup;

    pthread_mutex_init(&dispatch.mu, NULL);
    pthread_cond_init(&dispatch.cv, NULL);
    dispatch_init = true;

    for (uint32_t i = 0; i < worker_count; i++) {
        Worker *w = &workers[i];
        w->core_id = worker_cores ? worker_cores[i] : -1;
        for (uint32_t d = 0; d < NUM_DISKS; d++) {
            w->ctrlr[d] = disks[d].ctrlr;
            w->ns[d] = disks[d].ns;
        }
        w->meta = &meta;
        w->sorted_ids = sorted_ids;
        w->queries = qs.data;
        w->query_dim = qs.dim;
        w->gt = &gt;
        w->results = results;
        w->dispatch = &dispatch;
        w->nprobe = cfg.nprobe;
        w->rerank_k = cfg.rerank_k;
        w->io_depth = cfg.io_depth;
        w->pq_read_lbas = cfg.pq_read_lbas;
        w->cluster_threads = cfg.cluster_threads;
        w->max_queries = max_queries;

        if (pthread_create(&w->tid, NULL, worker_main, w) != 0) {
            dispatch.fatal_rc = -1;
            created_workers = i;
            goto join_workers;
        }
        created_workers = i + 1;
    }

    pthread_mutex_lock(&dispatch.mu);
    while (dispatch.ready_workers < worker_count && dispatch.fatal_rc == 0) {
        pthread_cond_wait(&dispatch.cv, &dispatch.mu);
    }
    run_begin = now_us();
    dispatch.start = true;
    pthread_cond_broadcast(&dispatch.cv);
    pthread_mutex_unlock(&dispatch.mu);

join_workers:
    for (uint32_t i = 0; i < created_workers; i++) {
        pthread_join(workers[i].tid, NULL);
    }
    if (dispatch.fatal_rc != 0) {
        fprintf(stderr, "worker execution failed\n");
        goto cleanup;
    }

    {
        uint32_t done = 0;
        uint64_t max_done_ts = run_begin;
        uint64_t latency_sum = 0, coarse_sum = 0, table_sum = 0, pq_io_sum = 0, pq_scan_sum = 0;
        uint64_t rerank_io_sum = 0, rerank_compute_sum = 0, pq_cand_sum = 0, rerank_cand_sum = 0;
        double recall_sum = 0.0;
        for (uint32_t i = 0; i < max_queries; i++) {
            if (results[i].rc != 0) continue;
            done++;
            latency_sum += results[i].latency_us;
            coarse_sum += results[i].coarse_us;
            table_sum += results[i].table_us;
            pq_io_sum += results[i].pq_io_us;
            pq_scan_sum += results[i].pq_scan_us;
            rerank_io_sum += results[i].rerank_io_us;
            rerank_compute_sum += results[i].rerank_compute_us;
            pq_cand_sum += results[i].pq_candidates;
            rerank_cand_sum += results[i].rerank_candidates;
            if (results[i].recall10 >= 0.0) recall_sum += results[i].recall10;
            if (results[i].done_ts_us > max_done_ts) max_done_ts = results[i].done_ts_us;
        }
        uint64_t wall_us = max_done_ts - run_begin;
        double qps = wall_us ? (double)done * 1000000.0 / (double)wall_us : 0.0;
        printf("[ivfpq] disks=%s,%s,%s,%s dim=%u nlist=%u nprobe=%u pq_m=%u rerank_k=%u io_depth=%u pq_read_lbas=%u cluster_threads=%u queries=%u threads=%u wall_ms=%.3f qps=%.3f\n",
               cfg.disk_traddr[0], cfg.disk_traddr[1], cfg.disk_traddr[2], cfg.disk_traddr[3],
               meta.hdr.dim, meta.hdr.nlist, cfg.nprobe, meta.hdr.pq_m, cfg.rerank_k,
               cfg.io_depth, cfg.pq_read_lbas, cfg.cluster_threads,
               done, worker_count, (double)wall_us / 1000.0, qps);
        if (done > 0) {
            printf("[ivfpq summary] avg_latency_ms=%.3f avg_coarse_ms=%.3f avg_table_ms=%.3f avg_pq_io_ms=%.3f avg_pq_scan_ms=%.3f avg_rerank_io_ms=%.3f avg_rerank_compute_ms=%.3f avg_pq_candidates=%.1f avg_rerank_candidates=%.1f avg_recall@10=%.4f\n",
                   (double)latency_sum / 1000.0 / done,
                   (double)coarse_sum / 1000.0 / done,
                   (double)table_sum / 1000.0 / done,
                   (double)pq_io_sum / 1000.0 / done,
                   (double)pq_scan_sum / 1000.0 / done,
                   (double)rerank_io_sum / 1000.0 / done,
                   (double)rerank_compute_sum / 1000.0 / done,
                   (double)pq_cand_sum / done,
                   (double)rerank_cand_sum / done,
                   recall_sum / done);
        }
    }

    rc = 0;

cleanup:
    if (dispatch_init) {
        pthread_cond_destroy(&dispatch.cv);
        pthread_mutex_destroy(&dispatch.mu);
    }
    free(workers);
    free(results);
    free(worker_cores);
    cleanup_disks(disks);
    free_query_set(&qs);
    free_gt(&gt);
    free(sorted_ids);
    free_meta(&meta);
    return rc;
}
