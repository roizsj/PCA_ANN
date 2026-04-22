#define _GNU_SOURCE
#include <spdk/env.h>
#include <spdk/nvme.h>

#include "coarse_search_module.h"
#include "query_loader.h"

#include <errno.h>
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

#define TOPK 10
#define NUM_SHARDS 4
#define MAGIC_META 0x49564633u

static const char *DEFAULT_META_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_1_disk.bin";
static const char *DEFAULT_SORTED_IDS_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_1_disk.bin";
static const char *DEFAULT_QUERY_FVECS_PATH = "/home/zhangshujie/ann_nic/gist/gist_query.fvecs";
static const char *DEFAULT_GT_IVECS_PATH = "/home/zhangshujie/ann_nic/gist/gist_groundtruth.ivecs";
static const char *DEFAULT_PCA_MEAN_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_mean.bin";
static const char *DEFAULT_PCA_COMPONENTS_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_components.bin";
static const char *DEFAULT_PCA_EV_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_explained_variance.bin";
static const char *DEFAULT_PCA_META_PATH = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_meta.bin";

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t shard_dim;
    uint32_t vectors_per_lba;
    uint32_t nlist;
    uint32_t num_vectors;
    uint32_t sector_size;
    uint64_t base_lba;
} MetaHeader;

typedef struct {
    uint32_t cluster_id;
    uint64_t start_lba;
    uint32_t num_vectors;
    uint32_t num_lbas;
} ClusterMetaOnDisk;

typedef struct {
    uint32_t cluster_id;
    uint64_t start_lba;
    uint32_t num_vectors;
    uint32_t num_lbas;
    uint32_t sorted_id_base;
    float *centroid;
} ClusterInfo;

typedef struct {
    MetaHeader header;
    ClusterInfo *clusters;
    uint32_t nlist;
    float *centroids;
} IvfMeta;

typedef struct {
    uint32_t vec_id;
    float dist;
} TopkItem;

typedef struct {
    TopkItem items[TOPK];
    uint32_t size;
} TopkState;

typedef struct {
    uint32_t n_queries;
    uint32_t k;
    int32_t *ids;
} GtSet;

typedef struct {
    uint32_t cluster_id;
    float dist;
} CoarseHit;

typedef struct {
    volatile bool done;
    int ok;
} IoWaiter;

typedef struct {
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
    const char *traddr;
    uint32_t sector_size;
} DiskCtx;

typedef struct {
    const char *meta_path;
    const char *sorted_ids_path;
    const char *query_fvecs_path;
    const char *gt_ivecs_path;
    const char *pca_mean_path;
    const char *pca_components_path;
    const char *pca_ev_path;
    const char *pca_meta_path;
    const char *disk_traddr[NUM_SHARDS];
    const char *cores_spec;
    const char *coarse_backend;
    uint32_t nprobe;
    uint32_t max_queries;
    uint32_t query_batch_size;
    uint32_t threads;
    bool threads_explicit;
    int base_core;
} Config;

typedef struct {
    int rc;
    uint64_t latency_us;
    uint64_t coarse_us;
    uint64_t io_us;
    uint64_t compute_us;
    uint64_t candidates;
    uint64_t bundles_read;
    uint64_t done_ts_us;
    double recall10;
    uint32_t disk_id;
} QueryResult;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    uint32_t ready_workers;
    uint32_t next_batch_begin[NUM_SHARDS];
    int fatal_rc;
    bool start;
} QueryDispatch;

typedef struct {
    pthread_t tid;
    int init_rc;
    int core_id;
    uint32_t assigned_disk_id;
    struct spdk_nvme_ctrlr *ctrlr[NUM_SHARDS];
    struct spdk_nvme_ns *ns[NUM_SHARDS];
    struct spdk_nvme_qpair *qpair[NUM_SHARDS];
    uint32_t sector_size;
    uint32_t max_lbas_per_read;
    uint8_t *dma_buf[NUM_SHARDS];
    size_t dma_bytes;
    CoarseHit *hits_buf;
    uint32_t *coarse_labels_buf;
    float *coarse_distances_buf;
    const coarse_search_module_t *coarse_module;
    pthread_mutex_t *coarse_mu;
    bool use_faiss_coarse;
    uint32_t nprobe;
    uint32_t max_queries;
    uint32_t query_batch_size;
    const IvfMeta *meta;
    const uint32_t *sorted_ids;
    const float *queries;
    uint32_t query_dim;
    const GtSet *gt;
    QueryResult *results;
    QueryDispatch *dispatch;
} QueryWorker;

static void free_ivf_meta(IvfMeta *meta);

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
        exit(1);
    }
}

static void io_done_cb(void *arg, const struct spdk_nvme_cpl *cpl)
{
    IoWaiter *wt = (IoWaiter *)arg;
    wt->ok = !spdk_nvme_cpl_is_error(cpl);
    wt->done = true;
}

static void topk_insert(TopkState *st, uint32_t vec_id, float dist)
{
    if (st->size < TOPK) {
        st->items[st->size].vec_id = vec_id;
        st->items[st->size].dist = dist;
        st->size++;
        return;
    }

    uint32_t worst = 0;
    for (uint32_t i = 1; i < st->size; i++) {
        if (st->items[i].dist > st->items[worst].dist) {
            worst = i;
        }
    }

    if (dist < st->items[worst].dist) {
        st->items[worst].vec_id = vec_id;
        st->items[worst].dist = dist;
    }
}

static int cmp_topk_item(const void *a, const void *b)
{
    const TopkItem *x = (const TopkItem *)a;
    const TopkItem *y = (const TopkItem *)b;
    if (x->dist < y->dist) return -1;
    if (x->dist > y->dist) return 1;
    if (x->vec_id < y->vec_id) return -1;
    if (x->vec_id > y->vec_id) return 1;
    return 0;
}

static void topk_finalize(TopkState *st)
{
    if (st && st->size > 1) {
        qsort(st->items, st->size, sizeof(st->items[0]), cmp_topk_item);
    }
}

static float topk_worst_dist(const TopkState *st)
{
    if (!st || st->size == 0) {
        return INFINITY;
    }

    float worst = st->items[0].dist;
    for (uint32_t i = 1; i < st->size; i++) {
        if (st->items[i].dist > worst) {
            worst = st->items[i].dist;
        }
    }
    return worst;
}

static float full_l2(const float *x, const float *q, uint32_t dim)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = x[i] - q[i];
        s += d * d;
    }
    return s;
}

static float bounded_l2(const float *x, const float *q, uint32_t dim, float limit)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = x[i] - q[i];
        s += d * d;
        if (s > limit) {
            return s;
        }
    }
    return s;
}

static void free_gt_set(GtSet *gt)
{
    if (!gt) {
        return;
    }
    free(gt->ids);
    memset(gt, 0, sizeof(*gt));
}

static int load_ivecs_topk(const char *path, GtSet *gt)
{
    if (!path || !gt) {
        return -1;
    }

    memset(gt, 0, sizeof(*gt));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen groundtruth");
        return -1;
    }

    int dim = 0;
    if (fread(&dim, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "load_ivecs_topk: failed to read dim from %s\n", path);
        fclose(fp);
        return -1;
    }
    if (dim <= 0) {
        fprintf(stderr, "load_ivecs_topk: invalid dim=%d in %s\n", dim, path);
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        perror("fseek end");
        fclose(fp);
        return -1;
    }

    long fsize = ftell(fp);
    if (fsize < 0) {
        perror("ftell");
        fclose(fp);
        return -1;
    }
    rewind(fp);

    long rec_size = (long)sizeof(int) + (long)dim * (long)sizeof(int32_t);
    if (rec_size <= 0 || fsize % rec_size != 0) {
        fprintf(stderr, "load_ivecs_topk: invalid file size=%ld rec_size=%ld\n", fsize, rec_size);
        fclose(fp);
        return -1;
    }

    uint32_t n_queries = (uint32_t)(fsize / rec_size);
    int32_t *ids = (int32_t *)malloc((size_t)n_queries * (size_t)dim * sizeof(int32_t));
    if (!ids) {
        perror("malloc groundtruth");
        fclose(fp);
        return -1;
    }

    for (uint32_t i = 0; i < n_queries; i++) {
        int row_dim = 0;
        if (fread(&row_dim, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "load_ivecs_topk: failed to read row dim %u\n", i);
            free(ids);
            fclose(fp);
            return -1;
        }
        if (row_dim != dim) {
            fprintf(stderr, "load_ivecs_topk: inconsistent row dim at %u got=%d expected=%d\n",
                    i, row_dim, dim);
            free(ids);
            fclose(fp);
            return -1;
        }
        if (fread(ids + (size_t)i * (size_t)dim, sizeof(int32_t), (size_t)dim, fp) != (size_t)dim) {
            fprintf(stderr, "load_ivecs_topk: failed to read row %u\n", i);
            free(ids);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    gt->n_queries = n_queries;
    gt->k = (uint32_t)dim;
    gt->ids = ids;
    return 0;
}

static double recall_at_k_overlap(const TopkState *pred, const GtSet *gt, uint32_t query_idx, uint32_t k)
{
    if (!pred || !gt || !gt->ids || query_idx >= gt->n_queries || k == 0) {
        return -1.0;
    }

    uint32_t eval_k = k;
    if (eval_k > pred->size) {
        eval_k = pred->size;
    }
    if (eval_k > gt->k) {
        eval_k = gt->k;
    }
    if (eval_k == 0) {
        return 0.0;
    }

    const int32_t *gt_row = gt->ids + (size_t)query_idx * (size_t)gt->k;
    bool gt_used[TOPK] = {false};
    uint32_t hits = 0;

    for (uint32_t i = 0; i < eval_k; i++) {
        int32_t pred_id = (int32_t)pred->items[i].vec_id;
        for (uint32_t j = 0; j < eval_k; j++) {
            if (!gt_used[j] && pred_id == gt_row[j]) {
                gt_used[j] = true;
                hits++;
                break;
            }
        }
    }

    return (double)hits / (double)eval_k;
}

static int parse_ivf_meta(const char *path, IvfMeta *meta)
{
    if (!path || !meta) {
        return -1;
    }

    memset(meta, 0, sizeof(*meta));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen meta");
        return -1;
    }

    if (fread(&meta->header, sizeof(meta->header), 1, fp) != 1) {
        fprintf(stderr, "parse_ivf_meta: failed to read header\n");
        fclose(fp);
        return -1;
    }

    if (meta->header.magic != MAGIC_META) {
        fprintf(stderr, "parse_ivf_meta: bad magic=0x%08x\n", meta->header.magic);
        fclose(fp);
        return -1;
    }
    if (meta->header.dim == 0 || meta->header.shard_dim != meta->header.dim) {
        fprintf(stderr, "parse_ivf_meta: invalid layout dim=%u shard_dim=%u\n",
                meta->header.dim, meta->header.shard_dim);
        fclose(fp);
        return -1;
    }

    meta->nlist = meta->header.nlist;
    meta->clusters = (ClusterInfo *)calloc(meta->nlist, sizeof(*meta->clusters));
    if (!meta->clusters) {
        perror("calloc clusters");
        fclose(fp);
        return -1;
    }
    meta->centroids = (float *)calloc((size_t)meta->nlist * meta->header.dim, sizeof(*meta->centroids));
    if (!meta->centroids) {
        perror("calloc centroids");
        free(meta->clusters);
        meta->clusters = NULL;
        fclose(fp);
        return -1;
    }

    uint32_t sorted_base = 0;
    for (uint32_t i = 0; i < meta->nlist; i++) {
        ClusterMetaOnDisk cm;
        if (fread(&cm.cluster_id, sizeof(cm.cluster_id), 1, fp) != 1 ||
            fread(&cm.start_lba, sizeof(cm.start_lba), 1, fp) != 1 ||
            fread(&cm.num_vectors, sizeof(cm.num_vectors), 1, fp) != 1 ||
            fread(&cm.num_lbas, sizeof(cm.num_lbas), 1, fp) != 1) {
            fprintf(stderr, "parse_ivf_meta: failed to read cluster %u\n", i);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }
        if (cm.cluster_id != i) {
            fprintf(stderr, "parse_ivf_meta: unsupported cluster id layout idx=%u cluster_id=%u\n",
                    i, cm.cluster_id);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }

        float *centroid = meta->centroids + (size_t)i * meta->header.dim;
        if (fread(centroid, sizeof(float), meta->header.dim, fp) != meta->header.dim) {
            fprintf(stderr, "parse_ivf_meta: failed to read centroid %u\n", i);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }

        meta->clusters[i].cluster_id = cm.cluster_id;
        meta->clusters[i].start_lba = cm.start_lba;
        meta->clusters[i].num_vectors = cm.num_vectors;
        meta->clusters[i].num_lbas = cm.num_lbas;
        meta->clusters[i].sorted_id_base = sorted_base;
        meta->clusters[i].centroid = centroid;
        sorted_base += cm.num_vectors;
    }

    fclose(fp);
    return 0;
}

static void free_ivf_meta(IvfMeta *meta)
{
    if (!meta) {
        return;
    }
    free(meta->centroids);
    free(meta->clusters);
    memset(meta, 0, sizeof(*meta));
}

static int load_sorted_ids_bin(const char *path, uint32_t *num_vectors_out, uint32_t **sorted_ids_out)
{
    if (!path || !num_vectors_out || !sorted_ids_out) {
        return -1;
    }

    *num_vectors_out = 0;
    *sorted_ids_out = NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen sorted ids");
        return -1;
    }

    uint32_t num_vectors = 0;
    if (fread(&num_vectors, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "load_sorted_ids_bin: failed to read count\n");
        fclose(fp);
        return -1;
    }

    uint32_t *sorted_ids = (uint32_t *)malloc((size_t)num_vectors * sizeof(uint32_t));
    if (!sorted_ids) {
        perror("malloc sorted ids");
        fclose(fp);
        return -1;
    }

    if (fread(sorted_ids, sizeof(uint32_t), num_vectors, fp) != num_vectors) {
        fprintf(stderr, "load_sorted_ids_bin: failed to read body\n");
        free(sorted_ids);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *num_vectors_out = num_vectors;
    *sorted_ids_out = sorted_ids;
    return 0;
}

static int coarse_search_topn(const float *query, const IvfMeta *meta, uint32_t nprobe, CoarseHit *out_hits)
{
    if (!query || !meta || !out_hits || nprobe == 0 || nprobe > meta->nlist) {
        return -1;
    }

    for (uint32_t i = 0; i < nprobe; i++) {
        out_hits[i].cluster_id = UINT32_MAX;
        out_hits[i].dist = INFINITY;
    }

    for (uint32_t cid = 0; cid < meta->nlist; cid++) {
        const float *c = meta->clusters[cid].centroid;
        float dist = full_l2(query, c, meta->header.dim);
        int pos = -1;
        for (uint32_t i = 0; i < nprobe; i++) {
            if (dist < out_hits[i].dist) {
                pos = (int)i;
                break;
            }
        }
        if (pos >= 0) {
            for (int j = (int)nprobe - 1; j > pos; j--) {
                out_hits[j] = out_hits[j - 1];
            }
            out_hits[pos].cluster_id = cid;
            out_hits[pos].dist = dist;
        }
    }
    return 0;
}

static int coarse_search_worker(QueryWorker *worker, const float *query)
{
    if (!worker || !query || !worker->hits_buf) {
        return -1;
    }

    if (!worker->use_faiss_coarse) {
        return coarse_search_topn(query, worker->meta, worker->nprobe, worker->hits_buf);
    }

    if (!worker->coarse_module || !worker->coarse_labels_buf || !worker->coarse_distances_buf ||
        !worker->coarse_mu) {
        return -1;
    }

    pthread_mutex_lock(worker->coarse_mu);
    int rc = coarse_search_module_search(worker->coarse_module,
                                         query,
                                         worker->nprobe,
                                         worker->coarse_labels_buf,
                                         worker->coarse_distances_buf);
    pthread_mutex_unlock(worker->coarse_mu);
    if (rc != 0) {
        return rc;
    }

    for (uint32_t i = 0; i < worker->nprobe; i++) {
        worker->hits_buf[i].cluster_id = worker->coarse_labels_buf[i];
        worker->hits_buf[i].dist = worker->coarse_distances_buf[i];
    }
    return 0;
}

static bool probe_cb(void *cb_ctx,
                     const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    DiskCtx *disk = (DiskCtx *)cb_ctx;
    return strcmp(trid->traddr, disk->traddr) == 0;
}

static void attach_cb(void *cb_ctx,
                      const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    DiskCtx *disk = (DiskCtx *)cb_ctx;

    if (strcmp(trid->traddr, disk->traddr) != 0) {
        return;
    }

    disk->ctrlr = ctrlr;
    disk->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
    if (!disk->ns || !spdk_nvme_ns_is_active(disk->ns)) {
        fprintf(stderr, "attach_cb: ns inactive for %s\n", disk->traddr);
        exit(1);
    }
    disk->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, NULL, 0);
    if (!disk->qpair) {
        fprintf(stderr, "attach_cb: alloc_io_qpair failed for %s\n", disk->traddr);
        exit(1);
    }
    disk->sector_size = spdk_nvme_ns_get_sector_size(disk->ns);
}

static int probe_disk(DiskCtx *disk)
{
    if (spdk_nvme_probe(NULL, disk, probe_cb, attach_cb, NULL) != 0) {
        fprintf(stderr, "spdk_nvme_probe failed\n");
        return -1;
    }
    if (!disk->ctrlr || !disk->ns || !disk->qpair) {
        fprintf(stderr, "probe_disk: failed to attach %s\n", disk->traddr);
        return -1;
    }
    return 0;
}

static void cleanup_disk(DiskCtx *disk)
{
    if (!disk) {
        return;
    }
    if (disk->qpair) {
        spdk_nvme_ctrlr_free_io_qpair(disk->qpair);
        disk->qpair = NULL;
    }
    if (disk->ctrlr) {
        spdk_nvme_detach(disk->ctrlr);
        disk->ctrlr = NULL;
    }
    disk->ns = NULL;
}

typedef struct {
    uint32_t next_hit;
    const ClusterInfo *cluster;
    uint64_t cur_lba;
    uint32_t remaining_lbas;
    uint32_t local_base;
    uint32_t active_cluster_id;
    uint32_t active_lba_count;
    uint32_t active_local_base;
    IoWaiter waiter;
    bool active;
    bool done;
} DiskScanState;

static bool load_next_cluster(DiskScanState *st,
                              const IvfMeta *meta,
                              const CoarseHit *hits,
                              uint32_t nprobe)
{
    while (st->next_hit < nprobe) {
        uint32_t hit_idx = st->next_hit++;
        if (hits[hit_idx].cluster_id == UINT32_MAX) {
            continue;
        }

        const ClusterInfo *ci = &meta->clusters[hits[hit_idx].cluster_id];
        if (ci->num_lbas == 0) {
            continue;
        }

        st->cluster = ci;
        st->cur_lba = ci->start_lba;
        st->remaining_lbas = ci->num_lbas;
        st->local_base = 0;
        return true;
    }

    st->cluster = NULL;
    st->done = true;
    return false;
}

static int submit_next_chunk(DiskScanState *st,
                             const IvfMeta *meta,
                             const CoarseHit *hits,
                             uint32_t nprobe,
                             struct spdk_nvme_ns *ns,
                             struct spdk_nvme_qpair *qpair,
                             uint8_t *dma_buf,
                             uint32_t max_lbas_per_read)
{
    if (st->active) {
        return 0;
    }

    while (!st->cluster || st->remaining_lbas == 0) {
        if (!load_next_cluster(st, meta, hits, nprobe)) {
            return 0;
        }
    }

    uint32_t chunk_lbas = st->remaining_lbas > max_lbas_per_read
                              ? max_lbas_per_read
                              : st->remaining_lbas;
    st->waiter.done = false;
    st->waiter.ok = 0;
    st->active_cluster_id = st->cluster->cluster_id;
    st->active_lba_count = chunk_lbas;
    st->active_local_base = st->local_base;

    int rc = spdk_nvme_ns_cmd_read(ns,
                                   qpair,
                                   dma_buf,
                                   st->cur_lba,
                                   chunk_lbas,
                                   io_done_cb,
                                   &st->waiter,
                                   0);
    if (rc != 0) {
        fprintf(stderr,
                "submit_next_chunk: submit failed cluster=%u lba=%" PRIu64 " count=%u rc=%d\n",
                st->cluster->cluster_id,
                st->cur_lba,
                chunk_lbas,
                rc);
        return -1;
    }

    st->active = true;
    st->cur_lba += chunk_lbas;
    st->remaining_lbas -= chunk_lbas;
    st->local_base += chunk_lbas * meta->header.vectors_per_lba;
    return 1;
}

static int process_completed_chunk(const DiskScanState *st,
                                   const IvfMeta *meta,
                                   const uint32_t *sorted_ids,
                                   const float *query,
                                   const uint8_t *dma_buf,
                                   uint32_t sector_size,
                                   TopkState *topk,
                                   uint64_t *compute_us,
                                   uint64_t *candidates,
                                   uint64_t *bundles_read)
{
    const ClusterInfo *ci = &meta->clusters[st->active_cluster_id];
    *bundles_read += st->active_lba_count;

    uint64_t compute_begin_us = now_us();
    for (uint32_t b = 0; b < st->active_lba_count; b++) {
        uint32_t base_local_idx = st->active_local_base + b * meta->header.vectors_per_lba;
        for (uint32_t lane = 0; lane < meta->header.vectors_per_lba; lane++) {
            uint32_t local_idx = base_local_idx + lane;
            if (local_idx >= ci->num_vectors) {
                break;
            }
            uint32_t sorted_pos = ci->sorted_id_base + local_idx;
            uint32_t vec_id = sorted_ids[sorted_pos];
            const float *vec = (const float *)(dma_buf + (size_t)b * sector_size +
                                               (size_t)lane * meta->header.dim * sizeof(float));
            float limit = topk->size < TOPK ? INFINITY : topk_worst_dist(topk);
            float dist = bounded_l2(vec, query, meta->header.dim, limit);
            topk_insert(topk, vec_id, dist);
            (*candidates)++;
        }
    }
    *compute_us += now_us() - compute_begin_us;
    return 0;
}

static int scan_query_single_disk_async(const IvfMeta *meta,
                                        const uint32_t *sorted_ids,
                                        const float *query,
                                        const CoarseHit *hits,
                                        uint32_t nprobe,
                                        struct spdk_nvme_ns *ns,
                                        struct spdk_nvme_qpair *qpair,
                                        uint8_t *dma_buf,
                                        uint32_t sector_size,
                                        uint32_t max_lbas_per_read,
                                        TopkState *topk,
                                        uint64_t *io_us,
                                        uint64_t *compute_us,
                                        uint64_t *candidates,
                                        uint64_t *bundles_read)
{
    if (!meta || !sorted_ids || !query || !hits || !dma_buf || !topk ||
        !io_us || !compute_us || !candidates || !bundles_read) {
        return -1;
    }

    memset(topk, 0, sizeof(*topk));
    *io_us = 0;
    *compute_us = 0;
    *candidates = 0;
    *bundles_read = 0;

    DiskScanState st;
    memset(&st, 0, sizeof(st));

    if (submit_next_chunk(&st,
                          meta,
                          hits,
                          nprobe,
                          ns,
                          qpair,
                          dma_buf,
                          max_lbas_per_read) < 0) {
        return -1;
    }

    while (st.active) {
        uint64_t wait_begin_us = now_us();
        while (!st.waiter.done) {
            int cpl_rc = spdk_nvme_qpair_process_completions(qpair, 0);
            if (cpl_rc < 0) {
                fprintf(stderr, "process_completions failed\n");
                st.waiter.ok = 0;
                st.waiter.done = true;
            }
        }
        *io_us += now_us() - wait_begin_us;

        st.active = false;
        if (!st.waiter.ok) {
            fprintf(stderr,
                    "I/O failed cluster=%u lba_count=%u\n",
                    st.active_cluster_id,
                    st.active_lba_count);
            return -1;
        }

        if (process_completed_chunk(&st,
                                    meta,
                                    sorted_ids,
                                    query,
                                    dma_buf,
                                    sector_size,
                                    topk,
                                    compute_us,
                                    candidates,
                                    bundles_read) != 0) {
            return -1;
        }

        if (submit_next_chunk(&st,
                              meta,
                              hits,
                              nprobe,
                              ns,
                              qpair,
                              dma_buf,
                              max_lbas_per_read) < 0) {
            return -1;
        }
    }

    topk_finalize(topk);
    return 0;
}

static int run_single_query(QueryWorker *worker, uint32_t query_idx, uint32_t disk_id)
{
    QueryResult *res = &worker->results[query_idx];
    memset(res, 0, sizeof(*res));
    res->rc = -1;
    res->disk_id = disk_id;

    const float *query = worker->queries + (size_t)query_idx * (size_t)worker->query_dim;
    TopkState topk = {0};

    uint64_t total_begin_us = now_us();
    uint64_t coarse_begin_us = now_us();
    if (coarse_search_worker(worker, query) != 0) {
        fprintf(stderr, "coarse_search failed for query %u\n", query_idx);
        return -1;
    }
    res->coarse_us = now_us() - coarse_begin_us;

    if (disk_id >= NUM_SHARDS) {
        return -1;
    }

    if (scan_query_single_disk_async(worker->meta,
                                     worker->sorted_ids,
                                     query,
                                     worker->hits_buf,
                                     worker->nprobe,
                                     worker->ns[disk_id],
                                     worker->qpair[disk_id],
                                     worker->dma_buf[disk_id],
                                     worker->sector_size,
                                     worker->max_lbas_per_read,
                                     &topk,
                                     &res->io_us,
                                     &res->compute_us,
                                     &res->candidates,
                                     &res->bundles_read) != 0) {
        return -1;
    }

    res->latency_us = now_us() - total_begin_us;
    res->done_ts_us = now_us();
    res->recall10 = recall_at_k_overlap(&topk, worker->gt, query_idx, TOPK);
    res->rc = 0;
    return 0;
}

static void *query_worker_main(void *arg)
{
    QueryWorker *worker = (QueryWorker *)arg;

    bind_to_core(worker->core_id);

    worker->init_rc = 0;
    for (int s = 0; s < NUM_SHARDS; s++) {
        worker->qpair[s] = spdk_nvme_ctrlr_alloc_io_qpair(worker->ctrlr[s], NULL, 0);
        if (!worker->qpair[s]) {
            worker->init_rc = -1;
            break;
        }
    }
    if (worker->init_rc == 0) {
        for (int s = 0; s < NUM_SHARDS; s++) {
            worker->dma_buf[s] = (uint8_t *)spdk_zmalloc(worker->dma_bytes,
                                                         worker->sector_size,
                                                         NULL,
                                                         SPDK_ENV_NUMA_ID_ANY,
                                                         SPDK_MALLOC_DMA);
            if (!worker->dma_buf[s]) {
                worker->init_rc = -1;
                break;
            }
        }
        if (worker->init_rc == 0) {
            worker->hits_buf = (CoarseHit *)calloc(worker->nprobe, sizeof(*worker->hits_buf));
            if (!worker->hits_buf) {
                perror("calloc hits_buf");
                worker->init_rc = -1;
            } else {
                if (worker->use_faiss_coarse) {
                    worker->coarse_labels_buf = (uint32_t *)calloc(worker->nprobe, sizeof(*worker->coarse_labels_buf));
                    worker->coarse_distances_buf = (float *)calloc(worker->nprobe, sizeof(*worker->coarse_distances_buf));
                    if (!worker->coarse_labels_buf || !worker->coarse_distances_buf) {
                        perror("calloc faiss coarse buffers");
                        free(worker->coarse_distances_buf);
                        free(worker->coarse_labels_buf);
                        worker->coarse_distances_buf = NULL;
                        worker->coarse_labels_buf = NULL;
                        free(worker->hits_buf);
                        worker->hits_buf = NULL;
                        worker->init_rc = -1;
                    } else {
                        worker->init_rc = 0;
                    }
                } else {
                    worker->init_rc = 0;
                }
            }
        }
    }

    pthread_mutex_lock(&worker->dispatch->mu);
    if (worker->init_rc != 0 && worker->dispatch->fatal_rc == 0) {
        worker->dispatch->fatal_rc = worker->init_rc;
    }
    worker->dispatch->ready_workers++;
    pthread_cond_broadcast(&worker->dispatch->cv);
    while (!worker->dispatch->start && worker->dispatch->fatal_rc == 0) {
        pthread_cond_wait(&worker->dispatch->cv, &worker->dispatch->mu);
    }
    pthread_mutex_unlock(&worker->dispatch->mu);

    if (worker->init_rc == 0) {
        const uint32_t disk_id = worker->assigned_disk_id;
        for (;;) {
            uint32_t batch_begin = UINT32_MAX;
            uint32_t batch_end = UINT32_MAX;

            pthread_mutex_lock(&worker->dispatch->mu);
            if (worker->dispatch->fatal_rc != 0 ||
                worker->dispatch->next_batch_begin[disk_id] >= worker->max_queries) {
                pthread_mutex_unlock(&worker->dispatch->mu);
                break;
            }
            batch_begin = worker->dispatch->next_batch_begin[disk_id];
            worker->dispatch->next_batch_begin[disk_id] += worker->query_batch_size * NUM_SHARDS;
            pthread_mutex_unlock(&worker->dispatch->mu);

            batch_end = batch_begin + worker->query_batch_size;
            if (batch_end > worker->max_queries) {
                batch_end = worker->max_queries;
            }

            for (uint32_t query_idx = batch_begin; query_idx < batch_end; query_idx++) {
                if (run_single_query(worker, query_idx, disk_id) != 0) {
                    pthread_mutex_lock(&worker->dispatch->mu);
                    if (worker->dispatch->fatal_rc == 0) {
                        worker->dispatch->fatal_rc = -1;
                    }
                    pthread_mutex_unlock(&worker->dispatch->mu);
                    break;
                }
            }

            pthread_mutex_lock(&worker->dispatch->mu);
            bool fatal = worker->dispatch->fatal_rc != 0;
            pthread_mutex_unlock(&worker->dispatch->mu);
            if (fatal) {
                break;
            }
        }
    }

    free(worker->hits_buf);
    worker->hits_buf = NULL;
    free(worker->coarse_labels_buf);
    worker->coarse_labels_buf = NULL;
    free(worker->coarse_distances_buf);
    worker->coarse_distances_buf = NULL;
    for (int s = 0; s < NUM_SHARDS; s++) {
        if (worker->dma_buf[s]) {
            spdk_free(worker->dma_buf[s]);
            worker->dma_buf[s] = NULL;
        }
    }
    for (int s = 0; s < NUM_SHARDS; s++) {
        if (worker->qpair[s]) {
            spdk_nvme_ctrlr_free_io_qpair(worker->qpair[s]);
            worker->qpair[s] = NULL;
        }
    }
    return NULL;
}

static int parse_core_list(const char *spec, int **cores_out, uint32_t *count_out)
{
    if (!spec || !cores_out || !count_out) {
        return -1;
    }

    *cores_out = NULL;
    *count_out = 0;

    char *tmp = strdup(spec);
    if (!tmp) {
        perror("strdup core list");
        return -1;
    }

    uint32_t count = 0;
    char *saveptr = NULL;
    for (char *tok = strtok_r(tmp, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr)) {
        count++;
    }
    free(tmp);

    if (count == 0) {
        fprintf(stderr, "parse_core_list: empty core list\n");
        return -1;
    }

    int *cores = (int *)malloc((size_t)count * sizeof(*cores));
    if (!cores) {
        perror("malloc core list");
        return -1;
    }

    tmp = strdup(spec);
    if (!tmp) {
        perror("strdup core list");
        free(cores);
        return -1;
    }

    saveptr = NULL;
    uint32_t idx = 0;
    for (char *tok = strtok_r(tmp, ",", &saveptr); tok; tok = strtok_r(NULL, ",", &saveptr)) {
        char *end = NULL;
        long core = strtol(tok, &end, 10);
        if (end == tok || *end != '\0' || core < 0 || core > INT32_MAX) {
            fprintf(stderr, "parse_core_list: invalid core id '%s'\n", tok);
            free(tmp);
            free(cores);
            return -1;
        }
        cores[idx++] = (int)core;
    }
    free(tmp);

    *cores_out = cores;
    *count_out = count;
    return 0;
}

static int resolve_worker_cores(const Config *cfg, uint32_t *worker_count_io, int **cores_out)
{
    if (!cfg || !worker_count_io || !cores_out) {
        return -1;
    }

    *cores_out = NULL;

    if (cfg->cores_spec) {
        uint32_t core_count = 0;
        if (parse_core_list(cfg->cores_spec, cores_out, &core_count) != 0) {
            return -1;
        }
        if (cfg->threads_explicit && *worker_count_io != core_count) {
            fprintf(stderr,
                    "resolve_worker_cores: --threads=%u mismatches --cores count=%u\n",
                    *worker_count_io,
                    core_count);
            free(*cores_out);
            *cores_out = NULL;
            return -1;
        }
        *worker_count_io = core_count;
        return 0;
    }

    if (cfg->base_core >= 0) {
        int *cores = (int *)malloc((size_t)(*worker_count_io) * sizeof(*cores));
        if (!cores) {
            perror("malloc base cores");
            return -1;
        }
        for (uint32_t i = 0; i < *worker_count_io; i++) {
            cores[i] = cfg->base_core + (int)i;
        }
        *cores_out = cores;
    }

    return 0;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --disk0 0000:e4:00.0 --disk1 0000:e5:00.0 --disk2 0000:e6:00.0 --disk3 0000:e7:00.0 \\\n"
            "     --nprobe 32 [--max-queries 1000] [--query-batch-size 32] [--threads 20] [--base-core 0] [--cores 0,1,2,3] [--coarse-backend brute|faiss]\n"
            "\n"
            "Built-in paths:\n"
            "  meta=%s\n"
            "  sorted_ids=%s\n"
            "  queries=%s\n"
            "  gt=%s\n"
            "  pca_mean=%s\n"
            "  pca_components=%s\n"
            "  pca_ev=%s\n"
            "  pca_meta=%s\n",
            prog,
            DEFAULT_META_PATH,
            DEFAULT_SORTED_IDS_PATH,
            DEFAULT_QUERY_FVECS_PATH,
            DEFAULT_GT_IVECS_PATH,
            DEFAULT_PCA_MEAN_PATH,
            DEFAULT_PCA_COMPONENTS_PATH,
            DEFAULT_PCA_EV_PATH,
            DEFAULT_PCA_META_PATH);
}

static int parse_args(int argc, char **argv, Config *cfg)
{
    if (!cfg) {
        return -1;
    }

    memset(cfg, 0, sizeof(*cfg));
    cfg->meta_path = DEFAULT_META_PATH;
    cfg->sorted_ids_path = DEFAULT_SORTED_IDS_PATH;
    cfg->query_fvecs_path = DEFAULT_QUERY_FVECS_PATH;
    cfg->gt_ivecs_path = DEFAULT_GT_IVECS_PATH;
    cfg->pca_mean_path = DEFAULT_PCA_MEAN_PATH;
    cfg->pca_components_path = DEFAULT_PCA_COMPONENTS_PATH;
    cfg->pca_ev_path = DEFAULT_PCA_EV_PATH;
    cfg->pca_meta_path = DEFAULT_PCA_META_PATH;
    cfg->nprobe = 32;
    cfg->max_queries = 0;
    cfg->query_batch_size = 32;
    cfg->threads = 1;
    cfg->base_core = -1;
    cfg->coarse_backend = "brute";

    static struct option long_opts[] = {
        {"meta", required_argument, 0, 'm'},
        {"sorted-ids", required_argument, 0, 's'},
        {"queries", required_argument, 0, 'q'},
        {"gt", required_argument, 0, 'g'},
        {"pca-mean", required_argument, 0, 1000},
        {"pca-components", required_argument, 0, 1001},
        {"pca-ev", required_argument, 0, 1002},
        {"pca-meta", required_argument, 0, 1003},
        {"disk0", required_argument, 0, 1100},
        {"disk1", required_argument, 0, 1101},
        {"disk2", required_argument, 0, 1102},
        {"disk3", required_argument, 0, 1103},
        {"nprobe", required_argument, 0, 'n'},
        {"max-queries", required_argument, 0, 1004},
        {"threads", required_argument, 0, 1005},
        {"base-core", required_argument, 0, 1006},
        {"cores", required_argument, 0, 1007},
        {"coarse-backend", required_argument, 0, 1008},
        {"query-batch-size", required_argument, 0, 1009},
        {0, 0, 0, 0}
    };

    int opt = 0;
    int idx = 0;
    while ((opt = getopt_long(argc, argv, "m:s:q:g:n:", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'm': cfg->meta_path = optarg; break;
            case 's': cfg->sorted_ids_path = optarg; break;
            case 'q': cfg->query_fvecs_path = optarg; break;
            case 'g': cfg->gt_ivecs_path = optarg; break;
            case 'n': cfg->nprobe = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1000: cfg->pca_mean_path = optarg; break;
            case 1001: cfg->pca_components_path = optarg; break;
            case 1002: cfg->pca_ev_path = optarg; break;
            case 1003: cfg->pca_meta_path = optarg; break;
            case 1004: cfg->max_queries = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1005:
                cfg->threads = (uint32_t)strtoul(optarg, NULL, 10);
                cfg->threads_explicit = true;
                break;
            case 1006:
                cfg->base_core = (int)strtol(optarg, NULL, 10);
                break;
            case 1007:
                cfg->cores_spec = optarg;
                break;
            case 1008:
                cfg->coarse_backend = optarg;
                break;
            case 1009:
                cfg->query_batch_size = (uint32_t)strtoul(optarg, NULL, 10);
                break;
            case 1100: cfg->disk_traddr[0] = optarg; break;
            case 1101: cfg->disk_traddr[1] = optarg; break;
            case 1102: cfg->disk_traddr[2] = optarg; break;
            case 1103: cfg->disk_traddr[3] = optarg; break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    if (!cfg->disk_traddr[0] || !cfg->disk_traddr[1] || !cfg->disk_traddr[2] ||
        !cfg->disk_traddr[3] || cfg->nprobe == 0 || cfg->threads == 0 ||
        cfg->query_batch_size == 0) {
        print_usage(argv[0]);
        return -1;
    }

    if (cfg->base_core < -1) {
        fprintf(stderr, "parse_args: base-core must be >= -1\n");
        return -1;
    }
    if (strcmp(cfg->coarse_backend, "brute") != 0 &&
        strcmp(cfg->coarse_backend, "faiss") != 0) {
        fprintf(stderr, "parse_args: coarse-backend must be 'brute' or 'faiss' (got=%s)\n",
                cfg->coarse_backend);
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    int rc = 1;
    Config cfg;
    IvfMeta meta;
    uint32_t *sorted_ids = NULL;
    query_set_t qs;
    GtSet gt;
    DiskCtx disks[NUM_SHARDS];
    QueryWorker *workers = NULL;
    QueryResult *results = NULL;
    QueryDispatch dispatch;
    int *worker_cores = NULL;
    uint32_t worker_count = 0;
    uint32_t created_workers = 0;
    uint32_t worker_count_by_disk[NUM_SHARDS] = {0};
    uint32_t max_queries = 0;
    uint32_t queries_ran = 0;
    uint64_t total_latency_us_sum = 0;
    uint64_t total_coarse_us_sum = 0;
    uint64_t total_io_us_sum = 0;
    uint64_t total_compute_us_sum = 0;
    uint64_t total_candidates_sum = 0;
    uint64_t total_bundles_sum = 0;
    uint64_t run_wall_us = 0;
    double total_recall_sum = 0.0;
    uint32_t num_sorted_ids = 0;
    uint32_t disk_query_counts[NUM_SHARDS] = {0};
    uint64_t run_begin_us = 0;
    bool dispatch_inited = false;
    coarse_search_module_t *coarse_module = NULL;
    pthread_mutex_t coarse_mu;
    bool coarse_mu_inited = false;
    bool use_faiss_coarse = false;

    memset(&meta, 0, sizeof(meta));
    memset(&qs, 0, sizeof(qs));
    memset(&gt, 0, sizeof(gt));
    memset(disks, 0, sizeof(disks));
    memset(&dispatch, 0, sizeof(dispatch));

    if (parse_args(argc, argv, &cfg) != 0) {
        return 1;
    }

    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "ivf_baseline_4way_raid_gist";
    opts.mem_size = 1024;
    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "spdk_env_init failed\n");
        return 1;
    }

    if (parse_ivf_meta(cfg.meta_path, &meta) != 0) {
        goto cleanup;
    }
    if (cfg.nprobe > meta.nlist) {
        fprintf(stderr, "invalid nprobe=%u nlist=%u\n", cfg.nprobe, meta.nlist);
        goto cleanup;
    }
    use_faiss_coarse = strcmp(cfg.coarse_backend, "faiss") == 0;
    if (use_faiss_coarse) {
        if (coarse_search_module_init(&coarse_module,
                                      meta.centroids,
                                      meta.header.dim,
                                      meta.nlist) != 0) {
            fprintf(stderr, "failed to initialize faiss coarse backend\n");
            goto cleanup;
        }
        if (pthread_mutex_init(&coarse_mu, NULL) != 0) {
            perror("pthread_mutex_init coarse_mu");
            goto cleanup;
        }
        coarse_mu_inited = true;
    }

    if (load_sorted_ids_bin(cfg.sorted_ids_path, &num_sorted_ids, &sorted_ids) != 0) {
        goto cleanup;
    }
    if (num_sorted_ids != meta.header.num_vectors) {
        fprintf(stderr, "sorted id count mismatch got=%u expected=%u\n", num_sorted_ids, meta.header.num_vectors);
        goto cleanup;
    }

    if (prepare_queries_with_pca(cfg.query_fvecs_path,
                                 cfg.pca_mean_path,
                                 cfg.pca_components_path,
                                 cfg.pca_ev_path,
                                 cfg.pca_meta_path,
                                 &qs) != 0) {
        goto cleanup;
    }
    if (qs.dim != meta.header.dim) {
        fprintf(stderr, "query dim mismatch got=%u expected=%u\n", qs.dim, meta.header.dim);
        goto cleanup;
    }

    if (load_ivecs_topk(cfg.gt_ivecs_path, &gt) != 0) {
        goto cleanup;
    }

    max_queries = cfg.max_queries;
    if (max_queries == 0 || max_queries > qs.n_queries) {
        max_queries = qs.n_queries;
    }
    if (max_queries > gt.n_queries) {
        max_queries = gt.n_queries;
    }

    for (int s = 0; s < NUM_SHARDS; s++) {
        disks[s].traddr = cfg.disk_traddr[s];
        if (probe_disk(&disks[s]) != 0) {
            goto cleanup;
        }
        if (disks[s].sector_size != meta.header.sector_size) {
            fprintf(stderr, "sector size mismatch disk%d=%u meta=%u\n",
                    s, disks[s].sector_size, meta.header.sector_size);
            goto cleanup;
        }
    }

    uint32_t max_lbas_per_read = spdk_nvme_ns_get_max_io_xfer_size(disks[0].ns) / disks[0].sector_size;
    if (max_lbas_per_read == 0) {
        max_lbas_per_read = 1;
    }
    for (int s = 1; s < NUM_SHARDS; s++) {
        uint32_t disk_max = spdk_nvme_ns_get_max_io_xfer_size(disks[s].ns) / disks[s].sector_size;
        if (disk_max == 0) {
            disk_max = 1;
        }
        if (disk_max < max_lbas_per_read) {
            max_lbas_per_read = disk_max;
        }
    }

    worker_count = cfg.threads;
    if (resolve_worker_cores(&cfg, &worker_count, &worker_cores) != 0) {
        goto cleanup;
    }
    if (worker_count == 0) {
        worker_count = 1;
    }

    results = (QueryResult *)calloc(max_queries > 0 ? max_queries : 1u, sizeof(*results));
    if (!results) {
        perror("calloc results");
        goto cleanup;
    }

    workers = (QueryWorker *)calloc(worker_count, sizeof(*workers));
    if (!workers) {
        perror("calloc workers");
        goto cleanup;
    }

    pthread_mutex_init(&dispatch.mu, NULL);
    pthread_cond_init(&dispatch.cv, NULL);
    dispatch_inited = true;
    for (uint32_t d = 0; d < NUM_SHARDS; d++) {
        dispatch.next_batch_begin[d] = d * cfg.query_batch_size;
    }

    size_t dma_bytes = (size_t)max_lbas_per_read * (size_t)disks[0].sector_size;
    for (uint32_t i = 0; i < worker_count; i++) {
        QueryWorker *worker = &workers[i];
        worker->core_id = worker_cores ? worker_cores[i] : -1;
        worker->assigned_disk_id = i % NUM_SHARDS;
        worker_count_by_disk[worker->assigned_disk_id]++;
        for (int s = 0; s < NUM_SHARDS; s++) {
            worker->ctrlr[s] = disks[s].ctrlr;
            worker->ns[s] = disks[s].ns;
        }
        worker->sector_size = disks[0].sector_size;
        worker->max_lbas_per_read = max_lbas_per_read;
        worker->dma_bytes = dma_bytes;
        worker->nprobe = cfg.nprobe;
        worker->query_batch_size = cfg.query_batch_size;
        worker->coarse_module = coarse_module;
        worker->coarse_mu = &coarse_mu;
        worker->use_faiss_coarse = use_faiss_coarse;
        worker->max_queries = max_queries;
        worker->meta = &meta;
        worker->sorted_ids = sorted_ids;
        worker->queries = qs.data;
        worker->query_dim = qs.dim;
        worker->gt = &gt;
        worker->results = results;
        worker->dispatch = &dispatch;

        if (pthread_create(&worker->tid, NULL, query_worker_main, worker) != 0) {
            fprintf(stderr, "pthread_create failed for worker %u\n", i);
            pthread_mutex_lock(&dispatch.mu);
            dispatch.fatal_rc = -1;
            dispatch.start = true;
            pthread_cond_broadcast(&dispatch.cv);
            pthread_mutex_unlock(&dispatch.mu);
            created_workers = i;
            goto join_workers;
        }
        created_workers = i + 1;
    }

    pthread_mutex_lock(&dispatch.mu);
    while (dispatch.ready_workers < worker_count && dispatch.fatal_rc == 0) {
        pthread_cond_wait(&dispatch.cv, &dispatch.mu);
    }
    if (dispatch.fatal_rc != 0) {
        dispatch.start = true;
        pthread_cond_broadcast(&dispatch.cv);
        pthread_mutex_unlock(&dispatch.mu);
        goto join_workers;
    }

    run_begin_us = now_us();
    dispatch.start = true;
    pthread_cond_broadcast(&dispatch.cv);
    pthread_mutex_unlock(&dispatch.mu);

join_workers:
    for (uint32_t i = 0; i < created_workers; i++) {
        pthread_join(workers[i].tid, NULL);
    }

    if (dispatch.fatal_rc != 0) {
        fprintf(stderr, "query-parallel baseline failed during worker execution\n");
        goto cleanup;
    }

    {
        uint64_t max_done_ts_us = run_begin_us;
        for (uint32_t qi = 0; qi < max_queries; qi++) {
            if (results[qi].rc != 0) {
                continue;
            }
            queries_ran++;
            if (results[qi].disk_id < NUM_SHARDS) {
                disk_query_counts[results[qi].disk_id]++;
            }
            total_latency_us_sum += results[qi].latency_us;
            total_coarse_us_sum += results[qi].coarse_us;
            total_io_us_sum += results[qi].io_us;
            total_compute_us_sum += results[qi].compute_us;
            total_candidates_sum += results[qi].candidates;
            total_bundles_sum += results[qi].bundles_read;
            if (results[qi].recall10 >= 0.0) {
                total_recall_sum += results[qi].recall10;
            }
            if (results[qi].done_ts_us > max_done_ts_us) {
                max_done_ts_us = results[qi].done_ts_us;
            }
        }
        run_wall_us = max_done_ts_us - run_begin_us;
    }

    printf("[baseline4way-raid] disks=%s,%s,%s,%s sector=%u vectors_per_lba=%u nlist=%u nprobe=%u max_queries=%u query_batch_size=%u threads=%u workers_per_disk=%u,%u,%u,%u max_lbas_per_read=%u coarse_backend=%s\n",
           disks[0].traddr,
           disks[1].traddr,
           disks[2].traddr,
           disks[3].traddr,
           disks[0].sector_size,
           meta.header.vectors_per_lba,
           meta.nlist,
           cfg.nprobe,
           max_queries,
           cfg.query_batch_size,
           worker_count,
           worker_count_by_disk[0],
           worker_count_by_disk[1],
           worker_count_by_disk[2],
           worker_count_by_disk[3],
           max_lbas_per_read,
           cfg.coarse_backend);
    if (worker_cores) {
        printf("[baseline4way-raid] worker_cores=");
        for (uint32_t i = 0; i < worker_count; i++) {
            printf(i == 0 ? "%d" : ",%d", worker_cores[i]);
        }
        printf("\n");
    }
    fflush(stdout);

    if (queries_ran > 0) {
        double qps = 0.0;
        if (run_wall_us > 0) {
            qps = (double)queries_ran * 1000000.0 / (double)run_wall_us;
        }

        printf("[baseline4way-raid summary] queries=%u nprobe=%u query_batch_size=%u threads=%u workers_per_disk=%u,%u,%u,%u total_wall_ms=%.3f qps=%.3f avg_latency_ms=%.3f avg_coarse_ms=%.3f avg_io_ms=%.3f avg_compute_ms=%.3f avg_candidates=%.1f avg_bundles=%.1f avg_recall@%d=%.4f disk_queries=%u,%u,%u,%u\n",
               queries_ran,
               cfg.nprobe,
               cfg.query_batch_size,
               worker_count,
               worker_count_by_disk[0],
               worker_count_by_disk[1],
               worker_count_by_disk[2],
               worker_count_by_disk[3],
               (double)run_wall_us / 1000.0,
               qps,
               (double)total_latency_us_sum / 1000.0 / (double)queries_ran,
               (double)total_coarse_us_sum / 1000.0 / (double)queries_ran,
               (double)total_io_us_sum / 1000.0 / (double)queries_ran,
               (double)total_compute_us_sum / 1000.0 / (double)queries_ran,
               (double)total_candidates_sum / (double)queries_ran,
               (double)total_bundles_sum / (double)queries_ran,
               TOPK,
               total_recall_sum / (double)queries_ran,
               disk_query_counts[0],
               disk_query_counts[1],
               disk_query_counts[2],
               disk_query_counts[3]);
        fflush(stdout);
    } else {
        printf("[baseline4way-raid summary] no queries completed\n");
        fflush(stdout);
    }

    rc = 0;

cleanup:
    if (dispatch_inited) {
        pthread_cond_destroy(&dispatch.cv);
        pthread_mutex_destroy(&dispatch.mu);
    }
    if (coarse_mu_inited) {
        pthread_mutex_destroy(&coarse_mu);
    }
    coarse_search_module_destroy(coarse_module);
    free(workers);
    free(results);
    free(worker_cores);
    for (int s = 0; s < NUM_SHARDS; s++) {
        cleanup_disk(&disks[s]);
    }
    free_gt_set(&gt);
    free_query_set(&qs);
    free(sorted_ids);
    free_ivf_meta(&meta);
    return rc;
}
