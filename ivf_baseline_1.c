#define _GNU_SOURCE
#include <spdk/env.h>
#include <spdk/nvme.h>

#include "query_loader.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DIM 128
#define TOPK 10
#define MAGIC_META 0x49564633u

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
    float centroid[DIM];
} ClusterMetaOnDisk;

typedef struct {
    uint32_t cluster_id;
    uint64_t start_lba;
    uint32_t num_vectors;
    uint32_t num_lbas;
    uint32_t sorted_id_base;
    float centroid[DIM];
} ClusterInfo;

typedef struct {
    MetaHeader header;
    ClusterInfo *clusters;
    uint32_t nlist;
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
    const char *disk_traddr;
    uint32_t nprobe;
    uint32_t max_queries;
} Config;

static inline uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
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

static float full_l2(const float *x, const float *q)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < DIM; i++) {
        float d = x[i] - q[i];
        s += d * d;
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
    if (meta->header.dim != DIM || meta->header.shard_dim != DIM) {
        fprintf(stderr, "parse_ivf_meta: dim mismatch dim=%u shard_dim=%u expected=%u\n",
                meta->header.dim, meta->header.shard_dim, DIM);
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

    uint32_t sorted_base = 0;
    for (uint32_t i = 0; i < meta->nlist; i++) {
        ClusterMetaOnDisk cm;
        if (fread(&cm, sizeof(cm), 1, fp) != 1) {
            fprintf(stderr, "parse_ivf_meta: failed to read cluster %u\n", i);
            free(meta->clusters);
            meta->clusters = NULL;
            fclose(fp);
            return -1;
        }
        if (cm.cluster_id != i) {
            fprintf(stderr, "parse_ivf_meta: unsupported cluster id layout idx=%u cluster_id=%u\n",
                    i, cm.cluster_id);
            free(meta->clusters);
            meta->clusters = NULL;
            fclose(fp);
            return -1;
        }
        meta->clusters[i].cluster_id = cm.cluster_id;
        meta->clusters[i].start_lba = cm.start_lba;
        meta->clusters[i].num_vectors = cm.num_vectors;
        meta->clusters[i].num_lbas = cm.num_lbas;
        meta->clusters[i].sorted_id_base = sorted_base;
        memcpy(meta->clusters[i].centroid, cm.centroid, sizeof(cm.centroid));
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
        float dist = full_l2(query, c);
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

static int read_lba_range(DiskCtx *disk, void *buf, uint64_t lba, uint32_t lba_count)
{
    IoWaiter waiter = {.done = false, .ok = 0};
    int rc = spdk_nvme_ns_cmd_read(disk->ns,
                                   disk->qpair,
                                   buf,
                                   lba,
                                   lba_count,
                                   io_done_cb,
                                   &waiter,
                                   0);
    if (rc != 0) {
        fprintf(stderr, "read_lba_range: submit failed lba=%" PRIu64 " count=%u rc=%d\n",
                lba, lba_count, rc);
        return -1;
    }

    while (!waiter.done) {
        int cpl_rc = spdk_nvme_qpair_process_completions(disk->qpair, 0);
        if (cpl_rc < 0) {
            fprintf(stderr, "read_lba_range: completion processing failed rc=%d\n", cpl_rc);
            return -1;
        }
    }

    return waiter.ok ? 0 : -1;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --meta ivf_meta.bin --sorted-ids sorted_vec_ids.bin --queries sift_query.fvecs \\\n"
            "     --gt sift_groundtruth.ivecs --disk0 0000:65:00.0 --nprobe 32 \\\n"
            "     --pca-mean mean.bin --pca-components comp.bin --pca-ev var.bin --pca-meta meta.bin \\\n"
            "     [--max-queries 1]\n",
            prog);
}

static int parse_args(int argc, char **argv, Config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->nprobe = 32;
    cfg->max_queries = 1;

    static struct option long_opts[] = {
        {"meta", required_argument, 0, 'm'},
        {"sorted-ids", required_argument, 0, 's'},
        {"queries", required_argument, 0, 'q'},
        {"gt", required_argument, 0, 'g'},
        {"pca-mean", required_argument, 0, 1000},
        {"pca-components", required_argument, 0, 1001},
        {"pca-ev", required_argument, 0, 1002},
        {"pca-meta", required_argument, 0, 1003},
        {"disk0", required_argument, 0, 'd'},
        {"nprobe", required_argument, 0, 'n'},
        {"max-queries", required_argument, 0, 1004},
        {0, 0, 0, 0}
    };

    int opt = 0;
    int idx = 0;
    while ((opt = getopt_long(argc, argv, "m:s:q:g:d:n:", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'm': cfg->meta_path = optarg; break;
            case 's': cfg->sorted_ids_path = optarg; break;
            case 'q': cfg->query_fvecs_path = optarg; break;
            case 'g': cfg->gt_ivecs_path = optarg; break;
            case 'd': cfg->disk_traddr = optarg; break;
            case 'n': cfg->nprobe = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1000: cfg->pca_mean_path = optarg; break;
            case 1001: cfg->pca_components_path = optarg; break;
            case 1002: cfg->pca_ev_path = optarg; break;
            case 1003: cfg->pca_meta_path = optarg; break;
            case 1004: cfg->max_queries = (uint32_t)strtoul(optarg, NULL, 10); break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    if (!cfg->meta_path || !cfg->sorted_ids_path || !cfg->query_fvecs_path || !cfg->gt_ivecs_path ||
        !cfg->pca_mean_path || !cfg->pca_components_path || !cfg->pca_ev_path || !cfg->pca_meta_path ||
        !cfg->disk_traddr || cfg->nprobe == 0) {
        print_usage(argv[0]);
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    Config cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        return 1;
    }

    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "ivf_baseline_single_disk";
    opts.mem_size = 1024;
    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "spdk_env_init failed\n");
        return 1;
    }

    IvfMeta meta;
    if (parse_ivf_meta(cfg.meta_path, &meta) != 0) {
        return 1;
    }
    if (cfg.nprobe > meta.nlist) {
        fprintf(stderr, "invalid nprobe=%u nlist=%u\n", cfg.nprobe, meta.nlist);
        free_ivf_meta(&meta);
        return 1;
    }

    uint32_t num_sorted_ids = 0;
    uint32_t *sorted_ids = NULL;
    if (load_sorted_ids_bin(cfg.sorted_ids_path, &num_sorted_ids, &sorted_ids) != 0) {
        free_ivf_meta(&meta);
        return 1;
    }
    if (num_sorted_ids != meta.header.num_vectors) {
        fprintf(stderr, "sorted id count mismatch got=%u expected=%u\n", num_sorted_ids, meta.header.num_vectors);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }

    query_set_t qs;
    memset(&qs, 0, sizeof(qs));
    if (prepare_queries_with_pca(cfg.query_fvecs_path,
                                 cfg.pca_mean_path,
                                 cfg.pca_components_path,
                                 cfg.pca_ev_path,
                                 cfg.pca_meta_path,
                                 &qs) != 0) {
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }
    if (qs.dim != DIM) {
        fprintf(stderr, "query dim mismatch got=%u expected=%u\n", qs.dim, DIM);
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }

    GtSet gt;
    if (load_ivecs_topk(cfg.gt_ivecs_path, &gt) != 0) {
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }

    uint32_t max_queries = cfg.max_queries;
    if (max_queries == 0 || max_queries > qs.n_queries) {
        max_queries = qs.n_queries;
    }
    if (max_queries > gt.n_queries) {
        max_queries = gt.n_queries;
    }

    DiskCtx disk = {.traddr = cfg.disk_traddr};
    if (probe_disk(&disk) != 0) {
        free_gt_set(&gt);
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }

    if (disk.sector_size != meta.header.sector_size) {
        fprintf(stderr, "sector size mismatch disk=%u meta=%u\n", disk.sector_size, meta.header.sector_size);
        cleanup_disk(&disk);
        free_gt_set(&gt);
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }

    uint32_t max_lbas_per_read = spdk_nvme_ns_get_max_io_xfer_size(disk.ns) / disk.sector_size;
    if (max_lbas_per_read == 0) {
        max_lbas_per_read = 1;
    }

    size_t dma_bytes = (size_t)max_lbas_per_read * (size_t)disk.sector_size;
    uint8_t *dma_buf = (uint8_t *)spdk_zmalloc(dma_bytes,
                                               disk.sector_size,
                                               NULL,
                                               SPDK_ENV_NUMA_ID_ANY,
                                               SPDK_MALLOC_DMA);
    if (!dma_buf) {
        fprintf(stderr, "spdk_zmalloc failed\n");
        cleanup_disk(&disk);
        free_gt_set(&gt);
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }

    printf("[baseline] disk=%s sector=%u vectors_per_lba=%u nlist=%u nprobe=%u max_queries=%u max_lbas_per_read=%u\n",
           disk.traddr,
           disk.sector_size,
           meta.header.vectors_per_lba,
           meta.nlist,
           cfg.nprobe,
           max_queries,
           max_lbas_per_read);
    fflush(stdout);

    uint64_t total_latency_us_sum = 0;
    uint64_t total_coarse_us_sum = 0;
    uint64_t total_io_us_sum = 0;
    uint64_t total_compute_us_sum = 0;
    uint64_t total_candidates_sum = 0;
    uint64_t total_bundles_sum = 0;
    double total_recall_sum = 0.0;
    uint32_t queries_ran = 0;

    for (uint32_t qi = 0; qi < max_queries; qi++) {
        const float *query = &qs.data[(size_t)qi * (size_t)qs.dim];
        TopkState topk = {0};
        uint64_t total_begin_us = now_us();
        uint64_t io_us = 0;
        uint64_t compute_us = 0;
        uint64_t coarse_begin_us = now_us();

        CoarseHit *hits = (CoarseHit *)calloc(cfg.nprobe, sizeof(*hits));
        if (!hits) {
            perror("calloc hits");
            break;
        }
        if (coarse_search_topn(query, &meta, cfg.nprobe, hits) != 0) {
            fprintf(stderr, "coarse_search_topn failed for query %u\n", qi);
            free(hits);
            break;
        }
        uint64_t coarse_us = now_us() - coarse_begin_us;

        uint64_t candidates = 0;
        uint64_t bundles_read = 0;

        for (uint32_t h = 0; h < cfg.nprobe; h++) {
            if (hits[h].cluster_id == UINT32_MAX) {
                continue;
            }

            const ClusterInfo *ci = &meta.clusters[hits[h].cluster_id];
            uint32_t remaining_lbas = ci->num_lbas;
            uint64_t cur_lba = ci->start_lba;
            uint32_t local_base = 0;

            while (remaining_lbas > 0) {
                uint32_t chunk_lbas = remaining_lbas > max_lbas_per_read ? max_lbas_per_read : remaining_lbas;

                uint64_t io_begin_us = now_us();
                if (read_lba_range(&disk, dma_buf, cur_lba, chunk_lbas) != 0) {
                    fprintf(stderr, "read_lba_range failed query=%u cluster=%u lba=%" PRIu64 " count=%u\n",
                            qi, ci->cluster_id, cur_lba, chunk_lbas);
                    free(hits);
                    spdk_free(dma_buf);
                    cleanup_disk(&disk);
                    free_gt_set(&gt);
                    free_query_set(&qs);
                    free(sorted_ids);
                    free_ivf_meta(&meta);
                    return 1;
                }
                io_us += now_us() - io_begin_us;
                bundles_read += chunk_lbas;

                uint64_t compute_begin_us = now_us();
                for (uint32_t b = 0; b < chunk_lbas; b++) {
                    uint32_t base_local_idx = local_base + b * meta.header.vectors_per_lba;
                    for (uint32_t lane = 0; lane < meta.header.vectors_per_lba; lane++) {
                        uint32_t local_idx = base_local_idx + lane;
                        if (local_idx >= ci->num_vectors) {
                            break;
                        }
                        uint32_t sorted_pos = ci->sorted_id_base + local_idx;
                        uint32_t vec_id = sorted_ids[sorted_pos];
                        const float *vec = (const float *)(dma_buf + (size_t)b * disk.sector_size +
                                                           (size_t)lane * DIM * sizeof(float));
                        float dist = full_l2(vec, query);
                        topk_insert(&topk, vec_id, dist);
                        candidates++;
                    }
                }
                compute_us += now_us() - compute_begin_us;

                cur_lba += chunk_lbas;
                remaining_lbas -= chunk_lbas;
                local_base += chunk_lbas * meta.header.vectors_per_lba;
            }
        }

        topk_finalize(&topk);
        uint64_t total_us = now_us() - total_begin_us;
        double recall10 = recall_at_k_overlap(&topk, &gt, qi, TOPK);

        total_latency_us_sum += total_us;
        total_coarse_us_sum += coarse_us;
        total_io_us_sum += io_us;
        total_compute_us_sum += compute_us;
        total_candidates_sum += candidates;
        total_bundles_sum += bundles_read;
        if (recall10 >= 0.0) {
            total_recall_sum += recall10;
        }
        queries_ran++;

        free(hits);
    }

    if (queries_ran > 0) {
        printf("[baseline summary] queries=%u nprobe=%u avg_latency_ms=%.3f avg_coarse_ms=%.3f avg_io_ms=%.3f avg_compute_ms=%.3f avg_candidates=%.1f avg_bundles=%.1f avg_recall@%d=%.4f\n",
               queries_ran,
               cfg.nprobe,
               (double)total_latency_us_sum / 1000.0 / (double)queries_ran,
               (double)total_coarse_us_sum / 1000.0 / (double)queries_ran,
               (double)total_io_us_sum / 1000.0 / (double)queries_ran,
               (double)total_compute_us_sum / 1000.0 / (double)queries_ran,
               (double)total_candidates_sum / (double)queries_ran,
               (double)total_bundles_sum / (double)queries_ran,
               TOPK,
               total_recall_sum / (double)queries_ran);
        fflush(stdout);
    }

    spdk_free(dma_buf);
    cleanup_disk(&disk);
    free_gt_set(&gt);
    free_query_set(&qs);
    free(sorted_ids);
    free_ivf_meta(&meta);
    return 0;
}
