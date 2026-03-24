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
#include <pthread.h>
#include <sched.h>

#define DIM 128
#define TOPK 10
#define MAGIC_META 0x49564633u

static const char *DEFAULT_META_PATH = "./preprocessing/ivf_output/ivf_meta_1_disk.bin";
static const char *DEFAULT_SORTED_IDS_PATH = "./preprocessing/ivf_output/sorted_vec_ids_1_disk.bin";
static const char *DEFAULT_QUERY_FVECS_PATH = "/home/zhangshujie/ann_nic/sift/sift_query.fvecs";
static const char *DEFAULT_GT_IVECS_PATH = "/home/zhangshujie/ann_nic/sift/sift_groundtruth.ivecs";
static const char *DEFAULT_PCA_MEAN_PATH = "./preprocessing/pca_output/pca_mean.bin";
static const char *DEFAULT_PCA_COMPONENTS_PATH = "./preprocessing/pca_output/pca_components.bin";
static const char *DEFAULT_PCA_EV_PATH = "./preprocessing/pca_output/pca_explained_variance.bin";
static const char *DEFAULT_PCA_META_PATH = "./preprocessing/pca_output/pca_meta.bin";

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
    IoWaiter waiter;
    bool in_use;
    uint64_t lba;
    uint32_t chunk_lbas;
    uint32_t local_base;
} AsyncReadSlot;

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
    const char *cores_spec;
    uint32_t nprobe;
    uint32_t max_queries;
    uint32_t threads;
    uint32_t async_depth;
    bool threads_explicit;
    int base_core;
} Config;

typedef struct {
    const IvfMeta *meta;
    const uint32_t *sorted_ids;
    const float *query;
    const CoarseHit *hits;
    uint32_t hit_begin;
    uint32_t hit_end;
} WorkerTask;

typedef struct {
    pthread_t tid;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t done_cv;
    bool stop;
    bool has_task;
    bool task_done;
    int init_rc;
    int core_id;
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
    uint32_t sector_size;
    uint32_t max_lbas_per_read;
    uint32_t async_depth;
    uint8_t *dma_buf;
    size_t dma_bytes;
    AsyncReadSlot *slots;
    WorkerTask task;
    TopkState topk;
    uint64_t io_us;
    uint64_t compute_us;
    uint64_t candidates;
    uint64_t bundles_read;
} BaselineWorker;

// 获取当前单调时钟的微秒时间戳，用于统计各阶段耗时。
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

// NVMe 读请求完成回调：记录请求是否成功，并通知轮询方结束等待。
static void io_done_cb(void *arg, const struct spdk_nvme_cpl *cpl)
{
    IoWaiter *wt = (IoWaiter *)arg;
    wt->ok = !spdk_nvme_cpl_is_error(cpl);
    wt->done = true;
}

// 向固定大小的 top-k 容器中插入一个候选结果，必要时替换当前最差项。
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

// top-k 排序比较函数：先按距离升序，再按向量 id 升序。
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

// 对 top-k 结果做最终排序，方便后续输出和评估。
static void topk_finalize(TopkState *st)
{
    if (st && st->size > 1) {
        qsort(st->items, st->size, sizeof(st->items[0]), cmp_topk_item);
    }
}

// 计算两个 128 维向量之间的 L2 距离平方。
static float full_l2(const float *x, const float *q)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < DIM; i++) {
        float d = x[i] - q[i];
        s += d * d;
    }
    return s;
}

// 释放 ground truth 集合占用的内存，并清空结构体状态。
static void free_gt_set(GtSet *gt)
{
    if (!gt) {
        return;
    }
    free(gt->ids);
    memset(gt, 0, sizeof(*gt));
}

// 从 .ivecs 文件加载 ground truth top-k 结果，供 recall 评估使用。
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

    // 步骤 1：读取首条记录的维度信息，确认每个查询对应多少个 ground truth id。
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

    // 步骤 2：结合文件总大小反推查询条数，并为全部结果一次性分配内存。
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

    // 步骤 3：逐条校验记录维度并读入每个查询对应的 top-k id。
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

// 计算单个查询的 recall@k，这里按预测结果和 ground truth 的 id 重合数统计。
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

// 解析 IVF 元数据文件，构建运行时使用的聚类布局信息。
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

    // 步骤 1：读取并校验文件头，确认魔数、维度等关键字段合法。
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

    // 步骤 2：按 nlist 分配聚类数组，准备把磁盘布局转成内存结构。
    meta->nlist = meta->header.nlist;
    meta->clusters = (ClusterInfo *)calloc(meta->nlist, sizeof(*meta->clusters));
    if (!meta->clusters) {
        perror("calloc clusters");
        fclose(fp);
        return -1;
    }

    // 步骤 3：逐个读取 cluster 元信息，并维护其在 sorted_ids 中的起始偏移。
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

// 释放 IVF 元数据中动态分配的聚类信息。
static void free_ivf_meta(IvfMeta *meta)
{
    if (!meta) {
        return;
    }
    free(meta->clusters);
    memset(meta, 0, sizeof(*meta));
}

// 加载按 cluster 排序后的向量 id 数组，供磁盘扫描结果回映射原始向量编号。
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

// 对所有聚类中心做粗搜，选出距离查询最近的 nprobe 个 cluster。
static int coarse_search_topn(const float *query, const IvfMeta *meta, uint32_t nprobe, CoarseHit *out_hits)
{
    if (!query || !meta || !out_hits || nprobe == 0 || nprobe > meta->nlist) {
        return -1;
    }

    for (uint32_t i = 0; i < nprobe; i++) {
        out_hits[i].cluster_id = UINT32_MAX;
        out_hits[i].dist = INFINITY;
    }

    // 线性扫描全部 centroid，并维护一个按距离升序排列的固定长度候选列表。
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

// SPDK probe 回调：只接受目标 PCI 地址对应的 NVMe 设备。
static bool probe_cb(void *cb_ctx,
                     const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    DiskCtx *disk = (DiskCtx *)cb_ctx;
    return strcmp(trid->traddr, disk->traddr) == 0;
}

// SPDK attach 回调：记录控制器、命名空间和默认 I/O qpair。
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

// 触发 SPDK 设备探测，并确认目标磁盘已经成功附着。
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

// 释放磁盘上下文中的 qpair 和控制器句柄。
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

// 异步提交一段连续 LBA 的读请求，由 worker 后续轮询 completion 并处理结果。
static int submit_async_read(struct spdk_nvme_ns *ns,
                             struct spdk_nvme_qpair *qpair,
                             AsyncReadSlot *slot,
                             void *buf,
                             uint64_t lba,
                             uint32_t lba_count,
                             uint32_t local_base)
{
    slot->waiter.done = false;
    slot->waiter.ok = 0;
    slot->in_use = true;
    slot->lba = lba;
    slot->chunk_lbas = lba_count;
    slot->local_base = local_base;

    int rc = spdk_nvme_ns_cmd_read(ns,
                                   qpair,
                                   buf,
                                   lba,
                                   lba_count,
                                   io_done_cb,
                                   &slot->waiter,
                                   0);
    if (rc != 0) {
        slot->in_use = false;
        fprintf(stderr, "submit_async_read: submit failed lba=%" PRIu64 " count=%u rc=%d\n",
                lba, lba_count, rc);
        return -1;
    }

    return 0;
}

// 将一个 worker 的局部 top-k 合并进全局 top-k。
static void topk_merge_into(TopkState *dst, const TopkState *src)
{
    if (!dst || !src) {
        return;
    }
    for (uint32_t i = 0; i < src->size; i++) {
        topk_insert(dst, src->items[i].vec_id, src->items[i].dist);
    }
}

// 处理一个 worker 分到的 coarse hit 区间：异步提交读请求并维护局部 top-k。
static int process_hit_range(BaselineWorker *worker)
{
    const WorkerTask *task = &worker->task;
    const IvfMeta *meta = task->meta;
    size_t slot_bytes = (size_t)worker->max_lbas_per_read * (size_t)worker->sector_size;

    memset(&worker->topk, 0, sizeof(worker->topk));
    worker->io_us = 0;
    worker->compute_us = 0;
    worker->candidates = 0;
    worker->bundles_read = 0;
    for (uint32_t s = 0; s < worker->async_depth; s++) {
        worker->slots[s].in_use = false;
    }

    for (uint32_t h = task->hit_begin; h < task->hit_end; h++) {
        if (task->hits[h].cluster_id == UINT32_MAX) {
            continue;
        }

        const ClusterInfo *ci = &meta->clusters[task->hits[h].cluster_id];
        uint32_t remaining_lbas = ci->num_lbas;
        uint64_t cur_lba = ci->start_lba;
        uint32_t local_base = 0;
        uint32_t inflight = 0;
        uint64_t cluster_begin_us = now_us();
        uint64_t cluster_compute_us = 0;

        while (remaining_lbas > 0 || inflight > 0) {
            while (remaining_lbas > 0 && inflight < worker->async_depth) {
                uint32_t slot_idx = 0;
                while (slot_idx < worker->async_depth && worker->slots[slot_idx].in_use) {
                    slot_idx++;
                }
                if (slot_idx >= worker->async_depth) {
                    break;
                }

                uint32_t chunk_lbas = remaining_lbas > worker->max_lbas_per_read
                                          ? worker->max_lbas_per_read
                                          : remaining_lbas;
                uint8_t *slot_buf = worker->dma_buf + (size_t)slot_idx * slot_bytes;
                if (submit_async_read(worker->ns,
                                      worker->qpair,
                                      &worker->slots[slot_idx],
                                      slot_buf,
                                      cur_lba,
                                      chunk_lbas,
                                      local_base) != 0) {
                    return -1;
                }

                inflight++;
                worker->bundles_read += chunk_lbas;
                cur_lba += chunk_lbas;
                remaining_lbas -= chunk_lbas;
                local_base += chunk_lbas * meta->header.vectors_per_lba;
            }

            int cpl_rc = spdk_nvme_qpair_process_completions(worker->qpair, 0);
            if (cpl_rc < 0) {
                fprintf(stderr, "process_hit_range: completion processing failed rc=%d\n", cpl_rc);
                return -1;
            }

            for (uint32_t slot_idx = 0; slot_idx < worker->async_depth; slot_idx++) {
                AsyncReadSlot *slot = &worker->slots[slot_idx];
                if (!slot->in_use || !slot->waiter.done) {
                    continue;
                }
                if (!slot->waiter.ok) {
                    fprintf(stderr,
                            "process_hit_range: async read failed cluster=%u lba=%" PRIu64 " count=%u\n",
                            ci->cluster_id,
                            slot->lba,
                            slot->chunk_lbas);
                    return -1;
                }

                uint64_t compute_begin_us = now_us();
                const uint8_t *slot_buf = worker->dma_buf + (size_t)slot_idx * slot_bytes;
                for (uint32_t b = 0; b < slot->chunk_lbas; b++) {
                    uint32_t base_local_idx = slot->local_base + b * meta->header.vectors_per_lba;
                    for (uint32_t lane = 0; lane < meta->header.vectors_per_lba; lane++) {
                        uint32_t local_idx = base_local_idx + lane;
                        if (local_idx >= ci->num_vectors) {
                            break;
                        }
                        uint32_t sorted_pos = ci->sorted_id_base + local_idx;
                        uint32_t vec_id = task->sorted_ids[sorted_pos];
                        const float *vec = (const float *)(slot_buf + (size_t)b * worker->sector_size +
                                                           (size_t)lane * DIM * sizeof(float));
                        float dist = full_l2(vec, task->query);
                        topk_insert(&worker->topk, vec_id, dist);
                        worker->candidates++;
                    }
                }
                cluster_compute_us += now_us() - compute_begin_us;
                slot->in_use = false;
                inflight--;
            }
        }

        uint64_t cluster_wall_us = now_us() - cluster_begin_us;
        worker->compute_us += cluster_compute_us;
        if (cluster_wall_us > cluster_compute_us) {
            worker->io_us += cluster_wall_us - cluster_compute_us;
        }
    }

    topk_finalize(&worker->topk);
    return 0;
}

// worker 线程主循环：初始化线程私有资源，等待任务，执行扫描并回传结果。
static void *baseline_worker_main(void *arg)
{
    BaselineWorker *worker = (BaselineWorker *)arg;
    size_t slot_bytes = (size_t)worker->max_lbas_per_read * (size_t)worker->sector_size;

    bind_to_core(worker->core_id);

    worker->qpair = spdk_nvme_ctrlr_alloc_io_qpair(worker->ctrlr, NULL, 0);
    if (!worker->qpair) {
        worker->init_rc = -1;
    } else {
        worker->dma_buf = (uint8_t *)spdk_zmalloc(worker->dma_bytes,
                                                  worker->sector_size,
                                                  NULL,
                                                  SPDK_ENV_NUMA_ID_ANY,
                                                  SPDK_MALLOC_DMA);
        if (!worker->dma_buf) {
            spdk_nvme_ctrlr_free_io_qpair(worker->qpair);
            worker->qpair = NULL;
            worker->init_rc = -1;
        } else {
            worker->slots = (AsyncReadSlot *)calloc(worker->async_depth, sizeof(*worker->slots));
            if (!worker->slots) {
                perror("calloc async slots");
                spdk_free(worker->dma_buf);
                worker->dma_buf = NULL;
                spdk_nvme_ctrlr_free_io_qpair(worker->qpair);
                worker->qpair = NULL;
                worker->init_rc = -1;
            } else {
                for (uint32_t i = 0; i < worker->async_depth; i++) {
                    worker->slots[i].in_use = false;
                }
                if (worker->dma_bytes < slot_bytes * worker->async_depth) {
                    worker->init_rc = -1;
                } else {
                    worker->init_rc = 0;
                }
            }
        }
    }

    pthread_mutex_lock(&worker->mu);
    worker->task_done = true;
    pthread_cond_broadcast(&worker->done_cv);
    while (!worker->stop) {
        while (!worker->stop && !worker->has_task) {
            pthread_cond_wait(&worker->cv, &worker->mu);
        }
        if (worker->stop) {
            break;
        }

        worker->has_task = false;
        worker->task_done = false;
        pthread_mutex_unlock(&worker->mu);

        int rc = worker->init_rc;
        if (rc == 0) {
            rc = process_hit_range(worker);
        }

        pthread_mutex_lock(&worker->mu);
        worker->init_rc = rc;
        worker->task_done = true;
        pthread_cond_broadcast(&worker->done_cv);
    }
    pthread_mutex_unlock(&worker->mu);

    free(worker->slots);
    worker->slots = NULL;
    if (worker->dma_buf) {
        spdk_free(worker->dma_buf);
        worker->dma_buf = NULL;
    }
    if (worker->qpair) {
        spdk_nvme_ctrlr_free_io_qpair(worker->qpair);
        worker->qpair = NULL;
    }
    return NULL;
}

// 初始化 worker 线程池，并等待每个线程完成自检。
static int init_workers(BaselineWorker *workers,
                        uint32_t nworkers,
                        struct spdk_nvme_ctrlr *ctrlr,
                        struct spdk_nvme_ns *ns,
                        uint32_t sector_size,
                        uint32_t max_lbas_per_read,
                        uint32_t async_depth,
                        const int *worker_cores)
{
    size_t slot_bytes = (size_t)max_lbas_per_read * (size_t)sector_size;
    size_t dma_bytes = slot_bytes * (size_t)async_depth;

    for (uint32_t i = 0; i < nworkers; i++) {
        BaselineWorker *worker = &workers[i];
        memset(worker, 0, sizeof(*worker));
        pthread_mutex_init(&worker->mu, NULL);
        pthread_cond_init(&worker->cv, NULL);
        pthread_cond_init(&worker->done_cv, NULL);
        worker->core_id = worker_cores ? worker_cores[i] : -1;
        worker->ctrlr = ctrlr;
        worker->ns = ns;
        worker->sector_size = sector_size;
        worker->max_lbas_per_read = max_lbas_per_read;
        worker->async_depth = async_depth;
        worker->dma_bytes = dma_bytes;

        if (pthread_create(&worker->tid, NULL, baseline_worker_main, worker) != 0) {
            fprintf(stderr, "pthread_create failed for worker %u\n", i);
            return -1;
        }

        pthread_mutex_lock(&worker->mu);
        while (!worker->task_done) {
            pthread_cond_wait(&worker->done_cv, &worker->mu);
        }
        int init_rc = worker->init_rc;
        pthread_mutex_unlock(&worker->mu);
        if (init_rc != 0) {
            fprintf(stderr, "worker %u initialization failed\n", i);
            return -1;
        }
    }

    return 0;
}

// 停止并回收全部 worker 线程及其同步原语。
static void destroy_workers(BaselineWorker *workers, uint32_t nworkers)
{
    if (!workers) {
        return;
    }

    for (uint32_t i = 0; i < nworkers; i++) {
        BaselineWorker *worker = &workers[i];
        pthread_mutex_lock(&worker->mu);
        worker->stop = true;
        pthread_cond_signal(&worker->cv);
        pthread_mutex_unlock(&worker->mu);
    }

    for (uint32_t i = 0; i < nworkers; i++) {
        pthread_join(workers[i].tid, NULL);
        pthread_mutex_destroy(&workers[i].mu);
        pthread_cond_destroy(&workers[i].cv);
        pthread_cond_destroy(&workers[i].done_cv);
    }
}

// 为单个查询调度 worker，并归并各线程扫描得到的局部结果。
static int run_query_workers(BaselineWorker *workers,
                             uint32_t nworkers,
                             const IvfMeta *meta,
                             const uint32_t *sorted_ids,
                             const float *query,
                             const CoarseHit *hits,
                             uint32_t nprobe,
                             TopkState *topk,
                             uint64_t *io_us,
                             uint64_t *compute_us,
                             uint64_t *candidates,
                             uint64_t *bundles_read)
{
    if (!workers || nworkers == 0 || !meta || !sorted_ids || !query || !hits || !topk || !io_us || !compute_us || !candidates || !bundles_read) {
        return -1;
    }

    uint32_t active_workers = nworkers < nprobe ? nworkers : nprobe;
    if (active_workers == 0) {
        active_workers = 1;
    }

    // 步骤 1：把 nprobe 个 coarse hit 尽量均匀地切分给可用 worker。
    uint32_t base = 0;
    // 按切分结果为每个 worker 下发本轮查询任务。
    for (uint32_t i = 0; i < active_workers; i++) {
        uint32_t span = nprobe / active_workers + (i < (nprobe % active_workers) ? 1u : 0u);
        BaselineWorker *worker = &workers[i];

        pthread_mutex_lock(&worker->mu);
        worker->init_rc = 0;
        worker->task.meta = meta;
        worker->task.sorted_ids = sorted_ids;
        worker->task.query = query;
        worker->task.hits = hits;
        worker->task.hit_begin = base;
        worker->task.hit_end = base + span;
        worker->task_done = false;
        worker->has_task = true;
        pthread_cond_signal(&worker->cv);
        pthread_mutex_unlock(&worker->mu);
        base += span;
    }

    // 步骤 2：清空全局统计项，准备接收各 worker 的局部结果。
    memset(topk, 0, sizeof(*topk));
    *io_us = 0;
    *compute_us = 0;
    *candidates = 0;
    *bundles_read = 0;

    // 步骤 3：等待 worker 完成，并把局部 top-k/耗时/候选数合并到全局。
    for (uint32_t i = 0; i < active_workers; i++) {
        BaselineWorker *worker = &workers[i];
        pthread_mutex_lock(&worker->mu);
        while (!worker->task_done) {
            pthread_cond_wait(&worker->done_cv, &worker->mu);
        }
        int rc = worker->init_rc;
        pthread_mutex_unlock(&worker->mu);
        if (rc != 0) {
            return -1;
        }

        topk_merge_into(topk, &worker->topk);
        *io_us += worker->io_us;
        *compute_us += worker->compute_us;
        *candidates += worker->candidates;
        *bundles_read += worker->bundles_read;
    }

    topk_finalize(topk);
    return 0;
}

// 打印命令行参数说明。
static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --disk0 0000:e4:00.0 --nprobe 32 [--max-queries 1000] [--threads 20] [--async-depth 4] [--base-core 0] [--cores 0,1,2,3]\n"
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

// 解析命令行参数，填充程序运行配置并做基础合法性检查。
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
    cfg->max_queries = 1;
    cfg->threads = 1;
    cfg->async_depth = 4;
    cfg->base_core = -1;

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
        {"threads", required_argument, 0, 1005},
        {"async-depth", required_argument, 0, 1006},
        {"base-core", required_argument, 0, 1007},
        {"cores", required_argument, 0, 1008},
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
            case 1005:
                cfg->threads = (uint32_t)strtoul(optarg, NULL, 10);
                cfg->threads_explicit = true;
                break;
            case 1006: cfg->async_depth = (uint32_t)strtoul(optarg, NULL, 10); break;
            case 1007: cfg->base_core = (int)strtol(optarg, NULL, 10); break;
            case 1008: cfg->cores_spec = optarg; break;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    if (!cfg->disk_traddr || cfg->nprobe == 0 || cfg->threads == 0 || cfg->async_depth == 0) {
        print_usage(argv[0]);
        return -1;
    }
    if (cfg->base_core < -1) {
        fprintf(stderr, "parse_args: base-core must be >= -1\n");
        return -1;
    }

    return 0;
}

// 程序入口：完成初始化、查询执行、统计输出以及资源清理。
int main(int argc, char **argv)
{
    Config cfg;

    // 步骤 1：解析命令行并初始化 SPDK 运行环境。
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

    // 步骤 2：加载 IVF 元数据、排序后的向量 id、PCA 查询数据和 ground truth。
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

    // 步骤 3：连接目标 NVMe 设备，并创建查询阶段会用到的 worker 线程池。
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

    uint32_t worker_count = cfg.threads;
    int *worker_cores = NULL;
    if (resolve_worker_cores(&cfg, &worker_count, &worker_cores) != 0) {
        cleanup_disk(&disk);
        free_gt_set(&gt);
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }
    if (worker_count > cfg.nprobe) {
        worker_count = cfg.nprobe;
    }
    if (worker_count == 0) {
        worker_count = 1;
    }

    BaselineWorker *workers = (BaselineWorker *)calloc(worker_count, sizeof(*workers));
    if (!workers) {
        perror("calloc workers");
        free(worker_cores);
        cleanup_disk(&disk);
        free_gt_set(&gt);
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }
    if (init_workers(workers, worker_count, disk.ctrlr, disk.ns, disk.sector_size, max_lbas_per_read, cfg.async_depth, worker_cores) != 0) {
        free(workers);
        free(worker_cores);
        cleanup_disk(&disk);
        free_gt_set(&gt);
        free_query_set(&qs);
        free(sorted_ids);
        free_ivf_meta(&meta);
        return 1;
    }

    printf("[baseline] disk=%s sector=%u vectors_per_lba=%u nlist=%u nprobe=%u max_queries=%u threads=%u async_depth=%u max_lbas_per_read=%u\n",
           disk.traddr,
           disk.sector_size,
           meta.header.vectors_per_lba,
           meta.nlist,
           cfg.nprobe,
           max_queries,
           worker_count,
           cfg.async_depth,
           max_lbas_per_read);
    if (worker_cores) {
        printf("[baseline] worker_cores=");
        for (uint32_t i = 0; i < worker_count; i++) {
            printf(i == 0 ? "%d" : ",%d", worker_cores[i]);
        }
        printf("\n");
    }
    fflush(stdout);

    // 步骤 4：逐个查询执行 coarse search 和磁盘扫描，同时累计统计指标。
    uint64_t total_latency_us_sum = 0;
    uint64_t total_coarse_us_sum = 0;
    uint64_t total_io_us_sum = 0;
    uint64_t total_compute_us_sum = 0;
    uint64_t total_candidates_sum = 0;
    uint64_t total_bundles_sum = 0;
    double total_recall_sum = 0.0;
    uint32_t queries_ran = 0;
    uint64_t run_begin_us = now_us();

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
        if (run_query_workers(workers, worker_count, &meta, sorted_ids, query, hits, cfg.nprobe, &topk,
                              &io_us, &compute_us, &candidates, &bundles_read) != 0) {
            fprintf(stderr, "run_query_workers failed for query %u\n", qi);
            free(hits);
            destroy_workers(workers, worker_count);
            free(workers);
            free(worker_cores);
            cleanup_disk(&disk);
            free_gt_set(&gt);
            free_query_set(&qs);
            free(sorted_ids);
            free_ivf_meta(&meta);
            return 1;
        }

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

    uint64_t run_wall_us = now_us() - run_begin_us;

    // 步骤 5：输出汇总结果，并在函数结尾统一释放所有资源。
    if (queries_ran > 0) {
        double qps = 0.0;
        if (run_wall_us > 0) {
            qps = (double)queries_ran * 1000000.0 / (double)run_wall_us;
        }

        printf("[baseline summary] queries=%u nprobe=%u threads=%u total_wall_ms=%.3f qps=%.3f avg_latency_ms=%.3f avg_coarse_ms=%.3f avg_io_ms=%.3f avg_compute_ms=%.3f avg_candidates=%.1f avg_bundles=%.1f avg_recall@%d=%.4f\n",
               queries_ran,
               cfg.nprobe,
               worker_count,
               (double)run_wall_us / 1000.0,
               qps,
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

    destroy_workers(workers, worker_count);
    free(workers);
    free(worker_cores);
    cleanup_disk(&disk);
    free_gt_set(&gt);
    free_query_set(&qs);
    free(sorted_ids);
    free_ivf_meta(&meta);
    return 0;
}
