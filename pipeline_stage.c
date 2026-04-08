#include "pipeline_stage.h"
#include "coarse_search_module.h"

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

struct io_waiter {
    volatile bool done;
    int ok;
};

struct bundle_run {
    uint32_t cluster_id;
    uint32_t start_bundle;
    uint32_t bundle_count;
    uint16_t begin_idx;
    uint16_t end_idx;
    uint64_t submit_us;
    void *buf;
    struct io_waiter waiter;
    int submit_rc;
    bool submitted;
};

static __thread uint32_t tls_vectors_per_lba_for_sort = 1;

static inline uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static const char *coarse_backend_name(coarse_backend_t backend)
{
    switch (backend) {
        case COARSE_BACKEND_BRUTE:
            return "brute";
        case COARSE_BACKEND_FAISS:
            return "faiss";
        default:
            return "unknown";
    }
}

static coarse_backend_t parse_coarse_backend_name(const char *name)
{
    if (!name || strcmp(name, "faiss") == 0) {
        return COARSE_BACKEND_FAISS;
    }
    if (strcmp(name, "brute") == 0) {
        return COARSE_BACKEND_BRUTE;
    }
    return (coarse_backend_t)-1;
}

static void io_done_cb(void *arg, const struct spdk_nvme_cpl *cpl)
{
    struct io_waiter *wt = (struct io_waiter *)arg;
    wt->ok = !spdk_nvme_cpl_is_error(cpl);
    wt->done = true;
}

static void free_query_tracker_segments(query_tracker_t *qt)
{
    if (!qt) {
        return;
    }

    for (int s = 0; s < NUM_STAGES; s++) {
        free(qt->query_segs[s]);
        qt->query_segs[s] = NULL;
    }
}

int parse_ivf_meta(const char* filename, ivf_meta_t *meta) {
    if (!meta) return -1;
    memset(meta, 0, sizeof(*meta));

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    ivf_meta_header_t header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fprintf(stderr, "Failed to read MetaHeader\n");
        fclose(fp);
        return -1;
    }

    if (header.magic != IVF_META_MAGIC_FLEX) {
        fprintf(stderr, "Invalid magic number: 0x%08X\n", header.magic);
        fclose(fp);
        return -1;
    }

    if (header.version != 1) {
        fprintf(stderr, "Unsupported IVF meta version: %u\n", header.version);
        fclose(fp);
        return -1;
    }

    if (header.num_shards != NUM_STAGES) {
        fprintf(stderr, "Unsupported num_shards: %u expected=%d\n",
                header.num_shards, NUM_STAGES);
        fclose(fp);
        return -1;
    }

    if (header.dim == 0 || header.nlist == 0 || header.vectors_per_lba == 0) {
        fprintf(stderr,
                "Invalid flex IVF header: dim=%u nlist=%u vectors_per_lba=%u\n",
                header.dim, header.nlist, header.vectors_per_lba);
        fclose(fp);
        return -1;
    }

    meta->header = header;
    meta->nlist = header.nlist;

    meta->clusters = malloc(header.nlist * sizeof(cluster_info_t));
    if (!meta->clusters) {
        perror("malloc clusters");
        fclose(fp);
        return -1;
    }

    meta->centroids = malloc((size_t)header.nlist * (size_t)header.dim * sizeof(float));
    if (!meta->centroids) {
        perror("malloc centroids");
        free(meta->clusters);
        meta->clusters = NULL;
        fclose(fp);
        return -1;
    }

    uint32_t sorted_id_base = 0;
    for (uint32_t i = 0; i < header.nlist; i++) {
        ClusterMetaOnDisk cm;

        if (fread(&cm, sizeof(cm), 1, fp) != 1) {
            fprintf(stderr, "Failed to read cluster %u metadata\n", i);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }

        if (cm.cluster_id != i) {
            fprintf(stderr,
                    "parse_ivf_meta: unsupported cluster_id layout at idx=%u cluster_id=%u\n",
                    i, cm.cluster_id);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }

        meta->clusters[i].cluster_id = cm.cluster_id;
        meta->clusters[i].start_lba = cm.start_lba;
        meta->clusters[i].num_vectors = cm.num_vectors;
        meta->clusters[i].num_lbas = cm.num_lbas;
        meta->clusters[i].sorted_id_base = sorted_id_base;
        sorted_id_base += cm.num_vectors;
    }

    {
        size_t want = (size_t)header.nlist * (size_t)header.dim;
        if (fread(meta->centroids, sizeof(float), want, fp) != want) {
            fprintf(stderr, "Failed to read centroids payload from %s\n", filename);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

int load_sorted_ids_bin(const char *path, uint32_t *num_vectors_out, uint32_t **sorted_ids_out)
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

// 释放ivf_meta_t结构体及其内部内存
void free_ivf_meta(ivf_meta_t *meta) {
    if (!meta) return;
    free(meta->clusters);
    free(meta->centroids);
    meta->clusters = NULL;
    meta->centroids = NULL;
    meta->nlist = 0;
    memset(&meta->header, 0, sizeof(meta->header));
}

// 通过qid查找query_tracker_t的辅助函数，要求调用时已经持有app->query_mu锁
static query_tracker_t *find_query_tracker_locked(pipeline_app_t *app, uint64_t qid)
{
    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        if (app->queries[i].qid == qid) {
            return &app->queries[i];
        }
    }
    return NULL;
}

static void maybe_mark_query_done_locked(query_tracker_t *qt)
{
    if (!qt) {
        return;
    }

    if (!qt->done && qt->submission_done && qt->outstanding_batches == 0) {
        qt->done = true;
        qt->done_ts_us = now_us();
    }
}



/* ============================================================
 * 1. 一个非常简单的指针队列
 *
 * 作用：
 *   - 每个 stage worker 有一个输入队列
 *   - topk worker 也有一个输入队列
 *   - 上游 stage 处理完 surviving batch 后，把 batch 指针塞到下游队列
 *
 * 这里为了先跑通功能，使用 pthread mutex + cond 实现。
 * 不追求极致性能，只追求清晰和稳定。
 * ============================================================ */

void queue_init(ptr_queue_t *q) {
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->cv, NULL);
}

void queue_close(ptr_queue_t *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    pthread_cond_broadcast(&q->cv);
    pthread_mutex_unlock(&q->mu);
}

void queue_push(ptr_queue_t *q, void *ptr) {
    qnode_t *n = calloc(1, sizeof(*n));
    if (!n) {
        perror("calloc qnode");
        exit(1);
    }
    n->ptr = ptr;

    pthread_mutex_lock(&q->mu);
    if (q->tail) {
        q->tail->next = n;
    } else {
        q->head = n;
    }
    q->tail = n;
    pthread_cond_signal(&q->cv);
    pthread_mutex_unlock(&q->mu);
}

void *queue_pop(ptr_queue_t *q) {
    pthread_mutex_lock(&q->mu);

    while (!q->head && !q->closed) {
        pthread_cond_wait(&q->cv, &q->mu);
    }

    /* 队列已关闭且为空 */
    if (!q->head) {
        pthread_mutex_unlock(&q->mu);
        return NULL;
    }

    qnode_t *n = q->head;
    q->head = n->next;
    if (!q->head) {
        q->tail = NULL;
    }

    pthread_mutex_unlock(&q->mu);

    void *p = n->ptr;
    free(n);
    return p;
}

/* ============================================================
 * 2. 一些基础 helper
 * ============================================================ */

/* 把当前 pthread 绑定到指定 CPU 核 */
void bind_to_core(int core_id) {
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

/* 计算一个 segment 与 query 对应 segment 的 L2 partial distance */
float partial_l2(const float *x, const float *q, uint32_t dim) {
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float d = x[i] - q[i];
        s += d * d;
    }
    return s;
}

/*
 * 作用：
 *   每个 stage 按自己的轮转游标分发 batch，避免同 cluster 的初始 batch 长时间黏在同一个 lane 上。
 */
static int pick_lane_rr(pipeline_app_t *app, uint32_t stage_id) {
    if (!app || stage_id >= NUM_STAGES) {
        return 0;
    }

    uint32_t worker_count = app->stage_worker_counts[stage_id];
    if (worker_count == 0) {
        return 0;
    }

    uint32_t ticket = __atomic_fetch_add(&app->stage_rr_cursor[stage_id], 1u, __ATOMIC_RELAXED);
    return (int)(ticket % worker_count);
}

/*
 * 根据 cluster_id 在 ivf_meta 中找到对应的 cluster_info_t。
 *
 * ivf_write_disk.c 当前写出的元数据满足 cluster_id == 数组下标，
 * 所以这里直接 O(1) 索引，避免热路径里的线性扫描。
 */
const cluster_info_t *find_cluster_info(const ivf_meta_t *meta, uint32_t cluster_id) {
    if (!meta || !meta->clusters) return NULL;
    if (cluster_id >= meta->nlist) {
        return NULL;
    }

    return &meta->clusters[cluster_id];
}


static int cmp_cand_item_by_bundle(const void *a, const void *b) {
    const cand_item_t *x = (const cand_item_t *)a;
    const cand_item_t *y = (const cand_item_t *)b;

    if (x->cluster_id < y->cluster_id) return -1;
    if (x->cluster_id > y->cluster_id) return 1;

    uint32_t x_bundle = x->local_idx / tls_vectors_per_lba_for_sort;
    uint32_t y_bundle = y->local_idx / tls_vectors_per_lba_for_sort;

    if (x_bundle < y_bundle) return -1;
    if (x_bundle > y_bundle) return 1;

    uint32_t x_lane = x->local_idx % tls_vectors_per_lba_for_sort;
    uint32_t y_lane = y->local_idx % tls_vectors_per_lba_for_sort;

    if (x_lane < y_lane) return -1;
    if (x_lane > y_lane) return 1;

    return 0;
}

/* ============================================================
 * 3. top-k 维护
 *
 * 现在用最简单的“固定容量数组 + 替换当前最差者”的方法。
 * 先不追求复杂 heap。
 * ============================================================ */

static void topk_insert(topk_state_t *st, uint32_t vec_id, float dist) {
    if (st->size < TOPK) {
        st->items[st->size].vec_id = vec_id;
        st->items[st->size].dist = dist;
        st->size++;
    } else {
        int worst = 0;
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
}

/* ============================================================
 * 4. 一个 worker 对自己的盘做一次同步读
 *
 * 当前实现按 LBA(bundle) 读取：
 *   - 每个 worker 拥有自己独占的 qpair
 *   - 提交读请求后，由当前 worker 自己轮询 completion
 *   - 一个 bundle 对应一个 4KB LBA，batch 内同 bundle 的 candidate 复用同一份数据
 * ============================================================ */

static bool plan_bundle_run(const batch_t *in,
                            uint16_t start_idx,
                            uint32_t vectors_per_lba,
                            uint32_t max_bundles_per_read,
                            uint32_t max_gap_bundles,
                            void *buf,
                            struct bundle_run *run)
{
    if (!in || !run || !buf || start_idx >= in->count) {
        return false;
    }

    uint32_t cluster_id = in->items[start_idx].cluster_id;
    uint32_t start_bundle = in->items[start_idx].local_idx / vectors_per_lba;
    uint32_t last_bundle = start_bundle;
    uint32_t bundle_count = 1;
    uint16_t j = start_idx + 1;

    while (j < in->count) {
        uint32_t next_cluster_id = in->items[j].cluster_id;
        uint32_t next_bundle_idx = in->items[j].local_idx / vectors_per_lba;

        if (next_cluster_id != cluster_id) {
            break;
        }

        if (next_bundle_idx <= last_bundle) {
            j++;
            continue;
        }

        uint32_t gap_bundles = next_bundle_idx - last_bundle - 1;
        uint32_t needed_bundle_count = next_bundle_idx - start_bundle + 1;

        if (gap_bundles > max_gap_bundles) {
            break;
        }
        if (needed_bundle_count > max_bundles_per_read) {
            break;
        }

        last_bundle = next_bundle_idx;
        bundle_count = needed_bundle_count;
        j++;
    }

    memset(run, 0, sizeof(*run));
    run->cluster_id = cluster_id;
    run->start_bundle = start_bundle;
    run->bundle_count = bundle_count;
    run->begin_idx = start_idx;
    run->end_idx = j;
    run->buf = buf;
    return true;
}

static int submit_vec_bundle_range(stage_worker_t *w, struct bundle_run *run)
{
    if (!w || !w->app || !w->disk || !w->disk->ns || !w->qpair) {
        fprintf(stderr, "[submit_vec_bundle_range] invalid worker context\n");
        return -1;
    }

    if (!run || !run->buf || run->bundle_count == 0) {
        fprintf(stderr, "[submit_vec_bundle_range] invalid run\n");
        return -1;
    }

    pipeline_app_t *app = w->app;
    ivf_meta_t *meta = &app->ivf_meta;

    const cluster_info_t *ci = find_cluster_info(meta, run->cluster_id);
    if (!ci) {
        fprintf(stderr, "[submit_vec_bundle_range] cluster_id=%u not found\n", run->cluster_id);
        return -1;
    }

    if (run->start_bundle >= ci->num_lbas || run->bundle_count > ci->num_lbas - run->start_bundle) {
        fprintf(stderr,
                "[submit_vec_bundle_range] bundle_idx=%u count=%u out of range cluster=%u num_lbas=%u\n",
                run->start_bundle, run->bundle_count, run->cluster_id, ci->num_lbas);
        return -1;
    }

    uint64_t lba = ci->start_lba + run->start_bundle;

    run->waiter.done = false;
    run->waiter.ok = 0;
    run->submit_us = now_us();
    run->submitted = true;

    int rc = spdk_nvme_ns_cmd_read(
        w->disk->ns,
        w->qpair,
        run->buf,
        lba,
        run->bundle_count,
        io_done_cb,
        &run->waiter,
        0
    );

    if (rc != 0) {
        run->submitted = false;
        fprintf(stderr,
                "[submit_vec_bundle_range] read submit failed rc=%d cluster=%u bundle=%u count=%u lba=%lu\n",
                rc, run->cluster_id, run->start_bundle, run->bundle_count, lba);
        return -1;
    }

    return 0;
}

static int wait_vec_bundle_range(stage_worker_t *w, struct bundle_run *run, uint64_t *io_us)
{
    if (!w || !run || !run->submitted) {
        return -1;
    }

    while (!run->waiter.done) {
        spdk_nvme_qpair_process_completions(w->qpair, 0);
    }

    if (io_us) {
        *io_us = now_us() - run->submit_us;
    }

    run->submitted = false;
    return run->waiter.ok ? 0 : -1;
}
/* ============================================================
 * 5. 把一个 batch 转发给下一个 stage
 *
 * 输入：
 *   - batch->stage 已经被上游设置为“下一个 stage”
 *
 * 行为：
 *   - 选一个 lane
 *   - 塞到对应 stage worker 的输入队列
 * ============================================================ */

static void forward_batch(pipeline_app_t *app, batch_t *b) {
    if (!b || b->count == 0) {
        free(b);
        return;
    }
    
    // 防一下非法stage的出现
    if (b->stage >= NUM_STAGES) {
        fprintf(stderr, "[forward_batch] invalid next stage=%u\n", b->stage);
        free(b);
        return;
    }

    int lane = pick_lane_rr(app, b->stage);
    queue_push(&app->workers[b->stage][lane].inq, b);
}

/* ============================================================
 * 6. topk 线程
 *
 * 作用：
 *   - 接收 stage3 输出的 final batch
 *   - 把里面的 (vec_id, full_dist) 插入 top-k
 *
 * 目前：
 *   - 维护 top-k
 *   - 同时把 top-k batch 也纳入 query 完成检测
 * ============================================================ */

static void *topk_thread_main(void *arg) {
    topk_worker_t *tw = arg;
    bind_to_core(tw->core_id);

    while (1) {
        batch_t *b = queue_pop(&tw->inq);
        if (!b) {
            break;  /* 队列关闭 */
        }

        uint64_t topk_begin_us = now_us();

        if (b->magic != MAGIC_BATCH) {
            fprintf(stderr, "[topk] bad batch magic\n");
            abort();
        }

        pthread_mutex_lock(&tw->app->query_mu);
        query_tracker_t *qt = find_query_tracker_locked(tw->app, b->qid);
        if (qt) {
            for (uint16_t i = 0; i < b->count; i++) {
                topk_insert(&qt->query_topk, b->items[i].vec_id, b->items[i].partial_sum);
            }
            qt->topk_batches++;
            qt->topk_items += b->count;
            qt->topk_wall_us += now_us() - topk_begin_us;
        }
        pthread_mutex_unlock(&tw->app->query_mu);

        mark_batch_finished(tw->app, b->qid, 0);

        free(b);
    }

    return NULL;
}

/* ============================================================
 * 7. stage worker 主循环
 *
 * 一个 stage worker 的职责：
 *   1. 从自己的输入队列取 batch
 *   2. 对 batch 里的每个 vec_id(cluster_id, local_idx)：
 *        - 从本 stage 对应的盘读 segment
 *        - 计算 partial distance
 *        - 与 partial_sum 累加
 *        - 超阈值则 prune
 *        - 未超阈值则进入下游 batch
 *   3. 如果这是 stage3，则进入 topk
 *      否则发到下一 stage
 *
 * 注意：
 *   - 每个 worker 启动时会先为自己创建独占 qpair
 *   - qpair 只由这个 worker 线程自己使用
 * ============================================================ */

static void *stage_worker_main(void *arg) {
    stage_worker_t *w = arg;
    bind_to_core(w->core_id);

    pipeline_app_t *app = w->app;

    /* 每个 worker 自己独占一个 qpair */
    w->qpair = spdk_nvme_ctrlr_alloc_io_qpair(w->disk->ctrlr, NULL, 0);
    if (!w->qpair) {
        fprintf(stderr,
        "alloc_io_qpair failed worker=%d stage=%d lane=%d disk=%s sector=%u\n",
        w->worker_id, w->stage_id, w->lane_id,
        w->disk->traddr, w->disk->sector_size);
        return NULL;
    }

    /*
     * 可调深度的多缓冲：当前按“顺序等待 + 提前提交后续 run”的方式工作。
     * 这样可以在保持处理顺序不变的同时，把 NVMe queue depth 从 2 提高到 read_depth。
     */
    const uint32_t read_depth = app->read_depth;
    const size_t buf_bytes = (size_t)w->disk->sector_size * MAX_BATCH;
    uint32_t max_bundles_per_read = spdk_nvme_ns_get_max_io_xfer_size(w->disk->ns) / w->disk->sector_size;
    if (max_bundles_per_read == 0) {
        max_bundles_per_read = 1;
    }
    if (max_bundles_per_read > MAX_BATCH) {
        max_bundles_per_read = MAX_BATCH;
    }

    void **read_bufs = calloc(read_depth, sizeof(*read_bufs));
    struct bundle_run *run_slots = calloc(read_depth, sizeof(*run_slots));
    if (!read_bufs || !run_slots) {
        fprintf(stderr,
                "worker=%d failed to allocate read-depth state depth=%u\n",
                w->worker_id,
                read_depth);
        free(run_slots);
        free(read_bufs);
        spdk_nvme_ctrlr_free_io_qpair(w->qpair);
        w->qpair = NULL;
        return NULL;
    }

    for (uint32_t i = 0; i < read_depth; i++) {
        read_bufs[i] = spdk_zmalloc(
            buf_bytes,
            4096,
            NULL,
            SPDK_ENV_NUMA_ID_ANY,
            SPDK_MALLOC_DMA
        );
        if (!read_bufs[i]) {
            fprintf(stderr,
                    "spdk_zmalloc failed worker=%d read_buf[%u] depth=%u\n",
                    w->worker_id,
                    i,
                    read_depth);
            for (uint32_t j = 0; j < i; j++) {
                spdk_free(read_bufs[j]);
            }
            free(run_slots);
            free(read_bufs);
            spdk_nvme_ctrlr_free_io_qpair(w->qpair);
            w->qpair = NULL;
            return NULL;
        }
    }

    while (1) {
        batch_t *in = queue_pop(&w->inq);
        if (!in) {
            break;  /* 队列关闭 */
        }

        if (in->magic != MAGIC_BATCH) {
            fprintf(stderr, "[stage %d] bad batch magic\n", w->stage_id);
            abort();
        }

        /*
         * 申请out，它是要发给下一个stage的新batch
         */
        batch_t *out = calloc(1, sizeof(*out));
        if (!out) {
            perror("calloc out batch");
            exit(1);
        }

        out->magic = MAGIC_BATCH;
        out->qid = in->qid;
        out->stage = in->stage + 1;

        // 从query_tracker_t里面读出query和该query的动态阈值
        const float *query_seg = NULL;
        float prune_threshold = app->threshold;
        pthread_mutex_lock(&app->query_mu);
        query_tracker_t *qt_for_query = find_query_tracker_locked(app, in->qid);
        if (qt_for_query) {
            query_seg = qt_for_query->query_segs[w->stage_id];
            if (qt_for_query->prune_threshold > 0.0f) {
                prune_threshold = qt_for_query->prune_threshold;
            }
        }
        pthread_mutex_unlock(&app->query_mu);

        if (!query_seg) {
            fprintf(stderr,
                    "[stage %d lane %d] query seg not found for qid=%lu\n",
                    w->stage_id, w->lane_id, in->qid);
            free(out);
            free(in);
            continue;
        }


        /* batch内profile */
        uint64_t batch_begin_us = now_us();
        uint64_t batch_qsort_us = 0;
        uint64_t batch_io_us = 0;
        uint32_t batch_in = in->count;
        uint32_t batch_pruned = 0;
        uint32_t batch_out = 0;              /* 统一定义为本 stage 保留下来的数量 */
        uint32_t child_batches = 0;
        uint32_t batch_bundles_read = 0;

        if (in->count > 1) {
            uint64_t qsort_begin_us = now_us();
            tls_vectors_per_lba_for_sort = app->ivf_meta.header.vectors_per_lba;
            qsort(in->items, in->count, sizeof(in->items[0]), cmp_cand_item_by_bundle);
            batch_qsort_us += now_us() - qsort_begin_us;
        }

        // -------- 优化版本：按LBA分批读 --------
        // -------- 没加向量计算并行，后续再加 -------
        const uint32_t vectors_per_lba = app->ivf_meta.header.vectors_per_lba;
        const uint32_t stage_dim = app->ivf_meta.header.shard_dims[w->stage_id];
        const uint32_t shard_bytes = app->ivf_meta.header.shard_bytes[w->stage_id];
        const uint32_t max_gap_bundles = (w->stage_id == 1) ? app->stage1_gap_merge_limit : 0u;

        if (in->count > 0) {
            uint16_t next_idx = 0;
            uint32_t inflight = 0;
            uint32_t head = 0;
            uint32_t tail = 0;

            while (next_idx < in->count && inflight < read_depth) {
                struct bundle_run *slot = &run_slots[tail];
                if (!plan_bundle_run(in,
                                     next_idx,
                                     vectors_per_lba,
                                     max_bundles_per_read,
                                     max_gap_bundles,
                                     read_bufs[tail],
                                     slot)) {
                    break;
                }
                slot->submit_rc = submit_vec_bundle_range(w, slot);
                next_idx = slot->end_idx;
                inflight++;
                tail = (tail + 1) % read_depth;
            }

            while (inflight > 0) {
                struct bundle_run *cur_run = &run_slots[head];
                uint64_t run_io_us = 0;
                int cur_io_rc = cur_run->submit_rc;

                if (cur_io_rc == 0) {
                    cur_io_rc = wait_vec_bundle_range(w, cur_run, &run_io_us);
                    batch_io_us += run_io_us;
                    if (cur_io_rc == 0) {
                        batch_bundles_read += cur_run->bundle_count;
                    }
                }

                if (cur_io_rc != 0) {
                    fprintf(stderr,
                            "[stage %d] merged bundle read failed cluster=%u start_bundle=%u bundle_count=%u disk=%s\n",
                            w->stage_id,
                            cur_run->cluster_id,
                            cur_run->start_bundle,
                            cur_run->bundle_count,
                            w->disk->traddr);
                } else {
                    for (uint16_t k = cur_run->begin_idx; k < cur_run->end_idx; k++) {
                        uint32_t vec_id = in->items[k].vec_id;
                        uint32_t cluster_id = in->items[k].cluster_id;
                        uint32_t local_idx = in->items[k].local_idx;
                        float acc = in->items[k].partial_sum;

                        uint32_t bundle_idx = local_idx / vectors_per_lba;
                        uint32_t lane_idx = local_idx % vectors_per_lba;
                        uint32_t bundle_offset = bundle_idx - cur_run->start_bundle;

                        float *seg = (float *)((uint8_t *)cur_run->buf +
                                               (size_t)bundle_offset * w->disk->sector_size +
                                               (size_t)lane_idx * shard_bytes);

                        float part = partial_l2(seg, query_seg, stage_dim);
                        acc += part;

                        if (acc > prune_threshold) {
                            batch_pruned++;
                            continue;
                        }
                        batch_out++;

                        if (w->stage_id == NUM_STAGES - 1) {
                            out->items[out->count].vec_id = vec_id;
                            out->items[out->count].cluster_id = cluster_id;
                            out->items[out->count].local_idx = local_idx;
                            out->items[out->count].partial_sum = acc;
                            out->count++;

                            if (out->count == MAX_BATCH) {
                                queue_push(&app->topk.inq, out);
                                child_batches++;

                                out = calloc(1, sizeof(*out));
                                if (!out) {
                                    perror("calloc final spill batch");
                                    exit(1);
                                }

                                out->magic = MAGIC_BATCH;
                                out->qid = in->qid;
                                out->stage = NUM_STAGES;
                            }
                        } else {
                            out->items[out->count].vec_id = vec_id;
                            out->items[out->count].cluster_id = cluster_id;
                            out->items[out->count].local_idx = local_idx;
                            out->items[out->count].partial_sum = acc;
                            out->count++;

                            if (out->count == MAX_BATCH) {
                                forward_batch(app, out);
                                child_batches++;

                                out = calloc(1, sizeof(*out));
                                if (!out) {
                                    perror("calloc spill batch");
                                    exit(1);
                                }

                                out->magic = MAGIC_BATCH;
                                out->qid = in->qid;
                                out->stage = in->stage + 1;
                            }
                        }
                    }
                }

                inflight--;
                head = (head + 1) % read_depth;

                while (next_idx < in->count && inflight < read_depth) {
                    struct bundle_run *slot = &run_slots[tail];
                    if (!plan_bundle_run(in,
                                         next_idx,
                                         vectors_per_lba,
                                         max_bundles_per_read,
                                         max_gap_bundles,
                                         read_bufs[tail],
                                         slot)) {
                        next_idx = in->count;
                        break;
                    }
                    slot->submit_rc = submit_vec_bundle_range(w, slot);
                    next_idx = slot->end_idx;
                    inflight++;
                    tail = (tail + 1) % read_depth;
                }
            }
        }

        /* 扫完输入 batch 后，把剩余 surviving batch 发出去 */

        if (w->stage_id != NUM_STAGES - 1) {
            if (out->count > 0) {
                forward_batch(app, out);
                child_batches++;
            } else {
                free(out);
            }
        } else {
            if (out->count > 0) {
                queue_push(&app->topk.inq, out);
                child_batches++;
            } else {
                free(out);
            }
        }

        uint64_t batch_wall_us = now_us() - batch_begin_us;

        __atomic_fetch_add(&app->stage_in[w->stage_id], batch_in, __ATOMIC_RELAXED);
        __atomic_fetch_add(&app->stage_out[w->stage_id], batch_out, __ATOMIC_RELAXED);
        __atomic_fetch_add(&app->stage_pruned[w->stage_id], batch_pruned, __ATOMIC_RELAXED);

        /* 先更新 per-query profiling */
        pthread_mutex_lock(&app->query_mu);
        query_tracker_t *qt = find_query_tracker_locked(app, in->qid);
        if (qt) {
            qt->stage_in[w->stage_id] += batch_in;
            qt->stage_out[w->stage_id] += batch_out;
            qt->stage_pruned[w->stage_id] += batch_pruned;
            qt->stage_batches[w->stage_id] += 1;
            qt->stage_bundles_read[w->stage_id] += batch_bundles_read;
            qt->stage_wall_us[w->stage_id] += batch_wall_us;
            qt->stage_io_us[w->stage_id] += batch_io_us;
            qt->stage_qsort_us[w->stage_id] += batch_qsort_us;
        }
        pthread_mutex_unlock(&app->query_mu);

        /* 再结束这个 batch 的生命周期 */
        mark_batch_finished(app, in->qid, child_batches);

        free(in);
    }

    for (uint32_t i = 0; i < read_depth; i++) {
        spdk_free(read_bufs[i]);
    }
    free(run_slots);
    free(read_bufs);
    spdk_nvme_ctrlr_free_io_qpair(w->qpair);
    w->qpair = NULL;
    return NULL;
}

/* ============================================================
 * 8. pipeline_init
 *
 * main.c 需要先准备好：
 *   disks[i].ctrlr
 *   disks[i].ns
 *   disks[i].sector_size
 *
 *   - 把已经 probe 好的 disks 拷贝进 app
 *   - 保存 query / threshold
 *   - 初始化各 worker 的配置和队列
 * ============================================================ */

int pipeline_init(
    pipeline_app_t *app,
    disk_ctx_t disks[NUM_STAGES],
    const uint32_t stage_worker_counts[NUM_STAGES],
    const int stage_cores[NUM_STAGES][MAX_WORKERS_PER_STAGE],
    int topk_core,
    uint32_t read_depth,
    uint32_t stage1_gap_merge_limit,
    const char *coarse_backend_name_arg,
    float threshold,
    const char *ivf_meta_path,
    const char *sorted_ids_path
)
{
    coarse_backend_t backend = parse_coarse_backend_name(coarse_backend_name_arg);

    memset(app, 0, sizeof(*app));

    if (read_depth == 0 || read_depth > MAX_BATCH) {
        fprintf(stderr,
                "pipeline_init: invalid read depth=%u valid_range=[1,%d]\n",
                read_depth,
                MAX_BATCH);
        return -1;
    }
    if (backend != COARSE_BACKEND_BRUTE && backend != COARSE_BACKEND_FAISS) {
        fprintf(stderr,
                "pipeline_init: invalid coarse backend '%s' (expected brute or faiss)\n",
                coarse_backend_name_arg ? coarse_backend_name_arg : "(null)");
        return -1;
    }

    fprintf(stderr,
            "[pipeline_init] begin read_depth=%u stage1_gap_merge_limit=%u coarse_backend=%s ivf_meta=%s sorted_ids=%s\n",
            read_depth,
            stage1_gap_merge_limit,
            coarse_backend_name(backend),
            ivf_meta_path ? ivf_meta_path : "(null)",
            sorted_ids_path ? sorted_ids_path : "(null)");

    for (int i = 0; i < NUM_STAGES; i++) {
        app->disks[i] = disks[i];

        if (!app->disks[i].ctrlr || !app->disks[i].ns) {
            fprintf(stderr, "pipeline_init: disk %d not initialized (traddr=%s)\n",
                    i, app->disks[i].traddr);
            return -1;
        }
    }

    app->threshold = threshold;
    app->read_depth = read_depth;
    app->stage1_gap_merge_limit = stage1_gap_merge_limit;
    app->coarse_backend = backend;

    fprintf(stderr, "[pipeline_init] before parse_ivf_meta\n");
    if (parse_ivf_meta(ivf_meta_path, &app->ivf_meta) != 0) {
        fprintf(stderr, "pipeline_init: failed to load ivf meta from %s\n", ivf_meta_path);
        return -1;
    }

    app->nlist = app->ivf_meta.nlist;
    app->centroids = app->ivf_meta.centroids;

    fprintf(stderr,
            "[pipeline_init] after parse_ivf_meta dim=%u nlist=%u num_vectors=%u sector_size=%u backend=%s\n",
            app->ivf_meta.header.dim,
            app->ivf_meta.nlist,
            app->ivf_meta.header.num_vectors,
            app->ivf_meta.header.sector_size,
            coarse_backend_name(app->coarse_backend));

    if (app->coarse_backend == COARSE_BACKEND_FAISS) {
        fprintf(stderr, "[pipeline_init] before coarse_search_module_init\n");
        if (coarse_search_module_init(&app->coarse_module,
                                      app->centroids,
                                      app->ivf_meta.header.dim,
                                      app->nlist) != 0) {
            fprintf(stderr, "pipeline_init: failed to build Faiss coarse index\n");
            free_ivf_meta(&app->ivf_meta);
            app->centroids = NULL;
            app->nlist = 0;
            return -1;
        }
        fprintf(stderr, "[pipeline_init] after coarse_search_module_init\n");
    } else {
        app->coarse_module = NULL;
        fprintf(stderr, "[pipeline_init] skip coarse_search_module_init for brute backend\n");
    }

    for (int s = 0; s < NUM_STAGES; s++) {
        if (app->disks[s].sector_size != app->ivf_meta.header.sector_size) {
            fprintf(stderr,
                    "pipeline_init: sector size mismatch stage=%d disk=%u meta=%u\n",
                    s,
                    app->disks[s].sector_size,
                    app->ivf_meta.header.sector_size);
            coarse_search_module_destroy(app->coarse_module);
            app->coarse_module = NULL;
            free_ivf_meta(&app->ivf_meta);
            app->centroids = NULL;
            app->nlist = 0;
            return -1;
        }
        if (app->ivf_meta.header.shard_bytes[s] > app->disks[s].sector_size) {
            fprintf(stderr,
                    "pipeline_init: shard bytes exceed sector size stage=%d shard_bytes=%u sector=%u\n",
                    s,
                    app->ivf_meta.header.shard_bytes[s],
                    app->disks[s].sector_size);
            coarse_search_module_destroy(app->coarse_module);
            app->coarse_module = NULL;
            free_ivf_meta(&app->ivf_meta);
            app->centroids = NULL;
            app->nlist = 0;
            return -1;
        }
    }

    fprintf(stderr, "[pipeline_init] before load_sorted_ids_bin\n");
    if (load_sorted_ids_bin(sorted_ids_path, &app->num_sorted_vec_ids, &app->sorted_vec_ids) != 0) {
        fprintf(stderr, "pipeline_init: failed to load sorted ids from %s\n", sorted_ids_path);
        coarse_search_module_destroy(app->coarse_module);
        app->coarse_module = NULL;
        free_ivf_meta(&app->ivf_meta);
        app->centroids = NULL;
        app->nlist = 0;
        return -1;
    }
    fprintf(stderr,
            "[pipeline_init] after load_sorted_ids_bin count=%u expected=%u\n",
            app->num_sorted_vec_ids,
            app->ivf_meta.header.num_vectors);

    if (app->num_sorted_vec_ids != app->ivf_meta.header.num_vectors) {
        fprintf(stderr,
                "pipeline_init: sorted ids count mismatch got=%u expected=%u\n",
                app->num_sorted_vec_ids, app->ivf_meta.header.num_vectors);
        free(app->sorted_vec_ids);
        app->sorted_vec_ids = NULL;
        app->num_sorted_vec_ids = 0;
        coarse_search_module_destroy(app->coarse_module);
        app->coarse_module = NULL;
        free_ivf_meta(&app->ivf_meta);
        app->centroids = NULL;
        app->nlist = 0;
        return -1;
    }

    fprintf(stderr, "[pipeline_init] before worker setup\n");
    pthread_mutex_init(&app->query_mu, NULL);
    memset(app->queries, 0, sizeof(app->queries));

    queue_init(&app->topk.inq);
    app->topk.app = app;
    app->topk.core_id = topk_core;

    int worker_id = 0;
    for (int s = 0; s < NUM_STAGES; s++) {
        uint32_t worker_count = stage_worker_counts[s];
        if (worker_count == 0 || worker_count > MAX_WORKERS_PER_STAGE) {
            fprintf(stderr,
                    "pipeline_init: invalid worker count stage=%d count=%u max=%d\n",
                    s, worker_count, MAX_WORKERS_PER_STAGE);
            free(app->sorted_vec_ids);
            app->sorted_vec_ids = NULL;
            app->num_sorted_vec_ids = 0;
            coarse_search_module_destroy(app->coarse_module);
            app->coarse_module = NULL;
            free_ivf_meta(&app->ivf_meta);
            app->centroids = NULL;
            app->nlist = 0;
            pthread_mutex_destroy(&app->query_mu);
            return -1;
        }

        app->stage_worker_counts[s] = worker_count;
        app->stage_rr_cursor[s] = 0;

        for (uint32_t lane = 0; lane < worker_count; lane++) {
            stage_worker_t *w = &app->workers[s][lane];
            w->app = app;
            w->worker_id = worker_id++;
            w->stage_id = s;
            w->lane_id = (int)lane;
            w->core_id = stage_cores[s][lane];
            w->disk = &app->disks[s];
            w->qpair = NULL;
            queue_init(&w->inq);
        }
    }

    fprintf(stderr, "[pipeline_init] done\n");
    return 0;
}

/* ============================================================
 * 9. 启动所有线程
 * ============================================================ */

void pipeline_start(pipeline_app_t *app) {
    pthread_create(&app->topk.tid, NULL, topk_thread_main, &app->topk);

    for (int s = 0; s < NUM_STAGES; s++) {
        for (uint32_t lane = 0; lane < app->stage_worker_counts[s]; lane++) {
            pthread_create(&app->workers[s][lane].tid, NULL,
                           stage_worker_main, &app->workers[s][lane]);
        }
    }
}

/* ============================================================
 * 10. 提交 stage0 的初始 batch
 * ============================================================ */

void pipeline_submit_initial_batch(pipeline_app_t *app, batch_t *b) {
    if (!b || b->magic != MAGIC_BATCH) {
        fprintf(stderr, "pipeline_submit_initial_batch: bad batch\n");
        abort();
    }

    int lane = pick_lane_rr(app, 0);
    queue_push(&app->workers[0][lane].inq, b);
    // 调试用输出
    // printf("[submit] qid=%lu stage=%u count=%u first_cluster=%u first_local=%u\n",
    //     b->qid, b->stage, b->count,
    //     b->items[0].cluster_id, b->items[0].local_idx);
    // fflush(stdout);
}

/* ============================================================
 * 11. 关闭队列，通知线程退出
 * ============================================================ */

void pipeline_stop(pipeline_app_t *app) {
    for (int s = 0; s < NUM_STAGES; s++) {
        for (uint32_t lane = 0; lane < app->stage_worker_counts[s]; lane++) {
            queue_close(&app->workers[s][lane].inq);
        }
    }
}

/* ============================================================
 * 12. 等待所有线程退出
 * ============================================================ */

void pipeline_join(pipeline_app_t *app) {
    /* 1. 先等所有 stage worker 退出 */
    for (int s = 0; s < NUM_STAGES; s++) {
        for (uint32_t lane = 0; lane < app->stage_worker_counts[s]; lane++) {
            pthread_join(app->workers[s][lane].tid, NULL);
        }
    }

    /* 2. 现在不会再有新的 final batch 进入 topk 了，再关闭 topk 队列 */
    queue_close(&app->topk.inq);

    /* 3. 最后等 topk 线程退出 */
    pthread_join(app->topk.tid, NULL);
}

/* ============================================================
 * 13. 清理内存空间
 * ============================================================ */
void pipeline_destroy(pipeline_app_t *app) {
    if (!app) return;

    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        free_query_tracker_segments(&app->queries[i]);
    }

    coarse_search_module_destroy(app->coarse_module);
    app->coarse_module = NULL;
    free_ivf_meta(&app->ivf_meta);
    app->centroids = NULL;
    app->nlist = 0;

    for (int s = 0; s < NUM_STAGES; s++) {
        for (uint32_t lane = 0; lane < app->stage_worker_counts[s]; lane++) {
            pthread_mutex_destroy(&app->workers[s][lane].inq.mu);
            pthread_cond_destroy(&app->workers[s][lane].inq.cv);
        }
    }

    pthread_mutex_destroy(&app->topk.inq.mu);
    pthread_cond_destroy(&app->topk.inq.cv);
    free(app->sorted_vec_ids);
    app->sorted_vec_ids = NULL;
    app->num_sorted_vec_ids = 0;
    pthread_mutex_destroy(&app->query_mu);

    memset(&app->topk_state, 0, sizeof(app->topk_state));
}

/* ============================================================
 * 14.query helper
 * ============================================================ */
// 注册一个新的 query，初始化query_tracker_t并返回其指针
query_tracker_t *register_query(pipeline_app_t *app,
                                uint64_t qid,
                                uint32_t nprobe,
                                uint32_t num_probed_clusters)
{
    if (!app || qid == 0) {
        return NULL;
    }

    pthread_mutex_lock(&app->query_mu);

    /* 先防重复 qid */
    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        if (app->queries[i].qid == qid) {
            pthread_mutex_unlock(&app->query_mu);
            fprintf(stderr, "register_query: duplicate qid=%lu\n", qid);
            return NULL;
        }
    }

    /* 找空槽 */
    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        if (app->queries[i].qid == 0) {
            query_tracker_t *qt = &app->queries[i];
            memset(qt, 0, sizeof(*qt));

            qt->qid = qid;
            qt->nprobe = nprobe;
            qt->num_probed_clusters = num_probed_clusters;
            qt->initial_candidates = 0;
            qt->submitted_batches = 0;
            qt->completed_batches = 0;
            qt->outstanding_batches = 0;
            qt->max_outstanding_batches = 0;
            qt->submission_done = false;
            qt->done = false;
            qt->prune_threshold = 0.0f;
            qt->coarse_search_us = 0;
            qt->submit_candidates_us = 0;
            memset(qt->stage_in, 0, sizeof(qt->stage_in));
            memset(qt->stage_out, 0, sizeof(qt->stage_out));
            memset(qt->stage_pruned, 0, sizeof(qt->stage_pruned));
            memset(qt->stage_batches, 0, sizeof(qt->stage_batches));
            memset(qt->stage_bundles_read, 0, sizeof(qt->stage_bundles_read));
            memset(qt->stage_wall_us, 0, sizeof(qt->stage_wall_us));
            memset(qt->stage_io_us, 0, sizeof(qt->stage_io_us));
            memset(qt->stage_qsort_us, 0, sizeof(qt->stage_qsort_us));
            qt->topk_batches = 0;
            qt->topk_items = 0;
            qt->topk_wall_us = 0;
            memset(&qt->query_topk, 0, sizeof(qt->query_topk));

            for (int s = 0; s < NUM_STAGES; s++) {
                uint32_t seg_dim = app->ivf_meta.header.shard_dims[s];
                qt->query_segs[s] = (float *)calloc(seg_dim, sizeof(float));
                if (!qt->query_segs[s]) {
                    perror("calloc query_seg");
                    free_query_tracker_segments(qt);
                    memset(qt, 0, sizeof(*qt));
                    pthread_mutex_unlock(&app->query_mu);
                    return NULL;
                }
            }

            qt->submit_ts_us = now_us();
            qt->done_ts_us = 0;

            pthread_mutex_unlock(&app->query_mu);
            return qt;
        }
    }

    pthread_mutex_unlock(&app->query_mu);
    fprintf(stderr, "register_query: no free query slot\n");
    return NULL;
}

// 一个 work item 完成后调用一次；spawned_batches 表示它向下游新产生了多少 batch
void mark_batch_finished(pipeline_app_t *app, uint64_t qid, uint32_t spawned_batches)
{
    if (!app || qid == 0) {
        return;
    }

    pthread_mutex_lock(&app->query_mu);

    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        query_tracker_t *qt = &app->queries[i];
        if (qt->qid == qid) {
            qt->completed_batches++;

            if (qt->outstanding_batches == 0) {
                pthread_mutex_unlock(&app->query_mu);
                fprintf(stderr, "mark_batch_finished: qid=%lu outstanding underflow\n", qid);
                return;
            }

            qt->outstanding_batches = qt->outstanding_batches - 1 + spawned_batches;
            if (qt->outstanding_batches > qt->max_outstanding_batches) {
                qt->max_outstanding_batches = qt->outstanding_batches;
            }

            maybe_mark_query_done_locked(qt);

            pthread_mutex_unlock(&app->query_mu);
            return;
        }
    }

    pthread_mutex_unlock(&app->query_mu);
    fprintf(stderr, "mark_batch_finished: qid=%lu not found\n", qid);
}


// 暴力搜top nprobe个聚类
int coarse_search_topn(const float *query,
                       const float *centroids,
                       uint32_t dim,
                       uint32_t nlist,
                       uint32_t nprobe,
                       coarse_hit_t *out_hits)
{
    if (!query || !centroids || !out_hits || dim == 0 || nprobe == 0 || nprobe > nlist) {
        return -1;
    }

    /* 初始化 top-nprobe 为 +inf */
    for (uint32_t i = 0; i < nprobe; i++) {
        out_hits[i].cluster_id = UINT32_MAX;
        out_hits[i].dist = __FLT_MAX__;
    }

    for (uint32_t cid = 0; cid < nlist; cid++) {
        const float *c = centroids + (size_t)cid * dim;

        float dist = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float diff = query[d] - c[d];
            dist += diff * diff;
        }

        /* 插入 top-nprobe（最简单的线性插入） */
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

// 等待query结束
int wait_query_done(pipeline_app_t *app, uint64_t qid, uint32_t timeout_ms)
{
    if (!app || qid == 0) {
        return -1;
    }

    const uint32_t sleep_us = 1000; /* 1ms */
    uint32_t waited_ms = 0;

    while (timeout_ms == 0 || waited_ms < timeout_ms) {
        pthread_mutex_lock(&app->query_mu);

        for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
            query_tracker_t *qt = &app->queries[i];
            if (qt->qid == qid) {
                bool done = qt->done;
                pthread_mutex_unlock(&app->query_mu);
                if (done) {
                    return 0;
                }
                goto not_done;
            }
        }

        pthread_mutex_unlock(&app->query_mu);
        fprintf(stderr, "wait_query_done: qid=%lu not found\n", qid);
        return -1;

not_done:
        usleep(sleep_us);
        waited_ms += 1;
    }

    return 1; /* timeout */
}

// 把一个cluster里面的candidate切成batches
int submit_cluster_candidates(pipeline_app_t *app,
                              uint64_t qid,
                              uint32_t cluster_id,
                              uint32_t max_batch)
{
    if (!app || qid == 0 || max_batch == 0) {
        return -1;
    }

    const cluster_info_t *ci = find_cluster_info(&app->ivf_meta, cluster_id);
    if (!ci) {
        fprintf(stderr, "submit_cluster_candidates: cluster %u not found\n", cluster_id);
        return -1;
    }

    uint32_t num_vec = ci->num_vectors;
    uint32_t local_idx = 0;
    uint32_t batches_submitted = 0;
    uint32_t sorted_base = ci->sorted_id_base;

    if (!app->sorted_vec_ids || sorted_base + num_vec > app->num_sorted_vec_ids) {
        fprintf(stderr,
                "submit_cluster_candidates: bad sorted id range cluster=%u base=%u num_vec=%u total=%u\n",
                cluster_id, sorted_base, num_vec, app->num_sorted_vec_ids);
        return -1;
    }

    while (local_idx < num_vec) {
        batch_t *b = (batch_t *)calloc(1, sizeof(*b));
        if (!b) {
            perror("calloc batch");
            return -1;
        }

        b->magic = MAGIC_BATCH;
        b->qid = qid;
        b->stage = 0;

        while (local_idx < num_vec && b->count < max_batch) {
            b->items[b->count].vec_id = app->sorted_vec_ids[sorted_base + local_idx];
            b->items[b->count].cluster_id = cluster_id;
            b->items[b->count].local_idx = local_idx;
            b->items[b->count].partial_sum = 0.0f;
            b->count++;
            local_idx++;
        }

        if (b->count == 0) {
            free(b);
            break;
        }

        pipeline_submit_initial_batch(app, b);
        batches_submitted++;
    }

    pthread_mutex_lock(&app->query_mu);
    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        query_tracker_t *qt = &app->queries[i];
        if (qt->qid == qid) {
            qt->initial_candidates += num_vec;
            qt->submitted_batches += batches_submitted;
            qt->outstanding_batches += batches_submitted;
            if (qt->outstanding_batches > qt->max_outstanding_batches) {
                qt->max_outstanding_batches = qt->outstanding_batches;
            }
            pthread_mutex_unlock(&app->query_mu);
            return 0;
        }
    }
    pthread_mutex_unlock(&app->query_mu);

    fprintf(stderr, "submit_cluster_candidates: qid=%lu not registered\n", qid);
    return -1;
}


// 这是外部接口，main.c会调用它来提交一个query
int submit_query(pipeline_app_t *app,
                 uint64_t qid,
                 const float *query,
                 uint32_t nprobe)
{
    if (!app || !query || qid == 0) {
        return -1;
    }

    if (!app->centroids || app->nlist == 0) {
        fprintf(stderr, "submit_query: coarse search state not ready\n");
        return -1;
    }
    if (app->coarse_backend == COARSE_BACKEND_FAISS && !app->coarse_module) {
        fprintf(stderr, "submit_query: coarse module not ready for faiss backend\n");
        return -1;
    }

    if (nprobe == 0 || nprobe > app->nlist) {
        fprintf(stderr, "submit_query: invalid nprobe=%u nlist=%u\n",
                nprobe, app->nlist);
        return -1;
    }

    /* 1) 注册 query tracker */
    // TODO 这里有个隐患，注册放在coerce_search的前面，如果coarse_search失败了，tracker就白注册了。后续可以改成先coarse_search，成功后再注册tracker。
    query_tracker_t *qt = register_query(app, qid, nprobe, nprobe);
    if (!qt) {
        fprintf(stderr, "submit_query: register_query failed qid=%lu\n", qid);
        return -1;
    }

    /* 2) 切 query segments */
    for (int s = 0; s < NUM_STAGES; s++) {
        uint32_t seg_dim = app->ivf_meta.header.shard_dims[s];
        uint32_t seg_offset = app->ivf_meta.header.shard_offsets[s];
        memcpy(qt->query_segs[s],
               query + seg_offset,
               (size_t)seg_dim * sizeof(float));
    }

    /* 3) coarse search 选 top-nprobe clusters */
    uint64_t coarse_begin_us = now_us();
    coarse_hit_t *hits = (coarse_hit_t *)calloc(nprobe, sizeof(*hits));
    if (!hits) {
        perror("calloc coarse hits");
        return -1;
    }

    if (app->coarse_backend == COARSE_BACKEND_FAISS) {
        uint32_t *hit_labels = (uint32_t *)calloc(nprobe, sizeof(*hit_labels));
        float *hit_distances = (float *)calloc(nprobe, sizeof(*hit_distances));
        if (!hit_labels || !hit_distances) {
            perror("calloc coarse search buffers");
            free(hit_distances);
            free(hit_labels);
            free(hits);
            return -1;
        }

        if (coarse_search_module_search(app->coarse_module,
                                        query,
                                        nprobe,
                                        hit_labels,
                                        hit_distances) != 0) {
            fprintf(stderr, "submit_query: coarse_search_module_search failed\n");
            free(hit_distances);
            free(hit_labels);
            free(hits);
            return -1;
        }

        for (uint32_t i = 0; i < nprobe; i++) {
            hits[i].cluster_id = hit_labels[i];
            hits[i].dist = hit_distances[i];
        }
        free(hit_distances);
        free(hit_labels);
    } else {
        if (coarse_search_topn(query,
                               app->centroids,
                               app->ivf_meta.header.dim,
                               app->nlist,
                               nprobe,
                               hits) != 0) {
            fprintf(stderr, "submit_query: coarse_search_topn failed\n");
            free(hits);
            return -1;
        }
    }
    qt->coarse_search_us = now_us() - coarse_begin_us;

    uint32_t threshold_rank = nprobe < 50 ? nprobe : 50; // CAUTION!!!! threshold setting
    if (threshold_rank > 0) {
        qt->prune_threshold = hits[threshold_rank - 1].dist;
    } else {
        qt->prune_threshold = app->threshold;
    }

    printf("[submit_query] qid=%lu nprobe=%u prune_threshold=%f rank=%u\n",
           qid, nprobe, qt->prune_threshold, threshold_rank);

    /* 4) 依次提交每个命中 cluster 的 candidates */
    uint64_t submit_begin_us = now_us();
    for (uint32_t i = 0; i < nprobe; i++) {
        uint32_t cluster_id = hits[i].cluster_id;

        if (cluster_id == UINT32_MAX) {
            continue;
        }

        if (submit_cluster_candidates(app, qid, cluster_id, MAX_BATCH) != 0) {
            fprintf(stderr,
                    "submit_query: submit_cluster_candidates failed qid=%lu cluster=%u\n",
                    qid, cluster_id);
            free(hits);
            return -1;
        }
    }
    qt->submit_candidates_us = now_us() - submit_begin_us;

    free(hits);

    /* 5) 检查至少提交了一个batch */
    pthread_mutex_lock(&app->query_mu);
    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        query_tracker_t *x = &app->queries[i];
        if (x->qid == qid) {
            uint32_t submitted = x->submitted_batches;
            x->submission_done = true;
            maybe_mark_query_done_locked(x);
            pthread_mutex_unlock(&app->query_mu);
            if (submitted == 0) {
                fprintf(stderr, "submit_query: qid=%lu submitted_batches=0\n", qid);
                return -1;
            }
            return 0;
        }
    }
    pthread_mutex_unlock(&app->query_mu);

    fprintf(stderr, "submit_query: tracker lost for qid=%lu\n", qid);
    return -1;
}
