#include "pipeline_stage.h"
#include "coarse_search_module.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
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
    uint64_t complete_us;
    void *buf;
    struct io_waiter waiter;
    int submit_rc;
    bool submitted;
    bool completion_seen;
};

struct worker_io_state {
    uint32_t read_depth;
    uint32_t max_bundles_per_read;
    void **read_bufs;
    struct bundle_run *run_slots;
    uint16_t *ready_slots;
};

struct worker_query_ctx {
    const float *query_seg;
    float prune_threshold;
};

struct worker_batch_stats {
    uint64_t batch_begin_us;
    uint64_t batch_qsort_us;
    uint64_t batch_io_us;
    uint32_t batch_in;
    uint32_t batch_pruned;
    uint32_t batch_out;
    uint32_t child_batches;
    uint32_t batch_bundles_read;
    uint32_t batch_nvme_reads;
    uint64_t batch_nvme_read_bytes;
};

static __thread uint32_t tls_vectors_per_lba_for_sort = 1;

static uint32_t stage_vectors_per_lba(const pipeline_app_t *app, uint32_t stage_id)
{
    if (!app || stage_id >= NUM_STAGES) {
        return 1;
    }
    uint32_t v = app->ivf_meta.header.shard_vectors_per_lba[stage_id];
    return v ? v : app->ivf_meta.header.vectors_per_lba;
}

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

static const char *prune_threshold_mode_name(prune_threshold_mode_t mode)
{
    switch (mode) {
        case PRUNE_THRESHOLD_CENTROID:
            return "centroid";
        case PRUNE_THRESHOLD_SAMPLED:
            return "sampled";
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

static prune_threshold_mode_t parse_prune_threshold_mode_name(const char *name)
{
    if (!name || strcmp(name, "sampled") == 0) {
        return PRUNE_THRESHOLD_SAMPLED;
    }
    if (strcmp(name, "centroid") == 0) {
        return PRUNE_THRESHOLD_CENTROID;
    }
    return (prune_threshold_mode_t)-1;
}

static void io_done_cb(void *arg, const struct spdk_nvme_cpl *cpl)
{
    struct io_waiter *wt = (struct io_waiter *)arg;
    wt->ok = !spdk_nvme_cpl_is_error(cpl);
    wt->done = true;
}

static int read_lba_sync(struct spdk_nvme_ns *ns,
                         struct spdk_nvme_qpair *qpair,
                         void *buf,
                         uint64_t lba,
                         uint32_t lba_count)
{
    struct io_waiter waiter = {.done = false, .ok = 0};
    int rc = spdk_nvme_ns_cmd_read(ns, qpair, buf, lba, lba_count, io_done_cb, &waiter, 0);
    if (rc != 0) {
        return rc;
    }

    while (!waiter.done) {
        int cpl_rc = spdk_nvme_qpair_process_completions(qpair, 0);
        if (cpl_rc < 0) {
            return cpl_rc;
        }
    }

    return waiter.ok ? 0 : -1;
}

static int build_prune_sample_vectors(pipeline_app_t *app)
{
    if (!app || !app->sorted_vec_ids || app->num_sorted_vec_ids == 0) {
        return -1;
    }

    const uint32_t total = app->ivf_meta.header.num_vectors;
    const uint32_t dim = app->ivf_meta.header.dim;
    const uint32_t sample_count = total < PRUNE_SAMPLE_SIZE ? total : PRUNE_SAMPLE_SIZE;
    struct spdk_nvme_qpair *qpairs[NUM_STAGES] = {0};
    void *sector_bufs[NUM_STAGES] = {0};
    int rc = -1;

    if (sample_count == 0 || dim == 0 || app->ivf_meta.header.vectors_per_lba == 0) {
        return -1;
    }

    app->prune_sample_vectors =
        (float *)calloc((size_t)sample_count * (size_t)dim, sizeof(float));
    if (!app->prune_sample_vectors) {
        perror("calloc prune_sample_vectors");
        return -1;
    }
    app->prune_sample_count = sample_count;

    for (uint32_t s = 0; s < app->active_stages; s++) {
        uint32_t disk_idx = app->stage_disk_indices[s][0];
        qpairs[s] = spdk_nvme_ctrlr_alloc_io_qpair(app->disks[disk_idx].ctrlr, NULL, 0);
        if (!qpairs[s]) {
            fprintf(stderr, "build_prune_sample_vectors: alloc_io_qpair failed stage=%u disk=%s\n",
                    s, app->disks[disk_idx].traddr);
            goto cleanup;
        }
        sector_bufs[s] = spdk_zmalloc(app->disks[disk_idx].sector_size,
                                      app->disks[disk_idx].sector_size,
                                      NULL,
                                      SPDK_ENV_NUMA_ID_ANY,
                                      SPDK_MALLOC_DMA);
        if (!sector_bufs[s]) {
            fprintf(stderr, "build_prune_sample_vectors: spdk_zmalloc failed stage=%u\n", s);
            goto cleanup;
        }
    }

    {
        uint32_t cluster_idx = 0;
        for (uint32_t i = 0; i < sample_count; i++) {
            uint64_t sorted_pos = ((uint64_t)i * (uint64_t)total) / (uint64_t)sample_count;
            while (cluster_idx + 1 < app->ivf_meta.nlist &&
                   sorted_pos >= (uint64_t)app->ivf_meta.clusters[cluster_idx].sorted_id_base +
                                     (uint64_t)app->ivf_meta.clusters[cluster_idx].num_vectors) {
                cluster_idx++;
            }

            const cluster_info_t *ci = &app->ivf_meta.clusters[cluster_idx];
            if (sorted_pos < ci->sorted_id_base ||
                sorted_pos >= (uint64_t)ci->sorted_id_base + (uint64_t)ci->num_vectors) {
                fprintf(stderr,
                        "build_prune_sample_vectors: sorted_pos mapping failed pos=%" PRIu64 " cluster=%u base=%u num=%u\n",
                        sorted_pos, cluster_idx, ci->sorted_id_base, ci->num_vectors);
                goto cleanup;
            }

            uint32_t local_idx = (uint32_t)(sorted_pos - (uint64_t)ci->sorted_id_base);
            float *dst = app->prune_sample_vectors + (size_t)i * (size_t)dim;

            for (uint32_t s = 0; s < app->active_stages; s++) {
                uint32_t disk_idx = app->stage_disk_indices[s][0];
                uint32_t vectors_per_lba = stage_vectors_per_lba(app, s);
                uint32_t bundle_idx = local_idx / vectors_per_lba;
                uint32_t lane_idx = local_idx % vectors_per_lba;
                uint64_t lba = ci->start_lba + bundle_idx;
                if (read_lba_sync(app->disks[disk_idx].ns, qpairs[s], sector_bufs[s], lba, 1) != 0) {
                    fprintf(stderr,
                            "build_prune_sample_vectors: read failed stage=%u disk=%u cluster=%u lba=%" PRIu64 "\n",
                            s, disk_idx, ci->cluster_id, lba);
                    goto cleanup;
                }

                memcpy((uint8_t *)dst + app->ivf_meta.header.shard_offsets[s] * sizeof(float),
                       (uint8_t *)sector_bufs[s] + (size_t)lane_idx * app->ivf_meta.header.shard_bytes[s],
                       app->ivf_meta.header.shard_bytes[s]);
            }
        }
    }

    rc = 0;

cleanup:
    for (uint32_t s = 0; s < app->active_stages; s++) {
        if (sector_bufs[s]) {
            spdk_free(sector_bufs[s]);
        }
        if (qpairs[s]) {
            spdk_nvme_ctrlr_free_io_qpair(qpairs[s]);
        }
    }

    if (rc != 0) {
        free(app->prune_sample_vectors);
        app->prune_sample_vectors = NULL;
        app->prune_sample_count = 0;
    }

    return rc;
}

static float compute_sampled_prune_threshold(const pipeline_app_t *app, const float *query)
{
    if (!app || !query || !app->prune_sample_vectors || app->prune_sample_count == 0) {
        return app ? app->threshold : 0.0f;
    }

    const uint32_t dim = app->ivf_meta.header.dim;
    const uint32_t k = app->prune_sample_count < PRUNE_SAMPLE_TOPK
                           ? app->prune_sample_count
                           : PRUNE_SAMPLE_TOPK;
    float topk_buf[PRUNE_SAMPLE_TOPK];

    for (uint32_t i = 0; i < k; i++) {
        topk_buf[i] = INFINITY;
    }

    for (uint32_t i = 0; i < app->prune_sample_count; i++) {
        const float *vec = app->prune_sample_vectors + (size_t)i * (size_t)dim;
        float dist = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float diff = vec[d] - query[d];
            dist += diff * diff;
        }

        int pos = -1;
        for (uint32_t j = 0; j < k; j++) {
            if (dist < topk_buf[j]) {
                pos = (int)j;
                break;
            }
        }
        if (pos >= 0) {
            for (int j = (int)k - 1; j > pos; j--) {
                topk_buf[j] = topk_buf[j - 1];
            }
            topk_buf[pos] = dist;
        }
    }

    return k > 0 ? topk_buf[k - 1] : app->threshold;
}

static float compute_centroid_prune_threshold(const coarse_hit_t *hits,
                                              uint32_t nprobe,
                                              float fallback_threshold)
{
    if (!hits || nprobe == 0) {
        return fallback_threshold;
    }

    uint32_t threshold_rank = nprobe < 50 ? nprobe : 50;
    if (threshold_rank == 0) {
        return fallback_threshold;
    }

    float threshold = hits[threshold_rank - 1].dist;
    if (!(threshold > 0.0f) || !isfinite(threshold)) {
        threshold = fallback_threshold;
    }

    return threshold;
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

static int collect_completed_runs(stage_worker_t *w,
                                  struct bundle_run *run_slots,
                                  uint32_t read_depth,
                                  uint16_t *ready_slots,
                                  uint32_t *ready_tail,
                                  uint32_t *ready_count)
{
    if (!w || !run_slots || !ready_slots || !ready_tail || !ready_count) {
        return -1;
    }

    int cpl_rc = spdk_nvme_qpair_process_completions(w->qpair, 0);
    if (cpl_rc < 0) {
        return cpl_rc;
    }

    uint64_t complete_ts_us = now_us();
    for (uint32_t i = 0; i < read_depth; i++) {
        struct bundle_run *slot = &run_slots[i];
        if (!slot->submitted || slot->completion_seen || !slot->waiter.done) {
            continue;
        }

        slot->completion_seen = true;
        slot->complete_us = complete_ts_us;
        ready_slots[*ready_tail] = (uint16_t)i;
        *ready_tail = (*ready_tail + 1) % read_depth;
        (*ready_count)++;
    }

    return 0;
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
    if (b->stage >= app->active_stages) {
        fprintf(stderr, "[forward_batch] invalid next stage=%u\n", b->stage);
        free(b);
        return;
    }

    int lane = pick_lane_rr(app, b->stage);
    queue_push(&app->workers[b->stage][lane].inq, b);
}

static batch_t *alloc_stage_out_batch(uint64_t qid, uint8_t stage)
{
    batch_t *out = (batch_t *)calloc(1, sizeof(*out));
    if (!out) {
        perror("calloc out batch");
        exit(1);
    }
    out->magic = MAGIC_BATCH;
    out->qid = qid;
    out->stage = stage;
    return out;
}

static void emit_out_batch(pipeline_app_t *app,
                           batch_t *out,
                           uint32_t *child_batches)
{
    if (!app || !out || out->count == 0) {
        return;
    }

    if (out->stage >= app->active_stages) {
        queue_push(&app->topk.inq, out);
    } else {
        forward_batch(app, out);
    }

    if (child_batches) {
        (*child_batches)++;
    }
}

static batch_t *append_bundle_group_to_out(pipeline_app_t *app,
                                           batch_t *out,
                                           const cand_item_t *bundle_items,
                                           uint16_t bundle_count,
                                           uint32_t *child_batches)
{
    if (!out || !bundle_items || bundle_count == 0) {
        return out;
    }

    if (out->count > 0 && out->count + bundle_count > MAX_BATCH) {
        emit_out_batch(app, out, child_batches);
        out = alloc_stage_out_batch(out->qid, out->stage);
    }

    if (bundle_count > MAX_BATCH) {
        fprintf(stderr, "append_bundle_group_to_out: bundle_count=%u exceeds MAX_BATCH=%u\n",
                bundle_count, MAX_BATCH);
        abort();
    }

    memcpy(&out->items[out->count], bundle_items, (size_t)bundle_count * sizeof(bundle_items[0]));
    out->count += bundle_count;
    return out;
}

static int stage_worker_alloc_qpair(stage_worker_t *w)
{
    w->qpair = spdk_nvme_ctrlr_alloc_io_qpair(w->disk->ctrlr, NULL, 0);
    if (!w->qpair) {
        fprintf(stderr,
                "alloc_io_qpair failed worker=%d stage=%d lane=%d disk=%s sector=%u\n",
                w->worker_id,
                w->stage_id,
                w->lane_id,
                w->disk->traddr,
                w->disk->sector_size);
        return -1;
    }
    return 0;
}

static uint32_t stage_worker_max_bundles_per_read(const pipeline_app_t *app,
                                                  const stage_worker_t *w)
{
    uint32_t max_bundles_per_read =
        spdk_nvme_ns_get_max_io_xfer_size(w->disk->ns) / w->disk->sector_size;

    if (max_bundles_per_read == 0) {
        max_bundles_per_read = 1;
    }
    if (app->max_read_lbas > 0 && app->max_read_lbas < max_bundles_per_read) {
        max_bundles_per_read = app->max_read_lbas;
    }
    if (w->stage_id == 0 &&
        app->stage0_max_read_lbas > 0 &&
        app->stage0_max_read_lbas < max_bundles_per_read) {
        max_bundles_per_read = app->stage0_max_read_lbas;
    }
    if (max_bundles_per_read > MAX_BATCH) {
        max_bundles_per_read = MAX_BATCH;
    }

    return max_bundles_per_read;
}

static int stage_worker_init_io_state(stage_worker_t *w, struct worker_io_state *io)
{
    const pipeline_app_t *app = w->app;
    const size_t buf_bytes = (size_t)w->disk->sector_size * MAX_BATCH;

    memset(io, 0, sizeof(*io));
    io->read_depth = app->read_depth;
    io->max_bundles_per_read = stage_worker_max_bundles_per_read(app, w);
    io->read_bufs = calloc(io->read_depth, sizeof(*io->read_bufs));
    io->run_slots = calloc(io->read_depth, sizeof(*io->run_slots));
    io->ready_slots = calloc(io->read_depth, sizeof(*io->ready_slots));
    if (!io->read_bufs || !io->run_slots || !io->ready_slots) {
        fprintf(stderr,
                "worker=%d failed to allocate read-depth state depth=%u\n",
                w->worker_id,
                io->read_depth);
        return -1;
    }

    for (uint32_t i = 0; i < io->read_depth; i++) {
        io->read_bufs[i] = spdk_zmalloc(buf_bytes,
                                        4096,
                                        NULL,
                                        SPDK_ENV_NUMA_ID_ANY,
                                        SPDK_MALLOC_DMA);
        if (!io->read_bufs[i]) {
            fprintf(stderr,
                    "spdk_zmalloc failed worker=%d read_buf[%u] depth=%u\n",
                    w->worker_id,
                    i,
                    io->read_depth);
            return -1;
        }
    }

    return 0;
}

static void stage_worker_cleanup_io_state(stage_worker_t *w, struct worker_io_state *io)
{
    if (io) {
        for (uint32_t i = 0; i < io->read_depth; i++) {
            if (io->read_bufs && io->read_bufs[i]) {
                spdk_free(io->read_bufs[i]);
            }
        }
        free(io->ready_slots);
        free(io->run_slots);
        free(io->read_bufs);
        memset(io, 0, sizeof(*io));
    }

    if (w->qpair) {
        spdk_nvme_ctrlr_free_io_qpair(w->qpair);
        w->qpair = NULL;
    }
}

static int stage_worker_load_query_ctx(stage_worker_t *w,
                                       uint64_t qid,
                                       struct worker_query_ctx *ctx)
{
    pipeline_app_t *app = w->app;

    memset(ctx, 0, sizeof(*ctx));
    ctx->prune_threshold = app->threshold;

    pthread_mutex_lock(&app->query_mu);
    query_tracker_t *qt = pipeline_find_query_tracker_locked(app, qid);
    if (qt) {
        ctx->query_seg = qt->query_segs[w->stage_id];
        if (qt->prune_threshold > 0.0f) {
            ctx->prune_threshold = qt->prune_threshold;
        }
    }
    pthread_mutex_unlock(&app->query_mu);

    ctx->prune_threshold *= app->prune_proportion[w->stage_id];

    if (!ctx->query_seg) {
        fprintf(stderr,
                "[stage %d lane %d] query seg not found for qid=%lu\n",
                w->stage_id,
                w->lane_id,
                qid);
        return -1;
    }

    return 0;
}

static void stage_worker_sort_batch(stage_worker_t *w,
                                    batch_t *in,
                                    struct worker_batch_stats *stats)
{
    if (!in || in->count <= 1) {
        return;
    }

    uint64_t qsort_begin_us = now_us();
    tls_vectors_per_lba_for_sort = stage_vectors_per_lba(w->app, (uint32_t)w->stage_id);
    qsort(in->items, in->count, sizeof(in->items[0]), cmp_cand_item_by_bundle);
    stats->batch_qsort_us += now_us() - qsort_begin_us;
}

static void stage_worker_finish_run(stage_worker_t *w,
                                    const batch_t *in,
                                    batch_t **out,
                                    const struct worker_query_ctx *query_ctx,
                                    struct worker_batch_stats *stats,
                                    cand_item_t *bundle_items,
                                    uint16_t *bundle_item_count,
                                    uint32_t *current_bundle_cluster,
                                    uint32_t *current_bundle_idx,
                                    struct bundle_run *run)
{
    pipeline_app_t *app = w->app;
    const uint32_t vectors_per_lba = stage_vectors_per_lba(app, (uint32_t)w->stage_id);
    const uint32_t stage_dim = app->ivf_meta.header.shard_dims[w->stage_id];
    const uint32_t shard_bytes = app->ivf_meta.header.shard_bytes[w->stage_id];
    uint64_t run_io_us = run->complete_us > run->submit_us
                             ? (run->complete_us - run->submit_us)
                             : 0;
    int io_rc = run->submit_rc;

    stats->batch_io_us += run_io_us;
    if (io_rc == 0 && run->waiter.ok) {
        stats->batch_bundles_read += run->bundle_count;
    } else if (io_rc == 0) {
        io_rc = -1;
    }

    if (io_rc != 0) {
        fprintf(stderr,
                "[stage %d] merged bundle read failed cluster=%u start_bundle=%u bundle_count=%u disk=%s\n",
                w->stage_id,
                run->cluster_id,
                run->start_bundle,
                run->bundle_count,
                w->disk->traddr);
        return;
    }

    for (uint16_t k = run->begin_idx; k < run->end_idx; k++) {
        uint32_t vec_id = in->items[k].vec_id;
        uint32_t cluster_id = in->items[k].cluster_id;
        uint32_t local_idx = in->items[k].local_idx;
        float acc = in->items[k].partial_sum;
        uint32_t bundle_idx = local_idx / vectors_per_lba;
        uint32_t lane_idx = local_idx % vectors_per_lba;
        uint32_t bundle_offset = bundle_idx - run->start_bundle;
        float *seg = (float *)((uint8_t *)run->buf +
                               (size_t)bundle_offset * w->disk->sector_size +
                               (size_t)lane_idx * shard_bytes);

        acc += partial_l2(seg, query_ctx->query_seg, stage_dim);
        if (acc > query_ctx->prune_threshold) {
            stats->batch_pruned++;
            continue;
        }

        stats->batch_out++;
        if (*current_bundle_cluster != cluster_id || *current_bundle_idx != bundle_idx) {
            *out = append_bundle_group_to_out(app,
                                              *out,
                                              bundle_items,
                                              *bundle_item_count,
                                              &stats->child_batches);
            *bundle_item_count = 0;
            *current_bundle_cluster = cluster_id;
            *current_bundle_idx = bundle_idx;
        }

        bundle_items[*bundle_item_count].vec_id = vec_id;
        bundle_items[*bundle_item_count].cluster_id = cluster_id;
        bundle_items[*bundle_item_count].local_idx = local_idx;
        bundle_items[*bundle_item_count].partial_sum = acc;
        (*bundle_item_count)++;
    }
}

static void stage_worker_process_batch(stage_worker_t *w,
                                       batch_t *in,
                                       const struct worker_io_state *io)
{
    pipeline_app_t *app = w->app;
    struct worker_query_ctx query_ctx;
    struct worker_batch_stats stats = {
        .batch_begin_us = now_us(),
        .batch_in = in->count,
    };
    cand_item_t bundle_items[MAX_BATCH];
    uint16_t bundle_item_count = 0;
    uint32_t current_bundle_cluster = UINT32_MAX;
    uint32_t current_bundle_idx = UINT32_MAX;
    batch_t *out = alloc_stage_out_batch(in->qid, in->stage + 1);

    if (stage_worker_load_query_ctx(w, in->qid, &query_ctx) != 0) {
        free(out);
        free(in);
        return;
    }

    stage_worker_sort_batch(w, in, &stats);

    if (in->count > 0) {
        const uint32_t vectors_per_lba = stage_vectors_per_lba(app, (uint32_t)w->stage_id);
        const uint32_t max_gap_bundles = (w->stage_id == 0)
                                             ? app->stage0_gap_merge_limit
                                             : app->stage1_gap_merge_limit;
        uint16_t next_idx = 0;
        uint32_t inflight = 0;
        uint32_t ready_head = 0;
        uint32_t ready_tail = 0;
        uint32_t ready_count = 0;

        while (next_idx < in->count || inflight > 0 || ready_count > 0) {
            for (uint32_t slot_idx = 0;
                 next_idx < in->count && inflight < io->read_depth && slot_idx < io->read_depth;
                 slot_idx++) {
                struct bundle_run *slot = &io->run_slots[slot_idx];
                if (slot->submitted) {
                    continue;
                }
                if (!plan_bundle_run(in,
                                     next_idx,
                                     vectors_per_lba,
                                     io->max_bundles_per_read,
                                     max_gap_bundles,
                                     io->read_bufs[slot_idx],
                                     slot)) {
                    next_idx = in->count;
                    break;
                }
                slot->submit_rc = submit_vec_bundle_range(w, slot);
                if (slot->submit_rc == 0) {
                    stats.batch_nvme_reads++;
                    stats.batch_nvme_read_bytes +=
                        (uint64_t)slot->bundle_count * (uint64_t)w->disk->sector_size;
                } else {
                    slot->submitted = false;
                }
                next_idx = slot->end_idx;
                inflight++;
            }

            if (ready_count == 0 && inflight > 0) {
                int collect_rc = collect_completed_runs(w,
                                                       io->run_slots,
                                                       io->read_depth,
                                                       io->ready_slots,
                                                       &ready_tail,
                                                       &ready_count);
                if (collect_rc != 0) {
                    fprintf(stderr,
                            "[stage %d] process completions failed rc=%d disk=%s\n",
                            w->stage_id,
                            collect_rc,
                            w->disk->traddr);
                    break;
                }
                if (ready_count == 0) {
                    continue;
                }
            }

            if (ready_count == 0) {
                continue;
            }

            uint16_t slot_idx = io->ready_slots[ready_head];
            struct bundle_run *cur_run = &io->run_slots[slot_idx];

            ready_head = (ready_head + 1) % io->read_depth;
            ready_count--;

            stage_worker_finish_run(w,
                                    in,
                                    &out,
                                    &query_ctx,
                                    &stats,
                                    bundle_items,
                                    &bundle_item_count,
                                    &current_bundle_cluster,
                                    &current_bundle_idx,
                                    cur_run);

            inflight--;
            cur_run->submitted = false;
            cur_run->completion_seen = false;
        }
    }

    out = append_bundle_group_to_out(app,
                                     out,
                                     bundle_items,
                                     bundle_item_count,
                                     &stats.child_batches);

    if (out->count > 0) {
        emit_out_batch(app, out, &stats.child_batches);
    } else {
        free(out);
    }

    uint64_t batch_wall_us = now_us() - stats.batch_begin_us;

    __atomic_fetch_add(&app->stage_in[w->stage_id], stats.batch_in, __ATOMIC_RELAXED);
    __atomic_fetch_add(&app->stage_out[w->stage_id], stats.batch_out, __ATOMIC_RELAXED);
    __atomic_fetch_add(&app->stage_pruned[w->stage_id], stats.batch_pruned, __ATOMIC_RELAXED);

    pthread_mutex_lock(&app->query_mu);
    query_tracker_t *qt = pipeline_find_query_tracker_locked(app, in->qid);
    if (qt) {
        qt->stage_in[w->stage_id] += stats.batch_in;
        qt->stage_out[w->stage_id] += stats.batch_out;
        qt->stage_pruned[w->stage_id] += stats.batch_pruned;
        qt->stage_batches[w->stage_id] += 1;
        qt->stage_bundles_read[w->stage_id] += stats.batch_bundles_read;
        qt->stage_nvme_reads[w->stage_id] += stats.batch_nvme_reads;
        qt->stage_nvme_read_bytes[w->stage_id] += stats.batch_nvme_read_bytes;
        qt->stage_wall_us[w->stage_id] += batch_wall_us;
        qt->stage_io_us[w->stage_id] += stats.batch_io_us;
        qt->stage_qsort_us[w->stage_id] += stats.batch_qsort_us;
    }
    pthread_mutex_unlock(&app->query_mu);

    mark_batch_finished(app, in->qid, stats.child_batches);
    free(in);
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
        query_tracker_t *qt = pipeline_find_query_tracker_locked(tw->app, b->qid);
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
    struct worker_io_state io;

    bind_to_core(w->core_id);

    if (stage_worker_alloc_qpair(w) != 0) {
        return NULL;
    }
    if (stage_worker_init_io_state(w, &io) != 0) {
        stage_worker_cleanup_io_state(w, &io);
        return NULL;
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

        stage_worker_process_batch(w, in, &io);
    }

    stage_worker_cleanup_io_state(w, &io);
    return NULL;
}

static int pipeline_validate_init_args(uint32_t read_depth,
                                       uint32_t stage1_gap_merge_limit,
                                       uint32_t stage0_gap_merge_limit,
                                       uint32_t stage0_max_read_lbas,
                                       uint32_t active_stages,
                                       const uint32_t stage_disk_counts[NUM_STAGES],
                                       const float prune_proportion[NUM_STAGES],
                                       coarse_backend_t backend,
                                       prune_threshold_mode_t prune_mode,
                                       const char *coarse_backend_name_arg,
                                       const char *prune_threshold_mode_name_arg)
{
    if (read_depth == 0 || read_depth > MAX_BATCH) {
        fprintf(stderr,
                "pipeline_init: invalid read depth=%u valid_range=[1,%d]\n",
                read_depth,
                MAX_BATCH);
        return -1;
    }
    if (active_stages == 0 || active_stages > NUM_STAGES) {
        fprintf(stderr,
                "pipeline_init: invalid active stages=%u valid_range=[1,%d]\n",
                active_stages,
                NUM_STAGES);
        return -1;
    }
    if (!stage_disk_counts) {
        fprintf(stderr, "pipeline_init: stage_disk_counts is required\n");
        return -1;
    }
    if (!prune_proportion) {
        fprintf(stderr, "pipeline_init: prune_proportion is required\n");
        return -1;
    }
    uint32_t total_disks = 0;
    for (uint32_t s = 0; s < active_stages; s++) {
        if (stage_disk_counts[s] == 0 || stage_disk_counts[s] > MAX_DISKS_PER_STAGE) {
            fprintf(stderr,
                    "pipeline_init: invalid disk count stage=%u count=%u valid_range=[1,%d]\n",
                    s,
                    stage_disk_counts[s],
                    MAX_DISKS_PER_STAGE);
            return -1;
        }
        total_disks += stage_disk_counts[s];
        if (!isfinite(prune_proportion[s]) || prune_proportion[s] < 0.0f) {
            fprintf(stderr,
                    "pipeline_init: invalid prune_proportion[%u]=%f, expected finite value >= 0\n",
                    s,
                    prune_proportion[s]);
            return -1;
        }
    }
    if (total_disks > NUM_STAGES) {
        fprintf(stderr,
                "pipeline_init: stage disk counts use %u disks but only %d are configured\n",
                total_disks,
                NUM_STAGES);
        return -1;
    }
    if (stage1_gap_merge_limit > MAX_BATCH || stage0_gap_merge_limit > MAX_BATCH) {
        fprintf(stderr,
                "pipeline_init: gap merge limits must be in [0,%d], got stage0=%u stage1plus=%u\n",
                MAX_BATCH,
                stage0_gap_merge_limit,
                stage1_gap_merge_limit);
        return -1;
    }
    if (stage0_max_read_lbas > MAX_BATCH) {
        fprintf(stderr,
                "pipeline_init: stage0-max-read-lbas must be in [0,%d], got=%u\n",
                MAX_BATCH,
                stage0_max_read_lbas);
        return -1;
    }
    if (backend != COARSE_BACKEND_BRUTE && backend != COARSE_BACKEND_FAISS) {
        fprintf(stderr,
                "pipeline_init: invalid coarse backend '%s' (expected brute or faiss)\n",
                coarse_backend_name_arg ? coarse_backend_name_arg : "(null)");
        return -1;
    }
    if (prune_mode != PRUNE_THRESHOLD_CENTROID && prune_mode != PRUNE_THRESHOLD_SAMPLED) {
        fprintf(stderr,
                "pipeline_init: invalid prune threshold mode '%s' (expected centroid or sampled)\n",
                prune_threshold_mode_name_arg ? prune_threshold_mode_name_arg : "(null)");
        return -1;
    }

    return 0;
}

static void pipeline_assign_stage_disk_config(pipeline_app_t *app,
                                              const uint32_t stage_disk_counts[NUM_STAGES])
{
    uint32_t next_disk = 0;

    memset(app->stage_disk_counts, 0, sizeof(app->stage_disk_counts));
    memset(app->stage_disk_indices, 0, sizeof(app->stage_disk_indices));

    for (uint32_t s = 0; s < app->active_stages; s++) {
        app->stage_disk_counts[s] = stage_disk_counts[s];
        for (uint32_t i = 0; i < stage_disk_counts[s]; i++) {
            app->stage_disk_indices[s][i] = next_disk++;
        }
    }
}

static int pipeline_copy_disk_ctxs(pipeline_app_t *app, disk_ctx_t disks[NUM_STAGES])
{
    for (int i = 0; i < NUM_STAGES; i++) {
        app->disks[i] = disks[i];
        if (!app->disks[i].ctrlr || !app->disks[i].ns) {
            fprintf(stderr, "pipeline_init: disk %d not initialized (traddr=%s)\n",
                    i, app->disks[i].traddr);
            return -1;
        }
    }

    return 0;
}

static void pipeline_assign_runtime_config(pipeline_app_t *app,
                                           float threshold,
                                           uint32_t read_depth,
                                           uint32_t stage1_gap_merge_limit,
                                           uint32_t stage0_gap_merge_limit,
                                           uint32_t stage0_max_read_lbas,
                                           uint32_t active_stages,
                                           coarse_backend_t backend,
                                           prune_threshold_mode_t prune_mode,
                                           const float prune_proportion[NUM_STAGES])
{
    app->threshold = threshold;
    app->read_depth = read_depth;
    app->stage1_gap_merge_limit = stage1_gap_merge_limit;
    app->stage0_gap_merge_limit = stage0_gap_merge_limit;
    app->stage0_max_read_lbas = stage0_max_read_lbas;
    app->active_stages = active_stages;
    app->coarse_backend = backend;
    app->prune_threshold_mode = prune_mode;
    for (uint32_t s = 0; s < NUM_STAGES; s++) {
        app->prune_proportion[s] = prune_proportion ? prune_proportion[s] : 1.0f;
    }
}

static int pipeline_check_meta_compatibility(pipeline_app_t *app)
{
    if (app->ivf_meta.header.num_shards != app->active_stages) {
        fprintf(stderr,
                "pipeline_init: metadata num_shards=%u does not match active_stages=%u\n",
                app->ivf_meta.header.num_shards,
                app->active_stages);
        return -1;
    }

    for (uint32_t s = 0; s < app->active_stages; s++) {
        for (uint32_t i = 0; i < app->stage_disk_counts[s]; i++) {
            uint32_t disk_idx = app->stage_disk_indices[s][i];
            if (disk_idx >= NUM_STAGES) {
                fprintf(stderr,
                        "pipeline_init: invalid disk index stage=%u disk_slot=%u disk_idx=%u\n",
                        s,
                        i,
                        disk_idx);
                return -1;
            }
            if (app->disks[disk_idx].sector_size != app->ivf_meta.header.sector_size) {
                fprintf(stderr,
                        "pipeline_init: sector size mismatch stage=%u disk=%u disk_sector=%u meta=%u\n",
                        s,
                        disk_idx,
                        app->disks[disk_idx].sector_size,
                        app->ivf_meta.header.sector_size);
                return -1;
            }
        }

        if (app->ivf_meta.header.shard_bytes[s] > app->ivf_meta.header.sector_size) {
            fprintf(stderr,
                    "pipeline_init: shard bytes exceed sector size stage=%d shard_bytes=%u sector=%u\n",
                    s,
                    app->ivf_meta.header.shard_bytes[s],
                    app->ivf_meta.header.sector_size);
            return -1;
        }
    }

    return 0;
}

static void pipeline_cleanup_init_state(pipeline_app_t *app, bool destroy_query_mu)
{
    free(app->sorted_vec_ids);
    app->sorted_vec_ids = NULL;
    app->num_sorted_vec_ids = 0;

    free(app->prune_sample_vectors);
    app->prune_sample_vectors = NULL;
    app->prune_sample_count = 0;

    coarse_search_module_destroy(app->coarse_module);
    app->coarse_module = NULL;

    free_ivf_meta(&app->ivf_meta);
    app->centroids = NULL;
    app->nlist = 0;

    if (destroy_query_mu) {
        pthread_mutex_destroy(&app->query_mu);
    }
}

static int pipeline_init_workers(pipeline_app_t *app,
                                 const uint32_t stage_worker_counts[NUM_STAGES],
                                 const int stage_cores[NUM_STAGES][MAX_WORKERS_PER_STAGE],
                                 int topk_core)
{
    pthread_mutex_init(&app->query_mu, NULL);
    memset(app->queries, 0, sizeof(app->queries));

    queue_init(&app->topk.inq);
    app->topk.app = app;
    app->topk.core_id = topk_core;

    int worker_id = 0;
    for (uint32_t s = 0; s < app->active_stages; s++) {
        uint32_t worker_count = stage_worker_counts[s];
        if (worker_count == 0 || worker_count > MAX_WORKERS_PER_STAGE) {
            fprintf(stderr,
                    "pipeline_init: invalid worker count stage=%u count=%u max=%d\n",
                    s, worker_count, MAX_WORKERS_PER_STAGE);
            return -1;
        }
        if (app->stage_disk_counts[s] == 0) {
            fprintf(stderr, "pipeline_init: stage=%u has no disks\n", s);
            return -1;
        }

        app->stage_worker_counts[s] = worker_count;
        app->stage_rr_cursor[s] = 0;

        for (uint32_t lane = 0; lane < worker_count; lane++) {
            uint32_t disk_slot = lane % app->stage_disk_counts[s];
            uint32_t disk_idx = app->stage_disk_indices[s][disk_slot];
            stage_worker_t *w = &app->workers[s][lane];
            w->app = app;
            w->worker_id = worker_id++;
            w->stage_id = s;
            w->lane_id = (int)lane;
            w->core_id = stage_cores[s][lane];
            w->disk_id = (int)disk_idx;
            w->disk = &app->disks[disk_idx];
            w->qpair = NULL;
            queue_init(&w->inq);
        }
    }

    return 0;
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
    const uint32_t stage_disk_counts[NUM_STAGES],
    const int stage_cores[NUM_STAGES][MAX_WORKERS_PER_STAGE],
    int topk_core,
    uint32_t read_depth,
    uint32_t stage1_gap_merge_limit,
    uint32_t stage0_gap_merge_limit,
    uint32_t stage0_max_read_lbas,
    uint32_t active_stages,
    const char *coarse_backend_name_arg,
    const char *prune_threshold_mode_name_arg,
    float threshold,
    const float prune_proportion[NUM_STAGES],
    const char *ivf_meta_path,
    const char *sorted_ids_path
)
{
    coarse_backend_t backend = parse_coarse_backend_name(coarse_backend_name_arg);
    prune_threshold_mode_t prune_mode = parse_prune_threshold_mode_name(prune_threshold_mode_name_arg);
    bool query_mu_initialized = false;

    memset(app, 0, sizeof(*app));

    if (pipeline_validate_init_args(read_depth,
                                    stage1_gap_merge_limit,
                                    stage0_gap_merge_limit,
                                    stage0_max_read_lbas,
                                    active_stages,
                                    stage_disk_counts,
                                    prune_proportion,
                                    backend,
                                    prune_mode,
                                    coarse_backend_name_arg,
                                    prune_threshold_mode_name_arg) != 0) {
        return -1;
    }

    fprintf(stderr,
            "[pipeline_init] begin read_depth=%u stage0_gap_merge_limit=%u stage1_gap_merge_limit=%u stage0_max_read_lbas=%u active_stages=%u coarse_backend=%s prune_threshold_mode=%s ivf_meta=%s sorted_ids=%s\n",
            read_depth,
            stage0_gap_merge_limit,
            stage1_gap_merge_limit,
            stage0_max_read_lbas,
            active_stages,
            coarse_backend_name(backend),
            prune_threshold_mode_name(prune_mode),
            ivf_meta_path ? ivf_meta_path : "(null)",
            sorted_ids_path ? sorted_ids_path : "(null)");

    if (pipeline_copy_disk_ctxs(app, disks) != 0) {
        return -1;
    }

    pipeline_assign_runtime_config(app,
                                   threshold,
                                   read_depth,
                                   stage1_gap_merge_limit,
                                   stage0_gap_merge_limit,
                                   stage0_max_read_lbas,
                                   active_stages,
                                   backend,
                                   prune_mode,
                                   prune_proportion);
    pipeline_assign_stage_disk_config(app, stage_disk_counts);

    fprintf(stderr, "[pipeline_init] prune_proportion:");
    for (uint32_t s = 0; s < active_stages; s++) {
        fprintf(stderr, " stage%u=%.6f", s, app->prune_proportion[s]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "[pipeline_init] before parse_ivf_meta\n");
    if (parse_ivf_meta(ivf_meta_path, &app->ivf_meta) != 0) {
        fprintf(stderr, "pipeline_init: failed to load ivf meta from %s\n", ivf_meta_path);
        goto fail;
    }

    app->nlist = app->ivf_meta.nlist;
    app->centroids = app->ivf_meta.centroids;

    fprintf(stderr,
            "[pipeline_init] after parse_ivf_meta version=%u dim=%u nlist=%u num_vectors=%u sector_size=%u backend=%s\n",
            app->ivf_meta.header.version,
            app->ivf_meta.header.dim,
            app->ivf_meta.nlist,
            app->ivf_meta.header.num_vectors,
            app->ivf_meta.header.sector_size,
            coarse_backend_name(app->coarse_backend));
    for (uint32_t s = 0; s < app->ivf_meta.header.num_shards; s++) {
        fprintf(stderr,
                "[pipeline_init] shard%u dim=%u bytes=%u vectors_per_lba=%u offset=%u\n",
                s,
                app->ivf_meta.header.shard_dims[s],
                app->ivf_meta.header.shard_bytes[s],
                stage_vectors_per_lba(app, s),
                app->ivf_meta.header.shard_offsets[s]);
    }

    if (app->coarse_backend == COARSE_BACKEND_FAISS) {
        fprintf(stderr, "[pipeline_init] before coarse_search_module_init\n");
        if (coarse_search_module_init(&app->coarse_module,
                                      app->centroids,
                                      app->ivf_meta.header.dim,
                                      app->nlist) != 0) {
            fprintf(stderr, "pipeline_init: failed to build Faiss coarse index\n");
            goto fail;
        }
        fprintf(stderr, "[pipeline_init] after coarse_search_module_init\n");
    } else {
        app->coarse_module = NULL;
        fprintf(stderr, "[pipeline_init] skip coarse_search_module_init for brute backend\n");
    }

    if (pipeline_check_meta_compatibility(app) != 0) {
        goto fail;
    }

    fprintf(stderr, "[pipeline_init] before load_sorted_ids_bin\n");
    if (load_sorted_ids_bin(sorted_ids_path, &app->num_sorted_vec_ids, &app->sorted_vec_ids) != 0) {
        fprintf(stderr, "pipeline_init: failed to load sorted ids from %s\n", sorted_ids_path);
        goto fail;
    }
    fprintf(stderr,
            "[pipeline_init] after load_sorted_ids_bin count=%u expected=%u\n",
            app->num_sorted_vec_ids,
            app->ivf_meta.header.num_vectors);

    if (app->num_sorted_vec_ids != app->ivf_meta.header.num_vectors) {
        fprintf(stderr,
                "pipeline_init: sorted ids count mismatch got=%u expected=%u\n",
                app->num_sorted_vec_ids, app->ivf_meta.header.num_vectors);
        goto fail;
    }

    if (app->prune_threshold_mode == PRUNE_THRESHOLD_SAMPLED) {
        fprintf(stderr, "[pipeline_init] before build_prune_sample_vectors\n");
        if (build_prune_sample_vectors(app) != 0) {
            fprintf(stderr, "pipeline_init: failed to build prune sample vectors\n");
            goto fail;
        }
        fprintf(stderr, "[pipeline_init] after build_prune_sample_vectors sample_count=%u topk=%u\n",
                app->prune_sample_count, PRUNE_SAMPLE_TOPK);
    } else {
        fprintf(stderr, "[pipeline_init] skip build_prune_sample_vectors for centroid mode\n");
    }

    fprintf(stderr, "[pipeline_init] before worker setup\n");
    if (pipeline_init_workers(app, stage_worker_counts, stage_cores, topk_core) != 0) {
        query_mu_initialized = true;
        goto fail;
    }
    query_mu_initialized = true;

    fprintf(stderr, "[pipeline_init] done\n");
    return 0;

fail:
    pipeline_cleanup_init_state(app, query_mu_initialized);
    return -1;
}

/* ============================================================
 * 9. 启动所有线程
 * ============================================================ */

void pipeline_start(pipeline_app_t *app) {
    pthread_create(&app->topk.tid, NULL, topk_thread_main, &app->topk);

    for (uint32_t s = 0; s < app->active_stages; s++) {
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
    for (uint32_t s = 0; s < app->active_stages; s++) {
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
    for (uint32_t s = 0; s < app->active_stages; s++) {
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
        pipeline_free_query_tracker_segments(&app->queries[i]);
    }

    coarse_search_module_destroy(app->coarse_module);
    app->coarse_module = NULL;
    free(app->prune_sample_vectors);
    app->prune_sample_vectors = NULL;
    app->prune_sample_count = 0;
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
    uint32_t sorted_base = ci->sorted_id_base;

    if (!app->sorted_vec_ids || sorted_base + num_vec > app->num_sorted_vec_ids) {
        fprintf(stderr,
                "submit_cluster_candidates: bad sorted id range cluster=%u base=%u num_vec=%u total=%u\n",
                cluster_id, sorted_base, num_vec, app->num_sorted_vec_ids);
        return -1;
    }

    pthread_mutex_lock(&app->query_mu);
    query_tracker_t *qt_for_submit = pipeline_find_query_tracker_locked(app, qid);
    if (!qt_for_submit) {
        pthread_mutex_unlock(&app->query_mu);
        fprintf(stderr, "submit_cluster_candidates: qid=%lu not registered\n", qid);
        return -1;
    }
    qt_for_submit->initial_candidates += num_vec;
    pthread_mutex_unlock(&app->query_mu);

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

        pthread_mutex_lock(&app->query_mu);
        qt_for_submit = pipeline_find_query_tracker_locked(app, qid);
        if (!qt_for_submit) {
            pthread_mutex_unlock(&app->query_mu);
            fprintf(stderr, "submit_cluster_candidates: qid=%lu not registered\n", qid);
            free(b);
            return -1;
        }
        qt_for_submit->submitted_batches++;
        qt_for_submit->outstanding_batches++;
        if (qt_for_submit->outstanding_batches > qt_for_submit->max_outstanding_batches) {
            qt_for_submit->max_outstanding_batches = qt_for_submit->outstanding_batches;
        }
        pthread_mutex_unlock(&app->query_mu);

        pipeline_submit_initial_batch(app, b);
    }

    return 0;
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
    for (uint32_t s = 0; s < app->active_stages; s++) {
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

    if (app->prune_threshold_mode == PRUNE_THRESHOLD_CENTROID) {
        qt->prune_threshold = compute_centroid_prune_threshold(hits, nprobe, app->threshold);
    } else {
        qt->prune_threshold = compute_sampled_prune_threshold(app, query);
        if (!(qt->prune_threshold > 0.0f) || !isfinite(qt->prune_threshold)) {
            qt->prune_threshold = app->threshold;
        }
    }

    // printf("[submit_query] qid=%lu nprobe=%u prune_mode=%s prune_threshold=%f sample_count=%u sample_topk=%u\n",
    //        qid,
    //        nprobe,
    //        prune_threshold_mode_name(app->prune_threshold_mode),
    //        qt->prune_threshold,
    //        app->prune_sample_count,
    //        PRUNE_SAMPLE_TOPK);

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
            pipeline_maybe_mark_query_done_locked(x);
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
