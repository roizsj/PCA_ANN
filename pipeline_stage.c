#include "pipeline_stage.h"
#include "app.h"

#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct io_waiter {
    volatile bool done;
    int ok;
};

static void io_done_cb(void *arg, const struct spdk_nvme_cpl *cpl)
{
    struct io_waiter *wt = (struct io_waiter *)arg;
    wt->ok = !spdk_nvme_cpl_is_error(cpl);
    wt->done = true;
}

/* ============================================================
 * 0. 读ivf_meta.bin
 *
 * 作用：
 *   - 从ivf_meta.bin文件中读取聚类元数据，构建ivf_meta_t结构体，供后续索引使用
 * ============================================================ */
int parse_ivf_meta(const char* filename, ivf_meta_t *meta) {
    if (!meta) return -1;
    memset(meta, 0, sizeof(*meta));

    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    // 读取MetaHeader (40 bytes)
    uint32_t magic, version, dim;
    ivf_meta_header_t header;
    
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&version, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&dim, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header.shard_dim, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header.vectors_per_lba, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header.nlist, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header.num_vectors, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header.sector_size, sizeof(uint32_t), 1, fp) != 1 ||
        fread(&header.base_lba, sizeof(uint64_t), 1, fp) != 1) {
        fprintf(stderr, "Failed to read MetaHeader\n");
        fclose(fp);
        return -1;
    }

    // 验证magic number
    if (magic != 0x49564633) {  // "IVF3"
        fprintf(stderr, "Invalid magic number: 0x%08X\n", magic);
        fclose(fp);
        return -1;
    }
    
    meta->header = header;
    meta->nlist = header.nlist;

    // 分配cluster_info_t数组
    meta->clusters = malloc(header.nlist * sizeof(cluster_info_t));
    if (!meta->clusters) {
        perror("malloc clusters");
        fclose(fp);
        return -1;
    }

    // 读取每个ClusterMeta
        for (uint32_t i = 0; i < header.nlist; i++) {
        ClusterMetaOnDisk cm;

        if (fread(&cm, sizeof(cm), 1, fp) != 1) {
            fprintf(stderr, "Failed to read cluster %u metadata\n", i);
            free_ivf_meta(meta);
            fclose(fp);
            return -1;
        }

        meta->clusters[i].cluster_id = cm.cluster_id;
        meta->clusters[i].start_lba = cm.start_lba;
        meta->clusters[i].num_vectors = cm.num_vectors;
        meta->clusters[i].num_lbas = cm.num_lbas;
    }

    fclose(fp);
    return 0;
}

// 释放ivf_meta_t结构体及其内部内存
void free_ivf_meta(ivf_meta_t *meta) {
    if (!meta) return;
    free(meta->clusters);
    meta->clusters = NULL;
    meta->nlist = 0;
    memset(&meta->header, 0, sizeof(meta->header));
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
float partial_l2(const float *x, const float *q) {
    float s = 0.0f;
    for (int i = 0; i < SEG_DIM; i++) {
        float d = x[i] - q[i];
        s += d * d;
    }
    return s;
}

/*
 * 作用：
 *   同一个 stage 有 4 个 worker，这里用 cluster_id和local_idx做一个简单的hash，决定发哪个worker。
 *   后面如果你想换成 batch hash / round-robin，都可以改这里。
 */
static int pick_lane(uint32_t cluster_id, uint32_t local_idx) {
    return (cluster_id ^ local_idx) % WORKERS_PER_STAGE;
}

/*
 * 根据 cluster_id 在 ivf_meta 中找到对应的 cluster_info_t。后续如果能保证cluster_id = 数组下标则可以做优化 TODO
 */
const cluster_info_t *find_cluster_info(const ivf_meta_t *meta, uint32_t cluster_id) {
    if (!meta || !meta->clusters) return NULL;

    for (uint32_t i = 0; i < meta->nlist; i++) {
        if (meta->clusters[i].cluster_id == cluster_id) {
            return &meta->clusters[i];
        }
    }
    return NULL;
}


static int cmp_cand_item_by_bundle(const void *a, const void *b) {
    const cand_item_t *x = (const cand_item_t *)a;
    const cand_item_t *y = (const cand_item_t *)b;

    if (x->cluster_id < y->cluster_id) return -1;
    if (x->cluster_id > y->cluster_id) return 1;

    /* bundle_idx = local_idx / vectors_per_lba */
    uint32_t x_bundle = x->local_idx / SEGMENTS_PER_LBA;
    uint32_t y_bundle = y->local_idx / SEGMENTS_PER_LBA;

    if (x_bundle < y_bundle) return -1;
    if (x_bundle > y_bundle) return 1;

    uint32_t x_lane = x->local_idx % SEGMENTS_PER_LBA;
    uint32_t y_lane = y->local_idx % SEGMENTS_PER_LBA;

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
    pthread_mutex_lock(&st->mu);

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

    pthread_mutex_unlock(&st->mu);
}

/* ============================================================
 * 4. 一个 worker 对自己的盘做一次同步读
 *
 * 说明：
 *   - 每个 worker 拥有自己独占的 qpair
 *   - 读请求提交后，当前 worker 自己轮询自己的 qpair completion
 *   - 现在是一个向量发起一次读盘请求，后面需要整合 TODO
 *
 * 这是最容易理解、最适合先把流水线跑起来的写法。
 * 后面你如果要做更高性能版本，可以再改成更异步的方式。
 * ============================================================ */

static int read_vec_segment(stage_worker_t *w,
                            uint32_t cluster_id,
                            uint32_t local_idx,
                            void *buf)
{
    if (!w || !w->app || !w->disk || !w->disk->ns || !w->qpair) {
        fprintf(stderr, "[read_vec_segment] invalid worker context\n");
        return -1;
    }

    pipeline_app_t *app = w->app;
    ivf_meta_t *meta = &app->ivf_meta;

    const cluster_info_t *ci = find_cluster_info(meta, cluster_id);
    if (!ci) {
        fprintf(stderr, "[read_vec_segment] cluster_id=%u not found\n", cluster_id);
        return -1;
    }

    if (local_idx >= ci->num_vectors) {
        fprintf(stderr,
                "[read_vec_segment] local_idx=%u out of range for cluster=%u num_vectors=%u\n",
                local_idx, cluster_id, ci->num_vectors);
        return -1;
    }

    uint32_t sector_size = w->disk->sector_size;
    uint32_t shard_bytes = SEG_DIM * sizeof(float);   // 32 * 4 = 128
    uint32_t vectors_per_lba = meta->header.vectors_per_lba;   // 应该是32

    if (sector_size == 0 || vectors_per_lba == 0) {
        fprintf(stderr, "[read_vec_segment] bad sector_size/vectors_per_lba\n");
        return -1;
    }

    uint32_t bundle_idx = local_idx / vectors_per_lba;
    uint32_t lane_idx   = local_idx % vectors_per_lba;
    uint64_t lba        = ci->start_lba + bundle_idx;

    uint64_t cluster_end_lba = ci->start_lba + ci->num_lbas;
    if (lba >= cluster_end_lba) {
        fprintf(stderr,
                "[read_vec_segment] OOB read: cluster=%u local_idx=%u bundle_idx=%u "
                "start_lba=%lu num_lbas=%u lba=%lu\n",
                cluster_id, local_idx, bundle_idx,
                ci->start_lba, ci->num_lbas, lba);
        return -1;
    }

    struct io_waiter waiter = {.done = false, .ok = 0};

    int rc = spdk_nvme_ns_cmd_read(
        w->disk->ns,
        w->qpair,
        buf,
        lba,
        1,              // 只读1个LBA
        io_done_cb,
        &waiter,
        0
    );

    if (rc != 0) {
        fprintf(stderr,
                "[read_vec_segment] read submit failed rc=%d cluster=%u local_idx=%u lba=%lu\n",
                rc, cluster_id, local_idx, lba);
        return -1;
    }

    while (!waiter.done) {
        spdk_nvme_qpair_process_completions(w->qpair, 0);
    }

    if (!waiter.ok) {
        return -1;
    }

    return lane_idx;   // 返回lane，调用方自己取buf里的对应128B
}

// 按LBA分批读，跟上面分candidate读是两种思路，最后留一个就行了
static int read_vec_bundle(stage_worker_t *w,
                           uint32_t cluster_id,
                           uint32_t bundle_idx,
                           void *bundle_buf)
{
    if (!w || !w->app || !w->disk || !w->disk->ns || !w->qpair) {
        fprintf(stderr, "[read_vec_bundle] invalid worker context\n");
        return -1;
    }

    pipeline_app_t *app = w->app;
    ivf_meta_t *meta = &app->ivf_meta;

    const cluster_info_t *ci = find_cluster_info(meta, cluster_id);
    if (!ci) {
        fprintf(stderr, "[read_vec_bundle] cluster_id=%u not found\n", cluster_id);
        return -1;
    }

    if (bundle_idx >= ci->num_lbas) {
        fprintf(stderr,
                "[read_vec_bundle] bundle_idx=%u out of range cluster=%u num_lbas=%u\n",
                bundle_idx, cluster_id, ci->num_lbas);
        return -1;
    }

    uint64_t lba = ci->start_lba + bundle_idx;

    struct io_waiter waiter = {.done = false, .ok = 0};

    int rc = spdk_nvme_ns_cmd_read(
        w->disk->ns,
        w->qpair,
        bundle_buf,
        lba,
        1,              /* 一次只读一个4KB LBA */
        io_done_cb,
        &waiter,
        0
    );

    if (rc != 0) {
        fprintf(stderr,
                "[read_vec_bundle] read submit failed rc=%d cluster=%u bundle=%u lba=%lu\n",
                rc, cluster_id, bundle_idx, lba);
        return -1;
    }

    while (!waiter.done) {
        spdk_nvme_qpair_process_completions(w->qpair, 0);
    }

    return waiter.ok ? 0 : -1;
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

    int lane = pick_lane(b->items[0].cluster_id, b->items[0].local_idx); // 这里只用vec[0]的id做分流
    queue_push(&app->workers[b->stage][lane].inq, b);
}

/* ============================================================
 * 6. topk 线程
 *
 * 作用：
 *   - 接收 stage3 输出的 final batch
 *   - 把里面的 (vec_id, full_dist) 插入 top-k
 *
 * 目前最小实现里：
 *   - 只做 top-k 维护
 *   - 不做 query 完成检测
 * ============================================================ */

static void *topk_thread_main(void *arg) {
    topk_worker_t *tw = arg;
    bind_to_core(tw->core_id);

    while (1) {
        batch_t *b = queue_pop(&tw->inq);
        if (!b) {
            break;  /* 队列关闭 */
        }

        if (b->magic != MAGIC_BATCH) {
            fprintf(stderr, "[topk] bad batch magic\n");
            abort();
        }

        for (uint16_t i = 0; i < b->count; i++) {
            topk_insert(&tw->app->topk_state, b->items[i].vec_id, b->items[i].partial_sum);
        }

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

    int actual_cpu = sched_getcpu();

    /* 每个 worker 自己独占一个 qpair */
    w->qpair = spdk_nvme_ctrlr_alloc_io_qpair(w->disk->ctrlr, NULL, 0);
    if (!w->qpair) {
        fprintf(stderr,
        "alloc_io_qpair failed worker=%d stage=%d lane=%d disk=%s sector=%u\n",
        w->worker_id, w->stage_id, w->lane_id,
        w->disk->traddr, w->disk->sector_size);
        return NULL;
    }

    printf("[worker] started worker=%d stage=%d lane=%d core=%d actual_cpu=%d disk=%s qpair=%p\n",
           w->worker_id, w->stage_id, w->lane_id,
           w->core_id, actual_cpu, w->disk->traddr, (void *)w->qpair);
    fflush(stdout);

    /*
     * 这个 worker 的临时 DMA buffer
     * 先用一个 buffer 反复读单个 segment
     */
    void *buf = spdk_zmalloc(
        w->disk->sector_size,
        4096,
        NULL,
        SPDK_ENV_NUMA_ID_ANY,
        SPDK_MALLOC_DMA
    );
    if (!buf) {
        fprintf(stderr, "spdk_zmalloc failed worker=%d\n", w->worker_id);
        return NULL;
    }

    while (1) {
        batch_t *in = queue_pop(&w->inq);
        if (!in) {
            break;  /* 队列关闭 */
        }
        printf("[stage %d lane %d] got batch qid=%lu count=%u first_cluster=%u first_local=%u\n",
            w->stage_id, w->lane_id, in->qid, in->count,
            in->items[0].cluster_id, in->items[0].local_idx);
        fflush(stdout);

        if (in->magic != MAGIC_BATCH) {
            fprintf(stderr, "[stage %d] bad batch magic\n", w->stage_id);
            abort();
        }

        pipeline_app_t *app = w->app;

        /*
         * out 是 surviving candidates 的新 batch
         * 将发给下一 stage
         */
        batch_t *out = calloc(1, sizeof(*out));
        if (!out) {
            perror("calloc out batch");
            exit(1);
        }

        out->magic = MAGIC_BATCH;
        out->qid = in->qid;
        out->stage = in->stage + 1;

        __sync_fetch_and_add(&app->stage_in[w->stage_id], in->count);

        if (in->count > 1) {
            qsort(in->items, in->count, sizeof(in->items[0]), cmp_cand_item_by_bundle);
        }

        // -------- 最简单的版本：逐candidate读 --------
        // for (uint16_t i = 0; i < in->count; i++) {
        //     uint32_t vec_id = in->items[i].vec_id; // 暂时保留 防止出错 确定数据布局了就可以删了 TODO
        //     uint32_t cluster_id = in->items[i].cluster_id; 
        //     uint32_t local_idx = in->items[i].local_idx; // 这两个是新加的索引元数据
        //     float acc = in->items[i].partial_sum;

        //     /* 1) 从当前 stage 对应的盘上读该向量的本段 segment */
        //     int lane_idx = read_vec_segment(w, cluster_id, local_idx, buf);
        //     if (lane_idx < 0) {
        //         fprintf(stderr,
        //                 "[stage %d] read failed cluster=%u local_idx=%u disk=%s\n",
        //                 w->stage_id, cluster_id, local_idx, w->disk->traddr);
        //         continue;
        //     }

        //     uint32_t shard_bytes = SEG_DIM * sizeof(float);
        //     float *seg = (float *)((uint8_t *)buf + lane_idx * shard_bytes); // 读出来的segment只是buf的一小段
        //     // TODO 这里需要整合盘上的读，不能每个segment都读一整个LB，放大太多了

        //     /* 2) 计算 partial distance 并累加 */
        //     float part = partial_l2(seg, app->query_segs[w->stage_id]);
        //     acc += part;

        //     /* 3) 提前终止：超过阈值则直接丢弃 */
        //     if (acc > app->threshold) {
        //         __sync_fetch_and_add(&app->stage_pruned[w->stage_id], 1);
        //         continue;
        //     }

        //     /* 4) 如果这是最后一个 stage，送到 top-k */
        //     if (w->stage_id == NUM_STAGES - 1) {
        //         batch_t *finalb = calloc(1, sizeof(*finalb));
        //         if (!finalb) {
        //             perror("calloc final batch");
        //             exit(1);
        //         }

        //         finalb->magic = MAGIC_BATCH;
        //         finalb->qid = in->qid;
        //         finalb->stage = NUM_STAGES;
        //         finalb->count = 1;

        //         finalb->items[0].vec_id = vec_id;  // 暂时保留防止出错 后面删掉 TODO
        //         finalb->items[0].cluster_id = cluster_id;
        //         finalb->items[0].local_idx = local_idx;
        //         finalb->items[0].partial_sum = acc;

        //         queue_push(&app->topk.inq, finalb);
        //     } else {
        //         /*
        //          * 5) 否则放入 surviving batch，等待发给下一 stage
        //          */
        //         out->items[out->count].vec_id = vec_id;   // 暂时保留 后面删掉 TODO
        //         out->items[out->count].cluster_id = cluster_id;
        //         out->items[out->count].local_idx = local_idx;
        //         out->items[out->count].partial_sum = acc;
        //         out->count++;

        //         /*
        //          * 如果 out 满了，就先发一个批次出去，再开新的 out
        //          */
        //         if (out->count == MAX_BATCH) {
        //             __sync_fetch_and_add(&app->stage_out[w->stage_id], out->count);
        //             forward_batch(app, out);

        //             out = calloc(1, sizeof(*out));
        //             if (!out) {
        //                 perror("calloc spill batch");
        //                 exit(1);
        //             }

        //             out->magic = MAGIC_BATCH;
        //             out->qid = in->qid;
        //             out->stage = in->stage + 1;
        //         }
        //     }
        // }


        // -------- 优化版本：按LBA分批读 --------
        // -------- 没加向量计算并行，后续再加 -------
        const uint32_t vectors_per_lba = app->ivf_meta.header.vectors_per_lba;
        const uint32_t shard_bytes = SEG_DIM * sizeof(float);

        uint32_t cur_cluster_id = UINT32_MAX;
        uint32_t cur_bundle_idx = UINT32_MAX;
        bool bundle_loaded = false;
        uint32_t bundles_read = 0; // 计数器，后面可删

        for (uint16_t i = 0; i < in->count; i++) {
            uint32_t vec_id = in->items[i].vec_id;
            uint32_t cluster_id = in->items[i].cluster_id;
            uint32_t local_idx = in->items[i].local_idx;
            float acc = in->items[i].partial_sum;

            uint32_t bundle_idx = local_idx / vectors_per_lba;
            uint32_t lane_idx   = local_idx % vectors_per_lba;

            /* 只有当 (cluster_id, bundle_idx) 变化时，才读一次新的LBA */
            if (!bundle_loaded ||
                cluster_id != cur_cluster_id ||
                bundle_idx != cur_bundle_idx) {

                if (read_vec_bundle(w, cluster_id, bundle_idx, buf) != 0) {
                    fprintf(stderr,
                            "[stage %d] bundle read failed cluster=%u bundle=%u disk=%s\n",
                            w->stage_id, cluster_id, bundle_idx, w->disk->traddr);
                    bundle_loaded = false;
                    continue;
                }

                cur_cluster_id = cluster_id;
                cur_bundle_idx = bundle_idx;
                bundle_loaded = true;
                bundles_read++; // 增加计数器
            }

            float *seg = (float *)((uint8_t *)buf + lane_idx * shard_bytes);

            /* 2) 计算 partial distance 并累加 */
            float part = partial_l2(seg, app->query_segs[w->stage_id]);
            acc += part;

            /* 3) 提前终止：超过阈值则直接丢弃 */
            if (acc > app->threshold) {
                __sync_fetch_and_add(&app->stage_pruned[w->stage_id], 1);
                continue;
            }

            /* 4) 如果这是最后一个 stage，送到 top-k */
            if (w->stage_id == NUM_STAGES - 1) {
                batch_t *finalb = calloc(1, sizeof(*finalb));
                if (!finalb) {
                    perror("calloc final batch");
                    exit(1);
                }

                finalb->magic = MAGIC_BATCH;
                finalb->qid = in->qid;
                finalb->stage = NUM_STAGES;
                finalb->count = 1;
                finalb->items[0].vec_id = vec_id;
                finalb->items[0].cluster_id = cluster_id;
                finalb->items[0].local_idx = local_idx;
                finalb->items[0].partial_sum = acc;
                queue_push(&app->topk.inq, finalb);
            } else {
                out->items[out->count].vec_id = vec_id;
                out->items[out->count].cluster_id = cluster_id;
                out->items[out->count].local_idx = local_idx;
                out->items[out->count].partial_sum = acc;
                out->count++;

                if (out->count == MAX_BATCH) {
                    __sync_fetch_and_add(&app->stage_out[w->stage_id], out->count);
                    forward_batch(app, out);

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

        /* 扫完输入 batch 后，把剩余 surviving batch 发出去 */
        if (w->stage_id != NUM_STAGES - 1) {
            __sync_fetch_and_add(&app->stage_out[w->stage_id], out->count);
            forward_batch(app, out);
        } else {
            free(out);
        }

        printf("[stage %d lane %d] processed batch qid=%lu count=%u bundles_read=%u\n",
            w->stage_id, w->lane_id, in->qid, in->count, bundles_read);
        fflush(stdout);
        free(in);
    }

    spdk_free(buf);
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
    const int stage_cores[NUM_STAGES][WORKERS_PER_STAGE],
    int topk_core,
    float *query_segs[NUM_STAGES],
    float threshold,
    const char *ivf_meta_path
)
{
    memset(app, 0, sizeof(*app));

    /* 使用 main 里已经初始化好的 disks */
    for (int i = 0; i < NUM_STAGES; i++) {
        app->disks[i] = disks[i];
        app->query_segs[i] = query_segs[i];

        if (!app->disks[i].ctrlr || !app->disks[i].ns) {
            fprintf(stderr, "pipeline_init: disk %d not initialized (traddr=%s)\n",
                    i, app->disks[i].traddr);
            return -1;
        }
    }

    app->threshold = threshold;

    /* 读入ivf_meta信息 */
    if (parse_ivf_meta(ivf_meta_path, &app->ivf_meta) != 0) {
        fprintf(stderr, "pipeline_init: failed to load ivf meta from %s\n", ivf_meta_path);
        return -1;
    }

    pthread_mutex_init(&app->topk_state.mu, NULL);

    /* 初始化 topk worker */
    queue_init(&app->topk.inq);
    app->topk.app = app;
    app->topk.core_id = topk_core;

    /* 初始化 4 x 4 个 stage worker */
    int worker_id = 0;
    for (int s = 0; s < NUM_STAGES; s++) {
        for (int lane = 0; lane < WORKERS_PER_STAGE; lane++) {
            stage_worker_t *w = &app->workers[s][lane];
            w->app = app;
            w->worker_id = worker_id++;
            w->stage_id = s;
            w->lane_id = lane;
            w->core_id = stage_cores[s][lane];
            w->disk = &app->disks[s];
            w->qpair = NULL;
            queue_init(&w->inq);
        }
    }

    return 0;
}

/* ============================================================
 * 9. 启动所有线程
 * ============================================================ */

void pipeline_start(pipeline_app_t *app) {
    pthread_create(&app->topk.tid, NULL, topk_thread_main, &app->topk);

    for (int s = 0; s < NUM_STAGES; s++) {
        for (int lane = 0; lane < WORKERS_PER_STAGE; lane++) {
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

    int lane = pick_lane(b->items[0].cluster_id, b->items[0].local_idx); // 这里只用vec[0]的id做分流
    queue_push(&app->workers[0][lane].inq, b);
    // 调试用输出
    printf("[submit] qid=%lu stage=%u count=%u first_cluster=%u first_local=%u\n",
        b->qid, b->stage, b->count,
        b->items[0].cluster_id, b->items[0].local_idx);
    fflush(stdout);
}

/* ============================================================
 * 11. 关闭队列，通知线程退出
 * ============================================================ */

void pipeline_stop(pipeline_app_t *app) {
    for (int s = 0; s < NUM_STAGES; s++) {
        for (int lane = 0; lane < WORKERS_PER_STAGE; lane++) {
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
        for (int lane = 0; lane < WORKERS_PER_STAGE; lane++) {
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

    free_ivf_meta(&app->ivf_meta);

    pthread_mutex_destroy(&app->topk_state.mu);

    for (int s = 0; s < NUM_STAGES; s++) {
        for (int lane = 0; lane < WORKERS_PER_STAGE; lane++) {
            pthread_mutex_destroy(&app->workers[s][lane].inq.mu);
            pthread_cond_destroy(&app->workers[s][lane].inq.cv);
        }
    }

    pthread_mutex_destroy(&app->topk.inq.mu);
    pthread_cond_destroy(&app->topk.inq.cv);

    memset(&app->topk_state, 0, sizeof(app->topk_state));
}
