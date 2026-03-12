#include "pipeline_stage.h"

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
 * 一个非常简单的 lane 选择策略：
 *   vec_id % WORKERS_PER_STAGE
 *
 * 作用：
 *   同一个 stage 有 4 个 worker，这里用 vec_id 做静态分流。
 *   后面如果你想换成 batch hash / round-robin，都可以改这里。
 */
static int pick_lane(uint32_t vec_id) {
    return vec_id % WORKERS_PER_STAGE;
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

static int read_vec_segment(stage_worker_t *w, uint32_t vec_id, void *buf) {
    if (!w || !w->disk || !w->disk->ns || !w->qpair) {
        fprintf(stderr, "[read_vec_segment] invalid worker context\n");
        return -1;
    }

    if (w->disk->sector_size == 0) {
        fprintf(stderr, "[read_vec_segment] sector_size is 0 for disk %s\n", w->disk->traddr);
        return -1;
    }

    if (SLOT_BYTES < w->disk->sector_size || (SLOT_BYTES % w->disk->sector_size) != 0) {
        fprintf(stderr,
                "[read_vec_segment] invalid SLOT_BYTES=%u for sector_size=%u disk=%s\n",
                (unsigned)SLOT_BYTES,
                w->disk->sector_size,
                w->disk->traddr);
        return -1;
    }

    uint32_t lba_count = SLOT_BYTES / w->disk->sector_size;
    uint64_t lba = (uint64_t)vec_id * lba_count; // TODO 这里假设向量顺序存储，且id与lba直接对应，后面需要改成真正的索引结构

    struct io_waiter waiter = {.done = false, .ok = 0};

    int rc = spdk_nvme_ns_cmd_read(
        w->disk->ns,
        w->qpair,
        buf,
        lba,
        lba_count,
        io_done_cb,
        &waiter,
        0
    );

    if (rc != 0) {
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

    int lane = pick_lane(b->items[0].vec_id); // 这里只用vec[0]的id做分流
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
 *   2. 对 batch 里的每个 vec_id：
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
        SLOT_BYTES,
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
        printf("[stage %d lane %d] got batch qid=%lu count=%u first_vec=%u\n",
            w->stage_id, w->lane_id, in->qid, in->count, in->items[0].vec_id);
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

        for (uint16_t i = 0; i < in->count; i++) {
            uint32_t vec_id = in->items[i].vec_id;
            float acc = in->items[i].partial_sum;

            /* 1) 从当前 stage 对应的盘上读该向量的本段 segment */
            if (read_vec_segment(w, vec_id, buf) != 0) {
                fprintf(stderr, "[stage %d] read failed vec=%u disk=%s\n",
                        w->stage_id, vec_id, w->disk->traddr);
                continue;
            }

            /* 2) 计算 partial distance 并累加 */
            float part = partial_l2((float *)buf, app->query_segs[w->stage_id]);
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
                finalb->items[0].partial_sum = acc;

                queue_push(&app->topk.inq, finalb);
            } else {
                /*
                 * 5) 否则放入 surviving batch，等待发给下一 stage
                 */
                out->items[out->count].vec_id = vec_id;
                out->items[out->count].partial_sum = acc;
                out->count++;

                /*
                 * 如果 out 满了，就先发一个批次出去，再开新的 out
                 */
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
    float threshold)
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

    int lane = pick_lane(b->items[0].vec_id);
    queue_push(&app->workers[0][lane].inq, b);
    // 调试用输出
    printf("[submit] qid=%lu stage=%u count=%u first_vec=%u\n",
       b->qid, b->stage, b->count, b->items[0].vec_id);
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