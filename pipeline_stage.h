#ifndef PIPELINE_STAGE_H
#define PIPELINE_STAGE_H

/*
 * 这个头文件定义的是一个最小可运行的 4-stage ANN pipeline 接口。
 *
 * 设计目标：
 *  1. main.c 先完成 SPDK 环境初始化和 NVMe probe
 *  2. pipeline_stage.c 只负责“流水线执行”
 *  3. 每个 stage 对应一块盘
 *  4. 每个 stage 有多个 worker（这里默认 4 个）
 *  5. 每个 worker 固定绑一个 CPU core，并且独占一个 qpair
 */

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include <spdk/env.h>
#include <spdk/nvme.h>

/* -----------------------------
 * 全局常量
 * ----------------------------- */

/* 4-stage pipeline */
#define NUM_STAGES 4

/* 每个 stage 绑定多少个 worker / CPU 核 */
#define WORKERS_PER_STAGE 4

/* 单个 batch 最多容纳多少个 candidate */
#define MAX_BATCH 128

/* top-k 大小 */
#define TOPK 10

/*
 * 每个向量 segment 在盘上占一个 slot，多大由你自己定义。
 * 为了和 NVMe 常见逻辑块大小兼容，先设成 4096 比较稳。
 */
#define SLOT_BYTES 4096

/* 向量总维度，4-stage 切开后每段维度 */
#define FULL_DIM 128
#define SEG_DIM (FULL_DIM / NUM_STAGES)

/*
 * 用于识别 batch 是否被写坏 / 传错指针。
 * debug 时非常有用。
 */
#define MAGIC_BATCH 0xBADC0DEu

/* -----------------------------
 * candidate / batch 数据结构
 * ----------------------------- */

/*
 * 一个 candidate 的状态：
 *  - vec_id: 向量 ID
 *  - partial_sum: 当前已经累计到的距离
 */
typedef struct {
    uint32_t vec_id;
    float partial_sum;
} cand_item_t;

/*
 * 一个 batch：
 *  - qid: query id
 *  - stage: 当前这个 batch 应该进入哪个 stage
 *  - count: 当前 batch 中有多少个 candidate
 *  - items: candidate 列表
 */
typedef struct batch {
    uint32_t magic;
    uint64_t qid;
    uint8_t stage;
    uint16_t count;
    cand_item_t items[MAX_BATCH];
} batch_t;

/* -----------------------------
 * top-k 数据结构
 * ----------------------------- */

typedef struct {
    uint32_t vec_id;
    float dist;
} topk_item_t;

/*
 * 一个最小 top-k 状态：
 *  - items: 当前 top-k 集合
 *  - size: 当前已有多少个元素
 *  - mu: 保护 top-k 更新的锁
 */
typedef struct {
    topk_item_t items[TOPK];
    uint32_t size;
    pthread_mutex_t mu;
} topk_state_t;

/* -----------------------------
 * 一个简单的线程安全指针队列
 * ----------------------------- */

typedef struct qnode {
    void *ptr;
    struct qnode *next;
} qnode_t;

/*
 * 最简单的 MPSC 风格队列（这里先用 mutex + cond 实现）
 *
 * 用途：
 *  - 每个 stage worker 有一个输入队列
 *  - top-k worker 也有一个输入队列
 */
typedef struct {
    qnode_t *head;
    qnode_t *tail;
    bool closed;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} ptr_queue_t;

/* -----------------------------
 * 磁盘上下文
 * ----------------------------- */

/*
 * main.c 在 probe 后要把这些字段填好：
 *  - traddr
 *  - ctrlr
 *  - ns
 *  - sector_size
 *
 * pipeline_stage.c 不负责 probe，只使用这里已经准备好的信息。
 */
typedef struct {
    const char *traddr;
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    uint32_t sector_size;
} disk_ctx_t;

struct pipeline_app;

/* -----------------------------
 * stage worker / topk worker
 * ----------------------------- */

/*
 * 一个 stage worker：
 *  - 归属于某个 stage
 *  - 绑到一个 CPU core
 *  - 绑定某块盘
 *  - 拥有一个独占 qpair
 *  - 从自己的输入队列中取 batch
 */
typedef struct {
    struct pipeline_app *app;

    int worker_id;      /* 全局 worker 编号 */
    int stage_id;       /* 0~3 */
    int lane_id;        /* 当前 stage 内的 worker 编号 */
    int core_id;        /* 绑到哪个 CPU 核 */

    disk_ctx_t *disk;   /* 这个 worker 对应哪块盘 */
    struct spdk_nvme_qpair *qpair;

    pthread_t tid;
    ptr_queue_t inq;    /* 输入队列 */
} stage_worker_t;

/*
 * top-k worker：
 *  - 接收 stage3 输出的完整距离
 *  - 维护 top-k
 */
typedef struct {
    struct pipeline_app *app;
    pthread_t tid;
    ptr_queue_t inq;
    int core_id;
} topk_worker_t;

/* -----------------------------
 * pipeline 总上下文
 * ----------------------------- */

typedef struct pipeline_app {
    /*
     * 4 个 stage 对应的 4 块盘
     * 这些盘对象在 main.c probe 完成后传进来
     */
    disk_ctx_t disks[NUM_STAGES];

    /*
     * workers[stage][lane]
     * 例如：
     *   workers[0][0..3] -> stage0 的 4 个 worker
     *   workers[1][0..3] -> stage1 的 4 个 worker
     */
    stage_worker_t workers[NUM_STAGES][WORKERS_PER_STAGE];

    /* top-k 线程 */
    topk_worker_t topk;

    /* query 被切成 4 段后的 segment 指针 */
    float *query_segs[NUM_STAGES];

    /* 提前终止阈值 */
    float threshold;

    /* 一些简单统计 */
    uint64_t stage_in[NUM_STAGES];
    uint64_t stage_out[NUM_STAGES];
    uint64_t stage_pruned[NUM_STAGES];

    /* top-k 状态 */
    topk_state_t topk_state;
} pipeline_app_t;

/* ============================================================
 * 队列接口
 * ============================================================ */

/* 初始化一个空队列 */
void queue_init(ptr_queue_t *q);

/* 关闭队列，唤醒所有阻塞中的消费者 */
void queue_close(ptr_queue_t *q);

/* 向队列中推入一个指针 */
void queue_push(ptr_queue_t *q, void *ptr);

/* 从队列中弹出一个指针；若队列已关闭且为空，则返回 NULL */
void *queue_pop(ptr_queue_t *q);

/* ============================================================
 * helper 接口
 * ============================================================ */

/* 把当前线程绑定到指定 CPU core */
void bind_to_core(int core_id);

/* 计算一个 segment 和 query 对应 segment 的 partial L2 距离 */
float partial_l2(const float *x, const float *q);

/* ============================================================
 * pipeline 模块接口
 * ============================================================ */

/*
 * 初始化 pipeline。
 *
 * 注意：
 *  - 这里不做 nvme_probe！
 *  - 传入的 disks[] 必须已经在 main.c 里 probe 完成
 *  - stage_cores 决定每个 stage 的 4 个 worker 各绑哪个核
 *
 * 参数说明：
 *  - app: pipeline 总上下文
 *  - disks: 已经 probe 完成的 4 块盘
 *  - stage_cores: 每个 stage 的 worker -> CPU core 映射
 *  - topk_core: top-k worker 绑定到哪个核
 *  - query_segs: query 的 4 个 segment
 *  - threshold: 提前终止阈值
 */
int pipeline_init(
    pipeline_app_t *app,
    disk_ctx_t disks[NUM_STAGES],
    const int stage_cores[NUM_STAGES][WORKERS_PER_STAGE],
    int topk_core,
    float *query_segs[NUM_STAGES],
    float threshold);

/*
 * 启动所有 worker 线程：
 *  - 16 个 stage worker
 *  - 1 个 top-k worker
 */
void pipeline_start(pipeline_app_t *app);

/*
 * 向 stage0 提交一个初始 batch。
 *
 * 要求：
 *  - b->magic == MAGIC_BATCH
 *  - b->stage == 0
 */
void pipeline_submit_initial_batch(pipeline_app_t *app, batch_t *b);

/*
 * 关闭所有队列，通知线程退出。
 * 注意：
 * 这只是发“停止信号”，不等待线程真正结束。
 */
void pipeline_stop(pipeline_app_t *app);

/*
 * 等待所有线程退出。
 * 一般与 pipeline_stop 配套使用。
 */
void pipeline_join(pipeline_app_t *app);

#endif /* PIPELINE_STAGE_H */