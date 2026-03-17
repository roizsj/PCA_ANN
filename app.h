#ifndef APP_H
#define APP_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include <spdk/stdinc.h>
#include <spdk/env.h>
#include <spdk/event.h>
#include <spdk/thread.h>
#include <spdk/nvme.h>
#include <spdk/util.h>

#define NUM_STAGES 4  // 流水线段数目
#define WORKERS_PER_DISK 4 // 每个盘绑多少个核
#define TOTAL_WORKERS (NUM_STAGES * WORKERS_PER_DISK) // 盘一共用多少个核
#define TOPK 10

#define FULL_DIM 128
#define SEG_DIM (FULL_DIM / NUM_STAGES)
#define SEG_BYTES (SEG_DIM * sizeof(float))

/* 先按每条 segment 占一个 4KiB slot */
#define IO_BYTES 4096

/* 单个 batch 最多多少条候选 */
#define BATCH_CAP 256

/* 一个LBA（最小读盘粒度）包含多少个向量的segment*/
#define SEGMENTS_PER_LBA (IO_BYTES / SEG_BYTES)

/* 最大并发查询数 */
#define MAX_QUERY_IN_FLIGHT 1024

struct stage_worker;
struct query_ctx;
struct cand_batch;
struct cand_item;

struct cand_item {
    uint32_t vec_id;
    float partial_sum;
};

struct cand_batch {
    uint64_t qid;
    uint8_t stage;          /* 当前要进入的 stage */
    uint16_t count;
    struct cand_item items[BATCH_CAP];
};

struct query_input {
    uint64_t qid;
    float threshold_init;
    float *query_segs[NUM_STAGES];
    uint32_t n_candidates;
    uint32_t *candidate_ids;
};

struct topk_item {
    uint32_t vec_id;
    float dist;
};

struct topk_state {
    struct topk_item items[TOPK];
    uint32_t size;
};

struct query_ctx {
    uint64_t qid;
    float threshold;
    float *query_segs[NUM_STAGES];
    struct topk_state topk;

    uint64_t input_candidates;
    uint64_t stage_in[NUM_STAGES];
    uint64_t stage_out[NUM_STAGES];
    uint64_t stage_pruned[NUM_STAGES];
    uint64_t stage_ios[NUM_STAGES];
    uint64_t stage_io_bytes[NUM_STAGES];
    uint64_t final_count;
};

struct io_req_ctx {
    struct cand_batch *parent_batch;
    struct stage_worker *worker;
    uint16_t item_idx;
    void *dma_buf;
};

struct stage_worker {
    uint8_t stage_id;
    uint32_t core_id;
    const char *traddr;

    struct spdk_thread *thread;
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
    struct spdk_poller *poller;

    struct cand_batch *active_batch;
    uint16_t inflight;
    uint16_t completed;

    /* 本轮 surviving candidates 暂存 */
    struct cand_batch *next_batch_accum;
};

struct disk_ctx {
    const char *traddr;                  // 例如 "0000:68:00.0"
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    uint32_t sector_size;
};

struct worker_ctx {
    int worker_id;
    int stage_id;                        // 0~3
    int lane_id;                         // 0~3
    int core_id;                         // 绑的核
    struct disk_ctx *disk;               // 指向所属盘
    struct spdk_nvme_qpair *qpair;       // 每个 worker 独占一个 qpair
    pthread_t tid;
};
#endif