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

#define NUM_STAGES 4
#define TOPK 10

#define FULL_DIM 128
#define SEG_DIM (FULL_DIM / NUM_STAGES)
#define SEG_BYTES (SEG_DIM * sizeof(float))

/* 先按每条 segment 占一个 4KiB slot */
#define IO_BYTES 4096

/* 单个 batch 最多多少条候选 */
#define BATCH_CAP 256

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

struct app_ctx {
    struct stage_worker stages[NUM_STAGES];
    struct spdk_thread *topk_thread;
};

extern struct app_ctx g_app;

#endif