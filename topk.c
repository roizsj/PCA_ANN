#include "topk.h"
#include "query_ctx.h"
#include <stdio.h>
#include <stdlib.h>

static void
topk_insert(struct topk_state *st, uint32_t vec_id, float dist)
{
    if (st->size < TOPK) {
        st->items[st->size].vec_id = vec_id;
        st->items[st->size].dist = dist;
        st->size++;
        return;
    }

    uint32_t worst = 0;
    for (uint32_t i = 1; i < st->size; i++) {
        if (st->items[i].dist > st->items[worst].dist) worst = i;
    }

    if (dist < st->items[worst].dist) {
        st->items[worst].vec_id = vec_id;
        st->items[worst].dist = dist;
    }
}

void
topk_accept_final_batch(void *arg)
{
    struct cand_batch *batch = arg;
    struct query_ctx *q = query_ctx_get(batch->qid);
    if (!q) {
        free(batch);
        return;
    }

    for (uint16_t i = 0; i < batch->count; i++) {
        topk_insert(&q->topk, batch->items[i].vec_id, batch->items[i].partial_sum);
        q->final_count++;
    }

    free(batch);
    // ---- 临时调试用 ----
    printf("[topk] qid=%lu final batch count=%u\n", batch->qid, batch->count);
    fflush(stdout);
}

void
topk_dump_query(struct query_ctx *q)
{
    printf("=== Query %lu summary ===\n", q->qid);
    printf("input=%lu final=%lu threshold=%.4f\n",
           q->input_candidates, q->final_count, q->threshold);

    for (int s = 0; s < NUM_STAGES; s++) {
        printf("stage%d: in=%lu out=%lu pruned=%lu ios=%lu io_bytes=%lu\n",
               s,
               q->stage_in[s],
               q->stage_out[s],
               q->stage_pruned[s],
               q->stage_ios[s],
               q->stage_io_bytes[s]);
    }

    printf("topk:\n");
    for (uint32_t i = 0; i < q->topk.size; i++) {
        printf("  vec=%u dist=%f\n", q->topk.items[i].vec_id, q->topk.items[i].dist);
    }
}