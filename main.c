#include "app.h"
#include "query_ctx.h"
#include "stage_batch.h"
#include "topk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern struct app_ctx g_app;

static void
init_stage_on_thread(void *arg)
{
    struct stage_worker *w = arg;
    if (stage_worker_init(w) != 0) {
        fprintf(stderr, "stage init failed %u\n", w->stage_id);
        abort();
    }
}

static void
dispatch_query_batches(void *arg)
{
    struct query_input *in = arg;
    struct query_ctx *q = query_ctx_create(in);
    if (!q) abort();

    uint32_t idx = 0;
    while (idx < in->n_candidates) {
        struct cand_batch *b = calloc(1, sizeof(*b));
        b->qid = in->qid;
        b->stage = 0;

        while (idx < in->n_candidates && b->count < BATCH_CAP) {
            b->items[b->count].vec_id = in->candidate_ids[idx];
            b->items[b->count].partial_sum = 0.0f;
            b->count++;
            idx++;
        }

        spdk_thread_send_msg(g_app.stages[0].thread, stage_submit_batch, b);
    }
}

static void
app_start(void *arg1)
{
    (void)arg1;

    g_app.stages[0] = (struct stage_worker){ .stage_id = 0, .core_id = 1, .traddr = "0000:5e:00.0" };
    g_app.stages[1] = (struct stage_worker){ .stage_id = 1, .core_id = 2, .traddr = "0000:af:00.0" };
    g_app.stages[2] = (struct stage_worker){ .stage_id = 2, .core_id = 3, .traddr = "0000:d8:00.0" };
    g_app.stages[3] = (struct stage_worker){ .stage_id = 3, .core_id = 4, .traddr = "0000:e1:00.0" };

    for (int i = 0; i < NUM_STAGES; i++) {
        char name[32];
        struct spdk_cpuset mask;

        snprintf(name, sizeof(name), "stage-%d", i);
        spdk_cpuset_zero(&mask);
        spdk_cpuset_set_cpu(&mask, g_app.stages[i].core_id, true);

        g_app.stages[i].thread = spdk_thread_create(name, &mask);
        if (!g_app.stages[i].thread) abort();

        spdk_thread_send_msg(g_app.stages[i].thread, init_stage_on_thread, &g_app.stages[i]);
    }

    {
        struct spdk_cpuset mask;
        spdk_cpuset_zero(&mask);
        spdk_cpuset_set_cpu(&mask, 5, true);
        g_app.topk_thread = spdk_thread_create("topk", &mask);
        if (!g_app.topk_thread) abort();
    }

    static float q0[SEG_DIM], q1[SEG_DIM], q2[SEG_DIM], q3[SEG_DIM];
    static uint32_t cand_ids[10000];

    for (uint32_t i = 0; i < 10000; i++) cand_ids[i] = i;

    struct query_input *qin = calloc(1, sizeof(*qin));
    qin->qid = 1;
    qin->threshold_init = 42.0f;
    qin->query_segs[0] = q0;
    qin->query_segs[1] = q1;
    qin->query_segs[2] = q2;
    qin->query_segs[3] = q3;
    qin->n_candidates = 10000;
    qin->candidate_ids = cand_ids;

    dispatch_query_batches(qin);
}

int
main(int argc, char **argv)
{
    struct spdk_app_opts opts;

    spdk_app_opts_init(&opts, sizeof(opts));
    opts.name = "cascade_ann_batch";
    opts.reactor_mask = "0x3f";

    return spdk_app_start(&opts, app_start, NULL);
}