#include "stage_batch.h"
#include "layout.h"
#include "distance.h"
#include "query_ctx.h"
#include "topk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct app_ctx g_app;

static bool
probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
         struct spdk_nvme_ctrlr_opts *opts)
{
    struct stage_worker *w = cb_ctx;
    return strcmp(trid->traddr, w->traddr) == 0;
}

static void
attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
          struct spdk_nvme_ctrlr *ctrlr,
          const struct spdk_nvme_ctrlr_opts *opts)
{
    struct stage_worker *w = cb_ctx;
    if (strcmp(trid->traddr, w->traddr) != 0) return;
    w->ctrlr = ctrlr;
    w->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
}

static int
stage_poller(void *arg)
{
    struct stage_worker *w = arg;
    int n = spdk_nvme_qpair_process_completions(w->qpair, 0);
    return n > 0 ? SPDK_POLLER_BUSY : SPDK_POLLER_IDLE;
}

static void stage_read_complete(void *arg, const struct spdk_nvme_cpl *cpl);
static void finalize_batch_if_done(struct stage_worker *w);

int
stage_worker_init(struct stage_worker *w)
{
    int rc = spdk_nvme_probe(NULL, w, probe_cb, attach_cb, NULL);
    if (rc != 0 || !w->ctrlr || !w->ns) {
        fprintf(stderr, "probe failed stage=%u rc=%d\n", w->stage_id, rc);
        return -1;
    }

    w->qpair = spdk_nvme_ctrlr_alloc_io_qpair(w->ctrlr, NULL, 0);
    if (!w->qpair) return -1;

    w->poller = spdk_poller_register(stage_poller, w, 0);
    if (!w->poller) return -1;

    return 0;
}

void
stage_worker_fini(struct stage_worker *w)
{
    if (w->poller) spdk_poller_unregister(&w->poller);
    if (w->qpair) spdk_nvme_ctrlr_free_io_qpair(w->qpair);
}

void
stage_submit_batch(void *arg)
{
    struct cand_batch *batch = arg;
    struct stage_worker *w = &g_app.stages[batch->stage];
    struct query_ctx *q = query_ctx_get(batch->qid);
    uint32_t lba_count;

    if (!q) {
        free(batch);
        return;
    }

    q->stage_in[w->stage_id] += batch->count;

    w->active_batch = batch;
    w->next_batch_accum = calloc(1, sizeof(struct cand_batch));
    w->next_batch_accum->qid = batch->qid;
    w->next_batch_accum->stage = batch->stage + 1;
    w->inflight = 0;
    w->completed = 0;

    lba_count = layout_lba_count(w);

    for (uint16_t i = 0; i < batch->count; i++) {
        struct io_req_ctx *req = calloc(1, sizeof(*req));
        req->parent_batch = batch;
        req->worker = w;
        req->item_idx = i;
        req->dma_buf = spdk_zmalloc(IO_BYTES, 4096, NULL,
                                    SPDK_ENV_NUMA_ID_ANY,
                                    SPDK_MALLOC_DMA);

        uint64_t lba = layout_lba_for_vec(batch->items[i].vec_id, w->stage_id);

        int rc = spdk_nvme_ns_cmd_read(
            w->ns,
            w->qpair,
            req->dma_buf,
            lba,
            lba_count,
            stage_read_complete,
            req,
            0
        );

        if (rc != 0) {
            fprintf(stderr, "read submit failed stage=%u vec=%u rc=%d\n",
                    w->stage_id, batch->items[i].vec_id, rc);
            spdk_free(req->dma_buf);
            free(req);
            q->stage_pruned[w->stage_id]++; /* 简化处理：提交失败也算丢弃 */
            continue;
        }

        w->inflight++;
        q->stage_ios[w->stage_id]++;
        q->stage_io_bytes[w->stage_id] += IO_BYTES;
    }

    finalize_batch_if_done(w);
}

static void
append_survivor(struct cand_batch *dst, uint32_t vec_id, float partial_sum)
{
    if (dst->count >= BATCH_CAP) return;
    dst->items[dst->count].vec_id = vec_id;
    dst->items[dst->count].partial_sum = partial_sum;
    dst->count++;
}

static void
stage_read_complete(void *arg, const struct spdk_nvme_cpl *cpl)
{
    struct io_req_ctx *req = arg;
    struct stage_worker *w = req->worker;
    struct cand_batch *batch = req->parent_batch;
    struct query_ctx *q = query_ctx_get(batch->qid);
    struct cand_item *item = &batch->items[req->item_idx];

    if (!q) {
        spdk_free(req->dma_buf);
        free(req);
        return;
    }

    if (!spdk_nvme_cpl_is_error(cpl)) {
        float part = compute_partial_l2_from_slot(req->dma_buf, q->query_segs[w->stage_id]);
        float total = item->partial_sum + part;

        if (total <= q->threshold) {
            if (w->stage_id == NUM_STAGES - 1) {
                append_survivor(w->next_batch_accum, item->vec_id, total);
            } else {
                append_survivor(w->next_batch_accum, item->vec_id, total);
            }
        } else {
            q->stage_pruned[w->stage_id]++;
        }
    } else {
        q->stage_pruned[w->stage_id]++;
    }

    spdk_free(req->dma_buf);
    free(req);

    w->completed++;
    finalize_batch_if_done(w);
}

static void
finalize_batch_if_done(struct stage_worker *w)
{
    struct cand_batch *batch = w->active_batch;
    struct cand_batch *out = w->next_batch_accum;
    struct query_ctx *q;

    if (!batch) return;
    if (w->completed < w->inflight) return;

    q = query_ctx_get(batch->qid);
    if (q) {
        q->stage_out[w->stage_id] += out->count;
    }

    if (w->stage_id == NUM_STAGES - 1) {
        spdk_thread_send_msg(g_app.topk_thread, topk_accept_final_batch, out);
    } else {
        if (out->count > 0) {
            spdk_thread_send_msg(g_app.stages[w->stage_id + 1].thread,
                                 stage_submit_batch,
                                 out);
        } else {
            free(out);
        }
    }

    free(batch);
    w->active_batch = NULL;
    w->next_batch_accum = NULL;
    w->inflight = 0;
    w->completed = 0;
}