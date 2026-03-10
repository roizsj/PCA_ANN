#include "query_ctx.h"
#include <stdlib.h>
#include <string.h>

#define MAX_Q 1024
static struct query_ctx *g_qtab[MAX_Q];

static inline uint32_t qslot(uint64_t qid) { return (uint32_t)(qid % MAX_Q); }

struct query_ctx *
query_ctx_create(const struct query_input *in)
{
    struct query_ctx *q = calloc(1, sizeof(*q));
    if (!q) return NULL;

    q->qid = in->qid;
    q->threshold = in->threshold_init;
    q->input_candidates = in->n_candidates;

    for (int s = 0; s < NUM_STAGES; s++) {
        q->query_segs[s] = in->query_segs[s];
    }

    g_qtab[qslot(q->qid)] = q;
    return q;
}

struct query_ctx *
query_ctx_get(uint64_t qid)
{
    struct query_ctx *q = g_qtab[qslot(qid)];
    return (q && q->qid == qid) ? q : NULL;
}

void
query_ctx_destroy(uint64_t qid)
{
    struct query_ctx *q = g_qtab[qslot(qid)];
    if (q && q->qid == qid) {
        free(q);
        g_qtab[qslot(qid)] = NULL;
    }
}