#include "pipeline_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static uint64_t query_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

void pipeline_free_query_tracker_segments(query_tracker_t *qt)
{
    if (!qt) {
        return;
    }

    for (int s = 0; s < NUM_STAGES; s++) {
        free(qt->query_segs[s]);
        qt->query_segs[s] = NULL;
    }
}

query_tracker_t *pipeline_find_query_tracker_locked(pipeline_app_t *app, uint64_t qid)
{
    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        if (app->queries[i].qid == qid) {
            return &app->queries[i];
        }
    }
    return NULL;
}

void pipeline_maybe_mark_query_done_locked(query_tracker_t *qt)
{
    if (!qt) {
        return;
    }

    if (!qt->done && qt->submission_done && qt->outstanding_batches == 0) {
        qt->done = true;
        qt->done_ts_us = query_now_us();
    }
}

query_tracker_t *register_query(pipeline_app_t *app,
                                uint64_t qid,
                                uint32_t nprobe,
                                uint32_t num_probed_clusters)
{
    if (!app || qid == 0) {
        return NULL;
    }

    pthread_mutex_lock(&app->query_mu);

    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        if (app->queries[i].qid == qid) {
            pthread_mutex_unlock(&app->query_mu);
            fprintf(stderr, "register_query: duplicate qid=%lu\n", qid);
            return NULL;
        }
    }

    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        if (app->queries[i].qid == 0) {
            query_tracker_t *qt = &app->queries[i];
            memset(qt, 0, sizeof(*qt));

            qt->qid = qid;
            qt->nprobe = nprobe;
            qt->num_probed_clusters = num_probed_clusters;
            qt->initial_candidates = 0;
            qt->submitted_batches = 0;
            qt->completed_batches = 0;
            qt->outstanding_batches = 0;
            qt->max_outstanding_batches = 0;
            qt->submission_done = false;
            qt->done = false;
            qt->prune_threshold = 0.0f;
            qt->coarse_search_us = 0;
            qt->submit_candidates_us = 0;
            memset(qt->stage_in, 0, sizeof(qt->stage_in));
            memset(qt->stage_out, 0, sizeof(qt->stage_out));
            memset(qt->stage_pruned, 0, sizeof(qt->stage_pruned));
            memset(qt->stage_batches, 0, sizeof(qt->stage_batches));
            memset(qt->stage_bundles_read, 0, sizeof(qt->stage_bundles_read));
            memset(qt->stage_wall_us, 0, sizeof(qt->stage_wall_us));
            memset(qt->stage_io_us, 0, sizeof(qt->stage_io_us));
            memset(qt->stage_qsort_us, 0, sizeof(qt->stage_qsort_us));
            qt->topk_batches = 0;
            qt->topk_items = 0;
            qt->topk_wall_us = 0;
            memset(&qt->query_topk, 0, sizeof(qt->query_topk));

            for (uint32_t s = 0; s < app->active_stages; s++) {
                uint32_t seg_dim = app->ivf_meta.header.shard_dims[s];
                qt->query_segs[s] = (float *)calloc(seg_dim, sizeof(float));
                if (!qt->query_segs[s]) {
                    perror("calloc query_seg");
                    pipeline_free_query_tracker_segments(qt);
                    memset(qt, 0, sizeof(*qt));
                    pthread_mutex_unlock(&app->query_mu);
                    return NULL;
                }
            }

            qt->submit_ts_us = query_now_us();
            qt->done_ts_us = 0;

            pthread_mutex_unlock(&app->query_mu);
            return qt;
        }
    }

    pthread_mutex_unlock(&app->query_mu);
    fprintf(stderr, "register_query: no free query slot\n");
    return NULL;
}

void mark_batch_finished(pipeline_app_t *app, uint64_t qid, uint32_t spawned_batches)
{
    if (!app || qid == 0) {
        return;
    }

    pthread_mutex_lock(&app->query_mu);

    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        query_tracker_t *qt = &app->queries[i];
        if (qt->qid == qid) {
            qt->completed_batches++;

            if (qt->outstanding_batches == 0) {
                pthread_mutex_unlock(&app->query_mu);
                fprintf(stderr, "mark_batch_finished: qid=%lu outstanding underflow\n", qid);
                return;
            }

            qt->outstanding_batches = qt->outstanding_batches - 1 + spawned_batches;
            if (qt->outstanding_batches > qt->max_outstanding_batches) {
                qt->max_outstanding_batches = qt->outstanding_batches;
            }

            pipeline_maybe_mark_query_done_locked(qt);

            pthread_mutex_unlock(&app->query_mu);
            return;
        }
    }

    pthread_mutex_unlock(&app->query_mu);
    fprintf(stderr, "mark_batch_finished: qid=%lu not found\n", qid);
}

int wait_query_done(pipeline_app_t *app, uint64_t qid, uint32_t timeout_ms)
{
    if (!app || qid == 0) {
        return -1;
    }

    const uint32_t sleep_us = 1000;
    uint32_t waited_ms = 0;

    while (timeout_ms == 0 || waited_ms < timeout_ms) {
        pthread_mutex_lock(&app->query_mu);

        for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
            query_tracker_t *qt = &app->queries[i];
            if (qt->qid == qid) {
                bool done = qt->done;
                pthread_mutex_unlock(&app->query_mu);
                if (done) {
                    return 0;
                }
                goto not_done;
            }
        }

        pthread_mutex_unlock(&app->query_mu);
        fprintf(stderr, "wait_query_done: qid=%lu not found\n", qid);
        return -1;

not_done:
        usleep(sleep_us);
        waited_ms += 1;
    }

    return 1;
}
