#include "pipeline_stage.h"
#include "query_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>

extern struct app_ctx g_app;

static void init_disk_config(disk_ctx_t disks[NUM_STAGES])
{
    disks[0].traddr = "0000:65:00.0";
    disks[1].traddr = "0000:66:00.0";
    disks[2].traddr = "0000:67:00.0";
    disks[3].traddr = "0000:68:00.0";
}


static bool probe_cb(void *cb_ctx,
         const struct spdk_nvme_transport_id *trid,
         struct spdk_nvme_ctrlr_opts *opts)
{
    (void)cb_ctx;
    disk_ctx_t *disks = (disk_ctx_t *)cb_ctx;

    printf("[probe_cb] see traddr=%s trtype=%s\n",
           trid->traddr, trid->trstring);
    fflush(stdout);

    // 只 attach 我们关心的 4 块盘
    for (int i = 0; i < NUM_STAGES; i++) {
        if (strcmp(trid->traddr, disks[i].traddr) == 0) {
            printf("[probe_cb] allow %s\n", trid->traddr);
            fflush(stdout);
            return true;
        }
    }

    return false;
}

static void attach_cb(void *cb_ctx,
          const struct spdk_nvme_transport_id *trid,
          struct spdk_nvme_ctrlr *ctrlr,
          const struct spdk_nvme_ctrlr_opts *opts)
{
    (void)cb_ctx;
    disk_ctx_t *disks = (disk_ctx_t *)cb_ctx;

    for (int i = 0; i < NUM_STAGES; i++) {
        if (strcmp(trid->traddr, disks[i].traddr) == 0) {
            disks[i].ctrlr = ctrlr;
            disks[i].ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);

            if (!disks[i].ns || !spdk_nvme_ns_is_active(disks[i].ns)) {
                fprintf(stderr, "[attach_cb] ns inactive for %s\n", disks[i].traddr);
                exit(1);
            }

            disks[i].sector_size = spdk_nvme_ns_get_sector_size(disks[i].ns);

            printf("[attach_cb] attached %s ctrlr=%p ns=%p sector=%u\n",
                   disks[i].traddr,
                   (void *)disks[i].ctrlr,
                   (void *)disks[i].ns,
                   disks[i].sector_size);
            fflush(stdout);
            return;
        }
    }

    printf("[attach_cb] got unexpected device %s\n", trid->traddr);
    fflush(stdout);
}

static int probe_all_disks(disk_ctx_t disks[NUM_STAGES])
{
    int rc = spdk_nvme_probe(NULL, disks, probe_cb, attach_cb, NULL);
    if (rc != 0) {
        fprintf(stderr, "spdk_nvme_probe failed rc=%d\n", rc);
        return -1;
    }

    for (int i = 0; i < NUM_STAGES; i++) {
        if (!disks[i].ctrlr || !disks[i].ns) {
            fprintf(stderr, "disk %d not attached: traddr=%s\n",
                    i, disks[i].traddr);
            return -1;
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "spdk_4stage_ann";
    opts.mem_size = 1024;

    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "spdk_env_init failed\n");
        return 1;
    }
    disk_ctx_t disks[NUM_STAGES];
    memset(disks, 0, sizeof(disks));
    init_disk_config(disks);
    // init_worker_config();

    if (probe_all_disks(disks) != 0) {
        fprintf(stderr, "probe_all_disks failed\n");
        return 1;
    }

    /* 3. core 配置 */
    const int stage_cores[NUM_STAGES][WORKERS_PER_STAGE] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    int topk_core = 17;

    /* 4. query 相关与输入参数配置 */
    static float q0[SEG_DIM], q1[SEG_DIM], q2[SEG_DIM], q3[SEG_DIM];
    float *query_segs[NUM_STAGES] = {q0, q1, q2, q3};

    for (int i = 0; i < SEG_DIM; i++) {
        q0[i] = 0.0f;
        q1[i] = 0.0f;
        q2[i] = 0.0f;
        q3[i] = 0.0f;
    }

    float threshold =80000.0f; // PARAM: 提前终止阈值
    const char *ivf_meta_path = 
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/ivf_meta.bin"; // ivf_meta的路径
    const char *centroids_path = 
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/centroids_4096.bin"; // centroids.bin
    const char *query_fvecs_path =
        "/home/zhangshujie/ann_nic/sift/sift_query.fvecs";
    const char *pca_mean_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_mean.bin";
    const char *pca_components_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_components.bin";
    const char *pca_ev_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_explained_variance.bin";
    const char *pca_meta_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/pca_meta.bin";

    /* 5. 创建 pipeline，并把 probe 好的 disks 传进去 */
    pipeline_app_t app;
    if (pipeline_init(&app, disks, stage_cores, topk_core, query_segs, threshold, ivf_meta_path) != 0) {
        fprintf(stderr, "pipeline_init failed\n");
        return 1;
    }

    pipeline_start(&app);

    // 加载centroids.bin
    if (load_centroids_bin(centroids_path, &app.nlist, &app.centroids) != 0) {
        fprintf(stderr, "load_centroids_bin failed\n");
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    }

    // 检查前几个cluster的信息，确认ivf_meta.bin读入正确
    for (uint32_t cid = 0; cid < 10 && cid < app.ivf_meta.nlist; cid++) {
        const cluster_info_t *x = find_cluster_info(&app.ivf_meta, cid);
        if (x) {
            printf("cluster %u: start_lba=%lu num_vectors=%u num_lbas=%u\n",
                cid, x->start_lba, x->num_vectors, x->num_lbas);
        }
    }
    fflush(stdout);

    // 6.** 用query_set 替换单个query
    query_set_t qs;
    memset(&qs, 0, sizeof(qs));
    if (prepare_queries_with_pca(query_fvecs_path,
                             pca_mean_path,
                             pca_components_path,
                             pca_ev_path,
                             pca_meta_path,
                             &qs) != 0) {
        fprintf(stderr, "prepare_queries_with_pca failed\n");
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    }

    printf("[main] loaded %u PCA queries, dim=%u\n", qs.n_queries, qs.dim);
    fflush(stdout);

    uint32_t nprobe = 8;   /* PARAM nprobe*/
    uint32_t max_queries_to_run = qs.n_queries < 1 ? qs.n_queries : 1;
    /* 联调时先只跑前 3 个；稳定后改成 qs.n_queries */

    for (uint32_t qi = 0; qi < max_queries_to_run; qi++) {
        uint64_t qid = (uint64_t)qi + 1;
        const float *query = &qs.data[(size_t)qi * (size_t)qs.dim];

        printf("\n[main] submit query qi=%u qid=%lu nprobe=%u\n", qi, qid, nprobe);
        // printf("[main] first 8 dims: ");
        // for (uint32_t d = 0; d < 8 && d < qs.dim; d++) {
        //     printf("%.6f ", query[d]);
        // }
        // printf("\n");
        // fflush(stdout);

        if (submit_query(&app, qid, query, nprobe) != 0) {
            fprintf(stderr, "submit_query failed for qid=%lu\n", qid);
            free_query_set(&qs);
            pipeline_stop(&app);
            pipeline_join(&app);
            pipeline_destroy(&app);
            return 1;
        }

        int wrc = wait_query_done(&app, qid, 10000);  /* 最多等 10 秒 */
        if (wrc == 1) {
            fprintf(stderr, "wait_query_done timeout for qid=%lu\n", qid);
            free_query_set(&qs);
            pipeline_stop(&app);
            pipeline_join(&app);
            pipeline_destroy(&app);
            return 1;
        } else if (wrc != 0) {
            fprintf(stderr, "wait_query_done error for qid=%lu\n", qid);
            free_query_set(&qs);
            pipeline_stop(&app);
            pipeline_join(&app);
            pipeline_destroy(&app);
            return 1;
        }

        /* 调试：打印 query tracker 信息 */
        pthread_mutex_lock(&app.query_mu);
        for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
            query_tracker_t *qt = &app.queries[i];
            if (qt->qid == qid) {
                double latency_ms = 0.0;
                if (qt->done_ts_us > qt->submit_ts_us) {
                    latency_ms = (qt->done_ts_us - qt->submit_ts_us) / 1000.0;
                }

                printf("[query tracker] qid=%lu nprobe=%u probed_clusters=%u "
                    "initial_candidates=%lu submitted_batches=%u completed_batches=%u "
                    "outstanding=%u max_outstanding=%u done=%d latency_ms=%.3f "
                    "coarse_ms=%.3f submit_ms=%.3f\n",
                    qt->qid,
                    qt->nprobe,
                    qt->num_probed_clusters,
                    qt->initial_candidates,
                    qt->submitted_batches,
                    qt->completed_batches,
                    qt->outstanding_batches,
                    qt->max_outstanding_batches,
                    qt->done ? 1 : 0,
                    latency_ms,
                    (double)qt->coarse_search_us / 1000.0,
                    (double)qt->submit_candidates_us / 1000.0);

                for (int s = 0; s < NUM_STAGES; s++) {
                    double prune_ratio = 0.0;
                    double wall_ms = (double)qt->stage_wall_us[s] / 1000.0;
                    double io_ms = (double)qt->stage_io_us[s] / 1000.0;
                    double qsort_ms = (double)qt->stage_qsort_us[s] / 1000.0;
                    double compute_ms = wall_ms - io_ms - qsort_ms;
                    if (compute_ms < 0.0) {
                        compute_ms = 0.0;
                    }
                    double avg_batch_ms = 0.0;
                    double avg_bundle_us = 0.0;
                    if (qt->stage_batches[s] > 0) {
                        avg_batch_ms = wall_ms / (double)qt->stage_batches[s];
                    }
                    if (qt->stage_bundles_read[s] > 0) {
                        avg_bundle_us = (double)qt->stage_io_us[s] / (double)qt->stage_bundles_read[s];
                    }
                    if (qt->stage_in[s] > 0) {
                        prune_ratio = 1.0 - (double)qt->stage_out[s] / (double)qt->stage_in[s];
                    }
                    printf("  stage%d in=%lu out=%lu pruned=%lu batches=%lu bundles=%lu "
                        "prune_ratio=%.4f wall_ms=%.3f io_ms=%.3f qsort_ms=%.3f "
                        "est_compute_ms=%.3f avg_batch_ms=%.3f avg_io_per_bundle_us=%.3f\n",
                        s,
                        qt->stage_in[s],
                        qt->stage_out[s],
                        qt->stage_pruned[s],
                        qt->stage_batches[s],
                        qt->stage_bundles_read[s],
                        prune_ratio,
                        wall_ms,
                        io_ms,
                        qsort_ms,
                        compute_ms,
                        avg_batch_ms,
                        avg_bundle_us);
                }
                printf("  topk batches=%lu items=%lu wall_ms=%.3f avg_batch_ms=%.3f\n",
                    qt->topk_batches,
                    qt->topk_items,
                    (double)qt->topk_wall_us / 1000.0,
                    qt->topk_batches > 0
                        ? ((double)qt->topk_wall_us / 1000.0) / (double)qt->topk_batches
                        : 0.0);
                break;
            }
        }
        pthread_mutex_unlock(&app.query_mu);
        fflush(stdout);

        /* 如果你后面要按 query 打 topk，这里再补 */
    }

    if (qs.dim != FULL_DIM) {
        fprintf(stderr, "query dim mismatch: got %u expected %d\n", qs.dim, FULL_DIM);
        free_query_set(&qs);
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    }



    // 这里必须严格顺序，不然会内存泄漏
    pipeline_stop(&app);
    pipeline_join(&app);

    for (int s = 0; s < NUM_STAGES; s++) {
        printf("stage%d in=%lu out=%lu pruned=%lu\n",
            s,
            app.stage_in[s],
            app.stage_out[s],
            app.stage_pruned[s]);
    }
    fflush(stdout);

    printf("topk size=%u\n", app.topk_state.size);
    for (uint32_t i = 0; i < app.topk_state.size; i++) {
        printf("  vec=%u dist=%f\n",
            app.topk_state.items[i].vec_id,
            app.topk_state.items[i].dist);
    }

    free_query_set(&qs);
    pipeline_destroy(&app);
    return 0;
}
