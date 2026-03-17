#include "pipeline_stage.h"

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

    /* 4. query */
    static float q0[SEG_DIM], q1[SEG_DIM], q2[SEG_DIM], q3[SEG_DIM];
    float *query_segs[NUM_STAGES] = {q0, q1, q2, q3};

    for (int i = 0; i < SEG_DIM; i++) {
        q0[i] = 0.0f;
        q1[i] = 0.0f;
        q2[i] = 0.0f;
        q3[i] = 0.0f;
    }

    float threshold =80000.0f; // 距离阈值 TODO
    const char *ivf_meta_path = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/ivf_meta.bin"; // ivf_meta的路径
    const char *centroids_path = "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/centroids_4096.bin"; // centroids.bin


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

    // /* 6. 构造初始 batch 并提交给 stage0 */
    // // 暂时是一个模拟版，只测试 cluster_id=0 的数据
    // batch_t *b = calloc(1, sizeof(*b));
    // if (!b) {
    //     perror("calloc batch");
    //     pipeline_stop(&app);
    //     pipeline_join(&app);
    //     pipeline_destroy(&app);
    //     return 1;
    // }

    // b->magic = MAGIC_BATCH;
    // b->qid = 1;
    // b->stage = 0;

    // /* 先固定测试一个cluster */
    // uint32_t test_cluster_id = 0;
    // const cluster_info_t *ci = find_cluster_info(&app.ivf_meta, test_cluster_id);
    // if (!ci) {
    //     fprintf(stderr, "cluster %u not found in ivf_meta\n", test_cluster_id);
    //     free(b);
    //     pipeline_stop(&app);
    //     pipeline_join(&app);
    //     pipeline_destroy(&app);
    //     return 1;
    // }

    // // 调试代码，看拿到了个什么聚类
    // printf("test_cluster_id=%u start_lba=%lu num_vectors=%u num_lbas=%u\n",
    //    test_cluster_id,
    //    ci->start_lba,
    //    ci->num_vectors,
    //    ci->num_lbas);
    // fflush(stdout);

    // uint32_t n = ci->num_vectors;
    // if (n > MAX_BATCH) {
    //     n = MAX_BATCH;
    // }

    // for (uint32_t local_idx = 0; local_idx < n; local_idx++) {
    //     b->items[b->count].vec_id = local_idx;   // 临时占位
    //     b->items[b->count].cluster_id = test_cluster_id;
    //     b->items[b->count].local_idx = local_idx;
    //     b->items[b->count].partial_sum = 0.0f;
    //     b->count++;
    // }

    // // 不提交空batch
    // if (b->count == 0) {
    //     fprintf(stderr, "initial batch is empty\n");
    //     free(b);
    //     pipeline_stop(&app);
    //     pipeline_join(&app);
    //     pipeline_destroy(&app);
    //     return 1;
    // }
    // pipeline_submit_initial_batch(&app, b);
    // sleep(5);

    /* 6. 构造一个真实 query，并通过 coarse IVF + submit_query 提交 */
    float query[FULL_DIM];
    for (int i = 0; i < FULL_DIM; i++) {
        query[i] = 0.0f;   /* 先用全0 query 联调；后面再换成真实query */
    }

    uint64_t qid = 1;
    uint32_t nprobe = 1;   /* 先从1开始最容易验证；后面可改成4/8/16 */

    if (submit_query(&app, qid, query, nprobe) != 0) {
        fprintf(stderr, "submit_query failed\n");
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    }

    int wrc = wait_query_done(&app, qid, 10000);  /* 最多等10秒 */
    if (wrc == 1) {
        fprintf(stderr, "wait_query_done timeout\n");
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    } else if (wrc != 0) {
        fprintf(stderr, "wait_query_done error\n");
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
            printf("[query tracker] qid=%lu nprobe=%u probed_clusters=%u "
                "initial_candidates=%lu submitted_batches=%u finished_batches=%u done=%d\n",
                qt->qid,
                qt->nprobe,
                qt->num_probed_clusters,
                qt->initial_candidates,
                qt->submitted_batches,
                qt->finished_batches,
                qt->done ? 1 : 0);
            break;
        }
    }
    pthread_mutex_unlock(&app.query_mu);
    fflush(stdout);

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

    pthread_mutex_lock(&app.topk_state.mu);
    printf("topk size=%u\n", app.topk_state.size);
    for (uint32_t i = 0; i < app.topk_state.size; i++) {
        printf("  vec=%u dist=%f\n",
            app.topk_state.items[i].vec_id,
            app.topk_state.items[i].dist);
    }
    pthread_mutex_unlock(&app.topk_state.mu);

    pipeline_destroy(&app); // 必须先unlock再destroy否则容易错
    return 0;
}