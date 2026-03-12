#include "pipeline_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

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

    float threshold = 2000.0f; // 距离阈值

    /* 5. 创建 pipeline，并把 probe 好的 disks 传进去 */
    pipeline_app_t app;
    if (pipeline_init(&app, disks, stage_cores, topk_core, query_segs, threshold) != 0) {
        fprintf(stderr, "pipeline_init failed\n");
        return 1;
    }

    pipeline_start(&app);

    /* 6. 构造初始 batch 并提交给 stage0 */
    batch_t *b = calloc(1, sizeof(*b));
    b->magic = MAGIC_BATCH;
    b->qid = 1;
    b->stage = 0;

    for (uint32_t i = 0; i < 100 && i < MAX_BATCH; i++) {
        b->items[b->count].vec_id = i;
        b->items[b->count].partial_sum = 0.0f;
        b->count++;
    }

    pipeline_submit_initial_batch(&app, b);

    sleep(5);

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

    return 0;
}