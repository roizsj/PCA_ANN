#include "app.h"
#include "query_ctx.h"
#include "topk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

extern struct app_ctx g_app;

static void init_disk_config(void)
{
    g_app.disks[0].traddr = "0000:68:00.0";
    g_app.disks[1].traddr = "0000:66:00.0";
    g_app.disks[2].traddr = "0000:67:00.0";
    g_app.disks[3].traddr = "0000:65:00.0";
}

static void init_worker_config(void)
{
    int worker_id = 0;
    int base_core[NUM_STAGES] = {1, 5, 9, 13};

    for (int s = 0; s < NUM_STAGES; s++) {
        for (int lane = 0; lane < WORKERS_PER_DISK; lane++) {
            struct worker_ctx *w = &g_app.workers[worker_id];
            w->worker_id = worker_id;
            w->stage_id = s;
            w->lane_id = lane;
            w->core_id = base_core[s] + lane;
            w->disk = &g_app.disks[s];
            w->qpair = NULL;
            worker_id++;
        }
    }
}

static bool probe_cb(void *cb_ctx,
         const struct spdk_nvme_transport_id *trid,
         struct spdk_nvme_ctrlr_opts *opts)
{
    (void)cb_ctx;
    (void)opts;

    printf("[probe_cb] see traddr=%s trtype=%s\n",
           trid->traddr, trid->trstring);
    fflush(stdout);

    // 只 attach 我们关心的 4 块盘
    for (int i = 0; i < NUM_STAGES; i++) {
        if (strcmp(trid->traddr, g_app.disks[i].traddr) == 0) {
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
    (void)opts;

    for (int i = 0; i < NUM_STAGES; i++) {
        struct disk_ctx *d = &g_app.disks[i];

        if (strcmp(trid->traddr, d->traddr) == 0) {
            d->ctrlr = ctrlr;
            d->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);

            if (!d->ns || !spdk_nvme_ns_is_active(d->ns)) {
                fprintf(stderr, "[attach_cb] ns inactive for %s\n", d->traddr);
                exit(1);
            }

            d->sector_size = spdk_nvme_ns_get_sector_size(d->ns);

            printf("[attach_cb] attached %s ctrlr=%p ns=%p sector=%u\n",
                   d->traddr,
                   (void *)d->ctrlr,
                   (void *)d->ns,
                   d->sector_size);
            fflush(stdout);
            return;
        }
    }

    printf("[attach_cb] got unexpected device %s\n", trid->traddr);
    fflush(stdout);
}

static int probe_all_disks(void)
{
    int rc = spdk_nvme_probe(NULL, NULL, probe_cb, attach_cb, NULL);
    if (rc != 0) {
        fprintf(stderr, "spdk_nvme_probe failed rc=%d\n", rc);
        return -1;
    }

    for (int i = 0; i < NUM_STAGES; i++) {
        if (!g_app.disks[i].ctrlr || !g_app.disks[i].ns) {
            fprintf(stderr, "disk %d not attached: traddr=%s\n",
                    i, g_app.disks[i].traddr);
            return -1;
        }
    }

    return 0;
}

static void bind_to_core(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        fprintf(stderr, "pthread_setaffinity_np failed core=%d rc=%d\n", core_id, rc);
        exit(1);
    }
}

struct smoke_waiter {
    volatile bool done;
    int success;
};

static void
smoke_read_done(void *arg, const struct spdk_nvme_cpl *cpl)
{
    struct smoke_waiter *w = (struct smoke_waiter *)arg;
    w->success = !spdk_nvme_cpl_is_error(cpl);
    w->done = true;
}

static int
smoke_read_one_slot(struct worker_ctx *w, uint32_t vec_id, void *buf, size_t buf_bytes)
{
    if (!w || !w->disk || !w->disk->ns || !w->qpair) {
        fprintf(stderr, "[smoke] invalid worker ctx\n");
        return -1;
    }

    uint32_t sector_size = spdk_nvme_ns_get_sector_size(w->disk->ns);
    uint32_t lba_count = (uint32_t)(buf_bytes / sector_size);
    uint64_t lba = (uint64_t)vec_id * lba_count;

    struct smoke_waiter waiter = {
        .done = false,
        .success = 0
    };

    int rc = spdk_nvme_ns_cmd_read(
        w->disk->ns,
        w->qpair,
        buf,
        lba,
        lba_count,
        smoke_read_done,
        &waiter,
        0
    );

    if (rc != 0) {
        fprintf(stderr,
                "[smoke] submit failed worker=%d stage=%d lane=%d disk=%s rc=%d\n",
                w->worker_id, w->stage_id, w->lane_id, w->disk->traddr, rc);
        return -1;
    }

    while (!waiter.done) {
        spdk_nvme_qpair_process_completions(w->qpair, 0);
    }

    return waiter.success ? 0 : -1;
}


static void *worker_main(void *arg)
{
    struct worker_ctx *w = (struct worker_ctx *)arg;

    bind_to_core(w->core_id);

    int actual_cpu = sched_getcpu(); // 验证是不是真的绑了核，后续可删

    w->qpair = spdk_nvme_ctrlr_alloc_io_qpair(w->disk->ctrlr, NULL, 0);
    if (!w->qpair) {
        fprintf(stderr, "alloc_io_qpair failed worker=%d stage=%d lane=%d disk=%s\n",
                w->worker_id, w->stage_id, w->lane_id, w->disk->traddr);
        return NULL;
    }

    printf("[worker] started worker=%d stage=%d lane=%d core=%d actual_cpu=%d disk=%s qpair=%p\n",
           w->worker_id, w->stage_id, w->lane_id, w->core_id, actual_cpu,
           w->disk->traddr, (void *)w->qpair);
    fflush(stdout);

    /* ---- smoke test begin ---- */
    void *buf = spdk_zmalloc(4096, 4096, NULL,
                             SPDK_ENV_NUMA_ID_ANY,
                             SPDK_MALLOC_DMA);
    if (!buf) {
        fprintf(stderr, "[smoke] spdk_zmalloc failed worker=%d\n", w->worker_id);
        spdk_nvme_ctrlr_free_io_qpair(w->qpair);
        w->qpair = NULL;
        return NULL;
    }

    memset(buf, 0, 512);

    int rc = smoke_read_one_slot(w, 0 /* vec_id=0 */, buf, 4096);
    if (rc == 0) {
        float *f = (float *)buf;
        printf("[worker smoke] OK worker=%d stage=%d lane=%d cpu=%d disk=%s "
               "sample=[%.4f %.4f %.4f %.4f]\n",
               w->worker_id, w->stage_id, w->lane_id, actual_cpu,
               w->disk->traddr,
               f[0], f[1], f[2], f[3]);
    } else {
        printf("[worker smoke] FAIL worker=%d stage=%d lane=%d cpu=%d disk=%s\n",
               w->worker_id, w->stage_id, w->lane_id, actual_cpu,
               w->disk->traddr);
    }
    fflush(stdout);

    spdk_free(buf);
    /* ---- smoke test end ---- */

    // /* 这里先挂个空循环，后面再塞 batch / stage 逻辑 */
    // while (1) {
    //     // 后面你可以在这里：
    //     // 1. 从队列取 batch
    //     // 2. 发 read
    //     // 3. process_completions
    //     // 4. partial distance
    //     // 5. prune / forward
    //     sleep(1);
    // }

    return NULL;
}

static void start_all_workers(void)
{
    for (int i = 0; i < TOTAL_WORKERS; i++) {
        int rc = pthread_create(&g_app.workers[i].tid, NULL, worker_main, &g_app.workers[i]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create failed for worker %d\n", i);
            exit(1);
        }
    }
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

    init_disk_config();
    init_worker_config();

    if (probe_all_disks() != 0) {
        fprintf(stderr, "probe_all_disks failed\n");
        return 1;
    }

    start_all_workers();

    for (int i = 0; i < TOTAL_WORKERS; i++) {
        pthread_join(g_app.workers[i].tid, NULL);
    }

    return 0;
}

// int main(int argc, char **argv)
// {
//     #ifdef FAKE_IO_TEST
//         printf("==== FAKE IO TEST MODE ====\n");
//     #endif
//     struct spdk_app_opts opts;

//     spdk_app_opts_init(&opts, sizeof(opts));
//     opts.name = "cascade_ann_batch";
//     opts.reactor_mask = "0x3f";

//     return spdk_app_start(&opts, app_start, NULL);
// }