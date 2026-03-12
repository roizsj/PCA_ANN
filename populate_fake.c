#include "pipeline_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * 这个程序的职责：
 *  1. 初始化 SPDK 环境
 *  2. probe 4 块盘
 *  3. 把前 num_vecs 个向量的 4 个 segment 分别写到 4 块盘
 *
 * 写盘规则（deterministic fake）：
 *   value = vec_id * 0.1f + stage + dim * 0.01f
 *
 * 例如 vec_id=0:
 *   stage0 segment 开头 = 0.00, 0.01, 0.02, ...
 *   stage1 segment 开头 = 1.00, 1.01, 1.02, ...
 */

/* ------------------------------------------------------------
 * fake data generation
 * ------------------------------------------------------------ */

static float fake_value(uint32_t vec_id, uint8_t stage, int dim_in_seg)
{
    return (float)(vec_id * 0.1f + stage + dim_in_seg * 0.01f);
}

static void fill_fake_segment(uint32_t vec_id, uint8_t stage, float *dst)
{
    for (int i = 0; i < SEG_DIM; i++) {
        dst[i] = fake_value(vec_id, stage, i);
    }
}

/* ------------------------------------------------------------
 * probe callbacks
 * ------------------------------------------------------------ */

static bool
probe_cb(void *cb_ctx,
         const struct spdk_nvme_transport_id *trid,
         struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    disk_ctx_t *disks = (disk_ctx_t *)cb_ctx;

    for (int i = 0; i < NUM_STAGES; i++) {
        if (disks[i].traddr && strcmp(trid->traddr, disks[i].traddr) == 0) {
            printf("[probe_cb] allow %s\n", trid->traddr);
            fflush(stdout);
            return true;
        }
    }

    return false;
}

static void
attach_cb(void *cb_ctx,
          const struct spdk_nvme_transport_id *trid,
          struct spdk_nvme_ctrlr *ctrlr,
          const struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    disk_ctx_t *disks = (disk_ctx_t *)cb_ctx;

    for (int i = 0; i < NUM_STAGES; i++) {
        if (strcmp(trid->traddr, disks[i].traddr) == 0) {
            disks[i].ctrlr = ctrlr;
            disks[i].ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);

            if (!disks[i].ns || !spdk_nvme_ns_is_active(disks[i].ns)) {
                fprintf(stderr, "[attach_cb] inactive ns for %s\n", disks[i].traddr);
                exit(1);
            }

            disks[i].sector_size = spdk_nvme_ns_get_sector_size(disks[i].ns);

            printf("[attach_cb] attached %s sector=%u\n",
                   disks[i].traddr, disks[i].sector_size);
            fflush(stdout);
            return;
        }
    }
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

/* ------------------------------------------------------------
 * synchronous write helper
 * ------------------------------------------------------------ */

struct io_waiter {
    volatile bool done;
    int ok;
};

static void io_done_cb(void *arg, const struct spdk_nvme_cpl *cpl)
{
    struct io_waiter *wt = (struct io_waiter *)arg;
    wt->ok = !spdk_nvme_cpl_is_error(cpl);
    wt->done = true;
}

static int write_vec_segment(disk_ctx_t *disk, struct spdk_nvme_qpair *qpair,
                             uint32_t vec_id, void *buf)
{
    if (!disk || !disk->ns || !qpair) {
        fprintf(stderr, "[write_vec_segment] invalid disk/qpair\n");
        return -1;
    }

    if (disk->sector_size == 0) {
        fprintf(stderr, "[write_vec_segment] sector_size is 0 for %s\n", disk->traddr);
        return -1;
    }

    if (SLOT_BYTES < disk->sector_size || (SLOT_BYTES % disk->sector_size) != 0) {
        fprintf(stderr,
                "[write_vec_segment] invalid SLOT_BYTES=%u for sector_size=%u disk=%s\n",
                (unsigned)SLOT_BYTES,
                disk->sector_size,
                disk->traddr);
        return -1;
    }

    uint32_t lba_count = SLOT_BYTES / disk->sector_size;
    uint64_t lba = (uint64_t)vec_id * lba_count;

    struct io_waiter waiter = {.done = false, .ok = 0};

    int rc = spdk_nvme_ns_cmd_write(
        disk->ns,
        qpair,
        buf,
        lba,
        lba_count,
        io_done_cb,
        &waiter,
        0
    );

    if (rc != 0) {
        fprintf(stderr, "[write_vec_segment] submit failed rc=%d disk=%s vec=%u\n",
                rc, disk->traddr, vec_id);
        return -1;
    }

    while (!waiter.done) {
        spdk_nvme_qpair_process_completions(qpair, 0);
    }

    return waiter.ok ? 0 : -1;
}

/* ------------------------------------------------------------
 * main
 * ------------------------------------------------------------ */

int main(int argc, char **argv)
{
    /*
     * 用法：
     *   ./populate_fake 1000
     *
     * 表示把 vec_id = 0..999 写到 4 块盘
     */
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_vecs>\n", argv[0]);
        return 1;
    }

    uint32_t num_vecs = (uint32_t)strtoul(argv[1], NULL, 10);
    if (num_vecs == 0) {
        fprintf(stderr, "num_vecs must be > 0\n");
        return 1;
    }

    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "populate_fake";
    opts.mem_size = 1024;

    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "spdk_env_init failed\n");
        return 1;
    }

    /* 4 块盘配置，和 main.c 保持一致 */
    disk_ctx_t disks[NUM_STAGES];
    memset(disks, 0, sizeof(disks));

    disks[0].traddr = "0000:68:00.0";
    disks[1].traddr = "0000:66:00.0";
    disks[2].traddr = "0000:67:00.0";
    disks[3].traddr = "0000:65:00.0";

    if (probe_all_disks(disks) != 0) {
        fprintf(stderr, "probe_all_disks failed\n");
        return 1;
    }

    /* 每块盘创建一个 qpair 用来写入 */
    struct spdk_nvme_qpair *qpairs[NUM_STAGES] = {0};
    for (int s = 0; s < NUM_STAGES; s++) {
        qpairs[s] = spdk_nvme_ctrlr_alloc_io_qpair(disks[s].ctrlr, NULL, 0);
        if (!qpairs[s]) {
            fprintf(stderr, "alloc_io_qpair failed for stage=%d disk=%s\n",
                    s, disks[s].traddr);
            return 1;
        }
    }

    /* 每块盘一个 DMA buffer */
    void *bufs[NUM_STAGES] = {0};
    for (int s = 0; s < NUM_STAGES; s++) {
        bufs[s] = spdk_zmalloc(
            SLOT_BYTES,
            4096,
            NULL,
            SPDK_ENV_NUMA_ID_ANY,
            SPDK_MALLOC_DMA
        );
        if (!bufs[s]) {
            fprintf(stderr, "spdk_zmalloc failed for stage=%d\n", s);
            return 1;
        }
    }

    /* 正式写盘 */
    for (uint32_t vec_id = 0; vec_id < num_vecs; vec_id++) {
        for (int s = 0; s < NUM_STAGES; s++) {
            memset(bufs[s], 0, SLOT_BYTES);

            /* 前 SEG_DIM 个 float 填 segment，剩下 padding 为 0 */
            fill_fake_segment(vec_id, (uint8_t)s, (float *)bufs[s]);

            if (write_vec_segment(&disks[s], qpairs[s], vec_id, bufs[s]) != 0) {
                fprintf(stderr, "write failed vec=%u stage=%d disk=%s\n",
                        vec_id, s, disks[s].traddr);
                return 1;
            }
        }

        if ((vec_id % 1000) == 0) {
            printf("wrote vec_id=%u\n", vec_id);
            fflush(stdout);
        }
    }

    printf("populate_fake done: wrote %u vectors across 4 stages\n", num_vecs);
    fflush(stdout);

    for (int s = 0; s < NUM_STAGES; s++) {
        if (bufs[s]) spdk_free(bufs[s]);
        if (qpairs[s]) spdk_nvme_ctrlr_free_io_qpair(qpairs[s]);
    }

    return 0;
}