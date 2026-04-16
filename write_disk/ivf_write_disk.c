#define _GNU_SOURCE
#include <spdk/env.h>
#include <spdk/nvme.h>

#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 128
#define NUM_SHARDS 4
#define SHARD_DIM 32
#define SHARD_BYTES (SHARD_DIM * sizeof(float))   // 128B
#define MAGIC_META 0x49564633u                    // "IVF3"

typedef struct {
    uint32_t n;
    uint32_t d;
    float *data;   // n * d
} FvecsData;

typedef struct {
    uint32_t nlist;
    uint32_t d;
    float *centroids; // nlist * d
} CentroidData;

typedef struct {
    uint32_t n;
    uint32_t *labels;   // labels[vec_id] = cluster_id
} CodebookData;

typedef struct {
    uint32_t cluster_id;
    uint64_t start_lba;     // cluster 从哪个 LBA 开始（4盘相同）
    uint32_t num_vectors;   // cluster 实际向量数
    uint32_t num_lbas;      // ceil(num_vectors / vectors_per_lba)
    float centroid[DIM];
} ClusterMeta;

typedef struct {
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
    char traddr[128];
    uint32_t sector_size;
} DiskTarget;

typedef struct {
    volatile int done;
    int status;
} IoCtx;

typedef struct {
    const char *input_fvecs;
    const char *centroids_bin;
    const char *codebook_bin;
    const char *meta_out;
    const char *sorted_ids_out;   // 可选，保存排序后的 vec_id 顺序，方便调试
    const char *disk_addrs[NUM_SHARDS];
    uint64_t base_lba;
} AppConfig;

typedef struct {
    AppConfig cfg;
    DiskTarget disks[NUM_SHARDS];
    int found_disks;
} AppState;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t shard_dim;
    uint32_t vectors_per_lba;
    uint32_t nlist;
    uint32_t num_vectors;
    uint32_t sector_size;
    uint64_t base_lba;
} MetaHeader;

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

static void die_msg(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t sz) {
    void *p = malloc(sz);
    if (!p) {
        fprintf(stderr, "malloc failed: %zu\n", sz);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "calloc failed: %zu x %zu\n", n, sz);
        exit(EXIT_FAILURE);
    }
    return p;
}

static FvecsData read_fvecs(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die("fopen fvecs");

    fseek(fp, 0, SEEK_END);
    long file_sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    int32_t dim = 0;
    if (fread(&dim, sizeof(int32_t), 1, fp) != 1) {
        die_msg("failed to read fvecs dim");
    }
    if (dim != DIM) {
        fprintf(stderr, "fvecs dim mismatch: got %d expected %d\n", dim, DIM);
        exit(EXIT_FAILURE);
    }

    const long row_bytes = sizeof(int32_t) + DIM * (long)sizeof(float);
    if (file_sz % row_bytes != 0) {
        fprintf(stderr, "invalid fvecs file size: %ld\n", file_sz);
        exit(EXIT_FAILURE);
    }

    uint32_t n = (uint32_t)(file_sz / row_bytes);
    float *data = (float *)xmalloc((size_t)n * DIM * sizeof(float));

    rewind(fp);
    for (uint32_t i = 0; i < n; i++) {
        int32_t row_dim = 0;
        if (fread(&row_dim, sizeof(int32_t), 1, fp) != 1) {
            die_msg("failed to read row dim");
        }
        if (row_dim != DIM) {
            fprintf(stderr, "row %u dim mismatch: got %d expected %d\n", i, row_dim, DIM);
            exit(EXIT_FAILURE);
        }
        if (fread(data + (size_t)i * DIM, sizeof(float), DIM, fp) != DIM) {
            die_msg("failed to read row vector");
        }
    }

    fclose(fp);

    FvecsData out = {
        .n = n,
        .d = DIM,
        .data = data
    };
    return out;
}

/*
 * centroids.bin:
 * [u32 nlist][u32 dim][float centroids[nlist * dim]]
 */
static CentroidData read_centroids_bin(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die("fopen centroids");

    uint32_t nlist = 0, d = 0;
    if (fread(&nlist, sizeof(uint32_t), 1, fp) != 1) die_msg("read nlist failed");
    if (fread(&d, sizeof(uint32_t), 1, fp) != 1) die_msg("read dim failed");

    if (d != DIM) {
        fprintf(stderr, "centroid dim mismatch: got %u expected %d\n", d, DIM);
        exit(EXIT_FAILURE);
    }

    float *centroids = (float *)xmalloc((size_t)nlist * DIM * sizeof(float));
    if (fread(centroids, sizeof(float), (size_t)nlist * DIM, fp) != (size_t)nlist * DIM) {
        die_msg("read centroids body failed");
    }

    fclose(fp);

    CentroidData out = {
        .nlist = nlist,
        .d = d,
        .centroids = centroids
    };
    return out;
}

/*
 * codebook.bin:
 * [u32 num_vectors][u32 label0][u32 label1]...[u32 labelN-1]
 */
static CodebookData read_codebook_bin(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die("fopen codebook");

    uint32_t n = 0;
    if (fread(&n, sizeof(uint32_t), 1, fp) != 1) {
        die_msg("read codebook n failed");
    }

    uint32_t *labels = (uint32_t *)xmalloc((size_t)n * sizeof(uint32_t));
    if (fread(labels, sizeof(uint32_t), n, fp) != n) {
        die_msg("read codebook labels failed");
    }

    fclose(fp);

    CodebookData out = {
        .n = n,
        .labels = labels
    };
    return out;
}

static void io_complete(void *arg, const struct spdk_nvme_cpl *cpl) {
    IoCtx *ctx = (IoCtx *)arg;
    ctx->status = spdk_nvme_cpl_is_error(cpl) ? -EIO : 0;
    ctx->done = 1;
}

static void wait_io(struct spdk_nvme_qpair *qpair, IoCtx *ctx) {
    while (!ctx->done) {
        int rc = spdk_nvme_qpair_process_completions(qpair, 0);
        if (rc < 0) {
            fprintf(stderr, "process completions failed rc=%d\n", rc);
            exit(EXIT_FAILURE);
        }
    }
    if (ctx->status != 0) {
        fprintf(stderr, "I/O completion error: %d\n", ctx->status);
        exit(EXIT_FAILURE);
    }
}

static bool probe_cb(void *cb_ctx,
                     const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts) {
    (void)opts;
    AppState *app = (AppState *)cb_ctx;

    for (int i = 0; i < NUM_SHARDS; i++) {
        if (strcmp(trid->traddr, app->cfg.disk_addrs[i]) == 0) {
            return true;
        }
    }
    return false;
}

static void attach_cb(void *cb_ctx,
                      const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts) {
    (void)opts;
    AppState *app = (AppState *)cb_ctx;

    for (int i = 0; i < NUM_SHARDS; i++) {
        if (strcmp(trid->traddr, app->cfg.disk_addrs[i]) == 0) {
            struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
            if (!ns || !spdk_nvme_ns_is_active(ns)) {
                fprintf(stderr, "disk %s ns1 inactive\n", trid->traddr);
                exit(EXIT_FAILURE);
            }

            app->disks[i].ctrlr = ctrlr;
            app->disks[i].ns = ns;
            app->disks[i].qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, NULL, 0);
            if (!app->disks[i].qpair) {
                fprintf(stderr, "alloc qpair failed for %s\n", trid->traddr);
                exit(EXIT_FAILURE);
            }
            snprintf(app->disks[i].traddr, sizeof(app->disks[i].traddr), "%s", trid->traddr);
            app->disks[i].sector_size = spdk_nvme_ns_get_sector_size(ns);
            app->found_disks++;

            fprintf(stderr, "[attach] disk[%d] %s sector=%u\n",
                    i, app->disks[i].traddr, app->disks[i].sector_size);
        }
    }
}

static void init_spdk_and_attach(AppState *app) {
    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "ivf_pack_writer_spdk_codebook";

    if (spdk_env_init(&opts) < 0) {
        die_msg("spdk_env_init failed");
    }

    if (spdk_nvme_probe(NULL, app, probe_cb, attach_cb, NULL) != 0) {
        die_msg("spdk_nvme_probe failed");
    }

    if (app->found_disks != NUM_SHARDS) {
        fprintf(stderr, "expected %d disks, found %d\n", NUM_SHARDS, app->found_disks);
        exit(EXIT_FAILURE);
    }

    uint32_t sector_size = app->disks[0].sector_size;
    if (sector_size != 4096) {
        fprintf(stderr,
                "this program currently requires 4096B sector/LBA; got %u\n",
                sector_size);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < NUM_SHARDS; i++) {
        if (app->disks[i].sector_size != sector_size) {
            die_msg("all 4 disks must have the same sector size");
        }
    }
}

static void cleanup_spdk(AppState *app) {
    for (int i = 0; i < NUM_SHARDS; i++) {
        if (app->disks[i].qpair) {
            spdk_nvme_ctrlr_free_io_qpair(app->disks[i].qpair);
            app->disks[i].qpair = NULL;
        }
    }
    for (int i = 0; i < NUM_SHARDS; i++) {
        if (app->disks[i].ctrlr) {
            spdk_nvme_detach(app->disks[i].ctrlr);
            app->disks[i].ctrlr = NULL;
        }
    }
}

static void save_meta_file(const char *path,
                           uint32_t nlist,
                           uint32_t num_vectors,
                           uint32_t sector_size,
                           uint32_t vectors_per_lba,
                           uint64_t base_lba,
                           const ClusterMeta *cluster_meta) {
    FILE *fp = fopen(path, "wb");
    if (!fp) die("fopen meta out");

    MetaHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = MAGIC_META;
    hdr.version = 1;
    hdr.dim = DIM;
    hdr.shard_dim = SHARD_DIM;
    hdr.vectors_per_lba = vectors_per_lba;
    hdr.nlist = nlist;
    hdr.num_vectors = num_vectors;
    hdr.sector_size = sector_size;
    hdr.base_lba = base_lba;

    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(cluster_meta, sizeof(ClusterMeta), nlist, fp);
    fclose(fp);
}

static void save_sorted_ids_file(const char *path,
                                 uint32_t num_vectors,
                                 const uint32_t *sorted_vec_ids) {
    FILE *fp = fopen(path, "wb");
    if (!fp) die("fopen sorted ids");

    fwrite(&num_vectors, sizeof(uint32_t), 1, fp);
    fwrite(sorted_vec_ids, sizeof(uint32_t), num_vectors, fp);
    fclose(fp);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --input base.fvecs --centroids centroids.bin --codebook codebook.bin --meta ivf_meta.bin \\\n"
        "     --disk0 0000:5e:00.0 --disk1 0000:60:00.0 --disk2 0000:61:00.0 --disk3 0000:62:00.0 \\\n"
        "     [--sorted-ids sorted_vec_ids.bin] [--base-lba 0]\n",
        prog);
}

static void parse_args(int argc, char **argv, AppConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    static struct option long_opts[] = {
        {"input", required_argument, 0, 'i'},
        {"centroids", required_argument, 0, 'c'},
        {"codebook", required_argument, 0, 'k'},
        {"meta", required_argument, 0, 'm'},
        {"sorted-ids", required_argument, 0, 's'},
        {"disk0", required_argument, 0, 1000},
        {"disk1", required_argument, 0, 1001},
        {"disk2", required_argument, 0, 1002},
        {"disk3", required_argument, 0, 1003},
        {"base-lba", required_argument, 0, 'b'},
        {0, 0, 0, 0}
    };

    int opt, idx = 0;
    while ((opt = getopt_long(argc, argv, "i:c:k:m:s:b:", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'i': cfg->input_fvecs = optarg; break;
            case 'c': cfg->centroids_bin = optarg; break;
            case 'k': cfg->codebook_bin = optarg; break;
            case 'm': cfg->meta_out = optarg; break;
            case 's': cfg->sorted_ids_out = optarg; break;
            case 'b': cfg->base_lba = strtoull(optarg, NULL, 10); break;
            case 1000: cfg->disk_addrs[0] = optarg; break;
            case 1001: cfg->disk_addrs[1] = optarg; break;
            case 1002: cfg->disk_addrs[2] = optarg; break;
            case 1003: cfg->disk_addrs[3] = optarg; break;
            default:
                usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!cfg->input_fvecs || !cfg->centroids_bin || !cfg->codebook_bin || !cfg->meta_out ||
        !cfg->disk_addrs[0] || !cfg->disk_addrs[1] ||
        !cfg->disk_addrs[2] || !cfg->disk_addrs[3]) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    AppState app;
    memset(&app, 0, sizeof(app));
    parse_args(argc, argv, &app.cfg);

    init_spdk_and_attach(&app);

    const uint32_t sector_size = app.disks[0].sector_size;
    const uint32_t vectors_per_lba = sector_size / SHARD_BYTES; // 4096 / 128 = 32
    if (vectors_per_lba == 0) {
        die_msg("vectors_per_lba computed as 0");
    }

    fprintf(stderr, "[info] sector_size=%u, shard_bytes=%u, vectors_per_lba=%u\n",
            sector_size, (unsigned)SHARD_BYTES, vectors_per_lba);

    FvecsData vecs = read_fvecs(app.cfg.input_fvecs);
    CentroidData cents = read_centroids_bin(app.cfg.centroids_bin);
    CodebookData codebook = read_codebook_bin(app.cfg.codebook_bin);

    if (cents.d != DIM) {
        die_msg("centroid dimension mismatch");
    }
    if (codebook.n != vecs.n) {
        fprintf(stderr, "codebook size mismatch: codebook=%u vectors=%u\n",
                codebook.n, vecs.n);
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "[info] loaded vectors=%u dim=%u, nlist=%u, codebook_n=%u\n",
            vecs.n, vecs.d, cents.nlist, codebook.n);

    /*
     * Step 1: 用 codebook 直接统计每个 cluster 的大小
     */
    uint32_t *cluster_sizes = (uint32_t *)xcalloc(cents.nlist, sizeof(uint32_t));

    for (uint32_t i = 0; i < vecs.n; i++) {
        uint32_t cid = codebook.labels[i];
        if (cid >= cents.nlist) {
            fprintf(stderr, "invalid cluster id in codebook: vec_id=%u cid=%u nlist=%u\n",
                    i, cid, cents.nlist);
            exit(EXIT_FAILURE);
        }
        cluster_sizes[cid]++;
    }

    /*
     * Step 2: counting sort by cluster_id
     * sorted_vec_ids[pos] = vec_id
     */
    uint64_t *cluster_offsets = (uint64_t *)xcalloc(cents.nlist + 1, sizeof(uint64_t));
    for (uint32_t c = 0; c < cents.nlist; c++) {
        cluster_offsets[c + 1] = cluster_offsets[c] + cluster_sizes[c];
    }

    uint64_t *write_ptrs = (uint64_t *)xmalloc((size_t)cents.nlist * sizeof(uint64_t));
    memcpy(write_ptrs, cluster_offsets, (size_t)cents.nlist * sizeof(uint64_t));

    uint32_t *sorted_vec_ids = (uint32_t *)xmalloc((size_t)vecs.n * sizeof(uint32_t));
    for (uint32_t vec_id = 0; vec_id < vecs.n; vec_id++) {
        uint32_t cid = codebook.labels[vec_id];
        uint64_t pos = write_ptrs[cid]++;
        sorted_vec_ids[pos] = vec_id;
    }

    free(write_ptrs);

    /*
     * Step 3: 计算每个 cluster 的 start_lba
     * 每个 cluster 从新的 LBA 边界开始
     */
    ClusterMeta *cluster_meta = (ClusterMeta *)xcalloc(cents.nlist, sizeof(ClusterMeta));
    uint64_t cur_lba = app.cfg.base_lba;

    for (uint32_t c = 0; c < cents.nlist; c++) {
        uint32_t sz = cluster_sizes[c];
        uint32_t num_lbas = (sz + vectors_per_lba - 1) / vectors_per_lba;

        cluster_meta[c].cluster_id = c;
        cluster_meta[c].start_lba = cur_lba;
        cluster_meta[c].num_vectors = sz;
        cluster_meta[c].num_lbas = num_lbas;
        memcpy(cluster_meta[c].centroid,
               cents.centroids + (size_t)c * DIM,
               DIM * sizeof(float));

        cur_lba += num_lbas;
    }

    const uint64_t total_lbas = cur_lba - app.cfg.base_lba;
    fprintf(stderr, "[info] total_lbas_to_write=%" PRIu64 "\n", total_lbas);

    if (app.cfg.sorted_ids_out) {
        save_sorted_ids_file(app.cfg.sorted_ids_out, vecs.n, sorted_vec_ids);
        fprintf(stderr, "[info] saved sorted ids to %s\n", app.cfg.sorted_ids_out);
    }

    /*
     * Step 4: 准备 4 个可复用 DMA buffer
     */
    uint8_t *dma_buf[NUM_SHARDS] = {0};
    for (int s = 0; s < NUM_SHARDS; s++) {
        dma_buf[s] = spdk_zmalloc(sector_size, sector_size, NULL,
                                  SPDK_ENV_LCORE_ID_ANY, SPDK_MALLOC_DMA);
        if (!dma_buf[s]) {
            die_msg("spdk_zmalloc failed");
        }
    }

    /*
     * Step 5: 写盘
     *
     * 对每个 cluster:
     *   每个 LBA bundle 最多打包 32 条向量
     *   每个盘一个 4KB buffer
     *   lane k 对应 bundle 内第 k 条向量
     *
     * 布局：
     *   disk s, LBA x:
     *     [vec0 shard_s][vec1 shard_s]...[vec31 shard_s]
     *     每个 shard 128B
     */
    for (uint32_t c = 0; c < cents.nlist; c++) {
        uint32_t num_vec = cluster_meta[c].num_vectors;
        uint32_t num_lbas = cluster_meta[c].num_lbas;
        uint64_t start_pos = cluster_offsets[c];
        uint64_t start_lba = cluster_meta[c].start_lba;

        for (uint32_t b = 0; b < num_lbas; b++) {
            for (int s = 0; s < NUM_SHARDS; s++) {
                memset(dma_buf[s], 0, sector_size);
            }

            for (uint32_t lane = 0; lane < vectors_per_lba; lane++) {
                uint32_t local_idx = b * vectors_per_lba + lane;
                if (local_idx >= num_vec) {
                    break;
                }

                uint32_t vec_id = sorted_vec_ids[start_pos + local_idx];
                const float *vec = vecs.data + (size_t)vec_id * DIM;

                for (int s = 0; s < NUM_SHARDS; s++) {
                    const uint8_t *src = (const uint8_t *)(vec + s * SHARD_DIM);
                    uint8_t *dst = dma_buf[s] + lane * SHARD_BYTES;
                    memcpy(dst, src, SHARD_BYTES);
                }
            }

            uint64_t lba = start_lba + b;
            IoCtx ctx[NUM_SHARDS];
            memset(ctx, 0, sizeof(ctx));

            for (int s = 0; s < NUM_SHARDS; s++) {
                int rc = spdk_nvme_ns_cmd_write(
                    app.disks[s].ns,
                    app.disks[s].qpair,
                    dma_buf[s],
                    lba,
                    1,
                    io_complete,
                    &ctx[s],
                    0
                );
                if (rc != 0) {
                    fprintf(stderr, "write submit failed: disk=%d lba=%" PRIu64 " rc=%d\n",
                            s, lba, rc);
                    exit(EXIT_FAILURE);
                }
            }

            for (int s = 0; s < NUM_SHARDS; s++) {
                wait_io(app.disks[s].qpair, &ctx[s]);
            }
        }

        if ((c % 128) == 0 || c + 1 == cents.nlist) {
            fprintf(stderr, "[progress] cluster %u / %u written\n", c + 1, cents.nlist);
        }
    }

    /*
     * 可选 flush
     */
    for (int s = 0; s < NUM_SHARDS; s++) {
        IoCtx ctx = {0};
        int rc = spdk_nvme_ns_cmd_flush(
            app.disks[s].ns,
            app.disks[s].qpair,
            io_complete,
            &ctx
        );
        if (rc == 0) {
            wait_io(app.disks[s].qpair, &ctx);
        }
    }

    /*
     * Step 6: 保存 metadata
     */
    save_meta_file(app.cfg.meta_out,
                   cents.nlist,
                   vecs.n,
                   sector_size,
                   vectors_per_lba,
                   app.cfg.base_lba,
                   cluster_meta);

    fprintf(stderr, "[done] metadata saved to %s\n", app.cfg.meta_out);

    for (int s = 0; s < NUM_SHARDS; s++) {
        spdk_free(dma_buf[s]);
    }

    free(cluster_meta);
    free(cluster_offsets);
    free(sorted_vec_ids);
    free(cluster_sizes);
    free(codebook.labels);
    free(cents.centroids);
    free(vecs.data);

    cleanup_spdk(&app);
    return 0;
}