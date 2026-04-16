#include <spdk/env.h>
#include <spdk/nvme.h>

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_SHARDS 4
#define MAGIC_META 0x49564634u /* "IVF4" */

typedef struct {
    uint32_t n;
    uint32_t d;
    float *data; /* n * d */
} FvecsData;

typedef struct {
    uint32_t nlist;
    uint32_t d;
    float *centroids; /* nlist * d */
} CentroidData;

typedef struct {
    uint64_t nlist;
    uint64_t total_ids;
    uint64_t *offsets; /* offsets[c] ... offsets[c + 1] are ids for cluster c */
    uint32_t *ids;     /* vector ids, duplicates allowed across clusters */
} MembersData;

typedef struct {
    uint32_t cluster_id;
    uint64_t start_lba;
    uint32_t num_vectors;
    uint32_t num_lbas;
} ClusterMetaEntry;

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
    const char *members_bin;
    const char *meta_out;
    const char *sorted_ids_out;
    const char *disk_addrs[NUM_SHARDS];
    uint32_t dim;
    uint32_t shard_dims[NUM_SHARDS];
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
    uint32_t num_shards;
    uint32_t shard_dims[NUM_SHARDS];
    uint32_t shard_offsets[NUM_SHARDS];
    uint32_t shard_bytes[NUM_SHARDS];
    uint32_t vectors_per_lba;
    uint32_t nlist;
    uint32_t num_vectors;
    uint32_t sector_size;
    uint64_t base_lba;
} MetaHeaderFlex;

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

static uint32_t parse_u32_arg(const char *name, const char *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (!value[0] || !end || *end != '\0' || parsed > UINT32_MAX) {
        fprintf(stderr, "invalid value for %s: %s\n", name, value);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)parsed;
}

static uint64_t parse_u64_arg(const char *name, const char *value) {
    char *end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (!value[0] || !end || *end != '\0') {
        fprintf(stderr, "invalid value for %s: %s\n", name, value);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)parsed;
}

static void compute_shard_layout(const AppConfig *cfg,
                                 uint32_t shard_offsets[NUM_SHARDS],
                                 uint32_t shard_bytes[NUM_SHARDS]) {
    uint64_t total_dim = 0;

    for (int s = 0; s < NUM_SHARDS; s++) {
        if (cfg->shard_dims[s] == 0) {
            fprintf(stderr, "disk%d-dim must be > 0\n", s);
            exit(EXIT_FAILURE);
        }
        shard_offsets[s] = (uint32_t)total_dim;
        shard_bytes[s] = cfg->shard_dims[s] * (uint32_t)sizeof(float);
        total_dim += cfg->shard_dims[s];
    }

    if (cfg->dim == 0) {
        die_msg("dim must be > 0");
    }

    if (total_dim != cfg->dim) {
        fprintf(stderr,
                "sum of shard dims mismatch: disk0+disk1+disk2+disk3=%" PRIu64
                ", expected dim=%u\n",
                total_dim, cfg->dim);
        exit(EXIT_FAILURE);
    }
}

static uint32_t compute_vectors_per_lba(uint32_t sector_size,
                                        const uint32_t shard_bytes[NUM_SHARDS]) {
    uint32_t vectors_per_lba = UINT32_MAX;

    for (int s = 0; s < NUM_SHARDS; s++) {
        uint32_t shard_byte = shard_bytes[s];
        uint32_t per_disk = sector_size / shard_byte;
        if (per_disk == 0) {
            fprintf(stderr,
                    "disk%d shard is too wide for one sector: sector=%u shard_bytes=%u\n",
                    s, sector_size, shard_byte);
            exit(EXIT_FAILURE);
        }
        if (per_disk < vectors_per_lba) {
            vectors_per_lba = per_disk;
        }
    }

    return vectors_per_lba;
}

static FvecsData read_fvecs(const char *path, uint32_t expected_dim) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen fvecs");
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        die("fseek fvecs end");
    }
    long file_sz = ftell(fp);
    if (file_sz < 0) {
        die("ftell fvecs");
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        die("fseek fvecs set");
    }

    int32_t dim = 0;
    if (fread(&dim, sizeof(int32_t), 1, fp) != 1) {
        die_msg("failed to read fvecs dim");
    }
    if ((uint32_t)dim != expected_dim) {
        fprintf(stderr, "fvecs dim mismatch: got %d expected %u\n", dim, expected_dim);
        exit(EXIT_FAILURE);
    }

    {
        long row_bytes = (long)sizeof(int32_t) + (long)expected_dim * (long)sizeof(float);
        if (file_sz % row_bytes != 0) {
            fprintf(stderr, "invalid fvecs file size: %ld\n", file_sz);
            exit(EXIT_FAILURE);
        }

        uint32_t n = (uint32_t)(file_sz / row_bytes);
        float *data = (float *)xmalloc((size_t)n * expected_dim * sizeof(float));

        rewind(fp);
        for (uint32_t i = 0; i < n; i++) {
            int32_t row_dim = 0;
            if (fread(&row_dim, sizeof(int32_t), 1, fp) != 1) {
                die_msg("failed to read row dim");
            }
            if ((uint32_t)row_dim != expected_dim) {
                fprintf(stderr, "row %u dim mismatch: got %d expected %u\n",
                        i, row_dim, expected_dim);
                exit(EXIT_FAILURE);
            }
            if (fread(data + (size_t)i * expected_dim, sizeof(float), expected_dim, fp) !=
                expected_dim) {
                die_msg("failed to read row vector");
            }
        }

        fclose(fp);

        {
            FvecsData out = {
                .n = n,
                .d = expected_dim,
                .data = data
            };
            return out;
        }
    }
}

/*
 * centroids.bin:
 * [u32 nlist][u32 dim][float centroids[nlist * dim]]
 */
static CentroidData read_centroids_bin(const char *path, uint32_t expected_dim) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen centroids");
    }

    uint32_t nlist = 0;
    uint32_t d = 0;
    if (fread(&nlist, sizeof(uint32_t), 1, fp) != 1) {
        die_msg("read nlist failed");
    }
    if (fread(&d, sizeof(uint32_t), 1, fp) != 1) {
        die_msg("read dim failed");
    }

    if (d != expected_dim) {
        fprintf(stderr, "centroid dim mismatch: got %u expected %u\n", d, expected_dim);
        exit(EXIT_FAILURE);
    }

    {
        float *centroids = (float *)xmalloc((size_t)nlist * expected_dim * sizeof(float));
        if (fread(centroids, sizeof(float), (size_t)nlist * expected_dim, fp) !=
            (size_t)nlist * expected_dim) {
            die_msg("read centroids body failed");
        }

        fclose(fp);

        {
            CentroidData out = {
                .nlist = nlist,
                .d = d,
                .centroids = centroids
            };
            return out;
        }
    }
}

/*
 * members.bin:
 * [u64 nlist][u64 total_ids][u64 offsets[nlist + 1]][u32 ids[total_ids]]
 *
 * This is the format produced by preprocessing/balanced_ivf_gist.py.
 * A vector id may appear in multiple clusters, so total_ids can be larger
 * than the number of base vectors.
 */
static MembersData read_members_bin(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen members");
    }

    uint64_t nlist = 0;
    uint64_t total_ids = 0;
    if (fread(&nlist, sizeof(uint64_t), 1, fp) != 1) {
        die_msg("read members nlist failed");
    }
    if (fread(&total_ids, sizeof(uint64_t), 1, fp) != 1) {
        die_msg("read members total_ids failed");
    }
    if (nlist == 0) {
        die_msg("members nlist must be > 0");
    }
    if (total_ids == 0) {
        die_msg("members total_ids must be > 0");
    }
    if (nlist > UINT32_MAX || total_ids > UINT32_MAX) {
        die_msg("this writer stores meta/sorted-id counts as uint32_t; nlist/total_ids exceed UINT32_MAX");
    }

    {
        uint64_t *offsets = (uint64_t *)xmalloc((size_t)(nlist + 1) * sizeof(uint64_t));
        uint32_t *ids = NULL;

        if (fread(offsets, sizeof(uint64_t), (size_t)nlist + 1, fp) != (size_t)nlist + 1) {
            die_msg("read members offsets failed");
        }
        if (offsets[0] != 0 || offsets[nlist] != total_ids) {
            fprintf(stderr,
                    "invalid members offsets: offsets[0]=%" PRIu64
                    " offsets[nlist]=%" PRIu64 " total_ids=%" PRIu64 "\n",
                    offsets[0], offsets[nlist], total_ids);
            exit(EXIT_FAILURE);
        }
        for (uint64_t i = 0; i < nlist; i++) {
            if (offsets[i] > offsets[i + 1]) {
                fprintf(stderr, "members offsets must be monotonic at idx=%" PRIu64 "\n", i);
                exit(EXIT_FAILURE);
            }
        }

        ids = (uint32_t *)xmalloc((size_t)total_ids * sizeof(uint32_t));
        if (total_ids > 0 &&
            fread(ids, sizeof(uint32_t), (size_t)total_ids, fp) != (size_t)total_ids) {
            die_msg("read members ids failed");
        }

        fclose(fp);

        {
            MembersData out = {
                .nlist = nlist,
                .total_ids = total_ids,
                .offsets = offsets,
                .ids = ids
            };
            return out;
        }
    }
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
    opts.name = "ivf_pack_writer_spdk_overlap_flex";

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

    {
        uint32_t sector_size = app->disks[0].sector_size;
        if (sector_size == 0) {
            die_msg("invalid sector size 0");
        }

        for (int i = 1; i < NUM_SHARDS; i++) {
            if (app->disks[i].sector_size != sector_size) {
                die_msg("all 4 disks must have the same sector size");
            }
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

/*
 * meta file layout:
 *   [MetaHeaderFlex]
 *   [ClusterMetaEntry x nlist]
 *   [float centroids[nlist * dim]]
 */
static void save_meta_file(const char *path,
                           const AppConfig *cfg,
                           uint32_t nlist,
                           uint32_t num_vectors,
                           uint32_t sector_size,
                           const uint32_t shard_offsets[NUM_SHARDS],
                           const uint32_t shard_bytes[NUM_SHARDS],
                           uint32_t vectors_per_lba,
                           const ClusterMetaEntry *cluster_meta,
                           const float *centroids) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        die("fopen meta out");
    }

    {
        MetaHeaderFlex hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.magic = MAGIC_META;
        hdr.version = 1;
        hdr.dim = cfg->dim;
        hdr.num_shards = NUM_SHARDS;
        memcpy(hdr.shard_dims, cfg->shard_dims, sizeof(hdr.shard_dims));
        memcpy(hdr.shard_offsets, shard_offsets, sizeof(hdr.shard_offsets));
        memcpy(hdr.shard_bytes, shard_bytes, sizeof(hdr.shard_bytes));
        hdr.vectors_per_lba = vectors_per_lba;
        hdr.nlist = nlist;
        hdr.num_vectors = num_vectors;
        hdr.sector_size = sector_size;
        hdr.base_lba = cfg->base_lba;

        if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
            die("write meta header");
        }
    }

    if (fwrite(cluster_meta, sizeof(ClusterMetaEntry), nlist, fp) != nlist) {
        die("write cluster meta");
    }

    if (fwrite(centroids, sizeof(float), (size_t)nlist * cfg->dim, fp) !=
        (size_t)nlist * cfg->dim) {
        die("write centroids");
    }

    fclose(fp);
}

static void save_sorted_ids_file(const char *path,
                                 uint32_t num_entries,
                                 const uint32_t *sorted_vec_ids) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        die("fopen sorted ids");
    }

    if (fwrite(&num_entries, sizeof(uint32_t), 1, fp) != 1) {
        die("write sorted ids count");
    }
    if (fwrite(sorted_vec_ids, sizeof(uint32_t), num_entries, fp) != num_entries) {
        die("write sorted ids body");
    }
    fclose(fp);
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --input base.fvecs --centroids centroids.bin --members members.bin \\\n"
            "     --meta ivf_meta_flex.bin --dim 96 --disk0-dim 24 --disk1-dim 24 \\\n"
            "     --disk2-dim 16 --disk3-dim 32 \\\n"
            "     --disk0 0000:5e:00.0 --disk1 0000:60:00.0 \\\n"
            "     --disk2 0000:61:00.0 --disk3 0000:62:00.0 \\\n"
            "     [--sorted-ids sorted_vec_ids.bin] [--base-lba 0]\n"
            "\n"
            "Notes:\n"
            "  - Exactly 4 disks are used.\n"
            "  - dim must equal disk0-dim + disk1-dim + disk2-dim + disk3-dim.\n"
            "  - The packed vectors_per_lba is limited by the widest shard.\n",
            prog);
}

static void parse_args(int argc, char **argv, AppConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    static struct option long_opts[] = {
        {"input", required_argument, 0, 'i'},
        {"centroids", required_argument, 0, 'c'},
        {"members", required_argument, 0, 'k'},
        {"meta", required_argument, 0, 'm'},
        {"sorted-ids", required_argument, 0, 's'},
        {"base-lba", required_argument, 0, 'b'},
        {"dim", required_argument, 0, 'd'},
        {"disk0", required_argument, 0, 1000},
        {"disk1", required_argument, 0, 1001},
        {"disk2", required_argument, 0, 1002},
        {"disk3", required_argument, 0, 1003},
        {"disk0-dim", required_argument, 0, 1100},
        {"disk1-dim", required_argument, 0, 1101},
        {"disk2-dim", required_argument, 0, 1102},
        {"disk3-dim", required_argument, 0, 1103},
        {0, 0, 0, 0}
    };

    int opt;
    int idx = 0;
    while ((opt = getopt_long(argc, argv, "i:c:k:m:s:b:d:", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'i':
                cfg->input_fvecs = optarg;
                break;
            case 'c':
                cfg->centroids_bin = optarg;
                break;
            case 'k':
                cfg->members_bin = optarg;
                break;
            case 'm':
                cfg->meta_out = optarg;
                break;
            case 's':
                cfg->sorted_ids_out = optarg;
                break;
            case 'b':
                cfg->base_lba = parse_u64_arg("base-lba", optarg);
                break;
            case 'd':
                cfg->dim = parse_u32_arg("dim", optarg);
                break;
            case 1000:
                cfg->disk_addrs[0] = optarg;
                break;
            case 1001:
                cfg->disk_addrs[1] = optarg;
                break;
            case 1002:
                cfg->disk_addrs[2] = optarg;
                break;
            case 1003:
                cfg->disk_addrs[3] = optarg;
                break;
            case 1100:
                cfg->shard_dims[0] = parse_u32_arg("disk0-dim", optarg);
                break;
            case 1101:
                cfg->shard_dims[1] = parse_u32_arg("disk1-dim", optarg);
                break;
            case 1102:
                cfg->shard_dims[2] = parse_u32_arg("disk2-dim", optarg);
                break;
            case 1103:
                cfg->shard_dims[3] = parse_u32_arg("disk3-dim", optarg);
                break;
            default:
                usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!cfg->input_fvecs || !cfg->centroids_bin || !cfg->members_bin || !cfg->meta_out ||
        !cfg->disk_addrs[0] || !cfg->disk_addrs[1] || !cfg->disk_addrs[2] ||
        !cfg->disk_addrs[3] || cfg->dim == 0 || cfg->shard_dims[0] == 0 ||
        cfg->shard_dims[1] == 0 || cfg->shard_dims[2] == 0 || cfg->shard_dims[3] == 0) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    AppState app;
    uint32_t shard_offsets[NUM_SHARDS];
    uint32_t shard_bytes[NUM_SHARDS];

    memset(&app, 0, sizeof(app));
    parse_args(argc, argv, &app.cfg);
    compute_shard_layout(&app.cfg, shard_offsets, shard_bytes);

    init_spdk_and_attach(&app);

    {
        uint32_t sector_size = app.disks[0].sector_size;
        uint32_t vectors_per_lba = compute_vectors_per_lba(sector_size, shard_bytes);

        fprintf(stderr, "[info] dim=%u\n", app.cfg.dim);
        for (int s = 0; s < NUM_SHARDS; s++) {
            fprintf(stderr,
                    "[info] disk%d dim=%u offset=%u shard_bytes=%u sector=%u\n",
                    s, app.cfg.shard_dims[s], shard_offsets[s], shard_bytes[s], sector_size);
        }
        fprintf(stderr, "[info] vectors_per_lba=%u\n", vectors_per_lba);

        {
            FvecsData vecs = read_fvecs(app.cfg.input_fvecs, app.cfg.dim);
            CentroidData cents = read_centroids_bin(app.cfg.centroids_bin, app.cfg.dim);
            MembersData members = read_members_bin(app.cfg.members_bin);

            if (members.nlist != cents.nlist) {
                fprintf(stderr, "members nlist mismatch: members=%" PRIu64 " centroids=%u\n",
                        members.nlist, cents.nlist);
                exit(EXIT_FAILURE);
            }

            for (uint64_t i = 0; i < members.total_ids; i++) {
                if (members.ids[i] >= vecs.n) {
                    fprintf(stderr,
                            "invalid vec id in members: entry=%" PRIu64 " vec_id=%u base_vectors=%u\n",
                            i, members.ids[i], vecs.n);
                    exit(EXIT_FAILURE);
                }
            }

            fprintf(stderr,
                    "[info] loaded base_vectors=%u dim=%u, nlist=%u, member_entries=%" PRIu64 "\n",
                    vecs.n, vecs.d, cents.nlist, members.total_ids);

            {
                uint32_t *cluster_sizes =
                    (uint32_t *)xcalloc(cents.nlist, sizeof(uint32_t));
                uint64_t *cluster_offsets =
                    (uint64_t *)xcalloc(cents.nlist + 1, sizeof(uint64_t));
                uint32_t *sorted_vec_ids =
                    (uint32_t *)xmalloc((size_t)members.total_ids * sizeof(uint32_t));
                ClusterMetaEntry *cluster_meta =
                    (ClusterMetaEntry *)xcalloc(cents.nlist, sizeof(ClusterMetaEntry));
                uint8_t *dma_buf[NUM_SHARDS] = {0};
                uint64_t cur_lba = app.cfg.base_lba;

                for (uint32_t c = 0; c < cents.nlist; c++) {
                    uint64_t start = members.offsets[c];
                    uint64_t end = members.offsets[c + 1];
                    uint64_t sz64 = end - start;
                    if (sz64 > UINT32_MAX) {
                        fprintf(stderr, "cluster %u too large: %" PRIu64 "\n", c, sz64);
                        exit(EXIT_FAILURE);
                    }
                    cluster_sizes[c] = (uint32_t)sz64;
                    cluster_offsets[c] = start;
                    if (sz64 > 0) {
                        memcpy(sorted_vec_ids + start,
                               members.ids + start,
                               (size_t)sz64 * sizeof(uint32_t));
                    }
                }
                cluster_offsets[cents.nlist] = members.total_ids;

                for (uint32_t c = 0; c < cents.nlist; c++) {
                    uint32_t sz = cluster_sizes[c];
                    uint32_t num_lbas = (sz + vectors_per_lba - 1) / vectors_per_lba;

                    cluster_meta[c].cluster_id = c;
                    cluster_meta[c].start_lba = cur_lba;
                    cluster_meta[c].num_vectors = sz;
                    cluster_meta[c].num_lbas = num_lbas;

                    cur_lba += num_lbas;
                }

                fprintf(stderr, "[info] total_lbas_to_write=%" PRIu64 "\n",
                        cur_lba - app.cfg.base_lba);

                if (app.cfg.sorted_ids_out) {
                    save_sorted_ids_file(app.cfg.sorted_ids_out,
                                         (uint32_t)members.total_ids,
                                         sorted_vec_ids);
                    fprintf(stderr, "[info] saved sorted ids to %s\n", app.cfg.sorted_ids_out);
                }

                for (int s = 0; s < NUM_SHARDS; s++) {
                    dma_buf[s] = spdk_zmalloc(sector_size, sector_size, NULL,
                                              SPDK_ENV_LCORE_ID_ANY, SPDK_MALLOC_DMA);
                    if (!dma_buf[s]) {
                        die_msg("spdk_zmalloc failed");
                    }
                }

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

                            {
                                uint32_t vec_id = sorted_vec_ids[start_pos + local_idx];
                                const float *vec = vecs.data + (size_t)vec_id * app.cfg.dim;

                                for (int s = 0; s < NUM_SHARDS; s++) {
                                    const uint8_t *src =
                                        (const uint8_t *)(vec + shard_offsets[s]);
                                    uint8_t *dst = dma_buf[s] + (size_t)lane * shard_bytes[s];
                                    memcpy(dst, src, shard_bytes[s]);
                                }
                            }
                        }

                        {
                            uint64_t lba = start_lba + b;
                            IoCtx ctx[NUM_SHARDS];

                            memset(ctx, 0, sizeof(ctx));
                            for (int s = 0; s < NUM_SHARDS; s++) {
                                int rc = spdk_nvme_ns_cmd_write(app.disks[s].ns,
                                                                app.disks[s].qpair,
                                                                dma_buf[s],
                                                                lba,
                                                                1,
                                                                io_complete,
                                                                &ctx[s],
                                                                0);
                                if (rc != 0) {
                                    fprintf(stderr,
                                            "write submit failed: disk=%d lba=%" PRIu64
                                            " rc=%d\n",
                                            s, lba, rc);
                                    exit(EXIT_FAILURE);
                                }
                            }

                            for (int s = 0; s < NUM_SHARDS; s++) {
                                wait_io(app.disks[s].qpair, &ctx[s]);
                            }
                        }
                    }

                    if ((c % 128) == 0 || c + 1 == cents.nlist) {
                        fprintf(stderr, "[progress] cluster %u / %u written\n",
                                c + 1, cents.nlist);
                    }
                }

                for (int s = 0; s < NUM_SHARDS; s++) {
                    IoCtx ctx = {0};
                    int rc = spdk_nvme_ns_cmd_flush(app.disks[s].ns,
                                                    app.disks[s].qpair,
                                                    io_complete,
                                                    &ctx);
                    if (rc == 0) {
                        wait_io(app.disks[s].qpair, &ctx);
                    }
                }

                save_meta_file(app.cfg.meta_out,
                               &app.cfg,
                               cents.nlist,
                               (uint32_t)members.total_ids,
                               sector_size,
                               shard_offsets,
                               shard_bytes,
                               vectors_per_lba,
                               cluster_meta,
                               cents.centroids);

                fprintf(stderr, "[done] metadata saved to %s\n", app.cfg.meta_out);

                for (int s = 0; s < NUM_SHARDS; s++) {
                    spdk_free(dma_buf[s]);
                }

                free(cluster_meta);
                free(cluster_offsets);
                free(sorted_vec_ids);
                free(cluster_sizes);
                free(members.offsets);
                free(members.ids);
                free(cents.centroids);
                free(vecs.data);
            }
        }
    }

    cleanup_spdk(&app);
    return 0;
}
