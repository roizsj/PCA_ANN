#define _GNU_SOURCE
#include <spdk/env.h>
#include <spdk/nvme.h>

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DISKS 4
#define META_MAGIC 0x4751504du /* "MPQG": GIST IVF-PQ metadata */
#define PQ_MAGIC 0x47515049u   /* "IPQG": preprocessing PQ table */

typedef struct {
    uint32_t n;
    uint32_t d;
    float *data;
} FvecsData;

typedef struct {
    uint32_t nlist;
    uint32_t d;
    float *centroids;
} CentroidData;

typedef struct {
    uint32_t n;
    uint32_t *labels;
} LabelData;

typedef struct {
    uint32_t n;
    uint32_t m;
    uint8_t *codes;
} PqCodes;

typedef struct {
    uint32_t dim;
    uint32_t m;
    uint32_t ksub;
    uint32_t nbits;
    uint32_t subdim;
    float *codebooks;
} PqTable;

typedef struct {
    uint32_t cluster_id;
    uint32_t disk_id;
    uint64_t pq_start_lba;
    uint32_t pq_num_lbas;
    uint64_t raw_start_lba;
    uint32_t raw_num_lbas;
    uint32_t num_vectors;
    uint32_t sorted_id_base;
} ClusterMeta;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t nlist;
    uint32_t num_vectors;
    uint32_t num_disks;
    uint32_t sector_size;
    uint32_t pq_m;
    uint32_t pq_ksub;
    uint32_t pq_nbits;
    uint32_t pq_subdim;
    uint32_t pq_code_bytes;
    uint32_t pq_codes_per_lba;
    uint32_t raw_vectors_per_lba;
    uint64_t base_lba;
    uint64_t raw_region_lba[NUM_DISKS];
} MetaHeader;

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
    const char *base_fvecs;
    const char *centroids_bin;
    const char *labels_bin;
    const char *pq_codes_bin;
    const char *pq_table_bin;
    const char *meta_out;
    const char *sorted_ids_out;
    const char *disk_addrs[NUM_DISKS];
    uint64_t base_lba;
} Config;

typedef struct {
    Config cfg;
    DiskTarget disks[NUM_DISKS];
    int found_disks;
} App;

static void die(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

static void die_msg(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "malloc failed size=%zu\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void *xcalloc(size_t n, size_t size)
{
    void *ptr = calloc(n, size);
    if (!ptr) {
        fprintf(stderr, "calloc failed n=%zu size=%zu\n", n, size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static FvecsData read_fvecs(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen fvecs");
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        die("fseek fvecs end");
    }
    long file_size = ftell(fp);
    if (file_size < 0) {
        die("ftell fvecs");
    }
    rewind(fp);

    int32_t dim = 0;
    if (fread(&dim, sizeof(dim), 1, fp) != 1 || dim <= 0) {
        die_msg("failed to read valid fvecs dim");
    }

    long row_bytes = (long)sizeof(int32_t) + (long)dim * (long)sizeof(float);
    if (row_bytes <= 0 || file_size % row_bytes != 0) {
        fprintf(stderr, "invalid fvecs file_size=%ld row_bytes=%ld\n", file_size, row_bytes);
        exit(EXIT_FAILURE);
    }

    uint32_t n = (uint32_t)(file_size / row_bytes);
    float *data = (float *)xmalloc((size_t)n * (size_t)dim * sizeof(float));
    rewind(fp);

    for (uint32_t i = 0; i < n; i++) {
        int32_t row_dim = 0;
        if (fread(&row_dim, sizeof(row_dim), 1, fp) != 1 || row_dim != dim) {
            fprintf(stderr, "bad fvecs row dim at row=%u got=%d expected=%d\n", i, row_dim, dim);
            exit(EXIT_FAILURE);
        }
        if (fread(data + (size_t)i * (size_t)dim, sizeof(float), (size_t)dim, fp) != (size_t)dim) {
            fprintf(stderr, "failed to read fvecs row=%u\n", i);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fp);
    FvecsData out = {.n = n, .d = (uint32_t)dim, .data = data};
    return out;
}

static CentroidData read_centroids(const char *path, uint32_t expected_dim)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen centroids");
    }

    uint32_t nlist = 0;
    uint32_t dim = 0;
    if (fread(&nlist, sizeof(nlist), 1, fp) != 1 ||
        fread(&dim, sizeof(dim), 1, fp) != 1) {
        die_msg("failed to read centroid header");
    }
    if (dim != expected_dim) {
        fprintf(stderr, "centroid dim mismatch got=%u expected=%u\n", dim, expected_dim);
        exit(EXIT_FAILURE);
    }

    float *centroids = (float *)xmalloc((size_t)nlist * dim * sizeof(float));
    if (fread(centroids, sizeof(float), (size_t)nlist * dim, fp) != (size_t)nlist * dim) {
        die_msg("failed to read centroid body");
    }
    fclose(fp);

    CentroidData out = {.nlist = nlist, .d = dim, .centroids = centroids};
    return out;
}

static LabelData read_labels(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen labels");
    }
    uint32_t n = 0;
    if (fread(&n, sizeof(n), 1, fp) != 1) {
        die_msg("failed to read label count");
    }
    uint32_t *labels = (uint32_t *)xmalloc((size_t)n * sizeof(uint32_t));
    if (fread(labels, sizeof(uint32_t), n, fp) != n) {
        die_msg("failed to read labels");
    }
    fclose(fp);
    LabelData out = {.n = n, .labels = labels};
    return out;
}

static PqCodes read_pq_codes(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen pq codes");
    }
    uint32_t n = 0;
    uint32_t m = 0;
    if (fread(&n, sizeof(n), 1, fp) != 1 ||
        fread(&m, sizeof(m), 1, fp) != 1) {
        die_msg("failed to read pq code header");
    }
    uint8_t *codes = (uint8_t *)xmalloc((size_t)n * m);
    if (fread(codes, 1, (size_t)n * m, fp) != (size_t)n * m) {
        die_msg("failed to read pq codes");
    }
    fclose(fp);
    PqCodes out = {.n = n, .m = m, .codes = codes};
    return out;
}

static PqTable read_pq_table(const char *path, uint32_t expected_dim)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        die("fopen pq table");
    }

    uint32_t magic = 0;
    PqTable table;
    memset(&table, 0, sizeof(table));
    if (fread(&magic, sizeof(magic), 1, fp) != 1 ||
        fread(&table.dim, sizeof(table.dim), 1, fp) != 1 ||
        fread(&table.m, sizeof(table.m), 1, fp) != 1 ||
        fread(&table.ksub, sizeof(table.ksub), 1, fp) != 1 ||
        fread(&table.nbits, sizeof(table.nbits), 1, fp) != 1 ||
        fread(&table.subdim, sizeof(table.subdim), 1, fp) != 1) {
        die_msg("failed to read pq table header");
    }
    if (magic != PQ_MAGIC) {
        fprintf(stderr, "bad pq table magic=0x%08x\n", magic);
        exit(EXIT_FAILURE);
    }
    if (table.dim != expected_dim || table.dim != table.m * table.subdim || table.ksub == 0) {
        fprintf(stderr,
                "invalid pq layout dim=%u expected=%u m=%u subdim=%u ksub=%u\n",
                table.dim, expected_dim, table.m, table.subdim, table.ksub);
        exit(EXIT_FAILURE);
    }
    if (table.ksub > 256 || table.nbits != 8) {
        die_msg("this writer/test currently expects 8-bit PQ codes with ksub <= 256");
    }

    size_t total = (size_t)table.m * table.ksub * table.subdim;
    table.codebooks = (float *)xmalloc(total * sizeof(float));
    if (fread(table.codebooks, sizeof(float), total, fp) != total) {
        die_msg("failed to read pq table body");
    }
    fclose(fp);
    return table;
}

static void io_complete(void *arg, const struct spdk_nvme_cpl *cpl)
{
    IoCtx *ctx = (IoCtx *)arg;
    ctx->status = spdk_nvme_cpl_is_error(cpl) ? -EIO : 0;
    ctx->done = 1;
}

static void wait_io(struct spdk_nvme_qpair *qpair, IoCtx *ctx)
{
    while (!ctx->done) {
        int rc = spdk_nvme_qpair_process_completions(qpair, 0);
        if (rc < 0) {
            fprintf(stderr, "process completions failed rc=%d\n", rc);
            exit(EXIT_FAILURE);
        }
    }
    if (ctx->status != 0) {
        fprintf(stderr, "I/O completion error=%d\n", ctx->status);
        exit(EXIT_FAILURE);
    }
}

static bool probe_cb(void *cb_ctx,
                     const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts)
{
    (void)opts;
    App *app = (App *)cb_ctx;
    for (int i = 0; i < NUM_DISKS; i++) {
        if (strcmp(trid->traddr, app->cfg.disk_addrs[i]) == 0) {
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
    (void)opts;
    App *app = (App *)cb_ctx;
    for (int i = 0; i < NUM_DISKS; i++) {
        if (strcmp(trid->traddr, app->cfg.disk_addrs[i]) == 0) {
            struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
            if (!ns || !spdk_nvme_ns_is_active(ns)) {
                fprintf(stderr, "inactive ns for %s\n", trid->traddr);
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
            fprintf(stderr, "[attach] disk%d=%s sector=%u\n", i, app->disks[i].traddr, app->disks[i].sector_size);
        }
    }
}

static void init_spdk(App *app)
{
    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "gist_ivfpq_writer";
    opts.mem_size = 1024;
    if (spdk_env_init(&opts) < 0) {
        die_msg("spdk_env_init failed");
    }
    if (spdk_nvme_probe(NULL, app, probe_cb, attach_cb, NULL) != 0) {
        die_msg("spdk_nvme_probe failed");
    }
    if (app->found_disks != NUM_DISKS) {
        fprintf(stderr, "expected %d disks, found %d\n", NUM_DISKS, app->found_disks);
        exit(EXIT_FAILURE);
    }
    uint32_t sector_size = app->disks[0].sector_size;
    if (sector_size != 4096) {
        fprintf(stderr, "expected 4096B sectors, got=%u\n", sector_size);
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i < NUM_DISKS; i++) {
        if (app->disks[i].sector_size != sector_size) {
            die_msg("all disks must have the same sector size");
        }
    }
}

static void cleanup_spdk(App *app)
{
    for (int i = 0; i < NUM_DISKS; i++) {
        if (app->disks[i].qpair) {
            spdk_nvme_ctrlr_free_io_qpair(app->disks[i].qpair);
            app->disks[i].qpair = NULL;
        }
    }
    for (int i = 0; i < NUM_DISKS; i++) {
        if (app->disks[i].ctrlr) {
            spdk_nvme_detach(app->disks[i].ctrlr);
            app->disks[i].ctrlr = NULL;
        }
    }
}

static void write_one_lba(DiskTarget *disk, uint64_t lba, void *buf)
{
    IoCtx ctx = {0};
    int rc = spdk_nvme_ns_cmd_write(disk->ns, disk->qpair, buf, lba, 1, io_complete, &ctx, 0);
    if (rc != 0) {
        fprintf(stderr, "write submit failed disk=%s lba=%" PRIu64 " rc=%d\n", disk->traddr, lba, rc);
        exit(EXIT_FAILURE);
    }
    wait_io(disk->qpair, &ctx);
}

static void save_sorted_ids(const char *path, uint32_t n, const uint32_t *sorted_ids)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        die("fopen sorted ids out");
    }
    fwrite(&n, sizeof(n), 1, fp);
    fwrite(sorted_ids, sizeof(uint32_t), n, fp);
    fclose(fp);
}

static void save_meta(const char *path,
                      const MetaHeader *hdr,
                      const ClusterMeta *clusters,
                      const float *centroids,
                      const float *pq_codebooks)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        die("fopen meta out");
    }
    if (fwrite(hdr, sizeof(*hdr), 1, fp) != 1) {
        die("write meta header");
    }
    if (fwrite(clusters, sizeof(ClusterMeta), hdr->nlist, fp) != hdr->nlist) {
        die("write cluster meta");
    }
    if (fwrite(centroids, sizeof(float), (size_t)hdr->nlist * hdr->dim, fp) != (size_t)hdr->nlist * hdr->dim) {
        die("write centroids");
    }
    size_t pq_floats = (size_t)hdr->pq_m * hdr->pq_ksub * hdr->pq_subdim;
    if (fwrite(pq_codebooks, sizeof(float), pq_floats, fp) != pq_floats) {
        die("write pq codebooks");
    }
    fclose(fp);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --base gist_base.fvecs --centroids gist_ivfpq_centroids.bin --labels gist_ivfpq_codebook.bin \\\n"
            "     --pq-codes gist_pq_codes.bin --pq-table gist_pq_table.bin --meta ivfpq_meta.bin \\\n"
            "     --sorted-ids sorted_vec_ids_ivfpq.bin \\\n"
            "     --disk0 0000:65:00.0 --disk1 0000:66:00.0 --disk2 0000:67:00.0 --disk3 0000:68:00.0 \\\n"
            "     [--base-lba 0]\n",
            prog);
}

static void parse_args(int argc, char **argv, Config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    static struct option opts[] = {
        {"base", required_argument, 0, 'b'},
        {"centroids", required_argument, 0, 'c'},
        {"labels", required_argument, 0, 'l'},
        {"pq-codes", required_argument, 0, 'p'},
        {"pq-table", required_argument, 0, 't'},
        {"meta", required_argument, 0, 'm'},
        {"sorted-ids", required_argument, 0, 's'},
        {"disk0", required_argument, 0, 1000},
        {"disk1", required_argument, 0, 1001},
        {"disk2", required_argument, 0, 1002},
        {"disk3", required_argument, 0, 1003},
        {"base-lba", required_argument, 0, 1004},
        {0, 0, 0, 0}
    };

    int opt = 0;
    int idx = 0;
    while ((opt = getopt_long(argc, argv, "b:c:l:p:t:m:s:", opts, &idx)) != -1) {
        switch (opt) {
            case 'b': cfg->base_fvecs = optarg; break;
            case 'c': cfg->centroids_bin = optarg; break;
            case 'l': cfg->labels_bin = optarg; break;
            case 'p': cfg->pq_codes_bin = optarg; break;
            case 't': cfg->pq_table_bin = optarg; break;
            case 'm': cfg->meta_out = optarg; break;
            case 's': cfg->sorted_ids_out = optarg; break;
            case 1000: cfg->disk_addrs[0] = optarg; break;
            case 1001: cfg->disk_addrs[1] = optarg; break;
            case 1002: cfg->disk_addrs[2] = optarg; break;
            case 1003: cfg->disk_addrs[3] = optarg; break;
            case 1004: cfg->base_lba = strtoull(optarg, NULL, 10); break;
            default:
                usage(argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if (!cfg->base_fvecs || !cfg->centroids_bin || !cfg->labels_bin ||
        !cfg->pq_codes_bin || !cfg->pq_table_bin || !cfg->meta_out ||
        !cfg->sorted_ids_out || !cfg->disk_addrs[0] || !cfg->disk_addrs[1] ||
        !cfg->disk_addrs[2] || !cfg->disk_addrs[3]) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    App app;
    memset(&app, 0, sizeof(app));
    parse_args(argc, argv, &app.cfg);
    init_spdk(&app);

    const uint32_t sector_size = app.disks[0].sector_size;
    FvecsData base = read_fvecs(app.cfg.base_fvecs);
    CentroidData cents = read_centroids(app.cfg.centroids_bin, base.d);
    LabelData labels = read_labels(app.cfg.labels_bin);
    PqCodes pq_codes = read_pq_codes(app.cfg.pq_codes_bin);
    PqTable pq_table = read_pq_table(app.cfg.pq_table_bin, base.d);

    if (labels.n != base.n || pq_codes.n != base.n) {
        fprintf(stderr, "count mismatch base=%u labels=%u pq_codes=%u\n", base.n, labels.n, pq_codes.n);
        exit(EXIT_FAILURE);
    }
    if (pq_codes.m != pq_table.m) {
        fprintf(stderr, "pq m mismatch codes=%u table=%u\n", pq_codes.m, pq_table.m);
        exit(EXIT_FAILURE);
    }

    uint32_t raw_bytes = base.d * (uint32_t)sizeof(float);
    uint32_t raw_per_lba = sector_size / raw_bytes;
    uint32_t pq_per_lba = sector_size / pq_codes.m;
    if (raw_per_lba == 0 || pq_per_lba == 0) {
        fprintf(stderr, "vector too large for one LBA raw_bytes=%u pq_code_bytes=%u sector=%u\n",
                raw_bytes, pq_codes.m, sector_size);
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "[info] base n=%u dim=%u nlist=%u pq_m=%u pq_per_lba=%u raw_per_lba=%u\n",
            base.n, base.d, cents.nlist, pq_codes.m, pq_per_lba, raw_per_lba);

    uint32_t *cluster_sizes = (uint32_t *)xcalloc(cents.nlist, sizeof(uint32_t));
    for (uint32_t i = 0; i < base.n; i++) {
        uint32_t cid = labels.labels[i];
        if (cid >= cents.nlist) {
            fprintf(stderr, "invalid label vec=%u cid=%u nlist=%u\n", i, cid, cents.nlist);
            exit(EXIT_FAILURE);
        }
        cluster_sizes[cid]++;
    }

    uint64_t *cluster_offsets = (uint64_t *)xcalloc((size_t)cents.nlist + 1, sizeof(uint64_t));
    for (uint32_t c = 0; c < cents.nlist; c++) {
        cluster_offsets[c + 1] = cluster_offsets[c] + cluster_sizes[c];
    }

    uint64_t *write_ptrs = (uint64_t *)xmalloc((size_t)cents.nlist * sizeof(uint64_t));
    memcpy(write_ptrs, cluster_offsets, (size_t)cents.nlist * sizeof(uint64_t));
    uint32_t *sorted_ids = (uint32_t *)xmalloc((size_t)base.n * sizeof(uint32_t));
    for (uint32_t vec_id = 0; vec_id < base.n; vec_id++) {
        uint32_t cid = labels.labels[vec_id];
        sorted_ids[write_ptrs[cid]++] = vec_id;
    }
    free(write_ptrs);

    ClusterMeta *clusters = (ClusterMeta *)xcalloc(cents.nlist, sizeof(ClusterMeta));
    uint64_t disk_vec_load[NUM_DISKS] = {0};
    uint64_t disk_pq_lbas[NUM_DISKS] = {0};
    uint64_t disk_raw_lbas[NUM_DISKS] = {0};

    for (uint32_t c = 0; c < cents.nlist; c++) {
        uint32_t best = 0;
        for (uint32_t d = 1; d < NUM_DISKS; d++) {
            if (disk_vec_load[d] < disk_vec_load[best]) {
                best = d;
            }
        }
        uint32_t sz = cluster_sizes[c];
        uint32_t pq_lbas = (sz + pq_per_lba - 1) / pq_per_lba;
        uint32_t raw_lbas = (sz + raw_per_lba - 1) / raw_per_lba;

        clusters[c].cluster_id = c;
        clusters[c].disk_id = best;
        clusters[c].num_vectors = sz;
        clusters[c].pq_num_lbas = pq_lbas;
        clusters[c].raw_num_lbas = raw_lbas;
        clusters[c].sorted_id_base = (uint32_t)cluster_offsets[c];
        clusters[c].pq_start_lba = app.cfg.base_lba + disk_pq_lbas[best];

        disk_vec_load[best] += sz;
        disk_pq_lbas[best] += pq_lbas;
        disk_raw_lbas[best] += raw_lbas;
    }

    uint64_t raw_region_lba[NUM_DISKS] = {0};
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        raw_region_lba[d] = app.cfg.base_lba + disk_pq_lbas[d];
    }
    uint64_t raw_cur[NUM_DISKS];
    memcpy(raw_cur, raw_region_lba, sizeof(raw_cur));
    for (uint32_t c = 0; c < cents.nlist; c++) {
        uint32_t d = clusters[c].disk_id;
        clusters[c].raw_start_lba = raw_cur[d];
        raw_cur[d] += clusters[c].raw_num_lbas;
    }

    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        fprintf(stderr,
                "[layout] disk%u vectors=%" PRIu64 " pq_lbas=%" PRIu64 " raw_lbas=%" PRIu64 " raw_region_lba=%" PRIu64 "\n",
                d, disk_vec_load[d], disk_pq_lbas[d], disk_raw_lbas[d], raw_region_lba[d]);
    }

    save_sorted_ids(app.cfg.sorted_ids_out, base.n, sorted_ids);

    uint8_t *dma[NUM_DISKS] = {0};
    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        dma[d] = spdk_zmalloc(sector_size, sector_size, NULL, SPDK_ENV_LCORE_ID_ANY, SPDK_MALLOC_DMA);
        if (!dma[d]) {
            die_msg("spdk_zmalloc failed");
        }
    }

    for (uint32_t c = 0; c < cents.nlist; c++) {
        ClusterMeta *cm = &clusters[c];
        uint32_t d = cm->disk_id;
        uint64_t sorted_base = cluster_offsets[c];

        for (uint32_t b = 0; b < cm->pq_num_lbas; b++) {
            memset(dma[d], 0, sector_size);
            for (uint32_t lane = 0; lane < pq_per_lba; lane++) {
                uint32_t local_idx = b * pq_per_lba + lane;
                if (local_idx >= cm->num_vectors) {
                    break;
                }
                uint32_t vec_id = sorted_ids[sorted_base + local_idx];
                memcpy(dma[d] + (size_t)lane * pq_codes.m,
                       pq_codes.codes + (size_t)vec_id * pq_codes.m,
                       pq_codes.m);
            }
            write_one_lba(&app.disks[d], cm->pq_start_lba + b, dma[d]);
        }

        for (uint32_t b = 0; b < cm->raw_num_lbas; b++) {
            memset(dma[d], 0, sector_size);
            for (uint32_t lane = 0; lane < raw_per_lba; lane++) {
                uint32_t local_idx = b * raw_per_lba + lane;
                if (local_idx >= cm->num_vectors) {
                    break;
                }
                uint32_t vec_id = sorted_ids[sorted_base + local_idx];
                memcpy(dma[d] + (size_t)lane * raw_bytes,
                       base.data + (size_t)vec_id * base.d,
                       raw_bytes);
            }
            write_one_lba(&app.disks[d], cm->raw_start_lba + b, dma[d]);
        }

        if ((c % 128) == 0 || c + 1 == cents.nlist) {
            fprintf(stderr, "[progress] cluster %u/%u written\n", c + 1, cents.nlist);
        }
    }

    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        IoCtx ctx = {0};
        int rc = spdk_nvme_ns_cmd_flush(app.disks[d].ns, app.disks[d].qpair, io_complete, &ctx);
        if (rc == 0) {
            wait_io(app.disks[d].qpair, &ctx);
        }
    }

    MetaHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = META_MAGIC;
    hdr.version = 1;
    hdr.dim = base.d;
    hdr.nlist = cents.nlist;
    hdr.num_vectors = base.n;
    hdr.num_disks = NUM_DISKS;
    hdr.sector_size = sector_size;
    hdr.pq_m = pq_table.m;
    hdr.pq_ksub = pq_table.ksub;
    hdr.pq_nbits = pq_table.nbits;
    hdr.pq_subdim = pq_table.subdim;
    hdr.pq_code_bytes = pq_codes.m;
    hdr.pq_codes_per_lba = pq_per_lba;
    hdr.raw_vectors_per_lba = raw_per_lba;
    hdr.base_lba = app.cfg.base_lba;
    memcpy(hdr.raw_region_lba, raw_region_lba, sizeof(hdr.raw_region_lba));

    save_meta(app.cfg.meta_out, &hdr, clusters, cents.centroids, pq_table.codebooks);
    fprintf(stderr, "[done] meta=%s sorted_ids=%s\n", app.cfg.meta_out, app.cfg.sorted_ids_out);

    for (uint32_t d = 0; d < NUM_DISKS; d++) {
        spdk_free(dma[d]);
    }
    free(clusters);
    free(sorted_ids);
    free(cluster_offsets);
    free(cluster_sizes);
    free(pq_table.codebooks);
    free(pq_codes.codes);
    free(labels.labels);
    free(cents.centroids);
    free(base.data);
    cleanup_spdk(&app);
    return 0;
}
