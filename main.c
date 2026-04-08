#include "pipeline_stage.h"
#include "query_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>

extern struct app_ctx g_app;

typedef struct {
    uint32_t n_queries;
    uint32_t k;
    int32_t *ids; /* [n_queries * k] */
} gt_set_t;

typedef struct {
    uint32_t max_queries;
    uint32_t nprobe;
    uint32_t read_depth;
    uint32_t stage1_gap_merge_limit;
    const char *coarse_backend;
    const char *iova_mode;
    bool print_per_query;
} runtime_options_t;

typedef struct {
    uint32_t queries;
    uint64_t latency_us_sum;
    uint64_t service_us_sum;
    uint64_t coarse_us_sum;
    uint64_t submit_us_sum;
    uint64_t topk_wall_us_sum;
    double recall_sum;
    uint64_t stage_in_sum[NUM_STAGES];
    uint64_t stage_out_sum[NUM_STAGES];
    uint64_t stage_pruned_sum[NUM_STAGES];
    uint64_t stage_batches_sum[NUM_STAGES];
    uint64_t stage_bundles_sum[NUM_STAGES];
    uint64_t stage_wall_us_sum[NUM_STAGES];
    uint64_t stage_io_us_sum[NUM_STAGES];
    uint64_t stage_qsort_us_sum[NUM_STAGES];
} run_summary_t;

static uint64_t query_service_us(const query_tracker_t *qt);

static inline uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static void init_runtime_options(runtime_options_t *opts)
{
    memset(opts, 0, sizeof(*opts));
    opts->max_queries = 1;
    opts->nprobe = 32;
    opts->read_depth = 8;
    opts->stage1_gap_merge_limit = 1;
    opts->coarse_backend = "faiss";
    opts->iova_mode = NULL;
    opts->print_per_query = false;
}

static int parse_runtime_options(int argc, char **argv, runtime_options_t *opts)
{
    if (!opts) {
        return -1;
    }

    init_runtime_options(opts);

    static struct option long_opts[] = {
        {"max-queries", required_argument, 0, 'm'},
        {"nprobe", required_argument, 0, 'n'},
        {"read-depth", required_argument, 0, 'r'},
        {"stage1-gap-merge", required_argument, 0, 'g'},
        {"coarse-backend", required_argument, 0, 'c'},
        {"iova-mode", required_argument, 0, 'i'},
        {"print-per-query", no_argument, 0, 'p'},
        {"summary-only", no_argument, 0, 's'},
        {0, 0, 0, 0}
    };

    int opt = 0;
    int idx = 0;
    while ((opt = getopt_long(argc, argv, "", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'm':
                opts->max_queries = (uint32_t)strtoul(optarg, NULL, 10);
                break;
            case 'n':
                opts->nprobe = (uint32_t)strtoul(optarg, NULL, 10);
                break;
            case 'r':
                opts->read_depth = (uint32_t)strtoul(optarg, NULL, 10);
                break;
            case 'g':
                opts->stage1_gap_merge_limit = (uint32_t)strtoul(optarg, NULL, 10);
                break;
            case 'c':
                opts->coarse_backend = optarg;
                break;
            case 'i':
                opts->iova_mode = optarg;
                break;
            case 'p':
                opts->print_per_query = true;
                break;
            case 's':
                opts->print_per_query = false;
                break;
            default:
                fprintf(stderr,
                        "Usage: %s [--max-queries N] [--nprobe N] [--read-depth N] [--stage1-gap-merge N] [--coarse-backend brute|faiss] [--iova-mode pa|va] [--print-per-query] [--summary-only]\n",
                        argv[0]);
                return -1;
        }
    }

    if (opts->nprobe == 0) {
        fprintf(stderr, "parse_runtime_options: nprobe must be > 0\n");
        return -1;
    }
    if (opts->read_depth == 0 || opts->read_depth > MAX_BATCH) {
        fprintf(stderr, "parse_runtime_options: read-depth must be in [1, %d]\n", MAX_BATCH);
        return -1;
    }
    if (opts->stage1_gap_merge_limit > MAX_BATCH) {
        fprintf(stderr, "parse_runtime_options: stage1-gap-merge must be in [0, %d]\n", MAX_BATCH);
        return -1;
    }
    if (strcmp(opts->coarse_backend, "brute") != 0 &&
        strcmp(opts->coarse_backend, "faiss") != 0) {
        fprintf(stderr,
                "parse_runtime_options: coarse-backend must be 'brute' or 'faiss' (got=%s)\n",
                opts->coarse_backend);
        return -1;
    }
    if (opts->iova_mode &&
        strcmp(opts->iova_mode, "pa") != 0 &&
        strcmp(opts->iova_mode, "va") != 0) {
        fprintf(stderr,
                "parse_runtime_options: iova-mode must be 'pa' or 'va' (got=%s)\n",
                opts->iova_mode);
        return -1;
    }

    return 0;
}

static void free_gt_set(gt_set_t *gt)
{
    if (!gt) {
        return;
    }
    free(gt->ids);
    memset(gt, 0, sizeof(*gt));
}

static int load_ivecs_topk(const char *path, gt_set_t *gt)
{
    if (!path || !gt) {
        return -1;
    }

    memset(gt, 0, sizeof(*gt));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror("fopen groundtruth");
        return -1;
    }

    int dim = 0;
    if (fread(&dim, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "load_ivecs_topk: failed to read dim from %s\n", path);
        fclose(fp);
        return -1;
    }
    if (dim <= 0) {
        fprintf(stderr, "load_ivecs_topk: invalid dim=%d in %s\n", dim, path);
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        perror("fseek end");
        fclose(fp);
        return -1;
    }

    long fsize = ftell(fp);
    if (fsize < 0) {
        perror("ftell");
        fclose(fp);
        return -1;
    }
    rewind(fp);

    long rec_size = (long)sizeof(int) + (long)dim * (long)sizeof(int32_t);
    if (rec_size <= 0 || fsize % rec_size != 0) {
        fprintf(stderr, "load_ivecs_topk: invalid file size=%ld rec_size=%ld\n", fsize, rec_size);
        fclose(fp);
        return -1;
    }

    uint32_t n_queries = (uint32_t)(fsize / rec_size);
    int32_t *ids = (int32_t *)malloc((size_t)n_queries * (size_t)dim * sizeof(int32_t));
    if (!ids) {
        perror("malloc groundtruth");
        fclose(fp);
        return -1;
    }

    for (uint32_t i = 0; i < n_queries; i++) {
        int row_dim = 0;
        if (fread(&row_dim, sizeof(int), 1, fp) != 1) {
            fprintf(stderr, "load_ivecs_topk: failed to read row dim %u\n", i);
            free(ids);
            fclose(fp);
            return -1;
        }
        if (row_dim != dim) {
            fprintf(stderr, "load_ivecs_topk: inconsistent row dim at %u got=%d expected=%d\n",
                    i, row_dim, dim);
            free(ids);
            fclose(fp);
            return -1;
        }
        if (fread(ids + (size_t)i * (size_t)dim, sizeof(int32_t), (size_t)dim, fp) != (size_t)dim) {
            fprintf(stderr, "load_ivecs_topk: failed to read row %u\n", i);
            free(ids);
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    gt->n_queries = n_queries;
    gt->k = (uint32_t)dim;
    gt->ids = ids;
    return 0;
}

static double recall_at_k_overlap(const topk_state_t *pred, const gt_set_t *gt, uint32_t query_idx, uint32_t k)
{
    if (!pred || !gt || !gt->ids || query_idx >= gt->n_queries || k == 0) {
        return -1.0;
    }

    uint32_t eval_k = k;
    if (eval_k > pred->size) {
        eval_k = pred->size;
    }
    if (eval_k > gt->k) {
        eval_k = gt->k;
    }
    if (eval_k == 0) {
        return 0.0;
    }

    const int32_t *gt_row = gt->ids + (size_t)query_idx * (size_t)gt->k;
    bool gt_used[TOPK] = {false};
    uint32_t hits = 0;

    for (uint32_t i = 0; i < eval_k; i++) {
        int32_t pred_id = (int32_t)pred->items[i].vec_id;
        for (uint32_t j = 0; j < eval_k; j++) {
            if (!gt_used[j] && pred_id == gt_row[j]) {
                gt_used[j] = true;
                hits++;
                break;
            }
        }
    }

    return (double)hits / (double)eval_k;
}

static void print_query_report(const pipeline_app_t *app, uint64_t qid, uint32_t query_idx, const gt_set_t *gt)
{
    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        const query_tracker_t *qt = &app->queries[i];
        if (qt->qid != qid) {
            continue;
        }

        double latency_ms = 0.0;
        double service_ms = (double)query_service_us(qt) / 1000.0;
        if (qt->done_ts_us > qt->submit_ts_us) {
            latency_ms = (qt->done_ts_us - qt->submit_ts_us) / 1000.0;
        }

        printf("[query tracker] qid=%lu nprobe=%u probed_clusters=%u "
            "initial_candidates=%lu submitted_batches=%u completed_batches=%u "
            "outstanding=%u max_outstanding=%u done=%d latency_ms=%.3f service_ms=%.3f "
            "coarse_ms=%.3f submit_ms=%.3f prune_threshold=%.3f\n",
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
            service_ms,
            (double)qt->coarse_search_us / 1000.0,
            (double)qt->submit_candidates_us / 1000.0,
            qt->prune_threshold);

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

        double recall10 = recall_at_k_overlap(&qt->query_topk, gt, query_idx, TOPK);
        if (recall10 >= 0.0) {
            printf("  recall@%d=%.4f\n", TOPK, recall10);
            printf("  query_topk:");
            for (uint32_t k = 0; k < qt->query_topk.size; k++) {
                printf(" %u", qt->query_topk.items[k].vec_id);
            }
            printf("\n");
        }
        return;
    }

    printf("[query tracker] qid=%lu not found\n", qid);
}

static const query_tracker_t *find_query_tracker_const(const pipeline_app_t *app, uint64_t qid)
{
    if (!app || qid == 0) {
        return NULL;
    }

    for (uint32_t i = 0; i < MAX_QUERIES_IN_FLIGHT; i++) {
        if (app->queries[i].qid == qid) {
            return &app->queries[i];
        }
    }
    return NULL;
}

static uint64_t query_service_us(const query_tracker_t *qt)
{
    if (!qt) {
        return 0;
    }

    uint64_t service_us = qt->coarse_search_us + qt->submit_candidates_us + qt->topk_wall_us;
    for (int s = 0; s < NUM_STAGES; s++) {
        service_us += qt->stage_wall_us[s];
    }
    return service_us;
}

static void accumulate_query_summary(run_summary_t *sum, const query_tracker_t *qt, double recall10)
{
    if (!sum || !qt) {
        return;
    }

    sum->queries++;
    if (qt->done_ts_us > qt->submit_ts_us) {
        sum->latency_us_sum += qt->done_ts_us - qt->submit_ts_us;
    }
    sum->service_us_sum += query_service_us(qt);
    sum->coarse_us_sum += qt->coarse_search_us;
    sum->submit_us_sum += qt->submit_candidates_us;
    sum->topk_wall_us_sum += qt->topk_wall_us;
    if (recall10 >= 0.0) {
        sum->recall_sum += recall10;
    }

    for (int s = 0; s < NUM_STAGES; s++) {
        sum->stage_in_sum[s] += qt->stage_in[s];
        sum->stage_out_sum[s] += qt->stage_out[s];
        sum->stage_pruned_sum[s] += qt->stage_pruned[s];
        sum->stage_batches_sum[s] += qt->stage_batches[s];
        sum->stage_bundles_sum[s] += qt->stage_bundles_read[s];
        sum->stage_wall_us_sum[s] += qt->stage_wall_us[s];
        sum->stage_io_us_sum[s] += qt->stage_io_us[s];
        sum->stage_qsort_us_sum[s] += qt->stage_qsort_us[s];
    }
}

static void print_run_summary(const run_summary_t *sum, uint32_t nprobe, uint64_t wall_us)
{
    if (!sum || sum->queries == 0) {
        printf("[run summary] no queries completed\n");
        return;
    }

    double qps = 0.0;
    if (wall_us > 0) {
        qps = (double)sum->queries * 1000000.0 / (double)wall_us;
    }

    printf("[run summary] queries=%u nprobe=%u total_wall_ms=%.3f qps=%.3f avg_latency_ms=%.3f avg_coarse_ms=%.3f avg_submit_ms=%.3f avg_recall@%d=%.4f\n",
           sum->queries,
           nprobe,
           (double)wall_us / 1000.0,
           qps,
           (double)sum->latency_us_sum / 1000.0 / (double)sum->queries,
           (double)sum->coarse_us_sum / 1000.0 / (double)sum->queries,
           (double)sum->submit_us_sum / 1000.0 / (double)sum->queries,
           TOPK,
           sum->recall_sum / (double)sum->queries);

    for (int s = 0; s < NUM_STAGES; s++) {
        double avg_wall_ms = (double)sum->stage_wall_us_sum[s] / 1000.0 / (double)sum->queries;
        double avg_io_ms = (double)sum->stage_io_us_sum[s] / 1000.0 / (double)sum->queries;
        double avg_qsort_ms = (double)sum->stage_qsort_us_sum[s] / 1000.0 / (double)sum->queries;
        double avg_compute_ms = avg_wall_ms - avg_io_ms - avg_qsort_ms;
        if (avg_compute_ms < 0.0) {
            avg_compute_ms = 0.0;
        }
        printf("  stage%d avg_in=%.1f avg_out=%.1f avg_pruned=%.1f avg_batches=%.1f avg_bundles=%.1f avg_wall_ms=%.3f avg_io_ms=%.3f avg_qsort_ms=%.3f avg_est_compute_ms=%.3f\n",
               s,
               (double)sum->stage_in_sum[s] / (double)sum->queries,
               (double)sum->stage_out_sum[s] / (double)sum->queries,
               (double)sum->stage_pruned_sum[s] / (double)sum->queries,
               (double)sum->stage_batches_sum[s] / (double)sum->queries,
               (double)sum->stage_bundles_sum[s] / (double)sum->queries,
               avg_wall_ms,
               avg_io_ms,
               avg_qsort_ms,
               avg_compute_ms);
    }
}

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
     /* 1. 读取参数 + 配置 */
    runtime_options_t runtime_opts;
    if (parse_runtime_options(argc, argv, &runtime_opts) != 0) {
        return 1;
    }

    /* 2. SPDK 初始化 */
    struct spdk_env_opts opts;
    memset(&opts, 0, sizeof(opts));
    spdk_env_opts_init(&opts);
    opts.opts_size = sizeof(opts);
    opts.name = "spdk_4stage_ann";
    opts.mem_size = 1024;
    if (runtime_opts.iova_mode) {
        opts.iova_mode = runtime_opts.iova_mode;
    }

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
    // TODO 如果后面要改每个stage的核数，就改这一行
    const uint32_t stage_worker_counts[NUM_STAGES] = {6, 4, 4, 4};
    const int stage_cores[NUM_STAGES][MAX_WORKERS_PER_STAGE] = {
        {1, 5, 9, 13, 17, 21},
        {2, 6, 10, 14, 18, 22},
        {3, 7, 11, 15, 19, 23},
        {4, 8, 12, 16, 20, 24}
    };
    int topk_core = 25;

    /* 4. query 相关与输入参数配置 */
    float threshold = 80000.0f; // fallback 阈值；正常情况下会被每个 query 的动态阈值覆盖
    const char *ivf_meta_path = 
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta.bin"; // ivf_meta_flex 的路径
    const char *query_fvecs_path =
        "/home/zhangshujie/ann_nic/gist/gist_query.fvecs";
    const char *gt_ivecs_path =
        "/home/zhangshujie/ann_nic/gist/gist_groundtruth.ivecs";
    const char *pca_mean_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_mean.bin";
    const char *pca_components_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_components.bin";
    const char *pca_ev_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_explained_variance.bin";
    const char *pca_meta_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_pca_meta.bin";
    const char *sorted_ids_path =
        "/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_ids.bin";

    /* 5. 创建 pipeline 并启动 */
    pipeline_app_t app;
    if (pipeline_init(&app, disks, stage_worker_counts, stage_cores, topk_core,
                      runtime_opts.read_depth, runtime_opts.stage1_gap_merge_limit,
                      runtime_opts.coarse_backend,
                      threshold, ivf_meta_path, sorted_ids_path) != 0) {
        fprintf(stderr, "pipeline_init failed\n");
        return 1;
    }
    /* 这个 init 函数里面包含了读 ivf_meta_flex.bin 和 sorted_ids.bin */

    pipeline_start(&app);

    // /* 5.2 检查前几个cluster的信息，确认ivf_meta.bin读入正确 */
    // for (uint32_t cid = 0; cid < 10 && cid < app.ivf_meta.nlist; cid++) {
    //     const cluster_info_t *x = find_cluster_info(&app.ivf_meta, cid);
    //     if (x) {
    //         printf("cluster %u: start_lba=%lu num_vectors=%u num_lbas=%u\n",
    //             cid, x->start_lba, x->num_vectors, x->num_lbas);
    //     }
    // }
    // fflush(stdout);

    /* 6.初始化query set */ 
    query_set_t qs;
    memset(&qs, 0, sizeof(qs));
    /* 6.1 读query并作PCA变换 */ 
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

    /* 6.2 读groundtruth ids */ 
    gt_set_t gt;
    if (load_ivecs_topk(gt_ivecs_path, &gt) != 0) {
        fprintf(stderr, "load_ivecs_topk failed\n");
        free_query_set(&qs);
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    }
    printf("[main] loaded groundtruth queries=%u k=%u\n", gt.n_queries, gt.k);
    fflush(stdout);

    /* 一些参数合法性检查 */ 
    uint32_t nprobe = runtime_opts.nprobe;
    uint32_t max_queries_to_run = runtime_opts.max_queries;
    if (max_queries_to_run == 0 || max_queries_to_run > qs.n_queries) {
        max_queries_to_run = qs.n_queries;
    }
    if (max_queries_to_run > gt.n_queries) {
        max_queries_to_run = gt.n_queries;
    }

    uint64_t *submitted_qids = (uint64_t *)calloc(max_queries_to_run, sizeof(*submitted_qids));
    if (!submitted_qids) {
        perror("calloc submitted_qids");
        free_gt_set(&gt);
        free_query_set(&qs);
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    }

    if (qs.dim != app.ivf_meta.header.dim) {
        fprintf(stderr, "query dim mismatch: got %u expected %u\n",
                qs.dim, app.ivf_meta.header.dim);
        free(submitted_qids);
        free_gt_set(&gt);
        free_query_set(&qs);
        pipeline_stop(&app);
        pipeline_join(&app);
        pipeline_destroy(&app);
        return 1;
    }

    /* 7. 正式运行，逐个提交query */
    uint32_t submitted_count = 0;
    uint64_t run_begin_us = now_us();
    for (uint32_t qi = 0; qi < max_queries_to_run; qi++) {
        uint64_t qid = (uint64_t)qi + 1;
        const float *query = &qs.data[(size_t)qi * (size_t)qs.dim];

        if (runtime_opts.print_per_query) {
            printf("\n[main] submit query qi=%u qid=%lu nprobe=%u\n", qi, qid, nprobe);
        }

        if (submit_query(&app, qid, query, nprobe) != 0) {
            fprintf(stderr, "submit_query failed for qid=%lu\n", qid);
            free(submitted_qids);
            free_gt_set(&gt);
            free_query_set(&qs);
            pipeline_stop(&app);
            pipeline_join(&app);
            pipeline_destroy(&app);
            return 1;
        }
        submitted_qids[submitted_count++] = qid;
    }

    run_summary_t summary;
    memset(&summary, 0, sizeof(summary));

    /* 8. 异步等待所有query返回 */
    for (uint32_t qi = 0; qi < submitted_count; qi++) {
        uint64_t qid = submitted_qids[qi];
        int wrc = wait_query_done(&app, qid, 10000);
        if (wrc == 1) {
            fprintf(stderr, "wait_query_done timeout for qid=%lu\n", qid);
            free(submitted_qids);
            free_gt_set(&gt);
            free_query_set(&qs);
            pipeline_stop(&app);
            pipeline_join(&app);
            pipeline_destroy(&app);
            return 1;
        }
        if (wrc != 0) {
            fprintf(stderr, "wait_query_done error for qid=%lu\n", qid);
            free(submitted_qids);
            free_gt_set(&gt);
            free_query_set(&qs);
            pipeline_stop(&app);
            pipeline_join(&app);
            pipeline_destroy(&app);
            return 1;
        }

        pthread_mutex_lock(&app.query_mu);
        const query_tracker_t *qt = find_query_tracker_const(&app, qid);
        if (!qt) {
            pthread_mutex_unlock(&app.query_mu);
            fprintf(stderr, "query tracker missing for qid=%lu\n", qid);
            free(submitted_qids);
            free_gt_set(&gt);
            free_query_set(&qs);
            pipeline_stop(&app);
            pipeline_join(&app);
            pipeline_destroy(&app);
            return 1;
        }

        double recall10 = recall_at_k_overlap(&qt->query_topk, &gt, qi, TOPK);
        accumulate_query_summary(&summary, qt, recall10);
        if (runtime_opts.print_per_query) {
            print_query_report(&app, qid, qi, &gt);
            fflush(stdout);
        }
        pthread_mutex_unlock(&app.query_mu);
    }
    uint64_t run_wall_us = now_us() - run_begin_us;

    // 这里必须严格顺序，不然会内存泄漏
    pipeline_stop(&app);
    pipeline_join(&app);

    print_run_summary(&summary, nprobe, run_wall_us);
    fflush(stdout);

    free(submitted_qids);
    free_query_set(&qs);
    free_gt_set(&gt);
    pipeline_destroy(&app);
    return 0;
}
