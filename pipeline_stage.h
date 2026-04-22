#ifndef PIPELINE_STAGE_H
#define PIPELINE_STAGE_H

/*
 * 这个头文件定义的是一个最小可运行的 4-stage ANN pipeline 接口。
 *
 * 设计目标：
 *  1. main.c 先完成 SPDK 环境初始化和 NVMe probe
 *  2. pipeline_stage.c 只负责“流水线执行”
 *  3. 每个 stage 对应一块盘
 *  4. 每个 stage 有多个 worker（这里默认 4 个）
 *  5. 每个 worker 固定绑一个 CPU core，并且独占一个 qpair
 */

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include <spdk/env.h>
#include <spdk/nvme.h>

typedef struct coarse_search_module coarse_search_module_t;

typedef enum {
    COARSE_BACKEND_BRUTE = 0,
    COARSE_BACKEND_FAISS = 1,
} coarse_backend_t;

typedef enum {
    PRUNE_THRESHOLD_CENTROID = 0,
    PRUNE_THRESHOLD_SAMPLED = 1,
} prune_threshold_mode_t;

/* -----------------------------
 * 全局常量
 * ----------------------------- */

/* 4-stage pipeline */
#define NUM_STAGES 4

/* 每个 stage 最多绑定多少个 worker / CPU 核 */
#define MAX_WORKERS_PER_STAGE 12

/* 单个 batch 最多容纳多少个 candidate */
#define MAX_BATCH 256

/* top-k 大小 */
#define TOPK 10

/*
 * 用于识别 batch 是否被写坏 / 传错指针。
 * debug 时非常有用。
 */
#define MAGIC_BATCH 0xBADC0DEu
#define IVF_META_MAGIC_FLEX 0x49564634u

#define MAX_QUERIES_IN_FLIGHT 1024

// prune阈值相关参数
#define PRUNE_SAMPLE_TOPK 10
#define PRUNE_SAMPLE_SIZE 10000

/* -----------------------------
 * cluster metadata 数据结构
 * ----------------------------- */

typedef struct {
    uint32_t cluster_id;
    uint64_t start_lba;
    uint32_t num_vectors;
    uint32_t num_lbas;
} ClusterMetaOnDisk;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;              // 全量向量维度
    uint32_t num_shards;       // 当前固定为 4
    uint32_t shard_dims[NUM_STAGES];
    uint32_t shard_offsets[NUM_STAGES];
    uint32_t shard_bytes[NUM_STAGES];
    uint32_t vectors_per_lba;  // 每个LBA的向量数
    uint32_t nlist;            // 聚类数量
    uint32_t num_vectors;      // 总向量数
    uint32_t sector_size;      // 扇区大小
    uint64_t base_lba;         // 基础LBA
} ivf_meta_header_t;

// 声明结构体，用于存储聚类信息
typedef struct {
    uint32_t cluster_id;    // 聚类ID
    uint64_t start_lba;     // 起始LBA
    uint32_t num_vectors;   // 向量数
    uint32_t num_lbas;      // 占用的LBA数
    uint32_t sorted_id_base; // 在 sorted_vec_ids.bin 中对应 cluster 的起始偏移
} cluster_info_t;

// 声明结构体，用于存储完整的IVF元数据
typedef struct {
    ivf_meta_header_t header;  // 头部信息
    cluster_info_t* clusters;  // 聚类信息数组
    uint32_t nlist;            // 聚类数量（与header.nlist相同）
    float *centroids;          // [nlist * dim]
} ivf_meta_t;


/* -----------------------------
 * candidate / batch 数据结构
 * ----------------------------- */

/*
 * 一个 candidate 的状态：
 *  - vec_id: 向量 ID TODO 改成sorted_id
 *  - cluster_id: 这个 candidate 属于哪个 cluster
 *  - local_idx: 这个 candidate 在这个cluster内的哪个位置（offset = local_idx * SEG_DIM * 4）
 *  - partial_sum: 当前已经累计到的距离
 */
typedef struct {
    uint32_t vec_id;
    uint32_t cluster_id;    
    uint32_t local_idx;
    float partial_sum;
} cand_item_t;

/*
 * 一个bundle：里面包含一系列向量，它们的segments位于同一个LBA上
 *  - cluster_id: 这些candidate属于哪个cluster
 *  - bundle_idx: 这些candidate对应的segment在cluster内的第几个bundle
 *  - item_indices: 这些candidate在batch中的位置索引
 *  - count: 这个bundle里有多少candidate
*/
typedef struct {
    uint32_t cluster_id;
    uint32_t bundle_idx;
    uint16_t item_indices[MAX_BATCH];   // 指向 in->items[] 中哪些候选落在这个bundle
    uint16_t count;
} bundle_group_t;


/*
 * 一个 batch：
 *  - qid: query id
 *  - stage: 当前这个 batch 应该进入哪个 stage
 *  - count: 当前 batch 中有多少个 candidate
 *  - items: candidate 列表
 */
typedef struct batch {
    uint32_t magic;
    uint64_t qid;
    uint8_t stage;
    uint16_t count;
    cand_item_t items[MAX_BATCH];
} batch_t;

/* -----------------------------
 * top-k 数据结构
 * ----------------------------- */

typedef struct {
    uint32_t vec_id;
    float dist;
} topk_item_t;

/*
 * 一个最小 top-k 状态：
 *  - items: 当前 top-k 集合
 *  - size: 当前已有多少个元素
 *
 * 当前实现里只有 top-k 线程会写它，主线程只会在 pipeline_join() 之后读取，
 * 所以这里不再为热路径维护额外的锁。
 */
typedef struct {
    topk_item_t items[TOPK];
    uint32_t size;
} topk_state_t;

/* -----------------------------
 * 一个简单的线程安全指针队列
 * ----------------------------- */

typedef struct qnode {
    void *ptr;
    struct qnode *next;
} qnode_t;

/*
 * 最简单的 MPSC 风格队列（这里先用 mutex + cond 实现）
 *
 * 用途：
 *  - 每个 stage worker 有一个输入队列
 *  - top-k worker 也有一个输入队列
 */
typedef struct {
    qnode_t *head;
    qnode_t *tail;
    bool closed;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} ptr_queue_t;

/* -----------------------------
 * 磁盘上下文
 * ----------------------------- */

/*
 * main.c 在 probe 后要把这些字段填好：
 *  - traddr
 *  - ctrlr
 *  - ns
 *  - sector_size
 *
 * pipeline_stage.c 不负责 probe，只使用这里已经准备好的信息。
 */
typedef struct {
    const char *traddr;
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
    uint32_t sector_size;
} disk_ctx_t;

struct pipeline_app;

/* -----------------------------
 * stage worker / topk worker
 * ----------------------------- */

/*
 * 一个 stage worker：
 *  - 归属于某个 stage
 *  - 绑到一个 CPU core
 *  - 绑定某块盘
 *  - 拥有一个独占 qpair
 *  - 从自己的输入队列中取 batch
 */
typedef struct {
    struct pipeline_app *app;

    int worker_id;      /* 全局 worker 编号 */
    int stage_id;       /* 0~3 */
    int lane_id;        /* 当前 stage 内的 worker 编号 */
    int core_id;        /* 绑到哪个 CPU 核 */

    disk_ctx_t *disk;   /* 这个 worker 对应哪块盘 */
    struct spdk_nvme_qpair *qpair;

    pthread_t tid;
    ptr_queue_t inq;    /* 输入队列 */
} stage_worker_t;

/*
 * top-k worker：
 *  - 接收 stage3 输出的完整距离
 *  - 维护 top-k
 */
typedef struct {
    struct pipeline_app *app;
    pthread_t tid;
    ptr_queue_t inq;
    int core_id;
} topk_worker_t;

// 聚类命中项；用于IVF取nprobe个聚类
typedef struct {
    uint32_t cluster_id;
    float dist;
} coarse_hit_t;

// query对象
typedef struct {
    uint64_t qid;
    float *full_query;
    float *query_segs[NUM_STAGES];
    uint32_t nprobe;
} query_ctx_t;

// query状态跟踪
typedef struct {
    uint64_t qid;

    /* query级输入信息 */
    uint32_t nprobe;
    uint32_t num_probed_clusters;

    /* initial batch 提交统计 */
    uint64_t initial_candidates;
    uint32_t submitted_batches;

    /* batch 生命周期统计 */
    uint32_t completed_batches;
    uint32_t outstanding_batches;
    uint32_t max_outstanding_batches;
    bool submission_done;

    /* 是否完成 */
    bool done;

    /* profile */
    float prune_threshold;
    uint64_t coarse_search_us;
    uint64_t submit_candidates_us;
    uint64_t stage_in[NUM_STAGES];
    uint64_t stage_out[NUM_STAGES];
    uint64_t stage_pruned[NUM_STAGES];
    uint64_t stage_batches[NUM_STAGES];
    uint64_t stage_bundles_read[NUM_STAGES];
    uint64_t stage_nvme_reads[NUM_STAGES];
    uint64_t stage_nvme_read_bytes[NUM_STAGES];
    uint64_t stage_wall_us[NUM_STAGES];
    uint64_t stage_io_us[NUM_STAGES];
    uint64_t stage_qsort_us[NUM_STAGES];
    uint64_t topk_batches;
    uint64_t topk_items;
    uint64_t topk_wall_us;
    topk_state_t query_topk;
    uint64_t submit_ts_us;
    uint64_t done_ts_us;

    /* 为了支持并发，query不能存在worker里，得分开存 */
    float *query_segs[NUM_STAGES];

} query_tracker_t;

/* -----------------------------
 * pipeline 总上下文
 * ----------------------------- */

typedef struct pipeline_app {
    /*
     * 4 个 stage 对应的 4 块盘
     * 这些盘对象在 main.c probe 完成后传进来
     */
    disk_ctx_t disks[NUM_STAGES];

    /*
     * workers[stage][lane]
     * 例如：
     *   workers[0][0..stage_worker_counts[0]-1] -> stage0 的 worker
     *   workers[1][0..stage_worker_counts[1]-1] -> stage1 的 worker
     */
    stage_worker_t workers[NUM_STAGES][MAX_WORKERS_PER_STAGE];
    uint32_t stage_worker_counts[NUM_STAGES];
    uint32_t stage_rr_cursor[NUM_STAGES];

    /* top-k 线程 */
    topk_worker_t topk;

    /* 提前终止阈值 */
    float threshold;

    /* 每个 stage worker 允许挂起的读请求深度 */
    uint32_t read_depth;

    /* 所有 stage 允许跨过的空 bundle 数量 */
    uint32_t stage1_gap_merge_limit;

    /* stage0 单独允许跨过的空 bundle 数量 */
    uint32_t stage0_gap_merge_limit;

    /* 单次连续读最多合并多少个 LBA；0 表示只使用设备上限和 MAX_BATCH */
    uint32_t max_read_lbas;

    /* stage0 单次连续读最多合并多少个 LBA；0 表示只使用 max_read_lbas/设备上限 */
    uint32_t stage0_max_read_lbas;

    /* 实际启用多少个 stage；最后一个 active stage 之后直接进入 top-k */
    uint32_t active_stages;

    /* coarse search backend */
    coarse_backend_t coarse_backend;

    /* prune threshold mode */
    prune_threshold_mode_t prune_threshold_mode;

    /* 一些简单统计 */
    uint64_t stage_in[NUM_STAGES];
    uint64_t stage_out[NUM_STAGES];
    uint64_t stage_pruned[NUM_STAGES];

    /* top-k 状态 */
    topk_state_t topk_state;
    /* 从 ivf_meta.bin 读到的 ivf metadata 信息 */
    ivf_meta_t ivf_meta; 


    /* coarse centroids / ivf metadata */
    uint32_t nlist;
    float *centroids;   // nlist * dim
    coarse_search_module_t *coarse_module;
    uint32_t *sorted_vec_ids;
    uint32_t num_sorted_vec_ids;
    float *prune_sample_vectors;      /* [prune_sample_count * dim] */
    uint32_t prune_sample_count;

    /* query 跟踪表 */
    query_tracker_t queries[MAX_QUERIES_IN_FLIGHT];
    pthread_mutex_t query_mu;

} pipeline_app_t;

/* ============================================================
 * 队列接口
 * ============================================================ */

/* 初始化一个空队列 */
void queue_init(ptr_queue_t *q);

/* 关闭队列，唤醒所有阻塞中的消费者 */
void queue_close(ptr_queue_t *q);

/* 向队列中推入一个指针 */
void queue_push(ptr_queue_t *q, void *ptr);

/* 从队列中弹出一个指针；若队列已关闭且为空，则返回 NULL */
void *queue_pop(ptr_queue_t *q);

/* ============================================================
 * helper 接口
 * ============================================================ */

/* 把当前线程绑定到指定 CPU core */
void bind_to_core(int core_id);

/* 计算一个 segment 和 query 对应 segment 的 partial L2 距离 */
float partial_l2(const float *x, const float *q, uint32_t dim);

/* 解析ivf_metadata.bin */
int parse_ivf_meta(const char *filename, ivf_meta_t *meta);

/* 释放ivf_meta_t 结构体及其内部内存 */
void free_ivf_meta(ivf_meta_t* meta);

/* 根据 cluster_id 在 ivf_meta 中找到对应的 cluster_info_t */
const cluster_info_t *find_cluster_info(const ivf_meta_t *meta, uint32_t cluster_id);

/* ============================================================
 * pipeline 模块接口
 * ============================================================ */

/*
 * 初始化 pipeline。
 *
 * 注意：
 *  - 这里不做 nvme_probe！
 *  - 传入的 disks[] 必须已经在 main.c 里 probe 完成
 *  - stage_worker_counts 决定每个 stage 启用多少个 worker
 *  - stage_cores 决定每个 stage 的 worker 各绑哪个核
 *
 * 参数说明：
 *  - app: pipeline 总上下文
 *  - disks: 已经 probe 完成的 4 块盘
 *  - stage_worker_counts: 每个 stage 启用的 worker 数
 *  - stage_cores: 每个 stage 的 worker -> CPU core 映射
 *  - topk_core: top-k worker 绑定到哪个核
 *  - read_depth: 每个 stage worker 的 in-flight read 深度
 *  - stage1_gap_merge_limit: 所有 stage 允许跨过的空 bundle 数量
 *  - stage0_gap_merge_limit: stage0 单独允许跨过的空 bundle 数量
 *  - stage0_max_read_lbas: stage0 单次读最多合并多少个 LBA
 *  - coarse_backend_name: coarse search 后端，brute 或 faiss
 *  - threshold: 提前终止阈值
 *  - ivf_meta_path: ivf_meta_flex.bin 的路径
 */
int pipeline_init(
    pipeline_app_t *app,
    disk_ctx_t disks[NUM_STAGES],
    const uint32_t stage_worker_counts[NUM_STAGES],
    const int stage_cores[NUM_STAGES][MAX_WORKERS_PER_STAGE],
    int topk_core,
    uint32_t read_depth,
    uint32_t stage1_gap_merge_limit,
    uint32_t stage0_gap_merge_limit,
    uint32_t stage0_max_read_lbas,
    uint32_t active_stages,
    const char *coarse_backend_name,
    const char *prune_threshold_mode_name,
    float threshold,
    const char *ivf_meta_path,
    const char *sorted_ids_path
);

/*
 * 启动所有 worker 线程：
 *  - 若干个 stage worker
 *  - 1 个 top-k worker
 */
void pipeline_start(pipeline_app_t *app);

/*
 * 向 stage0 提交一个初始 batch。
 *
 * 要求：
 *  - b->magic == MAGIC_BATCH
 *  - b->stage == 0
 */
void pipeline_submit_initial_batch(pipeline_app_t *app, batch_t *b);

/*
 * 关闭所有队列，通知线程退出。
 * 注意：
 * 这只是发“停止信号”，不等待线程真正结束。
 */
void pipeline_stop(pipeline_app_t *app);

/*
 * 等待所有线程退出。
 * 一般与 pipeline_stop 配套使用。
 */
void pipeline_join(pipeline_app_t *app);
/*
 * 清理资源
 * 一般与 pipeline_stop, pipeline_join 配套使用。
 */
void pipeline_destroy(pipeline_app_t *app);


/* ============================================================
 * query 相关接口 3.17更新
 * ============================================================ */

 
 
int load_sorted_ids_bin(const char *path, uint32_t *num_vectors_out, uint32_t **sorted_ids_out);

int coarse_search_topn(const float *query,
                       const float *centroids,
                       uint32_t dim,
                       uint32_t nlist,
                       uint32_t nprobe,
                       coarse_hit_t *out_hits);

query_tracker_t *register_query(pipeline_app_t *app,
                                uint64_t qid,
                                uint32_t nprobe,
                                uint32_t num_probed_clusters);

void pipeline_free_query_tracker_segments(query_tracker_t *qt);

query_tracker_t *pipeline_find_query_tracker_locked(pipeline_app_t *app, uint64_t qid);

void pipeline_maybe_mark_query_done_locked(query_tracker_t *qt);

void mark_batch_finished(pipeline_app_t *app, uint64_t qid, uint32_t spawned_batches);

int wait_query_done(pipeline_app_t *app, uint64_t qid, uint32_t timeout_ms);

int submit_cluster_candidates(pipeline_app_t *app,
                              uint64_t qid,
                              uint32_t cluster_id,
                              uint32_t max_batch);

int submit_query(pipeline_app_t *app,
                 uint64_t qid,
                 const float *query,
                 uint32_t nprobe);

#endif /* PIPELINE_STAGE_H */
