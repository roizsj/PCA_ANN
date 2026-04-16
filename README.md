# PCA ANN / IVF SPDK Baselines

This directory contains SPDK-based IVF ANN experiments over PCA-transformed vectors.

The current layout keeps shared pipeline/query code at the project root, baseline query programs under `baseline/`, and disk layout writers under `write_disk/`. All binaries are still emitted to `build/bin/`.

## Directory Layout

```text
pca_ann/
  Makefile
  main.c
  pipeline_stage.c
  pipeline_stage.h
  query_loader.c
  query_loader.h
  coarse_search_faiss.cpp
  baseline/
    ivf_baseline_1.c
    ivf_baseline_2.c
    ivf_baseline_2_gist.c
    ivf_baseline_4way_gist.c
    ivf_pq_baseline_gist.c
  write_disk/
    ivf_write_disk.c
    ivf_write_disk_1.c
    ivf_write_disk_flex.c
    ivf_write_disk_overlap_flex.c
    ivf_write_disk_4way_gist.c
    ivf_pq_write_disk_gist.c
  preprocessing/
    pca_sift.py
    pca_gist.py
    ivf.py
    ivf_pq_gist.py
    balanced_ivf_gist.py
    export_pca_output.py
    pca_output/
    ivf_output/
    gist1m_output/
```

## Build

Build everything:

```bash
cd /home/zhangshujie/ann_ssd/pca_ann
make all
```

Build one target:

```bash
make ivf_baseline_4way_gist
make ivf_write_disk_4way_gist
make ivf_write_disk_overlap_flex
make ivf_pq_write_disk_gist ivf_pq_baseline_gist
```

## Balanced GIST IVF Preprocessing

To build a GIST-1M IVF partition with 50,000 lists and about 20 vectors per list:

```bash
python3 preprocessing/balanced_ivf_gist.py \
  --input /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --outdir /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_balanced_ivf_50k \
  --nlist 50000 \
  --target-size 20 \
  --max-size 20 \
  --candidate-k 64 \
  --train-iter 25 \
  --gpu
```

Outputs:

```text
gist_balanced_ivf_centroids.bin  [u32 nlist][u32 dim][float32 centroids]
gist_balanced_codebook.bin       [u32 n][u32 labels[n]]
gist_balanced_members.bin        [u64 nlist][u64 total_ids][u64 offsets[nlist+1]][u32 ids[total_ids]]
```

`gist_balanced_codebook.bin` is compatible with the current single-assignment IVF disk writers. `gist_balanced_members.bin` is for future IVF variants that want explicit per-list ids and optional redundant ids.

Write the balanced/overlap list layout to four disks:

```bash
./build/bin/ivf_write_disk_overlap_flex \
  --input /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_balanced_ivf_50k/gist_balanced_ivf_centroids.bin \
  --members /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_balanced_ivf_50k/gist_balanced_members.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_balanced_ivf_50k/ivf_meta_overlap.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_balanced_ivf_50k/sorted_vec_ids_overlap.bin \
  --dim 960 \
  --disk0-dim 240 \
  --disk1-dim 240 \
  --disk2-dim 240 \
  --disk3-dim 240 \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --base-lba 0
```

For overlap layouts, metadata `num_vectors` and sorted-id count mean total IVF list entries, so duplicate vector ids are expected when a base vector appears in multiple lists.

Clean generated objects and binaries:

```bash
make clean
```

The default dependency locations are:

```text
SPDK_DIR=/home/wq/spdk
FAISS_DIR=/home/zhangshujie/opt/faiss
```

Override them when needed:

```bash
make SPDK_DIR=/path/to/spdk FAISS_DIR=/path/to/faiss all
```

## Common Data Paths

SIFT PCA / IVF files currently used by the SIFT baselines:

```text
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/sift_base_pca128.fvecs
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/centroids_4096.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/codebook_4096.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/ivf_meta_1_disk.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/sorted_vec_ids_1_disk.bin
/home/zhangshujie/ann_nic/sift/sift_query.fvecs
/home/zhangshujie/ann_nic/sift/sift_groundtruth.ivecs
```

GIST PCA / IVF files currently used by the GIST programs:

```text
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_base_pca960.fvecs
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_ivf_centroids.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_codebook.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_1_disk.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_4way_cluster.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_ids.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_1_disk.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_4way_cluster.bin
/home/zhangshujie/ann_nic/gist/gist_query.fvecs
/home/zhangshujie/ann_nic/gist/gist_groundtruth.ivecs
```

Replace all PCIe BDFs such as `0000:65:00.0` with the real NVMe controller addresses on the machine.

## Generated Binaries

After `make all`, `build/bin/` contains:

```text
build/bin/pca_ann
build/bin/ivf_write_disk
build/bin/ivf_write_disk_flex
build/bin/ivf_write_disk_overlap_flex
build/bin/ivf_write_disk_1
build/bin/ivf_write_disk_4way_gist
build/bin/ivf_pq_write_disk_gist
build/bin/ivf_baseline_1
build/bin/ivf_baseline_2
build/bin/ivf_baseline_2_gist
build/bin/ivf_baseline_4way_gist
build/bin/ivf_pq_baseline_gist
```

## Executable Examples

### `pca_ann`

Main four-stage pipeline query program.

Important current behavior:

- Disk BDFs are hardcoded in `main.c:init_disk_config()` as `0000:65:00.0`, `0000:66:00.0`, `0000:67:00.0`, and `0000:68:00.0`.
- GIST query, groundtruth, PCA model, IVF meta, and sorted ids paths are hardcoded in `main.c`.
- Use this with the four-disk segmented layout metadata at `preprocessing/gist1m_output/ivf_meta.bin`.

Run with defaults:

```bash
./build/bin/pca_ann
```

Run a small test:

```bash
./build/bin/pca_ann \
  --max-queries 100 \
  --nprobe 32 \
  --read-depth 16 \
  --stage1-gap-merge 16 \
  --active-stages 4 \
  --coarse-backend brute \
  --prune-threshold-mode centroid \
  --summary-only
```

Run with FAISS coarse search and per-query output:

```bash
./build/bin/pca_ann \
  --max-queries 1000 \
  --nprobe 64 \
  --read-depth 32 \
  --coarse-backend faiss \
  --prune-threshold-mode sampled \
  --iova-mode va \
  --print-per-query
```

### `ivf_write_disk`

Writes a four-disk segmented IVF layout. Each vector is split by fixed equal dimensions across four disks.

Example for SIFT PCA-128 with 32 dimensions per disk:

```bash
./build/bin/ivf_write_disk \
  --input /home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/sift_base_pca128.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/centroids_4096.bin \
  --codebook /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/codebook_4096.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/ivf_meta.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/sorted_vec_ids.bin \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --base-lba 0
```

### `ivf_write_disk_flex`

Writes a four-disk segmented IVF layout with configurable dimensions per disk. This is useful for GIST PCA-960 or non-uniform segment sizes.

Example for GIST PCA-960 split evenly as 240 dimensions per disk:

```bash
./build/bin/ivf_write_disk_flex \
  --input /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_base_pca960.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_ivf_centroids.bin \
  --codebook /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_codebook.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_ids.bin \
  --dim 960 \
  --disk0-dim 240 \
  --disk1-dim 240 \
  --disk2-dim 240 \
  --disk3-dim 240 \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --base-lba 0
```

### `ivf_write_disk_1`

Writes a one-disk full-vector IVF layout. Vectors are not split.

Example for GIST PCA-960 one-disk baseline:

```bash
./build/bin/ivf_write_disk_1 \
  --input /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_base_pca960.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_ivf_centroids.bin \
  --codebook /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_codebook.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_1_disk.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_1_disk.bin \
  --disk0 0000:65:00.0 \
  --base-lba 0
```

### `ivf_write_disk_4way_gist`

Writes a four-disk GIST baseline layout where each cluster is assigned to exactly one disk. Vectors are full PCA vectors and are not dimension-split. Different clusters are greedily balanced across the four disks by vector count.

Example:

```bash
./build/bin/ivf_write_disk_4way_gist \
  --input /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_base_pca960.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_ivf_centroids.bin \
  --codebook /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_codebook.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_4way_cluster.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_4way_cluster.bin \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --base-lba 0
```

### `ivf_baseline_1`

SIFT one-disk baseline with asynchronous read depth support.

Built-in default paths point to SIFT query, SIFT groundtruth, SIFT PCA model, and SIFT one-disk IVF metadata. You still need to pass at least `--disk0` and `--nprobe`.

Example:

```bash
./build/bin/ivf_baseline_1 \
  --disk0 0000:65:00.0 \
  --nprobe 32 \
  --max-queries 1000 \
  --threads 20 \
  --async-depth 4 \
  --base-core 0
```

Example with explicit CPU core list:

```bash
./build/bin/ivf_baseline_1 \
  --disk0 0000:65:00.0 \
  --nprobe 64 \
  --max-queries 1000 \
  --threads 8 \
  --async-depth 8 \
  --cores 0,1,2,3,4,5,6,7
```

### `ivf_baseline_2`

SIFT one-disk baseline with query-level parallel workers. It uses the same SIFT default paths as `ivf_baseline_1`.

Example:

```bash
./build/bin/ivf_baseline_2 \
  --disk0 0000:65:00.0 \
  --nprobe 32 \
  --max-queries 1000 \
  --threads 20 \
  --base-core 0
```

Example with explicit metadata paths:

```bash
./build/bin/ivf_baseline_2 \
  --disk0 0000:65:00.0 \
  --nprobe 32 \
  --max-queries 1000 \
  --threads 16 \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/ivf_meta_1_disk.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/sorted_vec_ids_1_disk.bin
```

### `ivf_baseline_2_gist`

GIST one-disk full-vector baseline with query-level parallel workers.

Example:

```bash
./build/bin/ivf_baseline_2_gist \
  --disk0 0000:65:00.0 \
  --nprobe 32 \
  --max-queries 1000 \
  --threads 20 \
  --base-core 0
```

Example with explicit GIST one-disk metadata:

```bash
./build/bin/ivf_baseline_2_gist \
  --disk0 0000:65:00.0 \
  --nprobe 64 \
  --max-queries 1000 \
  --threads 16 \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_1_disk.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_1_disk.bin
```

### `ivf_baseline_4way_gist`

GIST four-disk cluster-balanced baseline. It reads metadata produced by `ivf_write_disk_4way_gist`; each probed cluster is fetched from the disk recorded in the metadata.

Example:

```bash
./build/bin/ivf_baseline_4way_gist \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --nprobe 32 \
  --max-queries 1000 \
  --threads 32 \
  --base-core 0
```

Example with explicit CPU cores and metadata:

```bash
./build/bin/ivf_baseline_4way_gist \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --nprobe 64 \
  --max-queries 1000 \
  --threads 8 \
  --cores 0,1,2,3,4,5,6,7 \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_4way_cluster.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_4way_cluster.bin
```

### `ivf_pq_write_disk_gist`

Writes the GIST IVF-PQ disk layout. This layout does not use PCA. It stores both PQ codes and raw full GIST vectors on NVMe. Each IVF cluster is assigned to exactly one of the four disks; PQ codes are written first and raw vectors are written after the PQ region on the same disk.

This binary consumes files generated by `preprocessing/ivf_pq_gist.py`.

Example:

```bash
./build/bin/ivf_pq_write_disk_gist \
  --base /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_centroids.bin \
  --labels /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_codebook.bin \
  --pq-codes /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_codes.bin \
  --pq-table /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_table.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/ivfpq_disk_meta.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/sorted_vec_ids_ivfpq.bin \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --base-lba 0
```

### `ivf_pq_baseline_gist`

Runs GIST IVF-PQ search with rerank. For each query it:

- Performs IVF coarse search over raw GIST centroids.
- Reads PQ codes for the probed clusters from the disk recorded in metadata.
- Uses ADC lookup tables to keep the best approximate candidates.
- Reads raw full vectors for the approximate candidates.
- Reranks by exact L2 distance and reports latency, QPS, candidate counts, and recall@10.

Example:

```bash
./build/bin/ivf_pq_baseline_gist \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --nprobe 32 \
  --rerank-k 100 \
  --max-queries 1000 \
  --threads 8 \
  --cluster-threads 2 \
  --io-depth 8 \
  --pq-read-lbas 16 \
  --base-core 0
```

Example with explicit metadata:

```bash
./build/bin/ivf_pq_baseline_gist \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --nprobe 64 \
  --rerank-k 200 \
  --max-queries 1000 \
  --threads 16 \
  --cluster-threads 2 \
  --io-depth 16 \
  --pq-read-lbas 16 \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/ivfpq_disk_meta.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/sorted_vec_ids_ivfpq.bin
```

## GIST No-PCA IVF-PQ Baseline Runbook

This baseline is intended to evaluate IVF-PQ on raw GIST-1M vectors, without PCA. It has three stages:

1. `preprocessing/ivf_pq_gist.py`: train/build IVF + PQ files on the filesystem.
2. `build/bin/ivf_pq_write_disk_gist`: write PQ codes and raw vectors to four NVMe disks through SPDK.
3. `build/bin/ivf_pq_baseline_gist`: run IVF-PQ search, read raw vectors for rerank, and report latency/QPS/recall.

### Step 0: Build Binaries

```bash
cd /home/zhangshujie/ann_ssd/pca_ann
make ivf_pq_write_disk_gist ivf_pq_baseline_gist
```

The expected binaries are:

```text
build/bin/ivf_pq_write_disk_gist
build/bin/ivf_pq_baseline_gist
```

### Step 1: Preprocess Raw GIST-1M

Default input is the raw GIST base vectors:

```text
/home/zhangshujie/ann_nic/gist/gist_base.fvecs
```

Run preprocessing:

```bash
python3 preprocessing/ivf_pq_gist.py \
  --input /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --outdir /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output \
  --nlist 4096 \
  --pq-m 60 \
  --pq-nbits 8 \
  --pq-train-size 200000 \
  --ivf-batch-size 10000 \
  --pq-batch-size 10000 \
  --ivf-max-iter 100 \
  --pq-max-iter 100 \
  --mmap
```

Generated files:

```text
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_centroids.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_codebook.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_table.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_codes.bin
```

Parameter notes:

- `--nlist 4096`: number of IVF coarse clusters.
- `--pq-m 60`: GIST dim is 960, so each PQ subvector has `960 / 60 = 16` dimensions.
- `--pq-nbits 8`: one byte per PQ subquantizer code, so each vector has `60` bytes of PQ code.
- `--pq-train-size 200000`: number of sampled vectors used for PQ training. Increase for quality, decrease for faster preprocessing.
- `--mmap`: avoids eagerly copying the whole `.fvecs` file while reading.

If IVF centroids already exist and you only want to reassign + retrain PQ, use:

```bash
python3 preprocessing/ivf_pq_gist.py \
  --input /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --outdir /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output \
  --nlist 4096 \
  --pq-m 60 \
  --pq-nbits 8 \
  --ivf-centroids-in /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_centroids.bin \
  --pq-train-size 200000 \
  --mmap
```

### Step 2: Write PQ Codes + Raw Vectors To Four Disks

This stage writes two regions to each disk:

- PQ code region: compact PQ codes grouped by IVF cluster.
- Raw vector region: original full GIST vectors grouped in the same cluster order, used for rerank.

Each IVF cluster is assigned to exactly one disk. Different clusters are greedily balanced across four disks by vector count.

Run writer:

```bash
./build/bin/ivf_pq_write_disk_gist \
  --base /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_centroids.bin \
  --labels /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_codebook.bin \
  --pq-codes /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_codes.bin \
  --pq-table /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_table.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/ivfpq_disk_meta.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/sorted_vec_ids_ivfpq.bin \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --base-lba 0
```

Generated filesystem metadata:

```text
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/ivfpq_disk_meta.bin
/home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/sorted_vec_ids_ivfpq.bin
```

Important safety notes:

- Replace all `0000:65:00.0` style BDFs with the real NVMe controller addresses.
- The writer directly writes to NVMe through SPDK. Make sure the target LBAs do not contain data you need.
- `--base-lba 0` starts writing at LBA 0 on each disk. Use a different base if you reserve disk regions manually.

### Step 3: Run IVF-PQ Search + Rerank

Run test:

```bash
./build/bin/ivf_pq_baseline_gist \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/ivfpq_disk_meta.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/sorted_vec_ids_ivfpq.bin \
  --queries /home/zhangshujie/ann_nic/gist/gist_query.fvecs \
  --gt /home/zhangshujie/ann_nic/gist/gist_groundtruth.ivecs \
  --nprobe 32 \
  --rerank-k 100 \
  --max-queries 1000 \
  --threads 8 \
  --cluster-threads 2 \
  --io-depth 8 \
  --pq-read-lbas 16 \
  --base-core 0
```

I/O optimizations in this baseline:

- `--pq-read-lbas N`: reads PQ-code cluster data in multi-LBA chunks instead of one LBA per SPDK read. The program caps this value by each namespace's maximum transfer size.
- `--io-depth N`: keeps up to `N` raw-vector rerank reads in flight per worker thread, then polls SPDK completions asynchronously.
- `--cluster-threads N`: splits one query's `nprobe` clusters across `N` scan shards. The main query worker scans one shard, and `N - 1` helper threads scan the other shards with their own SPDK qpairs and PQ buffers.
- Rerank candidates are sorted by `(disk, raw_lba, lane)` before exact rerank, so candidates sharing the same raw-vector LBA are read once and reused.
- PQ scanning is chunked but still synchronous within each probed cluster. There is currently no cross-query PQ/raw-vector cache.

Output includes:

- `wall_ms`: total wall-clock time for completed queries.
- `qps`: throughput.
- `avg_latency_ms`: end-to-end per-query latency.
- `avg_coarse_ms`: IVF coarse search time.
- `avg_table_ms`: PQ ADC lookup table construction time.
- `avg_pq_io_ms`: time spent reading PQ code LBAs.
- `avg_pq_scan_ms`: ADC scanning time over PQ codes.
- `avg_rerank_io_ms`: time spent reading raw vector LBAs for rerank.
- `avg_rerank_compute_ms`: exact L2 rerank time.
- `avg_pq_candidates`: number of PQ candidates scanned before rerank.
- `avg_rerank_candidates`: number of approximate candidates reranked, normally close to `--rerank-k`.
- `avg_recall@10`: overlap recall against `gist_groundtruth.ivecs`.

The run header also prints `io_depth`, `pq_read_lbas`, and `cluster_threads`, which are the effective values after argument parsing and max-transfer capping.

When `--cluster-threads` is greater than 1, total pthread count is roughly `--threads * --cluster-threads`: every query worker owns `cluster_threads - 1` scan helpers. This can reduce single-query PQ scan latency, but it may reduce total QPS if it oversubscribes CPU cores or SPDK qpairs.

Useful sweeps:

```bash
for nprobe in 8 16 32 64; do
  ./build/bin/ivf_pq_baseline_gist \
    --disk0 0000:65:00.0 \
    --disk1 0000:66:00.0 \
    --disk2 0000:67:00.0 \
    --disk3 0000:68:00.0 \
    --nprobe "$nprobe" \
    --rerank-k 100 \
    --max-queries 1000 \
    --threads 8 \
    --cluster-threads 1 \
    --io-depth 8 \
    --pq-read-lbas 16
done
```

```bash
for rerank_k in 50 100 200 500; do
  ./build/bin/ivf_pq_baseline_gist \
    --disk0 0000:65:00.0 \
    --disk1 0000:66:00.0 \
    --disk2 0000:67:00.0 \
    --disk3 0000:68:00.0 \
    --nprobe 32 \
    --rerank-k "$rerank_k" \
    --max-queries 1000 \
    --threads 8 \
    --cluster-threads 1 \
    --io-depth 8 \
    --pq-read-lbas 16
done
```

```bash
for cluster_threads in 1 2 4 8; do
  ./build/bin/ivf_pq_baseline_gist \
    --disk0 0000:65:00.0 \
    --disk1 0000:66:00.0 \
    --disk2 0000:67:00.0 \
    --disk3 0000:68:00.0 \
    --nprobe 32 \
    --rerank-k 100 \
    --max-queries 1000 \
    --threads 4 \
    --cluster-threads "$cluster_threads" \
    --io-depth 8 \
    --pq-read-lbas 16
done
```

```bash
for io_depth in 1 2 4 8 16 32; do
  ./build/bin/ivf_pq_baseline_gist \
    --disk0 0000:65:00.0 \
    --disk1 0000:66:00.0 \
    --disk2 0000:67:00.0 \
    --disk3 0000:68:00.0 \
    --nprobe 32 \
    --rerank-k 100 \
    --max-queries 1000 \
    --threads 8 \
    --cluster-threads 1 \
    --io-depth "$io_depth" \
    --pq-read-lbas 16
done
```

```bash
for pq_read_lbas in 1 4 8 16 32; do
  ./build/bin/ivf_pq_baseline_gist \
    --disk0 0000:65:00.0 \
    --disk1 0000:66:00.0 \
    --disk2 0000:67:00.0 \
    --disk3 0000:68:00.0 \
    --nprobe 32 \
    --rerank-k 100 \
    --max-queries 1000 \
    --threads 8 \
    --cluster-threads 1 \
    --io-depth 8 \
    --pq-read-lbas "$pq_read_lbas"
done
```

## Typical Workflows

Generate or refresh the SIFT one-disk layout:

```bash
make ivf_write_disk_1 ivf_baseline_2
./build/bin/ivf_write_disk_1 \
  --input /home/zhangshujie/ann_ssd/pca_ann/preprocessing/pca_output/sift_base_pca128.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/centroids_4096.bin \
  --codebook /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/codebook_4096.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/ivf_meta_1_disk.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/ivf_output/sorted_vec_ids_1_disk.bin \
  --disk0 0000:65:00.0
./build/bin/ivf_baseline_2 \
  --disk0 0000:65:00.0 \
  --nprobe 32 \
  --max-queries 1000 \
  --threads 20
```

Generate or refresh the GIST four-way cluster-balanced layout:

```bash
make ivf_write_disk_4way_gist ivf_baseline_4way_gist
./build/bin/ivf_write_disk_4way_gist \
  --input /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_base_pca960.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_ivf_centroids.bin \
  --codebook /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/gist_codebook.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/ivf_meta_4way_cluster.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_output/sorted_vec_ids_4way_cluster.bin \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0
./build/bin/ivf_baseline_4way_gist \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --nprobe 32 \
  --max-queries 1000 \
  --threads 32
```

Generate and test the GIST no-PCA IVF-PQ + rerank baseline:

```bash
python3 preprocessing/ivf_pq_gist.py \
  --input /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --outdir /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output \
  --nlist 4096 \
  --pq-m 60 \
  --pq-nbits 8 \
  --pq-train-size 200000 \
  --mmap
make ivf_pq_write_disk_gist ivf_pq_baseline_gist
./build/bin/ivf_pq_write_disk_gist \
  --base /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --centroids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_centroids.bin \
  --labels /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_ivfpq_codebook.bin \
  --pq-codes /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_codes.bin \
  --pq-table /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/gist_pq_table.bin \
  --meta /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/ivfpq_disk_meta.bin \
  --sorted-ids /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output/sorted_vec_ids_ivfpq.bin \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0
./build/bin/ivf_pq_baseline_gist \
  --disk0 0000:65:00.0 \
  --disk1 0000:66:00.0 \
  --disk2 0000:67:00.0 \
  --disk3 0000:68:00.0 \
  --nprobe 32 \
  --rerank-k 100 \
  --max-queries 1000 \
  --threads 8
```

## Notes

The writer binaries directly write to NVMe namespaces through SPDK. Double-check disk BDFs and `--base-lba` before running them.

The query baselines read vectors from NVMe through SPDK and load query/PCA/groundtruth files from the filesystem.

For `--cores`, the number of comma-separated cores must match `--threads` when `--threads` is explicitly provided.

For `ivf_baseline_4way_gist`, `--threads` controls the number of query worker threads. If `--max-queries 1 --threads 32` is used, 32 workers are created but only one query is available to process.
