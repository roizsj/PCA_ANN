# Data API

This document describes the data contract between the three major stages in this repo:

1. Preprocessing: build IVF/PQ files on the filesystem.
2. Write disk: transform filesystem files into the SPDK/NVMe on-disk layout.
3. Search: load metadata, read vectors from NVMe, and run IVF/IVF-PQ queries.

The most complete current pipeline is the GIST-1M no-PCA IVF-PQ baseline:

- Preprocessing: `preprocessing/ivf_pq_gist.py`
- Write disk: `build/bin/ivf_pq_write_disk_gist`
- Search: `build/bin/ivf_pq_baseline_gist`

All integer fields below are little-endian because the files are written by Python/NumPy or C on the same target platform.

## Stage 1: Preprocessing

### Program

```bash
python3 preprocessing/ivf_pq_gist.py \
  --input /home/zhangshujie/ann_nic/gist/gist_base.fvecs \
  --outdir /home/zhangshujie/ann_ssd/pca_ann/preprocessing/gist1m_ivfpq_output \
  --nlist 4096 \
  --pq-m 60 \
  --pq-nbits 8 \
  --pq-train-size 200000 \
  --mmap
```

### Input

`gist_base.fvecs`

- Standard `.fvecs` layout.
- Each row is `int32 dim` followed by `dim` float32 values.
- For GIST-1M, `dim = 960`.
- This stage uses raw GIST vectors directly; no PCA is applied.

### Output Files

`gist_ivfpq_centroids.bin`

- Produced by `save_ivf_centroids`.
- Used by the disk writer and copied into the final metadata file.
- Binary format:

```text
uint32 nlist
uint32 dim
float32 centroids[nlist][dim]
```

`gist_ivfpq_codebook.bin`

- Despite the name, this file stores IVF labels, not PQ codebooks.
- One label per base vector.
- Used by the disk writer to group vectors by cluster.
- Binary format:

```text
uint32 n
uint32 labels[n]
```

`gist_pq_table.bin`

- Stores PQ codebooks.
- Used by the disk writer and copied into the final metadata file.
- `MAGIC_PQ = 0x47515049`, written by the script as the marker for the preprocessing PQ table.
- Current code expects `pq_nbits = 8` and `ksub <= 256`.
- Binary format:

```text
uint32 magic
uint32 dim
uint32 pq_m
uint32 pq_ksub
uint32 pq_nbits
uint32 pq_subdim
float32 codebooks[pq_m][pq_ksub][pq_subdim]
```

For the default GIST setup:

```text
dim = 960
pq_m = 60
pq_ksub = 256
pq_nbits = 8
pq_subdim = 16
```

`gist_pq_codes.bin`

- Stores the compressed PQ code for every base vector.
- Used by the disk writer to write the PQ region on NVMe.
- Binary format:

```text
uint32 n
uint32 pq_m
uint8 codes[n][pq_m]
```

### Optional Overlap IVF Members

`preprocessing/balanced_ivf_gist.py` also writes `gist_balanced_members.bin` for IVF layouts where a base vector may appear in more than one IVF list.

Binary format:

```text
uint64 nlist
uint64 total_ids
uint64 offsets[nlist + 1]
uint32 ids[total_ids]
```

Meaning:

```text
ids[offsets[c] ... offsets[c + 1]) = base vector ids stored in IVF list c
```

The same base vector id may appear in multiple lists. In this format `total_ids` is the number of IVF list entries, not the number of unique base vectors.

The overlap disk writer is:

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

It writes the same `IVF4` metadata shape as the flex writer so the current search-side parser can load it. For overlap layouts, `MetaHeaderFlex.num_vectors`, `ClusterMetaEntry.num_vectors`, and the sorted-id file count are all entry counts after duplication.

### Stage 1 Data Meaning

After preprocessing:

- `centroids[c]` is the raw-vector IVF centroid for cluster `c`.
- `labels[i]` is the cluster id assigned to base vector `i`.
- `codebooks[part][k]` is the PQ centroid for subquantizer `part`, code `k`.
- `codes[i]` is the PQ code sequence for base vector `i`.

At this point all data is still on the filesystem. The NVMe disk layout does not exist yet.

## Stage 2: Write Disk

### Program

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

### Inputs

Filesystem inputs:

- Raw base vectors: `gist_base.fvecs`
- IVF centroids: `gist_ivfpq_centroids.bin`
- IVF labels: `gist_ivfpq_codebook.bin`
- PQ codes: `gist_pq_codes.bin`
- PQ codebooks: `gist_pq_table.bin`

Device inputs:

- Four NVMe controller BDFs passed by `--disk0` through `--disk3`.
- All disks must expose the same sector size.
- Current writer expects 4096-byte sectors.

### Cluster-To-Disk Assignment

The writer keeps each cluster atomic:

- All PQ codes and raw vectors for one cluster are written to one disk.
- A cluster is never split across multiple disks.
- Different clusters may be assigned to different disks.
- Assignment uses a greedy balance by vector count: each cluster goes to the disk with the current smallest vector load.

This is the key invariant used by search: once the probed cluster id is known, the metadata gives exactly one disk to read.

### In-Memory Ordering

Before writing, the writer builds `sorted_ids`:

```text
sorted_ids[cluster_offsets[c] ... cluster_offsets[c + 1]) = original vector ids in cluster c
```

Within a cluster, `local_idx` means the vector position in that cluster-local order.

The mapping used later is:

```text
vec_id = sorted_ids[clusters[c].sorted_id_base + local_idx]
```

### NVMe Layout

For each disk:

```text
base_lba
  |
  +-- PQ region for clusters assigned to this disk
  |
  +-- raw-vector region for clusters assigned to this disk
```

The writer first lays out all PQ regions. The raw-vector region on each disk begins at:

```text
raw_region_lba[d] = base_lba + total_pq_lbas_on_disk_d
```

For each cluster `c`:

```text
clusters[c].disk_id       = assigned disk
clusters[c].pq_start_lba  = first LBA of cluster c's PQ codes
clusters[c].pq_num_lbas   = number of PQ-code LBAs for cluster c
clusters[c].raw_start_lba = first LBA of cluster c's raw vectors
clusters[c].raw_num_lbas  = number of raw-vector LBAs for cluster c
```

PQ code packing inside an LBA:

```text
pq_code_bytes = pq_m
pq_codes_per_lba = sector_size / pq_code_bytes
local_idx -> pq_lba = pq_start_lba + local_idx / pq_codes_per_lba
local_idx -> pq_lane = local_idx % pq_codes_per_lba
byte offset in LBA = pq_lane * pq_code_bytes
```

Raw vector packing inside an LBA:

```text
raw_vector_bytes = dim * sizeof(float)
raw_vectors_per_lba = sector_size / raw_vector_bytes
local_idx -> raw_lba = raw_start_lba + local_idx / raw_vectors_per_lba
local_idx -> raw_lane = local_idx % raw_vectors_per_lba
byte offset in LBA = raw_lane * raw_vector_bytes
```

For raw GIST-1M with `dim = 960` and 4096-byte sectors:

```text
raw_vector_bytes = 3840
raw_vectors_per_lba = 1
```

For default PQ with `pq_m = 60`:

```text
pq_code_bytes = 60
pq_codes_per_lba = 4096 / 60 = 68
```

Unused bytes at the end of an LBA are zero-filled.

### Output Files

`sorted_vec_ids_ivfpq.bin`

- Filesystem metadata.
- Maps cluster-local vector positions back to original base-vector ids.
- Binary format:

```text
uint32 n
uint32 sorted_ids[n]
```

`ivfpq_disk_meta.bin`

- Filesystem metadata.
- This is the main search-time API file.
- It contains global layout metadata, per-cluster disk/LBA metadata, IVF centroids, and PQ codebooks.
- `META_MAGIC = 0x4751504d`.
- Binary format:

```text
MetaHeader header
ClusterMeta clusters[header.nlist]
float32 centroids[header.nlist][header.dim]
float32 pq_codebooks[header.pq_m][header.pq_ksub][header.pq_subdim]
```

`MetaHeader` layout:

```c
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
    uint64_t raw_region_lba[4];
} MetaHeader;
```

`ClusterMeta` layout:

```c
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
```

NVMe device data:

- PQ code LBAs are written to the per-disk PQ regions.
- Raw vector LBAs are written to the per-disk raw-vector regions.
- These are not normal filesystem files; they are direct SPDK writes to the selected namespaces.

## Stage 3: Search

### Program

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
  --cluster-threads 1 \
  --io-depth 8 \
  --pq-read-lbas 16 \
  --base-core 0
```

### Inputs

Filesystem inputs:

- `ivfpq_disk_meta.bin`
- `sorted_vec_ids_ivfpq.bin`
- Query vectors: `gist_query.fvecs`
- Ground truth: `gist_groundtruth.ivecs`

Device inputs:

- Four NVMe disks containing the layout written by `ivf_pq_write_disk_gist`.
- The disk order must match the writer's `disk0` through `disk3` order.

### Search-Time Loaded Metadata

At startup, the search binary loads from `ivfpq_disk_meta.bin`:

- `MetaHeader`
- `ClusterMeta[]`
- IVF centroids
- PQ codebooks

It also loads `sorted_vec_ids_ivfpq.bin`.

The search binary validates:

- Metadata magic.
- Query dimension equals `header.dim`.
- Disk sector size equals `header.sector_size`.
- `nprobe <= header.nlist`.

### Per-Query Data Flow

For one query:

1. Coarse IVF search on CPU.
2. Build PQ ADC lookup table on CPU.
3. Read PQ codes for the `nprobe` clusters from NVMe.
4. ADC-scan PQ codes and keep approximate top `rerank_k`.
5. Sort approximate candidates by raw-vector `(disk_id, raw_lba, lane)`.
6. Read raw vectors for rerank from NVMe.
7. Exact L2 rerank on CPU.
8. Compare top-10 result with `.ivecs` ground truth.

The metadata links the steps:

```text
query -> nearest cluster ids
cluster id -> clusters[c].disk_id
cluster id -> clusters[c].pq_start_lba / pq_num_lbas
local_idx -> sorted_ids[clusters[c].sorted_id_base + local_idx]
local_idx -> clusters[c].raw_start_lba + local_idx / raw_vectors_per_lba
```

### Search Parallelism Parameters

`--threads`

- Number of query worker threads.
- Each worker processes different queries independently.
- Each worker owns one SPDK qpair per disk.

`--cluster-threads`

- Number of scan shards per query worker.
- `1` preserves the original behavior.
- If greater than `1`, the `nprobe` clusters for one query are split approximately evenly across scan shards.
- The main query worker scans one shard; `cluster_threads - 1` helper threads scan the other shards.
- Each helper owns one SPDK qpair per disk and its own PQ buffer.
- Total pthread count is roughly `threads * cluster_threads`.

`--pq-read-lbas`

- Maximum number of contiguous PQ-code LBAs read per SPDK read call.
- The runtime caps this by the NVMe namespace maximum transfer size.

`--io-depth`

- Number of raw-vector rerank reads allowed in flight per query worker.
- Applies to raw-vector rerank reads, not to PQ scan reads.

## File Dependency Graph

```text
gist_base.fvecs
  |
  | preprocessing/ivf_pq_gist.py
  v
gist_ivfpq_centroids.bin
gist_ivfpq_codebook.bin
gist_pq_table.bin
gist_pq_codes.bin
  |
  | build/bin/ivf_pq_write_disk_gist
  v
NVMe PQ regions
NVMe raw-vector regions
ivfpq_disk_meta.bin
sorted_vec_ids_ivfpq.bin
  |
  | build/bin/ivf_pq_baseline_gist
  v
latency / QPS / recall output
```

## Invariants To Preserve

These conventions are relied on across stages:

- `labels.n == pq_codes.n == number of base vectors`.
- `centroids.dim == pq_table.dim == base vector dim`.
- `pq_codes.m == pq_table.pq_m`.
- `pq_table.pq_nbits == 8`.
- `pq_table.pq_ksub <= 256`.
- `dim == pq_m * pq_subdim`.
- Every cluster id must satisfy `0 <= cluster_id < nlist`.
- Each cluster is assigned to exactly one disk.
- Search-time disk order must match write-time disk order.
- `sorted_ids` must be generated from the same labels used to write the disk layout.
- `ivfpq_disk_meta.bin` must match the actual data already written on NVMe.

If any of these invariants changes, update all three stages together.

## Naming Notes

`gist_ivfpq_codebook.bin` is an IVF label file in the current implementation. The name is historical and can be confusing:

```text
gist_ivfpq_codebook.bin -> IVF labels, uint32 labels[n]
gist_pq_table.bin      -> actual PQ codebooks
gist_pq_codes.bin      -> encoded PQ codes for every base vector
```

When extending this pipeline, prefer naming new files explicitly, for example `gist_ivfpq_labels.bin`.
