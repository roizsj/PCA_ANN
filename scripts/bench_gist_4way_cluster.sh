#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhangshujie/ann_ssd/pca_ann}"
BIN_DIR="${BIN_DIR:-${ROOT_DIR}/build/bin}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/preprocessing/gist1m_output}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/benchmark_results}"

BASE_FVECS="${BASE_FVECS:-${OUT_DIR}/gist_base_pca960.fvecs}"
CENTROIDS="${CENTROIDS:-${OUT_DIR}/gist_ivf_centroids.bin}"
CODEBOOK="${CODEBOOK:-${OUT_DIR}/gist_codebook.bin}"
META="${META:-${OUT_DIR}/ivf_meta_4way_cluster.bin}"
SORTED_IDS="${SORTED_IDS:-${OUT_DIR}/sorted_vec_ids_4way_cluster.bin}"

DISK0="${DISK0:-0000:e3:00.0}"
DISK1="${DISK1:-0000:e4:00.0}"
DISK2="${DISK2:-0000:e5:00.0}"
DISK3="${DISK3:-0000:e6:00.0}"
BASE_LBA="${BASE_LBA:-0}"

NPROBE="${NPROBE:-32}"
THREADS_LIST="${THREADS_LIST:-1 2 4 8 16 32}"
BASE_CORE="${BASE_CORE:-0}"
RUN_WRITE="${RUN_WRITE:-1}"
MAKE_TARGETS="${MAKE_TARGETS:-ivf_write_disk_4way_gist ivf_baseline_4way_gist}"

mkdir -p "${RESULT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="${RESULT_DIR}/gist_4way_cluster_${TS}.csv"
LOG_DIR="${RESULT_DIR}/gist_4way_cluster_${TS}_logs"
mkdir -p "${LOG_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing file: ${path}" >&2
    exit 1
  fi
}

extract_metric() {
  local key="$1"
  local file="$2"
  awk -v key="${key}" '
    index($0, "[baseline4way-cluster summary]") {
      for (i = 1; i <= NF; i++) {
        if ($i ~ ("^" key "=")) {
          split($i, a, "=")
          print a[2]
        }
      }
    }
  ' "${file}" | tail -n 1
}

calc_bandwidth_mb_s() {
  local candidates="$1"
  local dim="$2"
  local qps="$3"
  awk -v candidates="${candidates}" -v dim="${dim}" -v qps="${qps}" \
    'BEGIN { printf "%.3f", candidates * dim * 4.0 * qps / 1000000.0 }'
}

echo "[info] root=${ROOT_DIR}"
echo "[info] build targets: ${MAKE_TARGETS}"
make -C "${ROOT_DIR}" ${MAKE_TARGETS}

require_file "${BASE_FVECS}"
require_file "${CENTROIDS}"
require_file "${CODEBOOK}"

if [[ "${RUN_WRITE}" != "0" ]]; then
  echo "[info] writing 4-way cluster-balanced layout to disks"
  "${BIN_DIR}/ivf_write_disk_4way_gist" \
    --input "${BASE_FVECS}" \
    --centroids "${CENTROIDS}" \
    --codebook "${CODEBOOK}" \
    --meta "${META}" \
    --sorted-ids "${SORTED_IDS}" \
    --disk0 "${DISK0}" \
    --disk1 "${DISK1}" \
    --disk2 "${DISK2}" \
    --disk3 "${DISK3}" \
    --base-lba "${BASE_LBA}" \
    2>&1 | tee "${LOG_DIR}/write_disk.log"
else
  echo "[info] RUN_WRITE=0, skip disk writing and reuse existing metadata/layout"
fi

require_file "${META}"
require_file "${SORTED_IDS}"

echo "threads,single_query_latency_ms,single_query_qps,single_query_read_mb_s,query1000_avg_latency_ms,query1000_qps,query1000_read_mb_s,query1000_wall_ms,query1000_recall10" > "${CSV}"

printf "%8s %18s %14s %18s %18s %14s %18s %14s %10s\n" \
  "threads" "1q_lat_ms" "1q_qps" "1q_MBps" "1000q_lat_ms" "1000q_qps" "1000q_MBps" "wall_ms" "recall@10"

for threads in ${THREADS_LIST}; do
  echo "[info] benchmark threads=${threads}: single query latency"
  single_log="${LOG_DIR}/threads_${threads}_single_query.log"
  "${BIN_DIR}/ivf_baseline_4way_gist" \
    --disk0 "${DISK0}" \
    --disk1 "${DISK1}" \
    --disk2 "${DISK2}" \
    --disk3 "${DISK3}" \
    --meta "${META}" \
    --sorted-ids "${SORTED_IDS}" \
    --nprobe "${NPROBE}" \
    --max-queries 1 \
    --threads 1 \
    --base-core "${BASE_CORE}" \
    > "${single_log}" 2>&1

  single_lat="$(extract_metric "avg_latency_ms" "${single_log}")"
  single_qps="$(extract_metric "qps" "${single_log}")"
  single_candidates="$(extract_metric "avg_candidates" "${single_log}")"
  single_bw="$(calc_bandwidth_mb_s "${single_candidates:-0}" 960 "${single_qps:-0}")"

  echo "[info] benchmark threads=${threads}: 1000 query throughput"
  q1000_log="${LOG_DIR}/threads_${threads}_1000_queries.log"
  "${BIN_DIR}/ivf_baseline_4way_gist" \
    --disk0 "${DISK0}" \
    --disk1 "${DISK1}" \
    --disk2 "${DISK2}" \
    --disk3 "${DISK3}" \
    --meta "${META}" \
    --sorted-ids "${SORTED_IDS}" \
    --nprobe "${NPROBE}" \
    --max-queries 1000 \
    --threads "${threads}" \
    --base-core "${BASE_CORE}" \
    > "${q1000_log}" 2>&1

  q1000_lat="$(extract_metric "avg_latency_ms" "${q1000_log}")"
  q1000_qps="$(extract_metric "qps" "${q1000_log}")"
  q1000_candidates="$(extract_metric "avg_candidates" "${q1000_log}")"
  q1000_wall="$(extract_metric "total_wall_ms" "${q1000_log}")"
  q1000_recall="$(extract_metric "avg_recall@10" "${q1000_log}")"
  q1000_bw="$(calc_bandwidth_mb_s "${q1000_candidates:-0}" 960 "${q1000_qps:-0}")"

  printf "%8s %18s %14s %18s %18s %14s %18s %14s %10s\n" \
    "${threads}" \
    "${single_lat:-NA}" \
    "${single_qps:-NA}" \
    "${single_bw:-NA}" \
    "${q1000_lat:-NA}" \
    "${q1000_qps:-NA}" \
    "${q1000_bw:-NA}" \
    "${q1000_wall:-NA}" \
    "${q1000_recall:-NA}"

  echo "${threads},${single_lat:-},${single_qps:-},${single_bw:-},${q1000_lat:-},${q1000_qps:-},${q1000_bw:-},${q1000_wall:-},${q1000_recall:-}" >> "${CSV}"
done

echo "[done] csv=${CSV}"
echo "[done] logs=${LOG_DIR}"
