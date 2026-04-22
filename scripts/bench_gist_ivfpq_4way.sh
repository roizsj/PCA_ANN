#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhangshujie/ann_ssd/pca_ann}"
BIN_DIR="${BIN_DIR:-${ROOT_DIR}/build/bin}"
RAW_GIST_DIR="${RAW_GIST_DIR:-/home/zhangshujie/ann_nic/gist}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/preprocessing/gist1m_ivfpq_output}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/benchmark_results}"

BASE_FVECS="${BASE_FVECS:-${RAW_GIST_DIR}/gist_base.fvecs}"
CENTROIDS="${CENTROIDS:-${OUT_DIR}/gist_ivfpq_centroids.bin}"
LABELS="${LABELS:-${OUT_DIR}/gist_ivfpq_codebook.bin}"
PQ_CODES="${PQ_CODES:-${OUT_DIR}/gist_pq_codes.bin}"
PQ_TABLE="${PQ_TABLE:-${OUT_DIR}/gist_pq_table.bin}"
META="${META:-${OUT_DIR}/ivfpq_disk_meta.bin}"
SORTED_IDS="${SORTED_IDS:-${OUT_DIR}/sorted_vec_ids_ivfpq.bin}"
QUERIES="${QUERIES:-${RAW_GIST_DIR}/gist_query.fvecs}"
GT="${GT:-${RAW_GIST_DIR}/gist_groundtruth.ivecs}"

DISK0="${DISK0:-0000:e3:00.0}"
DISK1="${DISK1:-0000:e4:00.0}"
DISK2="${DISK2:-0000:e5:00.0}"
DISK3="${DISK3:-0000:e6:00.0}"
BASE_LBA="${BASE_LBA:-0}"

NPROBE="${NPROBE:-32}"
RERANK_K="${RERANK_K:-100}"
THREADS_LIST="${THREADS_LIST:-1 2 4 8 16 32}"
CLUSTER_THREADS="${CLUSTER_THREADS:-1}"
IO_DEPTH="${IO_DEPTH:-8}"
PQ_READ_LBAS="${PQ_READ_LBAS:-16}"
BASE_CORE="${BASE_CORE:-0}"
RUN_WRITE="${RUN_WRITE:-1}"
MAKE_TARGETS="${MAKE_TARGETS:-ivf_pq_write_disk_gist ivf_pq_baseline_gist}"

mkdir -p "${RESULT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="${RESULT_DIR}/gist_ivfpq_4way_${TS}.csv"
LOG_DIR="${RESULT_DIR}/gist_ivfpq_4way_${TS}_logs"
mkdir -p "${LOG_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing file: ${path}" >&2
    exit 1
  fi
}

extract_header_metric() {
  local key="$1"
  local file="$2"
  awk -v key="${key}" '
    index($0, "[ivfpq]") && !index($0, "[ivfpq summary]") {
      for (i = 1; i <= NF; i++) {
        if ($i ~ ("^" key "=")) {
          split($i, a, "=")
          print a[2]
        }
      }
    }
  ' "${file}" | tail -n 1
}

extract_summary_metric() {
  local key="$1"
  local file="$2"
  awk -v key="${key}" '
    index($0, "[ivfpq summary]") {
      for (i = 1; i <= NF; i++) {
        if ($i ~ ("^" key "=")) {
          split($i, a, "=")
          print a[2]
        }
      }
    }
  ' "${file}" | tail -n 1
}

calc_payload_bytes_per_query() {
  local pq_candidates="$1"
  local pq_m="$2"
  local rerank_candidates="$3"
  local dim="$4"
  awk -v pq_candidates="${pq_candidates}" \
      -v pq_m="${pq_m}" \
      -v rerank_candidates="${rerank_candidates}" \
      -v dim="${dim}" \
    'BEGIN { printf "%.3f", pq_candidates * pq_m + rerank_candidates * dim * 4.0 }'
}

calc_bandwidth_mb_s() {
  local bytes_per_query="$1"
  local qps="$2"
  awk -v bytes_per_query="${bytes_per_query}" -v qps="${qps}" \
    'BEGIN { printf "%.3f", bytes_per_query * qps / 1000000.0 }'
}

run_ivfpq() {
  local max_queries="$1"
  local threads="$2"
  local log_file="$3"
  "${BIN_DIR}/ivf_pq_baseline_gist" \
    --disk0 "${DISK0}" \
    --disk1 "${DISK1}" \
    --disk2 "${DISK2}" \
    --disk3 "${DISK3}" \
    --meta "${META}" \
    --sorted-ids "${SORTED_IDS}" \
    --queries "${QUERIES}" \
    --gt "${GT}" \
    --nprobe "${NPROBE}" \
    --rerank-k "${RERANK_K}" \
    --max-queries "${max_queries}" \
    --threads "${threads}" \
    --cluster-threads "${CLUSTER_THREADS}" \
    --io-depth "${IO_DEPTH}" \
    --pq-read-lbas "${PQ_READ_LBAS}" \
    --base-core "${BASE_CORE}" \
    > "${log_file}" 2>&1
}

echo "[info] root=${ROOT_DIR}"
echo "[info] build targets: ${MAKE_TARGETS}"
make -C "${ROOT_DIR}" ${MAKE_TARGETS}

require_file "${BASE_FVECS}"
require_file "${CENTROIDS}"
require_file "${LABELS}"
require_file "${PQ_CODES}"
require_file "${PQ_TABLE}"
require_file "${QUERIES}"
require_file "${GT}"

if [[ "${RUN_WRITE}" != "0" ]]; then
  echo "[info] writing IVF-PQ 4-way layout to disks"
  "${BIN_DIR}/ivf_pq_write_disk_gist" \
    --base "${BASE_FVECS}" \
    --centroids "${CENTROIDS}" \
    --labels "${LABELS}" \
    --pq-codes "${PQ_CODES}" \
    --pq-table "${PQ_TABLE}" \
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

echo "threads,single_query_latency_ms,single_query_qps,single_query_payload_read_mb_s,single_query_payload_bytes_per_query,single_query_pq_io_ms,single_query_pq_scan_ms,single_query_rerank_io_ms,single_query_rerank_compute_ms,single_query_pq_candidates,single_query_rerank_candidates,single_query_recall10,query1000_avg_latency_ms,query1000_qps,query1000_payload_read_mb_s,query1000_payload_bytes_per_query,query1000_wall_ms,query1000_pq_io_ms,query1000_pq_scan_ms,query1000_rerank_io_ms,query1000_rerank_compute_ms,query1000_pq_candidates,query1000_rerank_candidates,query1000_recall10" > "${CSV}"

printf "%8s %18s %14s %18s %18s %14s %14s %18s %22s %16s %20s %10s %18s %14s %18s %18s %14s %14s %14s %18s %22s %16s %20s %10s\n" \
  "threads" "1q_lat_ms" "1q_qps" "1q_MBps" "1q_B/q" "1q_pq_io" "1q_pq_scan" "1q_rerank_io" "1q_rerank_compute" "1q_pq_cand" "1q_rerank_cand" "1q_recall" \
  "1000q_lat_ms" "1000q_qps" "1000q_MBps" "1000q_B/q" "wall_ms" "pq_io" "pq_scan" "rerank_io" "rerank_compute" "pq_cand" "rerank_cand" "recall@10"

for threads in ${THREADS_LIST}; do
  echo "[info] benchmark threads=${threads}: single query latency"
  single_log="${LOG_DIR}/threads_${threads}_single_query.log"
  run_ivfpq 1 1 "${single_log}"

  single_dim="$(extract_header_metric "dim" "${single_log}")"
  single_pq_m="$(extract_header_metric "pq_m" "${single_log}")"
  single_qps="$(extract_header_metric "qps" "${single_log}")"
  single_lat="$(extract_summary_metric "avg_latency_ms" "${single_log}")"
  single_pq_io="$(extract_summary_metric "avg_pq_io_ms" "${single_log}")"
  single_pq_scan="$(extract_summary_metric "avg_pq_scan_ms" "${single_log}")"
  single_rerank_io="$(extract_summary_metric "avg_rerank_io_ms" "${single_log}")"
  single_rerank_compute="$(extract_summary_metric "avg_rerank_compute_ms" "${single_log}")"
  single_pq_candidates="$(extract_summary_metric "avg_pq_candidates" "${single_log}")"
  single_rerank_candidates="$(extract_summary_metric "avg_rerank_candidates" "${single_log}")"
  single_recall="$(extract_summary_metric "avg_recall@10" "${single_log}")"
  single_bytes="$(calc_payload_bytes_per_query "${single_pq_candidates:-0}" "${single_pq_m:-60}" "${single_rerank_candidates:-0}" "${single_dim:-960}")"
  single_bw="$(calc_bandwidth_mb_s "${single_bytes:-0}" "${single_qps:-0}")"

  echo "[info] benchmark threads=${threads}: 1000 query throughput"
  q1000_log="${LOG_DIR}/threads_${threads}_1000_queries.log"
  run_ivfpq 1000 "${threads}" "${q1000_log}"

  q1000_dim="$(extract_header_metric "dim" "${q1000_log}")"
  q1000_pq_m="$(extract_header_metric "pq_m" "${q1000_log}")"
  q1000_qps="$(extract_header_metric "qps" "${q1000_log}")"
  q1000_wall="$(extract_header_metric "wall_ms" "${q1000_log}")"
  q1000_lat="$(extract_summary_metric "avg_latency_ms" "${q1000_log}")"
  q1000_pq_io="$(extract_summary_metric "avg_pq_io_ms" "${q1000_log}")"
  q1000_pq_scan="$(extract_summary_metric "avg_pq_scan_ms" "${q1000_log}")"
  q1000_rerank_io="$(extract_summary_metric "avg_rerank_io_ms" "${q1000_log}")"
  q1000_rerank_compute="$(extract_summary_metric "avg_rerank_compute_ms" "${q1000_log}")"
  q1000_pq_candidates="$(extract_summary_metric "avg_pq_candidates" "${q1000_log}")"
  q1000_rerank_candidates="$(extract_summary_metric "avg_rerank_candidates" "${q1000_log}")"
  q1000_recall="$(extract_summary_metric "avg_recall@10" "${q1000_log}")"
  q1000_bytes="$(calc_payload_bytes_per_query "${q1000_pq_candidates:-0}" "${q1000_pq_m:-60}" "${q1000_rerank_candidates:-0}" "${q1000_dim:-960}")"
  q1000_bw="$(calc_bandwidth_mb_s "${q1000_bytes:-0}" "${q1000_qps:-0}")"

  printf "%8s %18s %14s %18s %18s %14s %14s %18s %22s %16s %20s %10s %18s %14s %18s %18s %14s %14s %14s %18s %22s %16s %20s %10s\n" \
    "${threads}" \
    "${single_lat:-NA}" "${single_qps:-NA}" "${single_bw:-NA}" "${single_bytes:-NA}" \
    "${single_pq_io:-NA}" "${single_pq_scan:-NA}" "${single_rerank_io:-NA}" "${single_rerank_compute:-NA}" \
    "${single_pq_candidates:-NA}" "${single_rerank_candidates:-NA}" "${single_recall:-NA}" \
    "${q1000_lat:-NA}" "${q1000_qps:-NA}" "${q1000_bw:-NA}" "${q1000_bytes:-NA}" "${q1000_wall:-NA}" \
    "${q1000_pq_io:-NA}" "${q1000_pq_scan:-NA}" "${q1000_rerank_io:-NA}" "${q1000_rerank_compute:-NA}" \
    "${q1000_pq_candidates:-NA}" "${q1000_rerank_candidates:-NA}" "${q1000_recall:-NA}"

  echo "${threads},${single_lat:-},${single_qps:-},${single_bw:-},${single_bytes:-},${single_pq_io:-},${single_pq_scan:-},${single_rerank_io:-},${single_rerank_compute:-},${single_pq_candidates:-},${single_rerank_candidates:-},${single_recall:-},${q1000_lat:-},${q1000_qps:-},${q1000_bw:-},${q1000_bytes:-},${q1000_wall:-},${q1000_pq_io:-},${q1000_pq_scan:-},${q1000_rerank_io:-},${q1000_rerank_compute:-},${q1000_pq_candidates:-},${q1000_rerank_candidates:-},${q1000_recall:-}" >> "${CSV}"
done

echo "[done] csv=${CSV}"
echo "[done] logs=${LOG_DIR}"
