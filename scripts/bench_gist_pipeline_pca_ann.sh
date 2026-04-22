#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhangshujie/ann_ssd/pca_ann}"
BIN_DIR="${BIN_DIR:-${ROOT_DIR}/build/bin}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/preprocessing/gist1m_output}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/benchmark_results}"

META="${META:-${OUT_DIR}/ivf_meta.bin}"
SORTED_IDS="${SORTED_IDS:-${OUT_DIR}/sorted_ids.bin}"
PCA_MEAN="${PCA_MEAN:-${OUT_DIR}/gist_pca_mean.bin}"
PCA_COMPONENTS="${PCA_COMPONENTS:-${OUT_DIR}/gist_pca_components.bin}"
PCA_EV="${PCA_EV:-${OUT_DIR}/gist_pca_explained_variance.bin}"
PCA_META="${PCA_META:-${OUT_DIR}/gist_pca_meta.bin}"

NPROBE="${NPROBE:-32}"
READ_DEPTH_LIST="${READ_DEPTH_LIST:-1 2 4 8 16 32}"
ACTIVE_STAGES="${ACTIVE_STAGES:-4}"
STAGE1_GAP_MERGE="${STAGE1_GAP_MERGE:-1}"
COARSE_BACKEND="${COARSE_BACKEND:-faiss}"
PRUNE_THRESHOLD_MODE="${PRUNE_THRESHOLD_MODE:-centroids}"
IOVA_MODE="${IOVA_MODE:-}"
STAGE_DIMS="${STAGE_DIMS:-240 240 240 240}"

mkdir -p "${RESULT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
CSV="${RESULT_DIR}/gist_pipeline_pca_ann_${TS}.csv"
LOG_DIR="${RESULT_DIR}/gist_pipeline_pca_ann_${TS}_logs"
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
    index($0, "[run summary]") {
      for (i = 1; i <= NF; i++) {
        if ($i ~ ("^" key "=")) {
          split($i, a, "=")
          print a[2]
        }
      }
    }
  ' "${file}" | tail -n 1
}

extract_stage_sum() {
  local file="$1"
  awk '
    /^  stage[0-9]+ / {
      reads = 0.0
      bytes = 0.0
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^avg_nvme_reads=/) {
          split($i, a, "=")
          reads = a[2] + 0.0
        } else if ($i ~ /^avg_read_bytes=/) {
          split($i, a, "=")
          bytes = a[2] + 0.0
        }
      }
      total += reads * bytes
    }
    END { printf "%.3f", total }
  ' "${file}"
}

extract_payload_bytes() {
  local file="$1"
  awk -v dims="${STAGE_DIMS}" '
    BEGIN {
      split(dims, dim_arr, " ")
    }
    /^  stage[0-9]+ / {
      stage = $1
      sub(/^stage/, "", stage)
      idx = stage + 1
      dim = dim_arr[idx] + 0
      avg_in = 0.0
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^avg_in=/) {
          split($i, a, "=")
          avg_in = a[2] + 0.0
        }
      }
      total += avg_in * dim * 4.0
    }
    END { printf "%.3f", total }
  ' "${file}"
}

calc_bandwidth_mb_s() {
  local bytes_per_query="$1"
  local qps="$2"
  awk -v bytes_per_query="${bytes_per_query}" -v qps="${qps}" \
    'BEGIN { printf "%.3f", bytes_per_query * qps / 1000000.0 }'
}

run_pca_ann() {
  local max_queries="$1"
  local read_depth="$2"
  local log_file="$3"
  local cmd=(
    "${BIN_DIR}/pca_ann"
    --max-queries "${max_queries}"
    --nprobe "${NPROBE}"
    --read-depth "${read_depth}"
    --stage1-gap-merge "${STAGE1_GAP_MERGE}"
    --active-stages "${ACTIVE_STAGES}"
    --coarse-backend "${COARSE_BACKEND}"
    --prune-threshold-mode "${PRUNE_THRESHOLD_MODE}"
    --summary-only
  )

  if [[ -n "${IOVA_MODE}" ]]; then
    cmd+=(--iova-mode "${IOVA_MODE}")
  fi

  "${cmd[@]}" > "${log_file}" 2>&1
}

echo "[info] root=${ROOT_DIR}"
echo "[info] build target: pca_ann"
make -C "${ROOT_DIR}" pca_ann

require_file "${META}"
require_file "${SORTED_IDS}"
require_file "${PCA_MEAN}"
require_file "${PCA_COMPONENTS}"
require_file "${PCA_EV}"
require_file "${PCA_META}"

cat <<EOF
[info] pca_ann uses hardcoded runtime paths in main.c:
       meta=${META}
       sorted_ids=${SORTED_IDS}
       PCA files under ${OUT_DIR}
[info] pca_ann also uses hardcoded disk BDFs in main.c. If disk BDFs changed, update main.c first.
[info] pca_ann stage worker counts are currently hardcoded in main.c as {4,4,4,4}; this script sweeps read-depth instead of thread count.
[info] payload bandwidth uses STAGE_DIMS="${STAGE_DIMS}" and sums avg_in * stage_dim * 4B across stages.
EOF

echo "read_depth,single_query_latency_ms,single_query_qps,single_query_device_read_mb_s,single_query_device_bytes_per_query,single_query_payload_read_mb_s,single_query_payload_bytes_per_query,query1000_avg_latency_ms,query1000_qps,query1000_device_read_mb_s,query1000_device_bytes_per_query,query1000_payload_read_mb_s,query1000_payload_bytes_per_query,query1000_wall_ms,query1000_recall10" > "${CSV}"

printf "%10s %18s %14s %18s %18s %18s %18s %18s %14s %18s %18s %18s %18s %14s %10s\n" \
  "read_depth" "1q_lat_ms" "1q_qps" "1q_dev_MBps" "1q_dev_B/q" "1q_payload_MBps" "1q_payload_B/q" "1000q_lat_ms" "1000q_qps" "1000q_dev_MBps" "1000q_dev_B/q" "1000q_payload_MBps" "1000q_payload_B/q" "wall_ms" "recall@10"

for read_depth in ${READ_DEPTH_LIST}; do
  echo "[info] benchmark read_depth=${read_depth}: single query latency"
  single_log="${LOG_DIR}/read_depth_${read_depth}_single_query.log"
  run_pca_ann 1 "${read_depth}" "${single_log}"

  single_lat="$(extract_metric "avg_latency_ms" "${single_log}")"
  single_qps="$(extract_metric "qps" "${single_log}")"
  single_bytes="$(extract_stage_sum "${single_log}")"
  single_bw="$(calc_bandwidth_mb_s "${single_bytes:-0}" "${single_qps:-0}")"
  single_payload_bytes="$(extract_payload_bytes "${single_log}")"
  single_payload_bw="$(calc_bandwidth_mb_s "${single_payload_bytes:-0}" "${single_qps:-0}")"

  echo "[info] benchmark read_depth=${read_depth}: 1000 query throughput"
  q1000_log="${LOG_DIR}/read_depth_${read_depth}_1000_queries.log"
  run_pca_ann 1000 "${read_depth}" "${q1000_log}"

  q1000_lat="$(extract_metric "avg_latency_ms" "${q1000_log}")"
  q1000_qps="$(extract_metric "qps" "${q1000_log}")"
  q1000_wall="$(extract_metric "total_wall_ms" "${q1000_log}")"
  q1000_recall="$(extract_metric "avg_recall@10" "${q1000_log}")"
  q1000_bytes="$(extract_stage_sum "${q1000_log}")"
  q1000_bw="$(calc_bandwidth_mb_s "${q1000_bytes:-0}" "${q1000_qps:-0}")"
  q1000_payload_bytes="$(extract_payload_bytes "${q1000_log}")"
  q1000_payload_bw="$(calc_bandwidth_mb_s "${q1000_payload_bytes:-0}" "${q1000_qps:-0}")"

  printf "%10s %18s %14s %18s %18s %18s %18s %18s %14s %18s %18s %18s %18s %14s %10s\n" \
    "${read_depth}" \
    "${single_lat:-NA}" \
    "${single_qps:-NA}" \
    "${single_bw:-NA}" \
    "${single_bytes:-NA}" \
    "${single_payload_bw:-NA}" \
    "${single_payload_bytes:-NA}" \
    "${q1000_lat:-NA}" \
    "${q1000_qps:-NA}" \
    "${q1000_bw:-NA}" \
    "${q1000_bytes:-NA}" \
    "${q1000_payload_bw:-NA}" \
    "${q1000_payload_bytes:-NA}" \
    "${q1000_wall:-NA}" \
    "${q1000_recall:-NA}"

  echo "${read_depth},${single_lat:-},${single_qps:-},${single_bw:-},${single_bytes:-},${single_payload_bw:-},${single_payload_bytes:-},${q1000_lat:-},${q1000_qps:-},${q1000_bw:-},${q1000_bytes:-},${q1000_payload_bw:-},${q1000_payload_bytes:-},${q1000_wall:-},${q1000_recall:-}" >> "${CSV}"
done

echo "[done] csv=${CSV}"
echo "[done] logs=${LOG_DIR}"
