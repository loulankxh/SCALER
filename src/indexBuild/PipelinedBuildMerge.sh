#!/bin/bash
# Pipelined build + merge.
#
# Builds shards on multiple GPUs (same scheduling as AssignByAvailGPU.sh) while
# a merge process runs in parallel. Each shard that finishes building touches
# build.done; the merge process consumes shards via Stage 1 worker pool, then
# does a final Stage 2 k-way merge.
#
# Reports:
#   build wall-clock  = time from pipeline start to the last build.done
#   total wall-clock  = time from pipeline start to merge finished
#   extra wall-clock  = total - build  (the part NOT overlapped)

set -u

# ---- GPU build config ------------------------------------------------------
declare -A GPU_PIDS

# ---- Process-tree cleanup --------------------------------------------------
# The merge binary and per-GPU builds run inside `( ... ) &` subshells so we
# can capture exit codes via files. But $! returns the WRAPPER subshell's pid;
# `kill $pid` alone tears down bash while the real workhorse (the binary) gets
# reparented to init and survives as an orphan. kill_proc_tree TERMs the
# child(ren) first, then the wrapper, with a SIGKILL fallback.
kill_proc_tree() {
    local pid="${1:-}"
    [[ -z "$pid" ]] && return 0
    kill -0 "$pid" 2>/dev/null || return 0
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    local _i
    for ((_i=0; _i<5; _i++)); do
        if ! kill -0 "$pid" 2>/dev/null && ! pgrep -P "$pid" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
}

# Catches Ctrl-C, terminal close (HUP), SIGTERM, and ordinary script exits.
# Without this, killing the parent shell would leave the merge binary and any
# in-flight RAFT_CAGRA builds running as orphans (the bug that produced the
# zombie executeDiskPipelinedMerge process the user had to manually kill).
cleanup_on_exit() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if [[ -n "${MERGE_PID:-}" ]] && kill -0 "${MERGE_PID}" 2>/dev/null; then
        echo "[pipeline] cleanup: terminating merge process tree ($MERGE_PID)" >&2
        kill_proc_tree "$MERGE_PID"
    fi
    local _slot _p
    for _slot in "${!GPU_PIDS[@]}"; do
        _p="${GPU_PIDS[$_slot]:-}"
        [[ -n "$_p" ]] && kill_proc_tree "$_p"
    done
    exit "$rc"
}
trap cleanup_on_exit EXIT INT TERM HUP
# Number of GPUs to use. 0 (or empty) means "use ALL currently-idle GPUs".
# Override via first CLI arg (e.g. `./PipelinedBuildMerge.sh 1`) or env var
# REQUESTED_NUM_GPUS. If the request exceeds the number of idle GPUs, the
# script falls back to the idle count and prints a warning.
REQUESTED_NUM_GPUS="${REQUESTED_NUM_GPUS:-0}"

RAFT_CAGRA="/home/lanlu/miniconda3/envs/rapids_raft/bin/ann/RAFT_CAGRA_ANN_BENCH"

# ---- Dataset config ------------------------------------------------------
Dataset="${Dataset:-msTuring100M}"
DatasetPath="/home/lanlu/scaleGANN/dataset/${Dataset}"
SHARD_FOLDER="${SHARD_FOLDER:-/ScaleGANN}"
N="${N:-20}"

# ---- Indexing degrees ------------------------------------------------------
BUILD_DEG="${BUILD_DEG:-64}"
INTERMEDIATE_DEG="${INTERMEDIATE_DEG:-128}"
MERGE_DEG=${BUILD_DEG}
INDEX_NAME="raft_cagra.graph_degree${BUILD_DEG}.intermediate_graph_degree${INTERMEDIATE_DEG}.graph_build_algoNN_DESCENT"

# ---- Build memory budget ---------------------------------------------------
# Same RAM ceiling that bounded each shard's build (partitioning was sized to
# fit). After all builds are done, Stage 2 can reclaim this whole budget.
# The auto-tuner in the merge binary uses this + K (known after Stage 1) to
# derive per_run_buf / output_buf / P. Override:
#   BUILD_MEMORY_GB=32 ./PipelinedBuildMerge.sh
BUILD_MEMORY_GB="${BUILD_MEMORY_GB:-16}"

# ---- Merge config ----------------------------------------------------------
MERGE_BIN="/home/lanlu/scaleGANN/build/executeDiskPipelinedMerge"
# Stage 1 parameters
MAX_MERGE_WORKERS="${MAX_MERGE_WORKERS:-1}"
SORT_BUF_MB="${SORT_BUF_MB:-1024}"
POLL_INTERVAL_MS="${POLL_INTERVAL_MS:-1000}"
# Set KEEP_SHARD_INDEX=1 to retain per-shard index files after Stage 1
# consumes them (e.g. for debugging). Default is to delete to reclaim disk.
KEEP_SHARD_INDEX="${KEEP_SHARD_INDEX:-0}"
# Set KEEP_PARTITION_INPUTS=1 to retain per-shard partition inputs
# (data.<suffix>, idmap.ibin) after Stage 1 consumes them. Default deletes to
# reclaim disk. WARNING: deleting them means you cannot re-run build/partition
# without re-creating partitions from the original full dataset.
KEEP_PARTITION_INPUTS="${KEEP_PARTITION_INPUTS:-0}"
# Stage 2 knobs. 0 = let the auto-tuner pick (recommended). Override via env:
#   STAGE2_PER_RUN_BUF_MB=4 STAGE2_OUTPUT_BUF_MB=16 ./PipelinedBuildMerge.sh
STAGE2_MAX_THREADS="${STAGE2_MAX_THREADS:-0}"
STAGE2_PER_RUN_BUF_MB="${STAGE2_PER_RUN_BUF_MB:-16}"
STAGE2_OUTPUT_BUF_MB="${STAGE2_OUTPUT_BUF_MB:-64}"

# ---- Paths -----------------------------------------------------------------
BASE_FOLDER="${DatasetPath}${SHARD_FOLDER}"
OUTPUT_DIR="${BASE_FOLDER}/mergedIndex"
TMP_DIR="${OUTPUT_DIR}/tmp_runs"
LOG_DIR="${BASE_FOLDER}"
BUILD_LOG="${LOG_DIR}/build_time.txt"
MERGE_LOG="${LOG_DIR}/merge_time.txt"
PIPELINE_LOG="${LOG_DIR}/pipeline_summary.txt"
MERGE_RC_FILE="${LOG_DIR}/.merge_rc"
BUILD_FAIL_FILE="${LOG_DIR}/.build_fail"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$TMP_DIR"
: > "$BUILD_LOG"
: > "$MERGE_LOG"
: > "$PIPELINE_LOG"
rm -f "$MERGE_RC_FILE" "$BUILD_FAIL_FILE"
echo -e "shard_id\tgpu\tbuild_time_s" >> "$BUILD_LOG"

# ---- Clean stale build.done markers ----------------------------------------
for ((i=0; i<N; i++)); do
    rm -f "${BASE_FOLDER}/partition${i}/build.done"
done

if [[ ! -x "$MERGE_BIN" ]]; then
    echo "[pipeline] ERROR: merge binary not found or not executable: $MERGE_BIN" >&2
    exit 1
fi

# ---- Detect currently-idle GPUs --------------------------------------------
detect_idle_gpus() {
    # Emits indices (one per line) of GPUs with NO running compute processes.
    local -A busy_uuids=()
    local line idx uuid
    while IFS= read -r line; do
        line="${line// /}"
        [[ -n "$line" ]] && busy_uuids["$line"]=1
    done < <(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)

    while IFS=',' read -r idx uuid; do
        idx="${idx// /}"
        uuid="${uuid// /}"
        [[ -z "$idx" ]] && continue
        if [[ -z "${busy_uuids[$uuid]:-}" ]]; then
            echo "$idx"
        fi
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)
}

mapfile -t AVAILABLE_GPU_IDS < <(detect_idle_gpus)
n_avail=${#AVAILABLE_GPU_IDS[@]}

if (( n_avail == 0 )); then
    echo "[pipeline] ERROR: no idle GPUs detected via nvidia-smi" >&2
    exit 1
fi

if (( REQUESTED_NUM_GPUS <= 0 )); then
    NUM_GPUS=$n_avail
    echo "[pipeline] no GPU count requested; using ALL $NUM_GPUS idle GPU(s): ${AVAILABLE_GPU_IDS[*]}"
elif (( REQUESTED_NUM_GPUS > n_avail )); then
    NUM_GPUS=$n_avail
    echo "[pipeline] WARNING: requested $REQUESTED_NUM_GPUS GPU(s) but only $n_avail idle; falling back to $n_avail GPU(s): ${AVAILABLE_GPU_IDS[*]}"
else
    NUM_GPUS=$REQUESTED_NUM_GPUS
    AVAILABLE_GPU_IDS=("${AVAILABLE_GPU_IDS[@]:0:$NUM_GPUS}")
    echo "[pipeline] using $NUM_GPUS of $n_avail idle GPU(s): ${AVAILABLE_GPU_IDS[*]}"
fi

# ---- Abort helper: log failure, then let the EXIT trap kill subprocesses ---
fail_pipeline() {
    local reason="${1:-build failure}"
    echo "" >&2
    echo "[pipeline] ERROR: ${reason} - aborting" >&2
    if [[ -s "$BUILD_FAIL_FILE" ]]; then
        echo "[pipeline] failed shards:" >&2
        cat "$BUILD_FAIL_FILE" >&2
    fi
    exit 1
}

start_time=$(date +%s)

# ---- Launch merge process in background ------------------------------------
# Wrapped in a subshell that writes the real exit code to MERGE_RC_FILE, so we
# don't have to rely on `wait $MERGE_PID` later. A bare `wait` (or a stray
# `wait -n` in the build loop) would otherwise drop MERGE_PID from the job
# table and `wait $MERGE_PID` would spuriously return 127.
echo "[pipeline] launching merge process (logs -> ${MERGE_LOG})"
EXTRA_MERGE_ARGS=()
if [[ "$KEEP_SHARD_INDEX" != "0" ]]; then
    EXTRA_MERGE_ARGS+=(--keep_shard_index)
fi
if [[ "$KEEP_PARTITION_INPUTS" != "0" ]]; then
    EXTRA_MERGE_ARGS+=(--keep_partition_inputs)
fi

# Yield I/O to concurrent RAFT_CAGRA builds so Stage 1 doesn't starve them on
# the shared dataset disk. NOTE: ionice classes are only honored by BFQ and
# CFQ schedulers. On mq-deadline / none / kyber this is silently a no-op.
# To make it effective:
#   sudo bash -c 'echo bfq > /sys/block/<dev>/queue/scheduler'
# (replace <dev> with the disk holding $BASE_FOLDER, e.g. sda)
IONICE=()
if command -v ionice >/dev/null 2>&1; then
    IONICE=(ionice -c 3)   # 3 = idle: run only when no other I/O is queued
fi

(
    "${IONICE[@]}" "$MERGE_BIN" \
        --base_folder="$BASE_FOLDER" \
        --index_name="$INDEX_NAME" \
        --nshards="$N" \
        --merge_degree="$MERGE_DEG" \
        --build_degree="$BUILD_DEG" \
        --max_merge_workers="$MAX_MERGE_WORKERS" \
        --sort_buf_mb="$SORT_BUF_MB" \
        --poll_interval_ms="$POLL_INTERVAL_MS" \
        --tmp_dir="$TMP_DIR" \
        --output_dir="$OUTPUT_DIR" \
        --stage2_memory_budget_mb=$((BUILD_MEMORY_GB * 1024)) \
        --stage2_max_threads="$STAGE2_MAX_THREADS" \
        --stage2_per_run_buf_mb="$STAGE2_PER_RUN_BUF_MB" \
        --stage2_output_buf_mb="$STAGE2_OUTPUT_BUF_MB" \
        ${EXTRA_MERGE_ARGS[@]+"${EXTRA_MERGE_ARGS[@]}"} \
        > "$MERGE_LOG" 2>&1
    echo $? > "$MERGE_RC_FILE"
) &
MERGE_PID=$!

# ---- Launch builds with GPU-availability scheduling ------------------------
# `slot` indexes GPU_PIDS[]; the actual CUDA device id is AVAILABLE_GPU_IDS[slot].
i=0
while (( i < N )); do
    [[ -e "$BUILD_FAIL_FILE" ]] && fail_pipeline "build failure detected during dispatch"

    isAvailSlot=-1
    for ((slot=0; slot<NUM_GPUS; slot++)); do
        pid=${GPU_PIDS[$slot]:-}
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            isAvailSlot=$slot
        fi
    done
    if (( isAvailSlot == -1)); then
        # Wait only on build pids, never on MERGE_PID, so a fast merge crash
        # doesn't masquerade as a freed build slot.
        active_pids=()
        for ((s=0; s<NUM_GPUS; s++)); do
            sp=${GPU_PIDS[$s]:-}
            [[ -n "$sp" ]] && active_pids+=("$sp")
        done
        if (( ${#active_pids[@]} > 0 )); then
            wait -n "${active_pids[@]}" 2>/dev/null || true
        fi
        [[ -e "$BUILD_FAIL_FILE" ]] && fail_pipeline "build failure detected during dispatch"
    fi

    for ((slot=0; slot<NUM_GPUS; slot++)); do
        pid=${GPU_PIDS[$slot]:-}
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            shard_gpu=${AVAILABLE_GPU_IDS[$slot]}
            TASK_FILE="${BASE_FOLDER}/partition$i/${Dataset}.json"
            OUTPUT_FILE="${BASE_FOLDER}/partition$i/${Dataset}.json.lock"
            DONE_FILE="${BASE_FOLDER}/partition$i/build.done"

            echo "[build] GPU $shard_gpu (slot $slot) available, launching shard $i"

            (
                shard_id=$i
                task_start_time=$(date +%s)
                CUDA_VISIBLE_DEVICES=$shard_gpu "$RAFT_CAGRA" \
                    --build --force \
                    --data_prefix="$DatasetPath" \
                    --benchmark_out_format=json \
                    --benchmark_counters_tabular=true \
                    --benchmark_out="$OUTPUT_FILE" \
                    --raft_log_level=3 \
                    "$TASK_FILE"
                rc=$?
                task_end_time=$(date +%s)
                task_duration=$((task_end_time - task_start_time))
                if (( rc == 0 )); then
                    # Signal completion to merge process. No fsync/sync needed:
                    # Stage 1 reads via page cache (same kernel), so the latest
                    # bytes are immediately visible without hitting disk. Skipping
                    # the global sync avoids waiting on Stage 1's dirty pages and
                    # on other parallel builds' dirty pages, which would otherwise
                    # hold the GPU slot and delay the next shard's launch.
                    touch "$DONE_FILE"
                    echo "[build] shard ${shard_id} (GPU ${shard_gpu}) done in ${task_duration} s"
                    printf "%d\t%d\t%d\n" "$shard_id" "$shard_gpu" "$task_duration" >> "$BUILD_LOG"
                else
                    echo "[build] shard ${shard_id} FAILED (rc=${rc})" >&2
                    printf "shard=%d gpu=%d rc=%d\n" "$shard_id" "$shard_gpu" "$rc" >> "$BUILD_FAIL_FILE"
                    exit "$rc"
                fi
            ) &

            GPU_PIDS[$slot]=$!
            echo "[build] shard $i running on GPU $shard_gpu (slot $slot, pid ${GPU_PIDS[$slot]})"
            ((i++))
            break
        fi
    done
done

# ---- Wait for the remaining builds (one per active slot) -------------------
# Use `wait -n pid...` so MERGE_PID is never reaped here, and so a failure on
# any slot is detected without first waiting for every other slot to finish.
remaining_pids=()
for ((slot=0; slot<NUM_GPUS; slot++)); do
    pid=${GPU_PIDS[$slot]:-}
    [[ -n "$pid" ]] && remaining_pids+=("$pid")
done
while (( ${#remaining_pids[@]} > 0 )); do
    wait -n "${remaining_pids[@]}" 2>/dev/null || true
    [[ -e "$BUILD_FAIL_FILE" ]] && fail_pipeline "build failure detected during drain"
    next=()
    for p in "${remaining_pids[@]}"; do
        kill -0 "$p" 2>/dev/null && next+=("$p")
    done
    remaining_pids=("${next[@]:-}")
    [[ -z "${remaining_pids[0]:-}" ]] && remaining_pids=()
done

build_end_time=$(date +%s)
build_elapsed=$((build_end_time - start_time))
echo "[pipeline] all builds finished in ${build_elapsed} s; waiting for merge..."

# ---- Wait for merge process ------------------------------------------------
wait "$MERGE_PID" 2>/dev/null || true
if [[ -s "$MERGE_RC_FILE" ]]; then
    merge_rc=$(<"$MERGE_RC_FILE")
else
    echo "[pipeline] WARNING: merge rc file missing or empty; treating merge as failed" >&2
    merge_rc=1
fi

end_time=$(date +%s)
total_elapsed=$((end_time - start_time))
extra_elapsed=$((end_time - build_end_time))

echo ""
echo "=== Pipeline Summary (shell-side wall-clock) ==="
echo "build wall-clock         : ${build_elapsed} s"
echo "total wall-clock         : ${total_elapsed} s"
echo "extra (= total - build)  : ${extra_elapsed} s"
echo "merge process exit code  : ${merge_rc}"
echo "(See ${MERGE_LOG} for the merge-internal timing breakdown)"

{
    echo "build_wall_clock_s=${build_elapsed}"
    echo "total_wall_clock_s=${total_elapsed}"
    echo "extra_wall_clock_s=${extra_elapsed}"
    echo "merge_rc=${merge_rc}"
} > "$PIPELINE_LOG"

exit "$merge_rc"
