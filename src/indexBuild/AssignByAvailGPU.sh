#!/bin/bash

declare -A GPU_PIDS
NUM_GPUS=4
DatasetPath="/home/lanlu/scaleGANN/dataset/sift100M"
DATASET="sift100M"
SHARD_FOLDER="/ScaleGANN"
N=10

RAFT_CAGRA="/home/lanlu/miniconda3/envs/rapids_raft/bin/ann/RAFT_CAGRA_ANN_BENCH"
LOG_FILE=${DatasetPath}/${SHARD_FOLDER}/time.txt

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
echo -e "shard_id\tgpu\tbuild_time_s" >> "$LOG_FILE"

start_time=$(date +%s)

i=0
while (( i < N )); do
    isAvailGPU=-1
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        pid=${GPU_PIDS[$gpu]}
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            isAvailGPU=$gpu
        fi
    done
    if (( isAvailGPU == -1)); then
        wait -n
    fi

    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        pid=${GPU_PIDS[$gpu]}
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
            TASK_FILE="${DatasetPath}/${SHARD_FOLDER}/partition$i/${DATASET}.json"
            OUTPUT_FILE="${DatasetPath}/${SHARD_FOLDER}/partition$i/${DATASET}.json.lock"

            echo "GPU $gpu is available. Launching task $i ..."

            (
                shard_id=$i
                shard_gpu=$gpu
                task_start_time=$(date +%s)
                CUDA_VISIBLE_DEVICES=$shard_gpu $RAFT_CAGRA \
                    --build --force \
                    --data_prefix=$DatasetPath \
                    --benchmark_out_format=json \
                    --benchmark_counters_tabular=true \
                    --benchmark_out="$OUTPUT_FILE" \
                    --raft_log_level=3 \
                    "$TASK_FILE"
                task_end_time=$(date +%s)
                task_duration=$((task_end_time - task_start_time))
                echo "Task $shard_id (GPU $shard_gpu) finished in ${task_duration} seconds" | tee -a "$LOG_FILE"
                printf "%d\t%d\t%d\n" "$shard_id" "$shard_gpu" "$task_duration" >> "$LOG_FILE"
            ) &

            GPU_PIDS[$gpu]=$!
            echo "Task $i running on GPU $gpu with PID ${GPU_PIDS[$gpu]}"
            ((i++))
            break
        fi
    done
done

wait

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "All tasks are done! Total execution time: ${elapsed_time} s." | tee -a "$LOG_FILE"