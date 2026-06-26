#!/bin/bash

GPU_ID=1
DatasetPath="/home/lanlu/scaleGANN/dataset/sift1B"
DATASET="sift1B"
SHARD_FOLDER="/NaiveCAGRA"
N=100


RAFT_CAGRA="/home/lanlu/miniconda3/envs/rapids_raft/bin/ann/RAFT_CAGRA_ANN_BENCH"
LOG_FILE=${DatasetPath}/${SHARD_FOLDER}/time.txt

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"
echo -e "shard_id\tgpu\tbuild_time_s" >> "$LOG_FILE"

start_time=$(date +%s)

for ((i=0; i<N; i++)); do
    TASK_FILE="${DatasetPath}/${SHARD_FOLDER}/partition$i/${DATASET}.json"
    OUTPUT_FILE="${DatasetPath}/${SHARD_FOLDER}/partition$i/${DATASET}.json.lock"

    echo "Starting task $i on GPU $GPU_ID ..."
    task_start_time=$(date +%s)

    CUDA_VISIBLE_DEVICES=$GPU_ID $RAFT_CAGRA \
        --build --force \
        --data_prefix=$DatasetPath \
        --benchmark_out_format=json \
        --benchmark_counters_tabular=true \
        --benchmark_out="$OUTPUT_FILE" \
        --raft_log_level=3 \
        "$TASK_FILE"


    task_end_time=$(date +%s)
    task_duration=$((task_end_time - task_start_time))

    echo "Task $i (GPU $GPU_ID) finished in ${task_duration} seconds" | tee -a "$LOG_FILE"
    printf "%d\t%d\t%d\n" "$i" "$GPU_ID" "$task_duration" >> "$LOG_FILE"
done

wait

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "All tasks are done! Total execution time: ${elapsed_time} s." | tee -a "$LOG_FILE"
