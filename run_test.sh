#!/bin/bash
set -e

# Configuration
CONDA_ENV="scalegann_env"
DATASET_PATH="$(pwd)/dataset/test_data/base.u8bin"
QUERY_PATH="$(pwd)/dataset/test_data/query.u8bin"
GT_PATH="$(pwd)/dataset/test_data/groundtruth.ibin"
BASE_FOLDER="$(pwd)/dataset/test_shards"
INDEX_NAME="raft_cagra"

echo "=== 1. Building ScaleGANN ==="
mkdir -p build
cd build
conda run -n $CONDA_ENV cmake ..
conda run -n $CONDA_ENV make -j executeDiskPartition executeDiskMerge searchScaleGANN
cd ..

echo "=== 2. Partitioning Dataset ==="
rm -rf $BASE_FOLDER
mkdir -p $BASE_FOLDER
# Running partition
build/executeDiskPartition --data_path $DATASET_PATH --base_folder $BASE_FOLDER -M 1 -R 32 -L 64 -W 2 -E 1.2 -T 8

echo "=== 3. Generating Fake Shard Indexes (CPU Fallback) ==="
python generate_fake_shards.py

echo "=== 4. Merging Shard Indexes ==="
# Using a smaller merge degree (16) than constructed degree (32) to ensure the assertion passes
build/executeDiskMerge --base_folder $BASE_FOLDER --index_name "$INDEX_NAME" -R 16 -B 32 -T 8

echo "=== 5. Searching ScaleGANN Index ==="
build/searchScaleGANN --data_file $DATASET_PATH --index_file $BASE_FOLDER/mergedIndex/$INDEX_NAME --query_file $QUERY_PATH --truth_file $GT_PATH -K 10 -L 32
