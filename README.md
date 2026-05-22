ScaleGANN is a highly efficient, scalable, and cost-effective system for ANNS indexing and querying. It is extendable to cloud-native resources or serverless usages. This Readme file gives instruction on how to build and test ScaleGANN.

# Build
## Vesions
In this work, we are using CUDA 12.5, python 3.11, cmake 3.28.
## Build Benchmarks
1. Build [CAGRA](https://github.com/rapidsai/raft/blob/branch-24.10/docs/source/build.md)
2. Build [DiskANN](https://github.com/microsoft/DiskANN/blob/main/README.md)
3. Build [GGNN](https://github.com/cgtuebingen/ggnn/tree/release_0.5)
## Build ScaleGANN
With all packages required for CAGRA and DiskANN, ScaleGANN is able to be built succesfully.
```
git clone --recurse-submodules https://github.com/loulankxh/scaleGANN.git
mkdir -p build && cd build
cmake .. && make -j
```

# Run 
## Run ScaleGANN
1. Compile the ScaleGANN to get the build folder and files
2. Run index construction (the time of each step will show in the last line of terminal output)
   a. Partition: build/executeDiskPartition
     ```
     E.g., ./executeDiskPartition --data_path “path_to_your_data” --base_folder “path_to_store_data_partitions” -M 16 -R 64 -L 128 -W 2 -E 1.2 -T 80
     ```
   b. Build:
      1. Run [src/indexBuild/generateConfig.sh](https://github.com/loulankxh/scaleGANN/blob/main/src/indexBuild/generateConfig.sh) to generate build configuration
         Line 1-8: change to your local setting
         Line 10-11: change to your build degree, e.g., use build_deg=64, build_intermediate_deg=128
      2. Run [src/indexBuild/SequentialAssign.sh](https://github.com/loulankxh/scaleGANN/blob/main/src/indexBuild/SequentialAssign.sh) for building using one GPU, or [src/indexBuild/AssignByAvailGPU.sh](https://github.com/loulankxh/scaleGANN/blob/main/src/indexBuild/AssignByAvailGPU.sh) for multi-GPU parallelism
         Change the “RAFT_CAGRA“ path to your path_to_cagra
         Change Line 3-11 to your local settings
   c. Merge: build/executeDiskMerge
   ```
    E.g., ./executeDiskMerge --base_folder “path_where_data_partitions_are_stored” --index_name “name_of_your_shard_index” -R 64 -B 64 -T 80
   ```
3. Index Search: build/searchScaleGANN

## Run DiskANN
DiskANN benchmark testing is implemented by function calls of its original repositary
1. Compile the ScaleGANN repo to get the build folder and files
2. Run index construction (the time of each step will show in the last line of terminal output) 
   a. Partition: build/diskannPartition 
   ```
   E.g., ./diskannPartition --data_path “path_to_your_data” --base_folder “path_to_store_data_partitions” –M 16 -R 64 -W 2 -T 80
   ```
   b. Build: build/diskannBuild 
   ```
   E.g., ./diskannBuild --data_type uint8 --dist_fn l2 --base_folder “path_where_data_partitions_are_stored” -N “enter_the_partition_num_you_got_in_previous_step” -R 64 -L 128 -T 80
   ```
   c. Merge: build/diskannMerge 
   ```
   E.g., ./diskannMerge --base_folder “path_where_data_partitions_are_stored” -R 64 -T 80
   ```
3. Index Search: build/searchDiskANN

## Run GGNN and Naive CAGRA
1. Compile the ScaleGANN repo to get the build folder and files
2. Run index construction (the time of each step will show in the last line of terminal output) 
   a. Partition (if necessary): build/executeDiskDirectPartition 
   b. Build: Run apps/buildGGNN.sh for GGNN build. For Naive CAGRA build, take ScaleGANN build for reference.
3. Index Search: build/searchGGNN, or build/searchCAGRA

# Experiment Results
In the paper, we compared ScaleGANN with three other benchmarks: [DiskANN](https://github.com/microsoft/DiskANN/tree/main), [GGNN](https://github.com/cgtuebingen/ggnn/tree/release_0.5), and Naive CAGRA which we implemented based on [CAGRA](https://github.com/rapidsai/raft/tree/branch-24.10).
## Build experiments
We built indexes using all the above four approaches on various datasets: Sift, Deep, SimSearchNet, Microsoft Turing, Laion. Most of the datasets are from [Bigann Benchamrks](https://big-ann-benchmarks.com/neurips21.html). While [Laion](https://laion.ai) is download at [Laion Download](https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion1B-nolang/).
Build data with details are in folder [experiments]().
## Search experiments
We record full search results including recall, time and query latency in [search results](https://docs.google.com/spreadsheets/d/1_rdrr2zPHzPDIhlvdY1N7BzFtH-y-3tpIPgv03B6Tw8/edit?usp=sharing). 
For more details, also see in folder [experiments]().
