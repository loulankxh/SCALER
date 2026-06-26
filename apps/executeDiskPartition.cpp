#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <boost/program_options.hpp>
#include <mkl.h>

#include "../src/partition/partition.h"
#include "../src/partition/disk_partition.h"
#include "../src/utils/fileUtils.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;

// [SAMPLE-FIX, scheme B] 15 → 10, matching DiskANN's partition_with_ram_budget
//   ([DiskANN/src/partition.cpp:529]). DiskANN's PQ training still uses 15
//   because it needs precise centroid placement; partition kmeans only needs
//   reasonable cluster boundaries, for which 10 iters + 1e-5 early-stop is
//   sufficient.
#define MAX_ITERATION 10
#define SAMPLING_RATE 0.05


void partitionDisk_kmeans(const std::string file_path, std::string baseFolder,
                        uint32_t memory_budget, uint32_t build_deg, uint32_t inter_build_deg, uint32_t gpu_threads,
                        uint32_t max_duplication, double epsilon,
                        uint32_t minimum_partition_required, uint32_t max_iteration, double sampling_rate){
    auto startTime = std::chrono::high_resolution_clock::now();

    uint32_t suffixType = suffixToType(file_path);
    if(suffixType == 0){ // float
        scaleGANN_partitions_with_ram_budget<float>(file_path, sampling_rate, memory_budget,
            build_deg, inter_build_deg, gpu_threads, baseFolder, max_duplication, minimum_partition_required, epsilon, max_iteration);
    } else if (suffixType == 2) { // uint8_t
        scaleGANN_partitions_with_ram_budget<uint8_t>(file_path, sampling_rate, memory_budget,
            build_deg, inter_build_deg, gpu_threads, baseFolder, max_duplication, minimum_partition_required, epsilon, max_iteration);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    printf("Prune factor epsilon is: %f\n", epsilon);
    printf("Disk partition duration: %lld milliseconds\n", overallDuration.count());
}


int main(int argc, char **argv) {
    std::string data_path, base_folder;
    uint32_t memory_budget, build_deg, inter_build_deg, max_duplication, minimum_partition_num, max_iteration;
    double epsilon, sampling_rate;
    uint32_t num_threads, gpu_threads;

    po::options_description desc{
        program_options_utils::make_program_description("scaleGANN_partition_disk_dataset", "Partition a vector dataset from disk to get data shards.")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       "File path of the dataset.");
        required_configs.add_options()("base_folder", po::value<std::string>(&base_folder)->required(),
                                       "Folder path where all the partioned data shards are stored.");
        required_configs.add_options()("memory_budget,M", po::value<uint32_t>(&memory_budget)->required(),
                                       "Memory budget for build.");
        required_configs.add_options()("build_degree,R", po::value<uint32_t>(&build_deg)->required(),
                                       "Expected build degree of each shard index.");
        required_configs.add_options()("inter_build_degree,L", po::value<uint32_t>(&inter_build_deg)->required(),
                                       "Maximum intermediate build degree of each shard index.");
        required_configs.add_options()("max_duplication,W", po::value<uint32_t>(&max_duplication)->required(),
                                       "Max duplication time of each vector.");
        required_configs.add_options()("epsilon,E", po::value<double>(&epsilon)->required(),
                                       "Selective duplication factor epsilon.");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        "Number of threads used.");
        optional_configs.add_options()("GPU_threads",
                                        po::value<uint32_t>(&gpu_threads)->default_value(5120),
                                        "Default GPU V100 has 5120 threads.");
        optional_configs.add_options()("sampling_rate",
                                        po::value<double>(&sampling_rate)->default_value(SAMPLING_RATE),
                                        "Suggested dataset sampling rate for kmeans training.");
        optional_configs.add_options()("max_iteration",
                                        po::value<uint32_t>(&max_iteration)->default_value(MAX_ITERATION),
                                        "Maximum iteration for kmeans training. Default is 15.");
        optional_configs.add_options()("minimum_partition_num,N", po::value<uint32_t>(&minimum_partition_num)->default_value(1),
                                        "Minimum number of partitioned data shards required by users.");

        desc.add(required_configs).add(optional_configs);


        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);
    printf("Using %d threads\n", num_threads);
    printf("Build using %d GPU threads\n", gpu_threads);
    printf("Expected build degree is %d\n", build_deg);
    printf("Maximum inter-build degree is %d\n", inter_build_deg);

    // Pre-processing: drop any stale partition output from a previous run BEFORE
    // starting the timer. Without this, opening multi-GB files with ios::trunc
    // (inside the timed region) causes synchronous block-release + journal
    // commits, AND old cached pages compete with new I/O for OS page cache.
    // This call uses unlink (async block release), so its cost stays small and
    // does not pollute the partition duration we report.
    cleanup_partition_dir(base_folder);

    partitionDisk_kmeans(data_path, base_folder,
                        memory_budget, build_deg, inter_build_deg, gpu_threads, max_duplication, epsilon,
                        minimum_partition_num, max_iteration, sampling_rate);
    return 0;
}