#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <variant>
#include <cassert>
#include <omp.h>
#include <boost/program_options.hpp>
#include <mkl.h>

#include "../../src/partition/partition.h"
#include "../../src/partition/disk_partition.h"
#include "../../src/utils/fileUtils.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;

// 15 → 10, matching DiskANN's own partition_with_ram_budget
// ([DiskANN/src/partition.cpp:529]).
#define MAX_ITERATION 10
#define SAMPLING_RATE 0.05


void partition(const std::string data_path, std::string baseFolder,
                uint32_t memory_budget, uint32_t build_deg, uint32_t max_duplication, 
                uint32_t max_iteration, double sampling_rate){
    auto startTime = std::chrono::high_resolution_clock::now();

    uint32_t suffixType = suffixToType(data_path);
    if(suffixType == 0){ // float
        diskANN_partitions_with_ram_budget<float>(data_path, sampling_rate, (double)memory_budget,
            (size_t)build_deg, baseFolder.c_str(), (size_t)max_duplication);
    } else if (suffixType == 2) { // uint8_t
        diskANN_partitions_with_ram_budget<uint8_t>(data_path, sampling_rate, (double)memory_budget,
            (size_t)build_deg, baseFolder.c_str(), (size_t)max_duplication);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    printf("DiskANN disk partition duration: %lld milliseconds\n", overallDuration.count());
}


int main(int argc, char **argv) {
    std::string data_path, base_folder;
    uint32_t memory_budget, build_deg, max_duplication, minimum_partition_num, max_iteration;
    double sampling_rate;
    uint32_t num_threads;

    po::options_description desc{
        program_options_utils::make_program_description("partition_disk_dataset", "Partition a vector dataset from disk to get data shards.")};
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
        required_configs.add_options()("max_duplication,W", po::value<uint32_t>(&max_duplication)->required(),
                                       "Max duplication time of each vector.");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("sampling_rate",
                                        po::value<double>(&sampling_rate)->default_value(SAMPLING_RATE),
                                        "Suggested dataset sampling rate for kmeans training.");
        optional_configs.add_options()("max_iteration",
                                        po::value<uint32_t>(&max_iteration)->default_value(MAX_ITERATION),
                                        "Maximum iteration for kmeans training. Default is 15.");

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
    printf("Build degree is %d\n", build_deg);

    // Pre-processing: drop any stale partition output from a previous run BEFORE
    // starting the timer. See apps/executeDiskPartition.cpp for the full rationale.
    cleanup_partition_dir(base_folder);

    partition(data_path, base_folder, memory_budget, build_deg, max_duplication, max_iteration, sampling_rate);
    return 0;
}

