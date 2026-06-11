#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <boost/program_options.hpp>
#include <mkl.h>

#include "../src/partition/disk_partition.h"
#include "../src/utils/fileUtils.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;

void partitionDisk_direct(const std::string file_path, std::string base_folder,
                        uint32_t shard_size, uint32_t dataset_size){
    auto startTime = std::chrono::high_resolution_clock::now();

    uint32_t suffixType = suffixToType(file_path);
    if(suffixType == 0){ // float
        direct_partitions<float>(file_path, base_folder, shard_size, dataset_size);
    } else if (suffixType == 2) { // uint8_t
        direct_partitions<uint8_t>(file_path, base_folder, shard_size, dataset_size);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("Disk direct partition duration: %lld milliseconds\n", overallDuration.count());
}


int main(int argc, char **argv) {
    std::string data_path, base_folder;
    uint32_t shard_size, dataset_size;
    uint32_t num_threads;

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
        required_configs.add_options()("dataset_size,S", po::value<uint32_t>(&dataset_size)->required(),
                                       "Number of vectors in the dataset to be partitioned.");


        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                        "Number of threads used.");
        optional_configs.add_options()("shard_size,C",
                                        po::value<uint32_t>(&shard_size)->default_value(1000000),
                                        "Number of points in a shard. Default is 1M.");

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

    // Pre-processing: drop any stale partition output from a previous run BEFORE
    // starting the timer. See executeDiskPartition.cpp for the full rationale.
    cleanup_partition_dir(base_folder);

    partitionDisk_direct(data_path, base_folder, shard_size, dataset_size);
    return 0;
}

