#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <variant>
#include <cassert>
#include <omp.h>
#include <mkl.h>
#include <boost/program_options.hpp>

#include "../src/merge/merge.hpp"
#include "../src/utils/fileUtils.h"
#include "../src/taskScheduler/TaskBroker.hpp"
#include "../src/taskScheduler/gpuManagement.h"
#include "../../DiskANN/include/program_options_utils.hpp"
#include "../src/utils/Logger.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;
using scalegann::Logger;

struct MergeContext {
    std::string indexName;
    std::string baseFolder;
    std::string datasetPath;
    uint32_t dataset_size;
    uint32_t ndim;
    uint32_t merge_deg;
    bool isGPU;
    omp_lock_t* locks;
};

void setupMergeResources(MergeContext& ctx, std::vector<std::vector<uint32_t>>& merged_index) {
    uint32_t header[2];
    readMetadata(ctx.datasetPath, header);
    ctx.dataset_size = header[0];
    ctx.ndim = header[1];
    Logger::info("Dataset size: {}, Dimension: {}", ctx.dataset_size, ctx.ndim);
    
    ctx.locks = new omp_lock_t[ctx.dataset_size];
    for (uint32_t i = 0; i < ctx.dataset_size; i++) omp_init_lock(&ctx.locks[i]);
    
    merged_index.clear();
    merged_index.resize(ctx.dataset_size);
    Logger::info("Generated locks and allocated merged index.");
}

void populateBroker(TaskBroker& broker, const MergeContext& ctx, uint32_t folderNum) {
    uint32_t n_shard = ctx.dataset_size / folderNum;
    for (uint32_t i = 0; i < folderNum; i++) {
        broker.addTask(i, n_shard, ctx.ndim, ctx.merge_deg);
    }
    
    if (ctx.isGPU) {
        for (int g = 0; g < GPU_NUM; g++) broker.addGPU(g);
    } else {
        for (int t = 0; t < omp_get_max_threads(); t++) broker.addGPU(t);
    }
}

void processShard(uint32_t i, MergeContext& ctx, std::vector<std::vector<uint32_t>>& merged_index, int gpu_id) {
    fs::path subFolder = "partition" + std::to_string(i);
    fs::path IndexPath = fs::path(ctx.baseFolder) / subFolder / "index" / ctx.indexName;
    
    if (fs::exists(IndexPath) && fs::is_regular_file(IndexPath)) {
        Logger::info("Found matching file: {}", IndexPath.string());

        std::vector<std::vector<uint32_t>> index;
        readIndex(IndexPath.string(), index);

        std::string idx_file = ctx.baseFolder + "/partition" + std::to_string(i) + "/idmap.ibin";
        uint32_t header[1];
        readMetadataOneDimension(idx_file, header);
        std::vector<uint32_t> idx_vec(header[0]);
        readFileOneDimension<uint32_t>(idx_file, idx_vec);

        Logger::task(i, "Read index and idx map.");

        if (ctx.isGPU) {
            Logger::task(i, "Merging using GPU {}", gpu_id);
            mergeShardAfterTranslationGPU(ctx.locks, merged_index, index, idx_vec, gpu_id);
        } else {
            Logger::task(i, "Merging using CPU thread {}", omp_get_thread_num());
            mergeShardAfterTranslation(ctx.locks, merged_index, index, idx_vec);
        }
        Logger::success("Merged shard {}.", i);
    } else {
        Logger::error("File not found: {}", IndexPath.string());
    }
}

void runMergeLoop(TaskBroker& broker, MergeContext& ctx, std::vector<std::vector<uint32_t>>& merged_index) {
    initGpuLocks();
    initEvents();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int gpu_id = tid % GPU_NUM;
        while (!broker.isAllDone()) {
            size_t cap = ctx.isGPU ? 16000000000ULL : 100000000000ULL;
            auto block = broker.Scheduler(tid, cap);
            for (uint32_t i : block) {
                processShard(i, ctx, merged_index, gpu_id);
                broker.updateStatus(i, ShardTask::COMPLETED);
            }
        }
    }

    destroyEvents();
    destroyGpuLocks();
}

void mergeIndex(const std::string indexName, std::string baseFolder, 
                const std::string& datasetPath, uint32_t folderNum, uint32_t merge_deg,
                std::vector<std::vector<uint32_t>>& merged_index, bool isGPU = 0) {
    
    MergeContext ctx = {indexName, baseFolder, datasetPath, 0, 0, merge_deg, isGPU, nullptr};
    setupMergeResources(ctx, merged_index);

    TaskBroker broker;
    populateBroker(broker, ctx, folderNum);

    auto startTime = std::chrono::high_resolution_clock::now();
    runMergeLoop(broker, ctx, merged_index);

    Logger::info("Starting final assignment and standardization...");
    auto assignTime = std::chrono::high_resolution_clock::now();
    auto assignDuration = std::chrono::duration_cast<std::chrono::milliseconds>(assignTime - startTime);

    standardizeNeighborList(merged_index, merge_deg);
    
    auto standardizeTime = std::chrono::high_resolution_clock::now();
    auto standardizeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(standardizeTime - assignTime);
    Logger::info("Assign time: {}ms, Standardize time: {}ms", assignDuration.count(), standardizeDuration.count());

    for (uint32_t i = 0; i < ctx.dataset_size; i++) omp_destroy_lock(&ctx.locks[i]);
    delete[] ctx.locks;
}

void mergeAllIndexInFolder(std::string baseFolder, std::string datasetPath, uint32_t merge_deg) {
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<fs::path> subfolders;
    for (const auto& entry : fs::directory_iterator(baseFolder)) {
        if (entry.is_directory() && entry.path().filename().string().find("partition") == 0) {
            subfolders.push_back(entry.path());
        }
    }

    if (subfolders.empty()) {
        Logger::error("No partitions found in {}", baseFolder);
        return;
    }

    fs::path firstSubfolder = subfolders[0];
    fs::path firstIndexFolder = firstSubfolder / "index";
    uint32_t folderNum = subfolders.size();

    auto prepareTime = std::chrono::high_resolution_clock::now();
    auto prepareDuration = std::chrono::duration_cast<std::chrono::milliseconds>(prepareTime - startTime);
    Logger::info("Prepare duration: {}ms", prepareDuration.count());

    auto lastIndexWrite = prepareTime;
    long long totalIndexMerge = 0;
    long long totalIndexWrite = 0;
    uint32_t iter_count = 0;

    for (const auto& file : fs::directory_iterator(firstIndexFolder)) {
        if (file.is_regular_file()) {
            std::string indexName = file.path().filename().string();
            std::vector<std::vector<uint32_t>> mergedIndex;
            
            Logger::info("Processing index: {}", indexName);
            mergeIndex(indexName, baseFolder, datasetPath, folderNum, merge_deg, mergedIndex, false);
            
            Logger::info("Index dimension: {} x {}", mergedIndex.size(), mergedIndex.empty() ? 0 : mergedIndex[0].size());
            auto indexMergeTime = std::chrono::high_resolution_clock::now();
            auto indexMergeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(indexMergeTime - lastIndexWrite);
            Logger::info("Index {} merge duration: {}ms", iter_count, indexMergeDuration.count());

            const std::string index_file = baseFolder + "/mergedIndex/" + indexName;
            fs::create_directories(baseFolder + "/mergedIndex");
            writeIndexMerged(index_file, mergedIndex);

            auto indexWriteTime = std::chrono::high_resolution_clock::now();
            auto indexWriteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(indexWriteTime - indexMergeTime);
            Logger::info("Index {} write duration: {}ms", iter_count, indexWriteDuration.count());
            lastIndexWrite = indexWriteTime;

            long long totalTimeOfThisIter = indexMergeDuration.count() + indexWriteDuration.count();
            Logger::info("Index {} total duration: {}ms", iter_count, totalTimeOfThisIter);

            totalIndexMerge += indexMergeDuration.count();
            totalIndexWrite += indexWriteDuration.count(); 
            iter_count++;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    Logger::success("Total merge time: {}ms, Total write time: {}ms", totalIndexMerge, totalIndexWrite);
    Logger::success("Overall duration: {}ms", overallDuration.count());
}

int main(int argc, char **argv) {
    std::string data_path, base_folder;
    uint32_t merge_deg, num_threads;

    po::options_description desc{program_options_utils::make_program_description("scaleGANN_merge", "Merge shard indices.")};
    desc.add_options()("help,h", "Print help");
    desc.add_options()("data_path", po::value<std::string>(&data_path)->required(), "Dataset path");
    desc.add_options()("base_folder", po::value<std::string>(&base_folder)->required(), "Shards folder");
    desc.add_options()("merge_degree,R", po::value<uint32_t>(&merge_deg)->required(), "Merge degree");
    desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()), "Threads");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) { std::cout << desc; return 0; }
    po::notify(vm);

    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);
    Logger::info("Using {} threads. Merge degree: {}", num_threads, merge_deg);

    mergeAllIndexInFolder(base_folder, data_path, merge_deg);
    return 0;
}
