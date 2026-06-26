#include <string>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cstdint>
#include <omp.h>
#include <boost/program_options.hpp>
#include <mkl.h>

#include "../src/merge/disk_merge.h"
#include "../../DiskANN/include/program_options_utils.hpp"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    std::string base_folder, index_name, output_dir, output_index_file, tmp_dir;
    uint32_t merge_deg = 0, build_deg = 0, num_threads = 0;
    uint32_t nshards = 0, max_merge_workers = 2;
    uint32_t sort_buf_mb = 512;
    uint32_t poll_interval_ms = 500;
    bool keep_runs = false;
    bool keep_shard_index = false;
    bool keep_partition_inputs = false;
    uint32_t stage2_memory_budget_mb = 16384;   // 16 GB default
    uint32_t stage2_max_threads = 0;            // 0 = hardware_concurrency
    uint32_t stage2_per_run_buf_mb = 8;
    uint32_t stage2_output_buf_mb = 32;

    po::options_description desc{
        program_options_utils::make_program_description(
            "scaleGANN_pipelined_merge_disk_index",
            "Pipelined merge: overlaps with per-shard build via build.done markers, "
            "memory-bounded external-sort + k-way merge.")};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        po::options_description required_configs("Required");
        required_configs.add_options()
            ("base_folder", po::value<std::string>(&base_folder)->required(),
             "Folder path where all the partitioned data shards are stored.")
            ("index_name", po::value<std::string>(&index_name)->required(),
             "Name of specific index inside each partition's index/ folder.")
            ("nshards", po::value<uint32_t>(&nshards)->required(),
             "Number of shards (partitions) to wait for and merge.")
            ("merge_degree,R", po::value<uint32_t>(&merge_deg)->required(),
             "Expected degree of the merged index.")
            ("build_degree,B", po::value<uint32_t>(&build_deg)->required(),
             "Build degree of the each shard index.");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()
            ("num_threads,T",
             po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
             "Number of threads used (omp/mkl).")
            ("max_merge_workers",
             po::value<uint32_t>(&max_merge_workers)->default_value(2),
             "Max concurrent Stage 1 workers.")
            ("sort_buf_mb",
             po::value<uint32_t>(&sort_buf_mb)->default_value(512),
             "Per-worker sort buffer in MB (bigger => fewer runs).")
            ("poll_interval_ms",
             po::value<uint32_t>(&poll_interval_ms)->default_value(500),
             "Filesystem poll interval for build.done markers.")
            ("tmp_dir",
             po::value<std::string>(&tmp_dir)->default_value(""),
             "Directory for intermediate run files. Defaults to <base_folder>/mergedIndex/tmp_runs.")
            ("output_dir",
             po::value<std::string>(&output_dir)->default_value(""),
             "Output directory. Defaults to <base_folder>/mergedIndex.")
            ("keep_runs",
             po::bool_switch(&keep_runs)->default_value(false),
             "Keep intermediate run files after merge (default: delete).")
            ("keep_shard_index",
             po::bool_switch(&keep_shard_index)->default_value(false),
             "Keep per-shard index files after Stage 1 consumes them "
             "(default: delete to reclaim disk).")
            ("keep_partition_inputs",
             po::bool_switch(&keep_partition_inputs)->default_value(false),
             "Keep per-shard partition inputs (data.<suffix>, idmap.ibin) "
             "after Stage 1 consumes them (default: delete to reclaim disk).")
            ("stage2_memory_budget_mb",
             po::value<uint32_t>(&stage2_memory_budget_mb)->default_value(16384),
             "Total RAM budget for Stage 2 in MB (build memory ceiling). "
             "Caps parallel worker count.")
            ("stage2_max_threads",
             po::value<uint32_t>(&stage2_max_threads)->default_value(0),
             "CPU-side cap on Stage 2 parallel workers. 0 = auto "
             "(hardware_concurrency).")
            ("stage2_per_run_buf_mb",
             po::value<uint32_t>(&stage2_per_run_buf_mb)->default_value(8),
             "Per-reader cache size in Stage 2 (each worker opens K such readers). "
             "0 = auto (derived from K + memory_budget + disk type).")
            ("stage2_output_buf_mb",
             po::value<uint32_t>(&stage2_output_buf_mb)->default_value(32),
             "Per-worker pwrite buffer for Stage 2 output. "
             "0 = auto (derived from K + memory_budget + disk type).");

        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    omp_set_num_threads(num_threads);
    mkl_set_num_threads(num_threads);

    if (output_dir.empty()) output_dir = base_folder + "/mergedIndex";
    if (tmp_dir.empty())    tmp_dir    = output_dir + "/tmp_runs";
    output_index_file      = output_dir + "/" + index_name;

    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    printf("=== Pipelined Merge Config ===\n");
    printf("base_folder         : %s\n", base_folder.c_str());
    printf("index_name          : %s\n", index_name.c_str());
    printf("nshards             : %u\n", nshards);
    printf("merge_degree (R)    : %u\n", merge_deg);
    printf("build_degree (B)    : %u\n", build_deg);
    printf("num_threads         : %u\n", num_threads);
    printf("max_merge_workers   : %u\n", max_merge_workers);
    printf("sort_buf            : %u MB\n", sort_buf_mb);
    printf("poll_interval_ms    : %u\n", poll_interval_ms);
    printf("tmp_dir             : %s\n", tmp_dir.c_str());
    printf("output_index_file   : %s\n", output_index_file.c_str());
    printf("keep_runs           : %s\n", keep_runs ? "true" : "false");
    printf("keep_shard_index    : %s\n", keep_shard_index ? "true" : "false");
    printf("keep_partition_inp. : %s\n", keep_partition_inputs ? "true" : "false");
    printf("stage2_memory_budget: %u MB\n", stage2_memory_budget_mb);
    printf("stage2_max_threads  : %u (0 = hw_concurrency)\n", stage2_max_threads);
    printf("stage2_per_run_buf  : %u MB\n", stage2_per_run_buf_mb);
    printf("stage2_output_buf   : %u MB\n", stage2_output_buf_mb);

    uint64_t sort_buf_bytes = (uint64_t)sort_buf_mb * 1024ULL * 1024ULL;
    uint64_t stage2_mem_bytes = (uint64_t)stage2_memory_budget_mb * 1024ULL * 1024ULL;
    uint64_t stage2_per_run_bytes = (uint64_t)stage2_per_run_buf_mb * 1024ULL * 1024ULL;
    uint64_t stage2_out_bytes = (uint64_t)stage2_output_buf_mb * 1024ULL * 1024ULL;

    int rc = scaleGANN_pipelined_merge(
        base_folder,
        index_name,
        output_index_file,
        tmp_dir,
        (uint64_t)nshards,
        merge_deg,
        build_deg,
        max_merge_workers,
        sort_buf_bytes,
        (int)poll_interval_ms,
        !keep_runs,
        !keep_shard_index,
        !keep_partition_inputs,
        stage2_mem_bytes,
        stage2_max_threads,
        stage2_per_run_bytes,
        stage2_out_bytes);

    return rc;
}
