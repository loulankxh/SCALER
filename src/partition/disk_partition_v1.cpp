#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <omp.h>
#include <vector>
#include <cfloat>
#include <chrono>   // [PROFILE] per-phase wall-clock timing

#include "../../DiskANN/include/partition.h"
#include "../../DiskANN/include/utils.h"
#include "../../DiskANN/include/math_utils.h"
#include "../../DiskANN/include/index.h"
#include "partition.h"
#include "AtomicWrapper.hpp"

// Lan: todo: add a sample maximum upper bound
#define MAX_SAMPLE 8388608 // 1 << 23
#define MAX_SAMPLE_FOR_KMEANS_TRAINING 256000 // 1 << 23
#define BLOCK_SIZE 5000000


bool ensure_directory_exists(const std::string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        if (mkdir(path.c_str(), 0777) == -1) {
            std::cerr << "Error: Failed to create directory " << path << std::endl;
            return false;
        }
    } else if (!(info.st_mode & S_IFDIR)) {
        std::cerr << "Error: " << path << " exists but is not a directory!" << std::endl;
        return false;
    }
    return true;
}

template <typename T>
void direct_partitions(const std::string data_file, std::string prefix_path, uint32_t shard_size, uint32_t dataset_size){
    std::string file_type = "";
    uint32_t dotPos = data_file.find_last_of('.');
    if (dotPos == std::string::npos) {
        throw std::invalid_argument("File does not have a valid suffix.");
    }
    std::string suffix = data_file.substr(dotPos + 1);
    if(suffix.size() >= 4 && suffix.substr(suffix.size() - 4) == "vecs") file_type = "vecs";
    else if(suffix.size() >= 3 && suffix.substr(suffix.size() - 3) == "bin") file_type = "bin";
    else throw std::invalid_argument("File does not have a valid suffix.");


    cached_ifstream base_reader(data_file, BUFFER_SIZE_FOR_CACHED_IO);
    uint32_t npts = dataset_size;
    uint32_t dim;
    if(file_type == "bin") base_reader.read((char *)&npts, sizeof(uint32_t));
    base_reader.read((char *)&dim, sizeof(uint32_t));
    if(file_type == "vecs") base_reader.seekg(0);

    uint32_t shard_num = (uint32_t)((npts - 1) / shard_size) + 1;
    printf("Directly partition dataset with %d points into %d shards, each with %d points at most\n", npts, shard_num, shard_size);

    std::vector<std::ofstream> shard_data_writer(shard_num);
    uint32_t actual_shard_size = shard_size;
    for(uint32_t i = 0; i < shard_num; i++){
        std::string partition_dir = prefix_path + "/partition" + std::to_string(i);
        ensure_directory_exists(partition_dir);
        
        std::string data_filename = partition_dir + "/data." + suffix;
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        if(file_type == "bin"){
            if (i == shard_num - 1) actual_shard_size = npts - i * shard_size;
            shard_data_writer[i].write((char *)&actual_shard_size, sizeof(uint32_t));
            shard_data_writer[i].write((char *)&dim, sizeof(uint32_t));
        }
    }

    std::unique_ptr<T[]> data_T = std::make_unique<T[]>(dim);
    for(uint32_t i = 0; i < shard_num; i++){
        uint32_t start_id = i * shard_size;
        uint32_t end_id= start_id + shard_size;
        if ( end_id > npts) end_id = npts;
        
        for(uint32_t p = start_id; p < end_id; p++){
            if(file_type == "vecs") {
                base_reader.read((char *)&dim, sizeof(uint32_t));
                shard_data_writer[i].write((char *)&dim, sizeof(uint32_t));
            }
            base_reader.read((char *)data_T.get(), sizeof(T) * dim);
            shard_data_writer[i].write((char *)data_T.get(), sizeof(T) * dim);
        }
    }

    for(uint32_t i = 0; i < shard_num; i++){
        shard_data_writer[i].close();
    }

}

template <typename T>
void diskANN_shard_data_into_clusters_with_ram_budget(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, std::string prefix_path){
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        // return -1;
        return;
    }

    std::unique_ptr<size_t[]> shard_counts = std::make_unique<size_t[]>(num_centers);
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    // uint32_t const_one = 1;

    for (size_t i = 0; i < num_centers; i++)
    {   
        std::string partition_dir = prefix_path + "/partition" + std::to_string(i);
        ensure_directory_exists(partition_dir);
        uint32_t dotPos = data_file.find_last_of('.');
        if (dotPos == std::string::npos) {
            throw std::invalid_argument("File does not have a valid suffix.");
        }
        std::string suffix = data_file.substr(dotPos + 1);
        std::string data_filename = partition_dir + "/data." + suffix;
        std::string idmap_filename = partition_dir + "/idmap.ibin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        // shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
        shard_counts[i] = 0;
    }

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * k_base);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

        math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, k_base,
                                            block_closest_centers.get());

        for (size_t p = 0; p < cur_blk_size; p++)
        {
            for (size_t p1 = 0; p1 < k_base; p1++)
            {
                size_t shard_id = block_closest_centers[p * k_base + p1];
                uint32_t original_point_map_id = (uint32_t)(start_id + p);
                shard_data_writer[shard_id].write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *)&original_point_map_id, sizeof(uint32_t));
                shard_counts[shard_id]++;
            }
        }
    }

    size_t total_count = 0;
    diskann::cout << "Actual shard sizes: " << std::flush;
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i];
        total_count += cur_shard_count;
        diskann::cout << cur_shard_count << " ";
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    diskann::cout << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get "
                  << total_count << " points across " << num_centers << " shards " << std::endl;
}

template <typename T>
void scaleGANN_shard_data_into_clusters_with_ram_budget_rankSequential(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, uint32_t size_limit, std::string prefix_path,
    float epsilon){
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        // return -1;
        return;
    }

    // use atomic for parallelism
    std::unique_ptr<AtomicWrapper<size_t>[]> shard_counts = std::make_unique<AtomicWrapper<size_t>[]>(num_centers);
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    // uint32_t const_one = 1;

    for (size_t i = 0; i < num_centers; i++)
    {   
        std::string partition_dir = prefix_path + "/partition" + std::to_string(i);
        ensure_directory_exists(partition_dir);
        
        uint32_t dotPos = data_file.find_last_of('.');
        if (dotPos == std::string::npos) {
            throw std::invalid_argument("File does not have a valid suffix.");
        }
        std::string suffix = data_file.substr(dotPos + 1);
        std::string data_filename = partition_dir + "/data." + suffix;
        std::string idmap_filename = partition_dir + "/idmap.ibin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        // shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
        shard_counts[i].store(0);
    }

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * num_centers);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    // To parallel the node assignment and reduction, extra data structures are used for maintain the results for final write (which can not be parallelled)
    // std::unique_ptr<float[]> distance_matrix = std::make_unique<float[]>(num_centers, );
    float *distance_matrix = new float[num_centers * block_size]; // row: num points, col: num centers
    std::vector<std::unique_ptr<uint32_t[]>> shard_to_ids(num_centers);
    std::unique_ptr<uint32_t[]> shard_counts_until_this_block = std::make_unique<uint32_t[]>(num_centers);
    std::unique_ptr<AtomicWrapper<uint32_t>[]> shard_counts_first_round = std::make_unique<AtomicWrapper<uint32_t>[]>(num_centers);
    std::unique_ptr<uint32_t[]> shard_counts_first_round_until_this_block = std::make_unique<uint32_t[]>(num_centers);
    std::unique_ptr<AtomicWrapper<uint32_t>[]> shard_counts_second_round = std::make_unique<AtomicWrapper<uint32_t>[]>(num_centers);
    uint32_t size_limit_first_round = (uint32_t)(size_limit * 0.8); // 0.8: bound proportion, a cluster should contain at most 80% points whose nearest center is in this cluster
    std::unique_ptr<float[]> data_distribution_first_round = std::make_unique<float[]>(num_centers);
    uint32_t size_lower_bound_second_round = size_limit - size_limit_first_round;
    std::unique_ptr<uint32_t[]> size_limit_second_round = std::make_unique<uint32_t[]>(num_centers);
    std::unique_ptr<uint32_t[]> first_round_assignment = std::make_unique<uint32_t[]>(block_size);
    std::unique_ptr<float[]> shard_radius = std::make_unique<float[]>(num_centers);
    for (size_t i = 0; i < num_centers; i++) {
        shard_to_ids[i] = std::make_unique<uint32_t[]>(block_size);
        shard_counts_until_this_block[i] = 0;

        shard_counts_first_round[i].store(0);
        shard_counts_first_round_until_this_block[i] = 0;
        shard_counts_second_round[i].store(0);
        data_distribution_first_round[i] = 1/(float)num_centers;
        size_limit_second_round[i] = 0;
        shard_radius[i] = 0.0;
    }
    for (size_t i = 0; i < block_size; i++){
        first_round_assignment[i] = num_centers; // initial: unassiged value
    }

    // [PROFILE] Per-phase wall-clock accumulators (sum across all blocks).
    //   Goal: identify which step actually dominates the partition runtime,
    //   so we can tell whether further CPU optimization or I/O changes pay off.
    auto __now = []() {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    };
    double t_io_read = 0, t_convert = 0, t_dist = 0;
    double t_round1 = 0, t_data_dist = 0, t_round2 = 0;
    double t_sort = 0, t_write = 0;
    double t_setup_start = __now();  // (function entry to first block) – approximate setup time

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        // [PROFILE] I/O read
        double __t = __now();
        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        double __t1 = __now(); t_io_read += __t1 - __t;
        // [PROFILE] convert types T -> float
        diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);
        double __t2 = __now(); t_convert += __t2 - __t1;

        // [PROFILE] distance compute (MKL-driven dense distance matrix)
        math_utils::compute_closest_centers_return_distance(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, num_centers,
                                            block_closest_centers.get(), distance_matrix, NULL, NULL);
        double __t3 = __now(); t_dist += __t3 - __t2;

        // [PROFILE] Round 1 assignment (atomic-heavy hot loop)
        double __t_r1 = __now();
        // Round 1 assignment: assign a point to its closest center
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < cur_blk_size; p++)
        {
            for (size_t p1 = 0; p1 < num_centers; p1++)
            {
                size_t shard_id = block_closest_centers[p * num_centers + p1];
                uint32_t current_r1_size = shard_counts_first_round[shard_id].load_relaxed(); // omp_get_num_procs: inaccuracy upper bound by concurrency after pragma & atomic
                uint32_t curret_size = shard_counts[shard_id].load_relaxed();
                if ((current_r1_size > size_limit_first_round) || (curret_size > size_limit)){
                    continue;
                }

                shard_counts_first_round[shard_id].fetch_add_relaxed(1);
                first_round_assignment[p] = shard_id;
                uint32_t original_point_map_id = (uint32_t)(start_id + p);
                uint32_t current_id = shard_counts[shard_id].fetch_add_relaxed(1) - shard_counts_until_this_block[shard_id];
                shard_to_ids[shard_id][current_id] = original_point_map_id;
                float dist = distance_matrix[p * num_centers + (uint32_t)shard_id];
                if (dist > shard_radius[shard_id]){
                    shard_radius[shard_id] = dist;
                }
                break;
            }
        }
        t_round1 += __now() - __t_r1;

        // [PROFILE] Data distribution / size_limit_second_round update
        double __t_dd = __now();
        // Update data distribution based on round 1, update round 2 size limit
        #pragma omp parallel for schedule(static)
        for (size_t cluster = 0; cluster < num_centers; cluster++){
            data_distribution_first_round[cluster] = (data_distribution_first_round[cluster] * (block + 1) +
                                                        ((shard_counts_first_round[cluster].load_relaxed() - shard_counts_first_round_until_this_block[cluster])/((float)cur_blk_size)))
                                                    / ((float)(block + 2));

            uint32_t size_remain_after_first_round = 0;
            if ((uint32_t)(data_distribution_first_round[cluster] * npts32) < size_limit){
                size_remain_after_first_round = size_limit - (uint32_t)(data_distribution_first_round[cluster] * npts32);
            }
            if (size_remain_after_first_round > size_lower_bound_second_round) size_limit_second_round[cluster] = size_remain_after_first_round;
            else size_limit_second_round[cluster] = size_lower_bound_second_round;
            if (block == num_blocks - 1){
                size_limit_second_round[cluster] = size_limit - shard_counts_first_round[cluster].load_relaxed();
            }
        }
        t_data_dist += __now() - __t_dd;

        // [PROFILE] Round 2 assignment (atomic-heavy hot loop)
        double __t_r2 = __now();
        // Round 2 assignment: assign points to other centers for duplication, while under reduction rules
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < cur_blk_size; p++)
        {
            uint32_t assigned_count = 1;
            for (size_t p1 = 0; p1 < num_centers; p1++)
            {
                if (assigned_count >= k_base)
                    break;
                uint32_t first_round_center = first_round_assignment[p];
                size_t shard_id = block_closest_centers[p * num_centers + p1];
                if (shard_id == first_round_center)
                    continue;
                uint32_t current_r2_size = shard_counts_second_round[shard_id].load_relaxed(); // omp_get_num_procs: inaccuracy upper bound by concurrency after pragma & atomic
                uint32_t curret_size = shard_counts[shard_id].load_relaxed();
                if ((current_r2_size > size_limit_second_round[shard_id]) || (curret_size > size_limit)){
                    continue;
                }

                float dist = distance_matrix[p * num_centers + (uint32_t)shard_id];
                if (dist < epsilon * distance_matrix[p * num_centers + first_round_center]){
                    if(dist < epsilon * (1 + (float)1 / (block + 1)) * shard_radius[shard_id]){
                        shard_counts_second_round[shard_id].fetch_add_relaxed(1);
                        uint32_t original_point_map_id = (uint32_t)(start_id + p);
                        uint32_t current_id = shard_counts[shard_id].fetch_add_relaxed(1) - shard_counts_until_this_block[shard_id];
                        shard_to_ids[shard_id][current_id] = original_point_map_id;
                        assigned_count++;
                    }
                } else{
                    break;
                }
            }
        }
        t_round2 += __now() - __t_r2;

        // [PROFILE] Sort phase (per shard: copy + std::sort + copy back)
        double __t_sort = __now();
        // Sorting for sequential disk layout
        #pragma omp parallel for schedule(static)
        for (size_t shard_id = 0; shard_id < num_centers; shard_id ++){
            auto& ptr = shard_to_ids[shard_id];
            uint32_t len = (uint32_t)(shard_counts[shard_id].load_relaxed() - shard_counts_until_this_block[shard_id]);
            std::vector<uint32_t> temp(len);
            for (size_t i = 0; i < len; ++i) {
                temp[i] = ptr[i];
            }
            std::sort(temp.begin(), temp.end());

            for (size_t i = 0; i < len; ++i) {
                ptr[i] = temp[i];
            }
        }
        t_sort += __now() - __t_sort;

        // [PROFILE] Write phase (per-point ofstream::write × 2, parallel by shard)
        double __t_write = __now();
        #pragma omp parallel for schedule(static)
        for (size_t shard_id = 0; shard_id < num_centers; shard_id ++){
            uint32_t shard_size_this_block = (uint32_t)(shard_counts[shard_id].load_relaxed() - shard_counts_until_this_block[shard_id]);
            shard_counts_until_this_block[shard_id] = (uint32_t)(shard_counts[shard_id].load_relaxed());
            shard_counts_first_round_until_this_block[shard_id] = (uint32_t)(shard_counts_first_round[shard_id].load_relaxed());
            for (uint32_t i = 0; i < shard_size_this_block; i++){
                uint32_t original_point_map_id = shard_to_ids[shard_id][i];
                uint32_t p = original_point_map_id - (uint32_t)start_id;
                shard_data_writer[shard_id].write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *)&original_point_map_id, sizeof(uint32_t));
            }
        }
        t_write += __now() - __t_write;
    }
    double t_block_loop_end = __now();

    // [PROFILE] Per-phase summary across all blocks.
    diskann::cout << "[PROFILE] Per-phase wall-clock totals (seconds, summed across "
                  << num_blocks << " blocks):\n"
                  << "  io_read           = " << t_io_read << "\n"
                  << "  convert_types     = " << t_convert << "\n"
                  << "  compute_distance  = " << t_dist << "\n"
                  << "  round1_assign     = " << t_round1 << "\n"
                  << "  data_dist_update  = " << t_data_dist << "\n"
                  << "  round2_assign     = " << t_round2 << "\n"
                  << "  sort_per_shard    = " << t_sort << "\n"
                  << "  write_per_shard   = " << t_write << "\n"
                  << "  block_loop_total  = "
                  << (t_io_read + t_convert + t_dist + t_round1 + t_data_dist
                      + t_round2 + t_sort + t_write) << "\n"
                  << "  setup_to_block_loop_end_wall = "
                  << (t_block_loop_end - t_setup_start) << std::endl;

    // [PROFILE] Per-shard close() timing — close() blocks until OS finishes
    //   writing this file's dirty pages. Long close times indicate the data was
    //   still in page cache (not yet on disk) when the block loop ended.
    size_t total_count = 0;
    diskann::cout << "Actual shard sizes:" << std::endl;
    double t_finalize_start = __now();
    for (size_t i = 0; i < num_centers; i++)
    {
        double __ts = __now();
        uint32_t cur_shard_count = (uint32_t)shard_counts[i].load_relaxed();
        total_count += cur_shard_count;
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        double __tc_d0 = __now();
        shard_data_writer[i].close();
        double __tc_d1 = __now();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        double __tc_i0 = __now();
        shard_idmap_writer[i].close();
        double __tc_i1 = __now();
        diskann::cout << "  shard[" << i << "] count=" << cur_shard_count
                      << " close_data=" << (__tc_d1 - __tc_d0) << "s"
                      << " close_idmap=" << (__tc_i1 - __tc_i0) << "s"
                      << " total=" << (__now() - __ts) << "s" << std::endl;
    }
    diskann::cout << "[PROFILE] Total finalize loop (all closes, serial) = "
                  << (__now() - t_finalize_start) << " seconds" << std::endl;

    diskann::cout << " (ScaleGANN) Partitioned " << num_points << " with replication factor " << k_base << " to get "
                  << total_count << " points across " << num_centers << " shards " << std::endl;
    delete[] distance_matrix;
}

template <typename T>
void scaleGANN_shard_data_into_clusters_with_ram_budget(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, uint32_t size_limit, std::string prefix_path,
    float epsilon){
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        // return -1;
        return;
    }

    // use atomic for parallelism
    std::unique_ptr<AtomicWrapper<size_t>[]> shard_counts = std::make_unique<AtomicWrapper<size_t>[]>(num_centers);
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    uint32_t const_one = 1;

    for (size_t i = 0; i < num_centers; i++)
    {   
        std::string partition_dir = prefix_path + "/partition" + std::to_string(i);
        ensure_directory_exists(partition_dir);
        
        uint32_t dotPos = data_file.find_last_of('.');
        if (dotPos == std::string::npos) {
            throw std::invalid_argument("File does not have a valid suffix.");
        }
        std::string suffix = data_file.substr(dotPos + 1);
        std::string data_filename = partition_dir + "/data." + suffix;
        std::string idmap_filename = partition_dir + "/idmap.ibin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        // shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
        shard_counts[i].store(0);
    }

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * num_centers);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    // To parallel the node assignment and reduction, extra data structures are used for maintain the results for final write (which can not be parallelled)
    // std::unique_ptr<float[]> distance_matrix = std::make_unique<float[]>(num_centers, );
    float *distance_matrix = new float[num_centers * block_size]; // row: num points, col: num centers
    std::vector<std::unique_ptr<uint32_t[]>> shard_to_ids(num_centers);
    std::unique_ptr<uint32_t[]> shard_counts_until_this_block = std::make_unique<uint32_t[]>(num_centers);
    std::unique_ptr<AtomicWrapper<uint32_t>[]> shard_counts_first_round = std::make_unique<AtomicWrapper<uint32_t>[]>(num_centers);
    std::unique_ptr<uint32_t[]> shard_counts_first_round_until_this_block = std::make_unique<uint32_t[]>(num_centers);
    std::unique_ptr<AtomicWrapper<uint32_t>[]> shard_counts_second_round = std::make_unique<AtomicWrapper<uint32_t>[]>(num_centers);
    uint32_t size_limit_first_round = (uint32_t)(size_limit * 0.8); // 0.8: bound proportion, a cluster should contain at most 80% points whose nearest center is in this cluster
    std::unique_ptr<float[]> data_distribution_first_round = std::make_unique<float[]>(num_centers);
    uint32_t size_lower_bound_second_round = size_limit - size_limit_first_round;
    std::unique_ptr<uint32_t[]> size_limit_second_round = std::make_unique<uint32_t[]>(num_centers);
    std::unique_ptr<uint32_t[]> first_round_assignment = std::make_unique<uint32_t[]>(block_size);
    std::unique_ptr<float[]> shard_radius = std::make_unique<float[]>(num_centers);
    for (size_t i = 0; i < num_centers; i++) {
        shard_to_ids[i] = std::make_unique<uint32_t[]>(block_size);
        shard_counts_until_this_block[i] = 0;

        shard_counts_first_round[i].store(0);
        shard_counts_first_round_until_this_block[i] = 0;
        shard_counts_second_round[i].store(0);
        data_distribution_first_round[i] = 1/(float)num_centers;
        size_limit_second_round[i] = 0;
        shard_radius[i] = 0.0;
    }
    for (size_t i = 0; i < block_size; i++){
        first_round_assignment[i] = num_centers; // initial: unassiged value
    }

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);


        math_utils::compute_closest_centers_return_distance(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, num_centers,
                                            block_closest_centers.get(), distance_matrix, NULL, NULL);

        // Round 1 assignment: assign a point to its closest center
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < cur_blk_size; p++)
        {   
            for (size_t p1 = 0; p1 < num_centers; p1++)
            {   
                size_t shard_id = block_closest_centers[p * num_centers + p1];
                uint32_t current_r1_size = shard_counts_first_round[shard_id].load_relaxed(); // omp_get_num_procs: inaccuracy upper bound by concurrency after pragma & atomic
                uint32_t curret_size = shard_counts[shard_id].load_relaxed();
                if ((current_r1_size > size_limit_first_round) || (curret_size > size_limit)){
                    continue;
                }

                shard_counts_first_round[shard_id].fetch_add_relaxed(1);
                first_round_assignment[p] = shard_id;
                uint32_t original_point_map_id = (uint32_t)(start_id + p);
                uint32_t current_id = shard_counts[shard_id].fetch_add_relaxed(1) - shard_counts_until_this_block[shard_id];
                shard_to_ids[shard_id][current_id] = original_point_map_id;
                float dist = distance_matrix[p * num_centers + (uint32_t)shard_id];
                if (dist > shard_radius[shard_id]){
                    shard_radius[shard_id] = dist;
                }
                break;
            }
        }
        // Update data distribution based on round 1, update round 2 size limit
        #pragma omp parallel for schedule(static)
        for (size_t cluster = 0; cluster < num_centers; cluster++){
            data_distribution_first_round[cluster] = (data_distribution_first_round[cluster] * (block + 1) + 
                                                        ((shard_counts_first_round[cluster].load_relaxed() - shard_counts_first_round_until_this_block[cluster])/((float)cur_blk_size))) 
                                                    / ((float)(block + 2));
            
            uint32_t size_remain_after_first_round = 0;
            if ((uint32_t)(data_distribution_first_round[cluster] * npts32) < size_limit){
                size_remain_after_first_round = size_limit - (uint32_t)(data_distribution_first_round[cluster] * npts32);
            }
            if (size_remain_after_first_round > size_lower_bound_second_round) size_limit_second_round[cluster] = size_remain_after_first_round;
            else size_limit_second_round[cluster] = size_lower_bound_second_round;
            if (block == num_blocks - 1){
                size_limit_second_round[cluster] = size_limit - shard_counts_first_round[cluster].load_relaxed();
            }
        }
        // Round 2 assignment: assign points to other centers for duplication, while under reduction rules
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < cur_blk_size; p++)
        {   
            uint32_t assigned_count = 1;
            for (size_t p1 = 0; p1 < num_centers; p1++)
            {   
                if (assigned_count >= k_base)
                    break;
                uint32_t first_round_center = first_round_assignment[p];
                size_t shard_id = block_closest_centers[p * num_centers + p1];
                if (shard_id == first_round_center) 
                    continue;
                uint32_t current_r2_size = shard_counts_second_round[shard_id].load_relaxed(); // omp_get_num_procs: inaccuracy upper bound by concurrency after pragma & atomic
                uint32_t curret_size = shard_counts[shard_id].load_relaxed();
                if ((current_r2_size > size_limit_second_round[shard_id]) || (curret_size > size_limit)){
                    continue;
                }

                float dist = distance_matrix[p * num_centers + (uint32_t)shard_id];
                // printf("node id: %d, block id: %d, firstRoundDsit: %f, cur dist: %f, epsilon: %f\n", p, block, distance_matrix[p * num_centers + first_round_center], dist, epsilon);
                if (dist < epsilon * distance_matrix[p * num_centers + first_round_center]){
                    if(dist < epsilon * (1 + (float)1 / (block + 1)) * shard_radius[shard_id]){
                        shard_counts_second_round[shard_id].fetch_add_relaxed(1);
                        uint32_t original_point_map_id = (uint32_t)(start_id + p);
                        uint32_t current_id = shard_counts[shard_id].fetch_add_relaxed(1) - shard_counts_until_this_block[shard_id];
                        shard_to_ids[shard_id][current_id] = original_point_map_id;
                        assigned_count++;
                    }
                } else{
                    // printf("node id: %d, block id: %d, firstRoundDsit: %f, cur dist: %f, epsilon: %f\n", p, block, distance_matrix[p * num_centers + first_round_center], dist, epsilon);
                    break;
                }
            }
        }
        #pragma omp parallel for schedule(static)
        for (size_t shard_id = 0; shard_id < num_centers; shard_id ++){
            uint32_t shard_size_this_block = (uint32_t)(shard_counts[shard_id].load_relaxed() - shard_counts_until_this_block[shard_id]);
            shard_counts_until_this_block[shard_id] = (uint32_t)(shard_counts[shard_id].load_relaxed());
            shard_counts_first_round_until_this_block[shard_id] = (uint32_t)(shard_counts_first_round[shard_id].load_relaxed());
            for (uint32_t i = 0; i < shard_size_this_block; i++){
                uint32_t original_point_map_id = shard_to_ids[shard_id][i];
                uint32_t p = original_point_map_id - (uint32_t)start_id;
                shard_data_writer[shard_id].write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *)&original_point_map_id, sizeof(uint32_t));
            }
        }
    }

    size_t total_count = 0;
    diskann::cout << "Actual shard sizes: " << std::flush;
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i].load_relaxed();
        total_count += cur_shard_count;
        diskann::cout << cur_shard_count << " ";
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    diskann::cout << "\n (ScaleGANN) Partitioned " << num_points << " with replication factor " << k_base << " to get "
                  << total_count << " points across " << num_centers << " shards " << std::endl;
    delete[] distance_matrix;
}


template <typename T>
void scaleGANN_non_selective_shard_with_ram_budget(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, uint32_t size_limit, std::string prefix_path){
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        // return -1;
        return;
    }

    // use atomic for parallelism
    std::unique_ptr<AtomicWrapper<size_t>[]> shard_counts = std::make_unique<AtomicWrapper<size_t>[]>(num_centers);
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;

    for (size_t i = 0; i < num_centers; i++)
    {   
        std::string partition_dir = prefix_path + "/partition" + std::to_string(i);
        ensure_directory_exists(partition_dir);
        
        uint32_t dotPos = data_file.find_last_of('.');
        if (dotPos == std::string::npos) {
            throw std::invalid_argument("File does not have a valid suffix.");
        }
        std::string suffix = data_file.substr(dotPos + 1);
        std::string data_filename = partition_dir + "/data." + suffix;
        std::string idmap_filename = partition_dir + "/idmap.ibin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_counts[i].store(0);
    }

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * num_centers);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    // To parallel the node assignment and reduction, extra data structures are used for maintain the results for final write (which can not be parallelled)
    std::vector<std::unique_ptr<uint32_t[]>> shard_to_ids(num_centers);
    std::unique_ptr<uint32_t[]> shard_counts_until_this_block = std::make_unique<uint32_t[]>(num_centers);
    for (size_t i = 0; i < num_centers; i++) {
        shard_to_ids[i] = std::make_unique<uint32_t[]>(block_size);
        shard_counts_until_this_block[i] = 0;
    }

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);


        math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, num_centers,
                                            block_closest_centers.get());

        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < cur_blk_size; p++)
        {   
            uint32_t assigned_count = 0;
            for (size_t p1 = 0; p1 < num_centers; p1++)
            {   
                // balance the size of each cluster
                if (assigned_count >= k_base) {
                    break;
                }

                size_t shard_id = block_closest_centers[p * num_centers + p1];
                uint32_t original_point_map_id = (uint32_t)(start_id + p);

                uint32_t partition_size_id = shard_counts[shard_id].load_relaxed(); // omp_get_num_procs: inaccuracy upper bound by concurrency after pragma & atomic
                if ((partition_size_id >= size_limit) && ((num_centers - p1) > (k_base - assigned_count))) {
                    continue;
                }

                uint32_t current_id = shard_counts[shard_id].fetch_add_relaxed(1) - shard_counts_until_this_block[shard_id];
                shard_to_ids[shard_id][current_id] = original_point_map_id; 
                assigned_count++;
            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t shard_id = 0; shard_id < num_centers; shard_id ++){
            uint32_t shard_size_this_block = (uint32_t)(shard_counts[shard_id].load_relaxed() - shard_counts_until_this_block[shard_id]);
            shard_counts_until_this_block[shard_id] = (uint32_t)(shard_counts[shard_id].load_relaxed());
            for (uint32_t i = 0; i < shard_size_this_block; i++){
                uint32_t original_point_map_id = shard_to_ids[shard_id][i];
                uint32_t p = original_point_map_id - (uint32_t)start_id;
                shard_data_writer[shard_id].write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *)&original_point_map_id, sizeof(uint32_t));
            }
        }
    }

    size_t total_count = 0;
    diskann::cout << "Actual shard sizes: " << std::flush;
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i].load_relaxed();
        total_count += cur_shard_count;
        diskann::cout << cur_shard_count << " ";
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    diskann::cout << "\n Partitioned " << num_points << " with replication factor " << k_base << " to get "
                  << total_count << " points across " << num_centers << " shards " << std::endl;
}


template <typename T>
void SOGAIC_shard_data_into_clusters_with_ram_budget(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, uint32_t size_limit, std::string prefix_path,
    float epsilon){
    size_t read_blk_size = 64 * 1024 * 1024;
    //  uint64_t write_blk_size = 64 * 1024 * 1024;
    // create cached reader + writer
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        // return -1;
        return;
    }

    // use atomic for parallelism
    std::unique_ptr<AtomicWrapper<size_t>[]> shard_counts = std::make_unique<AtomicWrapper<size_t>[]>(num_centers);
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    uint32_t const_one = 1;

    for (size_t i = 0; i < num_centers; i++)
    {   
        std::string partition_dir = prefix_path + "/partition" + std::to_string(i);
        ensure_directory_exists(partition_dir);

        uint32_t dotPos = data_file.find_last_of('.');
        if (dotPos == std::string::npos) {
            throw std::invalid_argument("File does not have a valid suffix.");
        }
        std::string suffix = data_file.substr(dotPos + 1);
        std::string data_filename = partition_dir + "/data." + suffix;
        std::string idmap_filename = partition_dir + "/idmap.ibin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        // shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
        shard_counts[i].store(0);
    }

    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    std::unique_ptr<uint32_t[]> block_closest_centers = std::make_unique<uint32_t[]>(block_size * num_centers);
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    // To parallel the node assignment and reduction, extra data structures are used for maintain the results for final write (which can not be parallelled)
    // std::unique_ptr<float[]> distance_matrix = std::make_unique<float[]>(num_centers, );
    float *distance_matrix = new float[num_centers * block_size]; // row: num points, col: num centers
    std::vector<std::unique_ptr<uint32_t[]>> shard_to_ids(num_centers);
    std::unique_ptr<uint32_t[]> shard_counts_until_this_block = std::make_unique<uint32_t[]>(num_centers);
    for (size_t i = 0; i < num_centers; i++) {
        shard_to_ids[i] = std::make_unique<uint32_t[]>(block_size);
        shard_counts_until_this_block[i] = 0;
    }

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);


        // math_utils::compute_closest_centers(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, num_centers,
        //                                     block_closest_centers.get());
        math_utils::compute_closest_centers_return_distance(block_data_float.get(), cur_blk_size, dim, pivots, num_centers, num_centers,
                                            block_closest_centers.get(), distance_matrix, NULL, NULL);

        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < cur_blk_size; p++)
        {   
            uint32_t curOLPcnt = 0;
            uint32_t curOLPFactor = 0;
            float accDist = 0;
            float curAVGDist = FLT_MAX;

            for (size_t p1 = 0; p1 < num_centers; p1++)
            {   
                if (curOLPcnt >= k_base)
                    break;

                size_t shard_id = block_closest_centers[p * num_centers + p1];
                float dist = distance_matrix[p * num_centers + (uint32_t)shard_id];
                
                // printf("curAVGDsit: %f, cur dist: %f, epsilon: %f\n", curAVGDist, dist, epsilon);
                if(dist <= (double) epsilon * curAVGDist){
                    curOLPFactor = curOLPFactor + 1;
                    accDist = accDist + dist;
                    curAVGDist = accDist / curOLPFactor;

                    uint32_t current_size = shard_counts[shard_id].load_relaxed();
                    if (current_size < size_limit){
                        curOLPcnt += 1;
                        uint32_t original_point_map_id = (uint32_t)(start_id + p);
                        uint32_t current_id = shard_counts[shard_id].fetch_add_relaxed(1) - shard_counts_until_this_block[shard_id];
                        shard_to_ids[shard_id][current_id] = original_point_map_id;
                    } else {
                        curAVGDist = FLT_MAX;
                    }
                }

            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t shard_id = 0; shard_id < num_centers; shard_id ++){
            uint32_t shard_size_this_block = (uint32_t)(shard_counts[shard_id].load_relaxed() - shard_counts_until_this_block[shard_id]);
            shard_counts_until_this_block[shard_id] = (uint32_t)(shard_counts[shard_id].load_relaxed());
            for (uint32_t i = 0; i < shard_size_this_block; i++){
                uint32_t original_point_map_id = shard_to_ids[shard_id][i];
                uint32_t p = original_point_map_id - (uint32_t)start_id;
                shard_data_writer[shard_id].write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                shard_idmap_writer[shard_id].write((char *)&original_point_map_id, sizeof(uint32_t));
            }
        }
    }

    size_t total_count = 0;
    diskann::cout << "Actual shard sizes: " << std::flush;
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i].load_relaxed();
        total_count += cur_shard_count;
        diskann::cout << cur_shard_count << " ";
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    diskann::cout << "\n (SOGAIC) Partitioned " << num_points << " with replication factor " << k_base << " to get "
                  << total_count << " points across " << num_centers << " shards " << std::endl;
    delete[] distance_matrix;
}


template <typename T>
void diskANN_partitions_with_ram_budget(const std::string data_file, double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base){
    size_t train_dim;
    size_t num_train;
    float *train_data_float;
    size_t max_k_means_reps = 15;

    int num_parts = 3;
    bool fit_in_ram = false;
    
    std::ifstream head_reader(data_file.c_str(), std::ios::binary);
    uint32_t npts32;
    uint32_t dim32;
    head_reader.read((char *)&npts32, sizeof(uint32_t));
    head_reader.read((char *)&dim32, sizeof(uint32_t));
    sampling_rate=((double)MAX_SAMPLE_FOR_KMEANS_TRAINING / (double) npts32);
    printf("Adjusting DiskANN sampling rate to %f\n", sampling_rate);

    gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

    size_t test_dim;
    size_t num_test;
    float *test_data_float;
    gen_random_slice<T>(data_file, sampling_rate, test_data_float, num_test, test_dim);

    float *pivot_data = nullptr;

    std::string cur_file = std::string(prefix_path);
    ensure_directory_exists(prefix_path);
    std::string output_file;

    // kmeans_partitioning on training data

    //  cur_file = cur_file + "_kmeans_partitioning-" +
    //  std::to_string(num_parts);
    output_file = cur_file + "/centroids.bin";

    while (!fit_in_ram)
    {
        fit_in_ram = true;

        double max_ram_usage = 0;
        if (pivot_data != nullptr)
            delete[] pivot_data;

        pivot_data = new float[num_parts * train_dim];
        // Process Global k-means for kmeans_partitioning Step
        diskann::cout << "Processing global k-means (kmeans_partitioning Step)" << std::endl;
        kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_parts);
        kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

        // now pivots are ready. need to stream base points and assign them to
        // closest clusters.

        std::vector<size_t> cluster_sizes;
        estimate_cluster_sizes(test_data_float, num_test, pivot_data, num_parts, train_dim, k_base, cluster_sizes);

        for (auto &p : cluster_sizes)
        {
            // to account for the fact that p is the size of the shard over the
            // testing sample.
            p = (uint64_t)(p / sampling_rate);
            double cur_shard_ram_estimate =
                diskann::estimate_ram_usage(p, (uint32_t)train_dim, sizeof(T), (uint32_t)graph_degree);

            if (cur_shard_ram_estimate > max_ram_usage)
                max_ram_usage = cur_shard_ram_estimate;
        }
        diskann::cout << "With " << num_parts
                      << " parts, max estimated RAM usage: " << max_ram_usage / (1024 * 1024 * 1024)
                      << "GB, budget given is " << ram_budget << std::endl;
        if (max_ram_usage > 1024 * 1024 * 1024 * ram_budget)
        {
            fit_in_ram = false;
            num_parts += 2;
        }
    }

    diskann::cout << "Saving global k-center pivots" << std::endl;
    diskann::save_bin<float>(output_file.c_str(), pivot_data, (size_t)num_parts, train_dim);

    diskANN_shard_data_into_clusters_with_ram_budget<T>(data_file, pivot_data, num_parts, train_dim, k_base, prefix_path);
    delete[] pivot_data;
    delete[] train_data_float;
    delete[] test_data_float;
    // return num_parts;
}


template <typename T>
void scaleGANN_partitions_with_ram_budget(const std::string data_file, double sampling_rate, double ram_budget,
        size_t graph_degree, uint32_t inter_degree, uint32_t threads,
        const std::string prefix_path, size_t k_base, uint32_t num_parts = 0, float epsilon=2, size_t max_k_means_reps = 15){
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    
    // Lan: todo: if partition-num == 1
    uint32_t partition_lower_bound = 0;
    uint32_t size_limit = (uint32_t) npts32;
    get_partition_num<T>(ram_budget, npts32, basedim32, graph_degree, inter_degree, threads, k_base, &partition_lower_bound, &size_limit);
    // uint32_t size_limit = (uint32_t) (1 + k_base * npts32 / partition_lower_bound);
    num_parts = partition_lower_bound > num_parts ? partition_lower_bound : num_parts;

    size_t train_dim;
    size_t num_train;
    float *train_data_float;
    if (sampling_rate > ((double)MAX_SAMPLE / npts32)){
        sampling_rate = (double)MAX_SAMPLE / npts32;
    }
    gen_random_slice<T>(data_file, sampling_rate, train_data_float, num_train, train_dim);

    float *pivot_data = nullptr;
    pivot_data = new float[num_parts * train_dim];
    // Process Global k-means for kmeans_partitioning Step
    diskann::cout << "Processing global k-means (kmeans_partitioning Step)" << std::endl;
    // Lan: todo: use GPU Kmeans clustering
    kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_parts);
    kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data, num_parts, max_k_means_reps, NULL, NULL);

    std::string cur_file = std::string(prefix_path);
    ensure_directory_exists(prefix_path);
    std::string output_file;
    output_file = cur_file + "/centroids.bin";

    diskann::cout << "Saving global k-center pivots" << std::endl;
    diskann::save_bin<float>(output_file.c_str(), pivot_data, (size_t)num_parts, train_dim);

    // scaleGANN_shard_data_into_clusters_with_ram_budget<T>(data_file, pivot_data, num_parts, train_dim, k_base, size_limit, prefix_path, epsilon);
    scaleGANN_shard_data_into_clusters_with_ram_budget_rankSequential<T>(data_file, pivot_data, num_parts, train_dim, k_base, size_limit, prefix_path, epsilon);
    // SOGAIC_shard_data_into_clusters_with_ram_budget<T>(data_file, pivot_data, num_parts, train_dim, k_base, size_limit, prefix_path, epsilon);
    // scaleGANN_non_selective_shard_with_ram_budget<T>(data_file, pivot_data, num_parts, train_dim, k_base, size_limit, prefix_path);
    delete[] pivot_data;
    delete[] train_data_float;
}



template void direct_partitions<float>(const std::string data_file, std::string prefix_path, uint32_t shard_size, uint32_t dataset_size);
template void direct_partitions<uint8_t>(const std::string data_file, std::string prefix_path, uint32_t shard_size, uint32_t dataset_size);


template void diskANN_shard_data_into_clusters_with_ram_budget<float>(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, std::string prefix_path);
// template void diskANN_shard_data_into_clusters_with_ram_budget<uint32_t>(const std::string data_file, float *pivots, const size_t num_centers,
//     const size_t dim, const size_t k_base, std::string prefix_path);
template void diskANN_shard_data_into_clusters_with_ram_budget<uint8_t>(const std::string data_file, float *pivots, const size_t num_centers,
    const size_t dim, const size_t k_base, std::string prefix_path);

template void diskANN_partitions_with_ram_budget<float>(const std::string data_file, double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);
// template void diskANN_partitions_with_ram_budget<uint32_t>(const std::string data_file, const double sampling_rate, double ram_budget,
//     size_t graph_degree, const std::string prefix_path, size_t k_base);
template void diskANN_partitions_with_ram_budget<uint8_t>(const std::string data_file, double sampling_rate, double ram_budget,
    size_t graph_degree, const std::string prefix_path, size_t k_base);


template void scaleGANN_partitions_with_ram_budget<float>(const std::string data_file, double sampling_rate, double ram_budget,
    size_t graph_degree, uint32_t inter_degree, uint32_t threads,
    const std::string prefix_path, size_t k_base, uint32_t num_parts = 0, float epsilon = 2, size_t max_k_means_reps = 15);
template void scaleGANN_partitions_with_ram_budget<uint8_t>(const std::string data_file, double sampling_rate, double ram_budget,
    size_t graph_degree, uint32_t inter_degree, uint32_t threads,
    const std::string prefix_path, size_t k_base, uint32_t num_parts = 0, float epsilon = 2, size_t max_k_means_reps = 15);