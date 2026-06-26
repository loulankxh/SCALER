#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <unordered_set>
#include <numeric>
#include <random>
#include <filesystem>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "../../DiskANN/include/disk_utils.h"
#include "../utils/indexIO.hpp"
#include "../utils/datasetIO.hpp"
#include "disk_merge.h"

void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs)
{
    uint32_t npts32;
    size_t actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *)&npts32, sizeof(uint32_t));
    if (actual_file_size != ((size_t)npts32) * sizeof(uint32_t) + sizeof(uint32_t))
    {
        std::stringstream stream;
        stream << "Error reading idmap file. Check if the file is bin file with "
                  "1 dimensional data. Actual: "
               << actual_file_size << ", expected: " << (size_t)npts32 * sizeof(uint32_t) + sizeof(uint32_t) << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    ivecs.resize(npts32);
    reader.read((char *)ivecs.data(), ((size_t)npts32) * sizeof(uint32_t));
    reader.close();
}


#define RAFT_NUMPY_MAGIC_STRING_LENGTH 6
#define RAFT_NUMPY_MAGIC_STRING        "\x93NUMPY"

void read_header(cached_ifstream& is){
    char magic_buf[RAFT_NUMPY_MAGIC_STRING_LENGTH + 2] = {0};
    is.read(magic_buf, RAFT_NUMPY_MAGIC_STRING_LENGTH + 2);
    // printf("magic buffer is: %s\n",magic_buf);

    std::uint8_t header_len_le16[2];
    is.read(reinterpret_cast<char*>(header_len_le16), 2);

    const std::uint32_t header_length = (header_len_le16[0] << 0) | (header_len_le16[1] << 8);
    std::vector<char> header_bytes(header_length);
    is.read(header_bytes.data(), header_length);
    std::string str = std::string(header_bytes.data(), header_length);
}

template <typename T>
T read_once(cached_ifstream& is){
    read_header(is);

    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
    return val;
}

template <typename T>
struct always_false : std::false_type {};

template <typename T>
void write_header(cached_ofstream& os, bool fortran_order, uint32_t shape_x = 0, uint32_t shape_y = 0, uint32_t shape_z = 0) {
    const std::string header_dict = header_to_string<T>(fortran_order, shape_x, shape_y, shape_z);

    os.write(RAFT_NUMPY_MAGIC_STRING, RAFT_NUMPY_MAGIC_STRING_LENGTH);
    // if (!os) {
    //     throw std::runtime_error("Failed to write magic string to the output stream.");
    // }
    // Use version 1.0
    char ch = 1;
    os.write(&ch, 1);
    ch = 0;
    os.write(&ch, 1);
    // os.put(1);
    // os.put(0);
    // if (!os) {
    //     throw std::runtime_error("Failed to write magic string to the output stream.");
    // }

    const std::uint32_t header_length = 118;
    std::uint8_t header_len_le16[2] = {
        static_cast<std::uint8_t>(header_length & 0xFF),
        static_cast<std::uint8_t>((header_length >> 8) & 0xFF)};
    os.write(reinterpret_cast<const char*>(header_len_le16), 2);
    // if (!os) {
    //     throw std::runtime_error("Failed to write header length to the output stream.");
    // }

    std::size_t preamble_length = RAFT_NUMPY_MAGIC_STRING_LENGTH + 2 + 2 + header_dict.length() + 1;
    // Enforce 64-byte alignment
    std::size_t padding_len = 64 - preamble_length % 64;
    std::string padding(padding_len, ' ');
    // os << header_dict << padding << "\n";
    os.write(header_dict.data(), header_dict.size());
    os.write(padding.data(), padding.size());
    char newline = '\n';
    os.write(&newline, 1);
    // if (!os) {
    //     throw std::runtime_error("Failed to write header data to the output stream.");
    // }
}

template <typename T>
void write_once(cached_ofstream& os, const T& value) {
    write_header<T>(os, false);

    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    // if (!os) {
    //     throw std::runtime_error("Failed to write the value to the output stream.");
    // }
}


// Due to data folder structure change, add new parameter "index_name", remove "vamana_suffix", "idmaps_suffix"
int DiskANN_merge(const std::string &vamana_prefix, const std::string &index_name, const std::string &idmaps_prefix,
                 const uint64_t nshards, uint32_t max_degree,
                 const std::string &output_vamana, const std::string &medoids_file, bool use_filters,
                 const std::string &labels_to_medoids_file)
{
    // Read ID maps
    std::vector<std::string> vamana_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        vamana_names[shard] = vamana_prefix + "/partition" + std::to_string(shard) + "/index/" + index_name;
        read_idmap(idmaps_prefix + "/partition" + std::to_string(shard) + "/idmap.ibin", idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
        diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, (uint32_t)shard));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    // will merge all the labels to medoids files of each shard into one
    // combined file
    if (use_filters)
    {
        std::unordered_map<uint32_t, std::vector<uint32_t>> global_label_to_medoids;

        for (size_t i = 0; i < nshards; i++)
        {
            std::ifstream mapping_reader;
            std::string map_file = vamana_names[i] + "_labels_to_medoids.txt";
            mapping_reader.open(map_file);

            std::string line, token;
            uint32_t line_cnt = 0;

            while (std::getline(mapping_reader, line))
            {
                std::istringstream iss(line);
                uint32_t cnt = 0;
                uint32_t medoid = 0;
                uint32_t label = 0;
                while (std::getline(iss, token, ','))
                {
                    token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());

                    uint32_t token_as_num = std::stoul(token);

                    if (cnt == 0)
                        label = token_as_num;
                    else
                        medoid = token_as_num;
                    cnt++;
                }
                global_label_to_medoids[label].push_back(idmaps[i][medoid]);
                line_cnt++;
            }
            mapping_reader.close();
        }

        std::ofstream mapping_writer(labels_to_medoids_file);
        assert(mapping_writer.is_open());
        for (auto iter : global_label_to_medoids)
        {
            mapping_writer << iter.first << ", ";
            auto &vec = iter.second;
            for (uint32_t idx = 0; idx < vec.size() - 1; idx++)
            {
                mapping_writer << vec[idx] << ", ";
            }
            mapping_writer << vec[vec.size() - 1] << std::endl;
        }
        mapping_writer.close();
    }

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (size_t i = 0; i < nshards; i++)
    {
        vamana_readers[i].open(vamana_names[i], BUFFER_SIZE_FOR_CACHED_IO);
        size_t expected_file_size;
        vamana_readers[i].read((char *)&expected_file_size, sizeof(uint64_t));
    }

    size_t vamana_metadata_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // expected file size + max degree +
                                                                                   // medoid_id + frozen_point info

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_vamana, BUFFER_SIZE_FOR_CACHED_IO);

    size_t merged_index_size = vamana_metadata_size; // we initialize the size of the merged index to
                                                     // the metadata size
    size_t merged_index_frozen = 0;
    merged_vamana_writer.write((char *)&merged_index_size,
                               sizeof(uint64_t)); // we will overwrite the index size at the end

    uint32_t output_width = max_degree;
    uint32_t max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
    for (auto &reader : vamana_readers)
    {
        uint32_t input_width;
        reader.read((char *)&input_width, sizeof(uint32_t));
        max_input_width = input_width > max_input_width ? input_width : max_input_width;
    }

    diskann::cout << "Max input width: " << max_input_width << ", output width: " << output_width << std::endl;

    merged_vamana_writer.write((char *)&output_width, sizeof(uint32_t));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    uint32_t nshards_u32 = (uint32_t)nshards;
    uint32_t one_val = 1;
    medoid_writer.write((char *)&nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *)&one_val, sizeof(uint32_t));

    uint64_t vamana_index_frozen = 0; // as of now the functionality to merge many overlapping vamana
                                      // indices is supported only for bulk indices without frozen point.
                                      // Hence the final index will also not have any frozen points.
    for (uint64_t shard = 0; shard < nshards; shard++)
    {
        uint32_t medoid;
        // read medoid
        vamana_readers[shard].read((char *)&medoid, sizeof(uint32_t));
        vamana_readers[shard].read((char *)&vamana_index_frozen, sizeof(uint64_t));
        assert(vamana_index_frozen == false);
        // rename medoid
        medoid = idmaps[shard][medoid];

        medoid_writer.write((char *)&medoid, sizeof(uint32_t));
        // write renamed medoid
        if (shard == (nshards - 1)) //--> uncomment if running hierarchical
            merged_vamana_writer.write((char *)&medoid, sizeof(uint32_t));
    }
    merged_vamana_writer.write((char *)&merged_index_frozen, sizeof(uint64_t));
    medoid_writer.close();

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second;
        if (cur_id < node_id)
        {
            // Gopal. random_shuffle() is deprecated.
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)max_degree);
            // write into merged ofstream
            merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
            merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
            merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                diskann::cout << "." << std::flush;
            }
            cur_id = node_id;
            nnbrs = 0;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        vamana_readers[shard_id].read((char *)&shard_nnbrs, sizeof(uint32_t));

        if (shard_nnbrs == 0)
        {
            diskann::cout << "WARNING: shard #" << shard_id << ", node_id " << node_id << " has 0 nbrs" << std::endl;
        }

        std::vector<uint32_t> shard_nhood(shard_nnbrs);
        if (shard_nnbrs > 0)
            vamana_readers[shard_id].read((char *)shard_nhood.data(), shard_nnbrs * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < shard_nnbrs; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }

    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (uint32_t)(std::min)(final_nhood.size(), (uint64_t)max_degree);
    // write into merged ofstream
    merged_vamana_writer.write((char *)&nnbrs, sizeof(uint32_t));
    if (nnbrs > 0)
    {
        merged_vamana_writer.write((char *)final_nhood.data(), nnbrs * sizeof(uint32_t));
    }
    merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    diskann::cout << "Expected size: " << merged_index_size << std::endl;

    merged_vamana_writer.reset();
    merged_vamana_writer.write((char *)&merged_index_size, sizeof(uint64_t));

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}

// In the partition step, points are handled sequentially such that within each shard, 
// the points' global id are ascending as in the complete dataset
int DiskANN_merge_sequentialShardRead(const std::string base_folder,
                const uint64_t nshards, uint32_t merge_degree, uint32_t constructed_deg,
                const std::string output_index_file,
                const std::string index_name)
{
    // Read ID maps
    std::vector<std::string> index_file_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {   
        index_file_names[shard] = base_folder + "/partition" + std::to_string(shard) + "/index/" + index_name;
        read_idmap(base_folder + "/partition" + std::to_string(shard) + "/idmap.ibin", idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", merge. degree: " << merge_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, uint32_t>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
        diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, (uint32_t)shard));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second < right.second);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    // create cached readers
    std::vector<cached_ifstream> index_readers(nshards);
    for (size_t i = 0; i < nshards; i++)
    {   
        index_readers[i].open(index_file_names[i], BUFFER_SIZE_FOR_CACHED_IO);
        char dtype_string[4];
        index_readers[i].read(dtype_string, 4);
        read_once<int>(index_readers[i]); // int version
        read_once<std::uint32_t>(index_readers[i]); // uint32_t rows
        read_once<std::uint32_t>(index_readers[i]); // uint32_t dim
        assert(constructed_deg == read_once<std::uint32_t>(index_readers[i])); // uint32_t deg
        read_once<unsigned short>(index_readers[i]); // unsigned short metric
        read_header(index_readers[i]);
    }

    size_t vamana_metadata_size =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t); // expected file size + max degree +
                                                                                   // medoid_id + frozen_point info

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_index_file, BUFFER_SIZE_FOR_CACHED_IO);
    // write header
    std::string dtype_string = "<f4";
    dtype_string.resize(4);
    merged_vamana_writer.write(dtype_string.c_str(), 4);
    int version = 4;
    write_once<int>(merged_vamana_writer, version);
    uint32_t rows = (uint32_t)nnodes;
    write_once<std::uint32_t>(merged_vamana_writer, rows);
    uint32_t dim = 0;
    write_once<std::uint32_t>(merged_vamana_writer, dim);
    uint32_t deg = merge_degree;
    write_once<std::uint32_t>(merged_vamana_writer, deg);
    unsigned short metric = 0;
    write_once<unsigned short>(merged_vamana_writer, metric);
    write_header<uint32_t>(merged_vamana_writer, false, rows, deg);

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    // uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second;
        if (cur_id < node_id)
        {
            // Gopal. random_shuffle() is deprecated.
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            assert((final_nhood.size() >= merge_degree));
            // write into merged ofstream
            merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                diskann::cout << "." << std::flush;
            }
            cur_id = node_id;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        std::vector<uint32_t> shard_nhood(constructed_deg);
        index_readers[shard_id].read((char *)shard_nhood.data(), constructed_deg * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < constructed_deg; j++)
        {
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }

    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    assert((final_nhood.size() >= merge_degree));
    // write into merged ofstream
    merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    // write pos metadata
    bool has_dataset = 0;
    write_once<bool>(merged_vamana_writer, has_dataset);

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}


// In the partition step, points are handled parallely such that within each shard, 
// the points' global id are random compared with the complete dataset.
// Thus, we need to utilize "seekg" to locate the current read position
// We use the buffered read, and may need to reload the read buffer.
int scaleGANN_merge(const std::string base_folder,
                const uint64_t nshards, uint32_t max_degree, uint32_t constructed_deg,
                const std::string output_index_file,
                const std::string index_name)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    // Read ID maps
    std::vector<std::string> index_file_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {   
        index_file_names[shard] = base_folder + "/partition" + std::to_string(shard) + "/index/" + index_name;
        read_idmap(base_folder + "/partition" + std::to_string(shard) + "/idmap.ibin", idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
        for (auto &id : idmap)
        {
            nnodes = std::max(nnodes, (size_t)id);
        }
        printf("Shard has # element %d\n", idmap.size());
        nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
        diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
        for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
        {
            size_t node_id = idmaps[shard][idx];
            node_shard.push_back(std::make_pair((uint32_t)node_id, std::make_pair((uint32_t)shard, idx)));
        }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
        return left.first < right.first || (left.first == right.first && left.second.first < right.second.first);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    auto idMapDealTime = std::chrono::high_resolution_clock::now();

    // create cached readers
    std::vector<cached_ifstream> index_readers(nshards);
    uint32_t header_size = 790; // CAGRA index header size
    for (size_t i = 0; i < nshards; i++)
    {   
        index_readers[i].open(index_file_names[i], BUFFER_SIZE_FOR_CACHED_IO);
        if(i==0){
            std::ifstream head_reader(index_file_names[i].c_str(), std::ios::binary);
            char dtype_string[4];
            head_reader.read(dtype_string, 4);
            read_once<int>(head_reader); // int version
            read_once<std::uint32_t>(head_reader); // uint32_t rows
            read_once<std::uint32_t>(head_reader); // uint32_t dim
            assert(constructed_deg == read_once<std::uint32_t>(head_reader)); // uint32_t deg
            read_once<unsigned short>(head_reader); // unsigned short metric
            read_header(head_reader);
            if (uint32_t(head_reader.tellg()) != header_size){
                header_size = uint32_t(head_reader.tellg());
                printf("updated header size is: %d\n", header_size);
            }
        }
    
    }

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_index_file, BUFFER_SIZE_FOR_CACHED_IO);
    // write header
    std::string dtype_string = "<f4";
    dtype_string.resize(4);
    merged_vamana_writer.write(dtype_string.c_str(), 4);
    int version = 4;
    write_once<int>(merged_vamana_writer, version);
    uint32_t rows = (uint32_t)nnodes;
    write_once<std::uint32_t>(merged_vamana_writer, rows);
    uint32_t dim = 0;
    write_once<std::uint32_t>(merged_vamana_writer, dim);
    uint32_t deg = max_degree;
    write_once<std::uint32_t>(merged_vamana_writer, deg);
    unsigned short metric = 0;
    write_once<unsigned short>(merged_vamana_writer, metric);
    write_header<uint32_t>(merged_vamana_writer, false, rows, deg);

    uint32_t nshards_u32 = (uint32_t)nshards;
    uint32_t one_val = 1;

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    // uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
        uint32_t node_id = id_shard.first;
        uint32_t shard_id = id_shard.second.first;
        uint32_t node_local_id = id_shard.second.second;
        if (cur_id < node_id)
        {
            // Gopal. random_shuffle() is deprecated.
            std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
            assert((final_nhood.size() >= max_degree));
            // write into merged ofstream
            merged_vamana_writer.write((char *)final_nhood.data(), max_degree * sizeof(uint32_t));
            if (cur_id % 499999 == 1)
            {
                diskann::cout << "." << std::flush;
            }
            cur_id = node_id;
            for (auto &p : final_nhood)
                nhood_set[p] = 0;
            final_nhood.clear();
        }
        // read from shard_id ifstream
        std::vector<uint32_t> shard_nhood(constructed_deg);
        std::streampos read_pos = header_size + node_local_id * constructed_deg * sizeof(uint32_t);
        index_readers[shard_id].seekg(read_pos);
        // assert(((index_readers[shard_id].tellg() + static_cast<std::streampos>(constructed_deg * sizeof(uint32_t))) < index_readers[shard_id].get_file_size()));
        index_readers[shard_id].read((char *)shard_nhood.data(), constructed_deg * sizeof(uint32_t));
        // rename nodes
        for (uint64_t j = 0; j < constructed_deg; j++)
        {   
            assert((shard_nhood[j] >= 0));
            assert((shard_nhood[j] < idmaps[shard_id].size()));
            assert(idmaps[shard_id][shard_nhood[j]] < nnodes);
            if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
            {
                nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
            }
        }
    }
    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    assert((final_nhood.size() >= max_degree));
    // write into merged ofstream
    merged_vamana_writer.write((char *)final_nhood.data(), max_degree * sizeof(uint32_t));
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();

    // write pos metadata
    bool has_dataset = 0;
    write_once<bool>(merged_vamana_writer, has_dataset);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto idMapDealDuration = std::chrono::duration_cast<std::chrono::milliseconds>(idMapDealTime - startTime);
    auto mergeReadWriteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - idMapDealTime);
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("idmap dealing duration: %lld milliseconds, merge & read $ write duration: %lld milliseconds\n", idMapDealDuration.count(), mergeReadWriteDuration.count());
    printf("overall duration: %lld milliseconds\n", overallDuration.count());

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}

// In the partition step, points are handled parallely such that within each shard, 
// the points' global id are random compared with the complete dataset.
// Thus, we need to utilize "seekg" to locate the current read position
// We don't use buffer in the read.
int scaleGANN_merge_unCachedRead(const std::string base_folder,
    const uint64_t nshards, uint32_t merge_degree, uint32_t constructed_deg,
    const std::string output_index_file,
    const std::string index_name)
{
    auto startTime = std::chrono::high_resolution_clock::now();

    // Read ID maps
    std::vector<std::string> index_file_names(nshards);
    std::vector<std::vector<uint32_t>> idmaps(nshards);
    for (uint64_t shard = 0; shard < nshards; shard++)
    {   
    index_file_names[shard] = base_folder + "/partition" + std::to_string(shard) + "/index/" + index_name;
    read_idmap(base_folder + "/partition" + std::to_string(shard) + "/idmap.ibin", idmaps[shard]);
    }

    // find max node id
    size_t nnodes = 0;
    size_t nelems = 0;
    for (auto &idmap : idmaps)
    {
    for (auto &id : idmap)
    {
    nnodes = std::max(nnodes, (size_t)id);
    }
    printf("Shard has # element %d\n", idmap.size());
    nelems += idmap.size();
    }
    nnodes++;
    diskann::cout << "# nodes: " << nnodes << ", merge. degree: " << merge_degree << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> node_shard;
    node_shard.reserve(nelems);
    for (size_t shard = 0; shard < nshards; shard++)
    {
    diskann::cout << "Creating inverse map -- shard #" << shard << std::endl;
    for (size_t idx = 0; idx < idmaps[shard].size(); idx++)
    {
    size_t node_id = idmaps[shard][idx];
    node_shard.push_back(std::make_pair((uint32_t)node_id, std::make_pair((uint32_t)shard, idx)));
    }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
    return left.first < right.first || (left.first == right.first && left.second.first < right.second.first);
    });
    diskann::cout << "Finished computing node -> shards map" << std::endl;

    auto idMapDealTime = std::chrono::high_resolution_clock::now();

    // create cached readers
    // std::vector<cached_ifstream> index_readers(nshards);
    std::vector<std::unique_ptr<std::ifstream>> index_readers(index_file_names.size());
    uint32_t header_size = 790; // CAGRA index header size
    for (size_t i = 0; i < nshards; i++)
    {   
    // index_readers[i].open(index_file_names[i], BUFFER_SIZE_FOR_CACHED_IO);
    index_readers[i] = std::make_unique<std::ifstream>(index_file_names[i], std::ios::binary);
    if (!index_readers[i]->is_open()) {
    std::cerr << "Failed to open file: " << index_file_names[i] << std::endl;
    }
    }

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_index_file, BUFFER_SIZE_FOR_CACHED_IO);
    // write header
    std::string dtype_string = "<f4";
    dtype_string.resize(4);
    merged_vamana_writer.write(dtype_string.c_str(), 4);
    int version = 4;
    write_once<int>(merged_vamana_writer, version);
    uint32_t rows = (uint32_t)nnodes;
    write_once<std::uint32_t>(merged_vamana_writer, rows);
    uint32_t dim = 0;
    write_once<std::uint32_t>(merged_vamana_writer, dim);
    uint32_t deg = merge_degree;
    write_once<std::uint32_t>(merged_vamana_writer, deg);
    unsigned short metric = 0;
    write_once<unsigned short>(merged_vamana_writer, metric);
    write_header<uint32_t>(merged_vamana_writer, false, rows, deg);

    diskann::cout << "Starting merge" << std::endl;

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<bool> nhood_set(nnodes, 0);
    std::vector<uint32_t> final_nhood;

    // uint32_t nnbrs = 0, shard_nnbrs = 0;
    uint32_t cur_id = 0;
    for (const auto &id_shard : node_shard)
    {
    uint32_t node_id = id_shard.first;
    uint32_t shard_id = id_shard.second.first;
    uint32_t node_local_id = id_shard.second.second;
    if (cur_id < node_id)
    {
    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    assert((final_nhood.size() >= merge_degree));
    // write into merged ofstream
    merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
    if (cur_id % 499999 == 1)
    {
        diskann::cout << "." << std::flush;
    }
    cur_id = node_id;
    for (auto &p : final_nhood)
        nhood_set[p] = 0;
    final_nhood.clear();
    }
    // read from shard_id ifstream
    std::vector<uint32_t> shard_nhood(constructed_deg);
    std::streampos read_pos = header_size + node_local_id * constructed_deg * sizeof(uint32_t);
    index_readers[shard_id]->seekg(read_pos);
    // assert(((index_readers[shard_id].tellg() + static_cast<std::streampos>(constructed_deg * sizeof(uint32_t))) < index_readers[shard_id].get_file_size()));
    index_readers[shard_id]->read((char *)shard_nhood.data(), constructed_deg * sizeof(uint32_t));
    // rename nodes
    for (uint64_t j = 0; j < constructed_deg; j++)
    {
    if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0)
    {
        nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
        final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
    }
    }
    }
    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    assert((final_nhood.size() >= merge_degree));
    // write into merged ofstream
    merged_vamana_writer.write((char *)final_nhood.data(), merge_degree * sizeof(uint32_t));
    for (auto &p : final_nhood)
    nhood_set[p] = 0;
    final_nhood.clear();

    // write pos metadata
    bool has_dataset = 0;
    write_once<bool>(merged_vamana_writer, has_dataset);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto idMapDealDuration = std::chrono::duration_cast<std::chrono::milliseconds>(idMapDealTime - startTime);
    auto mergeReadWriteDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - idMapDealTime);
    auto overallDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    printf("idmap dealing duration: %lld milliseconds, merge & read $ write duration: %lld milliseconds\n", idMapDealDuration.count(), mergeReadWriteDuration.count());
    printf("overall duration: %lld milliseconds\n", overallDuration.count());

    diskann::cout << "Finished merge" << std::endl;
    return 0;
}


// ============================================================================
// Pipelined merge: overlap with build via build.done markers.
// Stage 1 (per shard, runs as shards become ready):
//   sequential read of shard index -> remap neighbor IDs to global ->
//   accumulate in bounded sort buffer -> sort by global id -> flush "runs"
//   to tmp_dir.
// Stage 2 (after all shards done):
//   k-way merge across all runs by global id, dedup/shuffle/truncate to
//   merge_degree, write final index.
// Memory bounded by max_merge_workers * (idmap_shard + sort_buf).
// ============================================================================

namespace {

// Counting semaphore (std::counting_semaphore is C++20; project is C++17).
class CountSemaphore {
public:
    explicit CountSemaphore(int n) : count_(n) {}
    void acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this]{ return count_ > 0; });
        --count_;
    }
    void release() {
        std::lock_guard<std::mutex> lk(m_);
        ++count_;
        cv_.notify_one();
    }
private:
    int count_;
    std::mutex m_;
    std::condition_variable cv_;
};

// Read CAGRA header from a shard index file to determine the exact header byte
// size and verify the constructed degree.
uint64_t read_cagra_header_size_and_verify(const std::string &index_file,
                                           uint32_t expected_deg) {
    std::ifstream f(index_file.c_str(), std::ios::binary);
    if (!f.is_open()) {
        throw diskann::ANNException("Failed to open shard index: " + index_file,
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    char dtype_string[4];
    f.read(dtype_string, 4);
    read_once<int>(f);
    read_once<std::uint32_t>(f); // rows
    read_once<std::uint32_t>(f); // dim
    uint32_t deg = read_once<std::uint32_t>(f);
    if (deg != expected_deg) {
        std::stringstream ss;
        ss << "Shard index deg=" << deg << " but expected constructed_deg="
           << expected_deg << " for " << index_file;
        throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    read_once<unsigned short>(f); // metric
    read_header(f);               // trailing numpy header
    return (uint64_t)f.tellg();
}

struct ShardStage1Result {
    std::vector<std::string> run_files;
    uint32_t local_max_gid;
    uint32_t shard_size;
    long long duration_ms;
};

// Process a single shard end-to-end (Stage 1). Sequential read, remap, sort,
// flush sorted runs to tmp_dir. Returns the list of run files produced.
ShardStage1Result stage1_process_shard(
    uint32_t shard_id,
    const std::string &base_folder,
    const std::string &index_name,
    const std::string &tmp_dir,
    uint32_t constructed_deg,
    uint64_t sort_buf_bytes,
    bool cleanup_shard_index,
    bool cleanup_partition_inputs) {

    auto t0 = std::chrono::high_resolution_clock::now();

    std::string shard_dir = base_folder + "/partition" + std::to_string(shard_id);
    std::string idmap_path = shard_dir + "/idmap.ibin";
    std::string index_path = shard_dir + "/index/" + index_name;

    // Load this shard's idmap (only resident while processing this shard).
    std::vector<uint32_t> idmap;
    read_idmap(idmap_path, idmap);
    uint32_t shard_size = (uint32_t)idmap.size();

    uint32_t local_max_gid = 0;
    for (auto g : idmap) {
        if (g > local_max_gid) local_max_gid = g;
    }

    uint64_t header_size = read_cagra_header_size_and_verify(index_path, constructed_deg);

    // Sort buffer: flat layout, N records each (1 + deg) uint32_t.
    const uint64_t record_words = 1ULL + constructed_deg;
    const uint64_t record_bytes = record_words * sizeof(uint32_t);
    uint64_t records_per_chunk = sort_buf_bytes / record_bytes;
    if (records_per_chunk == 0) records_per_chunk = 1;

    std::vector<std::string> run_files;
    uint32_t run_idx = 0;

    // Scope the reader so its fd is closed before we (optionally) unlink
    // the shard index file below.
    {
    cached_ifstream index_reader;
    index_reader.open(index_path, BUFFER_SIZE_FOR_CACHED_IO);
    {
        // skip header bytes
        std::vector<char> tmp(header_size);
        index_reader.read(tmp.data(), header_size);
    }

    std::vector<uint32_t> chunk_buf;
    chunk_buf.reserve(records_per_chunk * record_words);
    std::vector<uint32_t> perm;
    perm.reserve(records_per_chunk);

    auto flush_chunk = [&]() {
        if (chunk_buf.empty()) return;
        uint64_t n_records = chunk_buf.size() / record_words;
        perm.resize(n_records);
        std::iota(perm.begin(), perm.end(), 0u);
        std::sort(perm.begin(), perm.end(),
                  [&](uint32_t a, uint32_t b) {
                      return chunk_buf[(uint64_t)a * record_words] <
                             chunk_buf[(uint64_t)b * record_words];
                  });

        std::string run_path = tmp_dir + "/run_shard" + std::to_string(shard_id)
                             + "_run" + std::to_string(run_idx) + ".bin";
        {
            cached_ofstream writer(run_path, 32ULL * 1024 * 1024);
            uint64_t nrec = n_records;
            uint32_t deg = constructed_deg;
            writer.write((const char *)&nrec, sizeof(uint64_t));
            writer.write((const char *)&deg, sizeof(uint32_t));
            for (uint64_t i = 0; i < n_records; i++) {
                const uint32_t *rec = &chunk_buf[(uint64_t)perm[i] * record_words];
                writer.write((const char *)rec, record_bytes);
            }
        }
        run_files.push_back(run_path);
        run_idx++;
        chunk_buf.clear();
        perm.clear();
    };

    std::vector<uint32_t> local_nhood(constructed_deg);
    const uint64_t local_nhood_bytes = (uint64_t)constructed_deg * sizeof(uint32_t);
    for (uint32_t local_id = 0; local_id < shard_size; local_id++) {
        index_reader.read((char *)local_nhood.data(), local_nhood_bytes);
        uint32_t gid = idmap[local_id];
        chunk_buf.push_back(gid);
        for (uint32_t j = 0; j < constructed_deg; j++) {
            uint32_t local_nb = local_nhood[j];
            assert(local_nb < shard_size);
            chunk_buf.push_back(idmap[local_nb]);
        }
        if ((uint64_t)chunk_buf.size() >= records_per_chunk * record_words) {
            flush_chunk();
        }
    }
    flush_chunk();
    } // index_reader destructed here -> fd closed

    if (cleanup_shard_index) {
        std::error_code ec;
        std::filesystem::remove(index_path, ec);
        if (ec) {
            fprintf(stderr, "[Stage1] shard %u: failed to remove %s (%s)\n",
                    shard_id, index_path.c_str(), ec.message().c_str());
        }
    }

    if (cleanup_partition_inputs) {
        std::error_code ec;
        std::filesystem::remove(idmap_path, ec);
        if (ec) {
            fprintf(stderr, "[Stage1] shard %u: failed to remove %s (%s)\n",
                    shard_id, idmap_path.c_str(), ec.message().c_str());
        }
        // Delete partition data file(s): data.u8bin / data.fbin / data.bin / etc.
        // (The dtype suffix depends on the dataset; partition step writes it as
        //  data.<suffix>, so glob the prefix.)
        std::error_code dir_ec;
        for (auto &entry : std::filesystem::directory_iterator(shard_dir, dir_ec)) {
            if (dir_ec) break;
            const std::string fname = entry.path().filename().string();
            if (fname.rfind("data.", 0) == 0) {
                std::error_code rm_ec;
                std::filesystem::remove(entry.path(), rm_ec);
                if (rm_ec) {
                    fprintf(stderr, "[Stage1] shard %u: failed to remove %s (%s)\n",
                            shard_id, entry.path().c_str(), rm_ec.message().c_str());
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    long long dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    ShardStage1Result r;
    r.run_files = std::move(run_files);
    r.local_max_gid = local_max_gid;
    r.shard_size = shard_size;
    r.duration_ms = dur;
    return r;
}

// Stage 2: k-way merge across all run files, write final index.
void stage2_kway_merge(
    const std::vector<std::string> &run_files,
    const std::string &output_index_file,
    uint32_t merge_degree,
    uint32_t constructed_deg,
    uint64_t nnodes,
    uint64_t per_run_buf_bytes) {

    const uint64_t record_words = 1ULL + constructed_deg;
    const uint64_t record_bytes = record_words * sizeof(uint32_t);

    const size_t K = run_files.size();
    std::vector<cached_ifstream> readers(K);
    std::vector<uint64_t> records_remaining(K, 0);
    std::vector<std::vector<uint32_t>> current(K, std::vector<uint32_t>(record_words));

    for (size_t i = 0; i < K; i++) {
        readers[i].open(run_files[i], per_run_buf_bytes);
        uint64_t nrec;
        uint32_t deg;
        readers[i].read((char *)&nrec, sizeof(uint64_t));
        readers[i].read((char *)&deg, sizeof(uint32_t));
        if (deg != constructed_deg) {
            std::stringstream ss;
            ss << "Run file " << run_files[i] << " has deg=" << deg
               << " but expected " << constructed_deg;
            throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        records_remaining[i] = nrec;
    }

    struct HeapEntry {
        uint32_t gid;
        size_t run_id;
    };
    auto cmp = [](const HeapEntry &a, const HeapEntry &b) {
        return a.gid > b.gid; // min-heap on gid
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> heap(cmp);

    for (size_t i = 0; i < K; i++) {
        if (records_remaining[i] > 0) {
            readers[i].read((char *)current[i].data(), record_bytes);
            records_remaining[i]--;
            heap.push({current[i][0], i});
        }
    }

    // Output writer + CAGRA header (matches scaleGANN_merge output format).
    cached_ofstream writer(output_index_file, BUFFER_SIZE_FOR_CACHED_IO);
    std::string dtype_string = "<f4";
    dtype_string.resize(4);
    writer.write(dtype_string.c_str(), 4);
    int version = 4;
    write_once<int>(writer, version);
    uint32_t rows = (uint32_t)nnodes;
    write_once<std::uint32_t>(writer, rows);
    uint32_t dim = 0;
    write_once<std::uint32_t>(writer, dim);
    write_once<std::uint32_t>(writer, merge_degree);
    unsigned short metric = 0;
    write_once<unsigned short>(writer, metric);
    write_header<uint32_t>(writer, false, rows, merge_degree);

    std::random_device rng;
    std::mt19937 urng(rng());

    std::vector<uint32_t> final_nhood;
    final_nhood.reserve((size_t)constructed_deg * 4);
    std::unordered_set<uint32_t> seen;
    seen.reserve((size_t)constructed_deg * 4);

    uint32_t cur_id = 0;

    auto flush_cur = [&]() {
        assert(final_nhood.size() >= merge_degree);
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        writer.write((const char *)final_nhood.data(),
                     (uint64_t)merge_degree * sizeof(uint32_t));
        final_nhood.clear();
        seen.clear();
    };

    while (!heap.empty()) {
        HeapEntry top = heap.top();
        heap.pop();
        uint32_t gid = top.gid;
        size_t run_id = top.run_id;

        if (gid != cur_id) {
            flush_cur();
            cur_id = gid;
        }

        const uint32_t *rec = current[run_id].data();
        for (uint32_t j = 0; j < constructed_deg; j++) {
            uint32_t n = rec[1 + j];
            if (seen.insert(n).second) {
                final_nhood.push_back(n);
            }
        }

        if (records_remaining[run_id] > 0) {
            readers[run_id].read((char *)current[run_id].data(), record_bytes);
            records_remaining[run_id]--;
            heap.push({current[run_id][0], run_id});
        }
    }
    flush_cur();

    bool has_dataset = 0;
    write_once<bool>(writer, has_dataset);
}


// ============================================================================
// Parallel Stage 2 (split-and-merge over disjoint gid ranges)
// ============================================================================

// Buffered writer that flushes via pwrite at a tracked file offset. Multiple
// instances can pwrite into the same fd as long as their ranges don't overlap.
class PwriteBufWriter {
public:
    PwriteBufWriter(int fd, uint64_t start_offset, size_t buf_bytes)
        : fd_(fd), file_offset_(start_offset), buf_(buf_bytes), buf_used_(0) {}

    PwriteBufWriter(const PwriteBufWriter&) = delete;
    PwriteBufWriter& operator=(const PwriteBufWriter&) = delete;

    ~PwriteBufWriter() {
        try { flush(); } catch (...) {}
    }

    void append(const void *data, size_t n) {
        const char *src = (const char *)data;
        while (n > 0) {
            if (buf_used_ == buf_.size()) flush();
            size_t take = std::min(n, buf_.size() - buf_used_);
            std::memcpy(buf_.data() + buf_used_, src, take);
            buf_used_ += take;
            src += take;
            n -= take;
        }
    }

    void flush() {
        if (buf_used_ == 0) return;
        size_t total = 0;
        while (total < buf_used_) {
            ssize_t w = ::pwrite(fd_, buf_.data() + total,
                                 buf_used_ - total,
                                 (off_t)(file_offset_ + total));
            if (w < 0) {
                if (errno == EINTR) continue;
                throw diskann::ANNException(
                    std::string("pwrite failed: ") + std::strerror(errno),
                    -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            total += (size_t)w;
        }
        file_offset_ += buf_used_;
        buf_used_ = 0;
    }

    uint64_t current_offset() const { return file_offset_ + buf_used_; }

private:
    int fd_;
    uint64_t file_offset_;
    std::vector<char> buf_;
    size_t buf_used_;
};


// Binary search in a run file (records sorted by gid) for the smallest record
// index i such that record[i].gid >= target_gid. Returns nrec if all gids < target.
// Each record begins with a 4-byte gid; run file layout: [u64 nrec][u32 deg][records].
uint64_t find_first_record_ge_gid(int fd, uint64_t nrec, uint32_t target_gid,
                                  uint64_t record_bytes) {
    const uint64_t run_header_bytes = sizeof(uint64_t) + sizeof(uint32_t);
    uint64_t lo = 0, hi = nrec;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint64_t off = run_header_bytes + mid * record_bytes;
        uint32_t gid_at_mid = 0;
        ssize_t r = ::pread(fd, &gid_at_mid, sizeof(uint32_t), (off_t)off);
        if (r != (ssize_t)sizeof(uint32_t)) {
            throw diskann::ANNException(
                std::string("pread failed during binary search: ") + std::strerror(errno),
                -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        if (gid_at_mid < target_gid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}


// Read [nrec, deg] from start of a run file and verify deg matches expected.
uint64_t read_run_file_header_pread(int fd, uint32_t expected_deg,
                                    const std::string &run_path) {
    uint64_t nrec = 0;
    uint32_t deg = 0;
    ssize_t r1 = ::pread(fd, &nrec, sizeof(uint64_t), 0);
    ssize_t r2 = ::pread(fd, &deg, sizeof(uint32_t), (off_t)sizeof(uint64_t));
    if (r1 != (ssize_t)sizeof(uint64_t) || r2 != (ssize_t)sizeof(uint32_t)) {
        throw diskann::ANNException(
            std::string("Failed to read run header for ") + run_path,
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (deg != expected_deg) {
        std::stringstream ss;
        ss << "Run file " << run_path << " has deg=" << deg
           << " but expected " << expected_deg;
        throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    return nrec;
}


// Compose the has_dataset trailer (numpy-style write_once<bool>) into a buffer
// without touching the file. Layout mirrors write_header<bool> + 1-byte value.
std::vector<char> compose_has_dataset_trailer(bool has_dataset) {
    std::vector<char> buf;
    auto wr = [&](const void *p, size_t n) {
        const char *cp = (const char *)p;
        buf.insert(buf.end(), cp, cp + n);
    };

    const std::string header_dict = header_to_string<bool>(false, 0, 0, 0);
    wr(RAFT_NUMPY_MAGIC_STRING, RAFT_NUMPY_MAGIC_STRING_LENGTH);
    char v1 = 1; wr(&v1, 1);
    char v0 = 0; wr(&v0, 1);
    const std::uint32_t header_length = 118;
    std::uint8_t header_len_le16[2] = {
        static_cast<std::uint8_t>(header_length & 0xFF),
        static_cast<std::uint8_t>((header_length >> 8) & 0xFF)};
    wr(header_len_le16, 2);
    std::size_t preamble_length = RAFT_NUMPY_MAGIC_STRING_LENGTH + 2 + 2
                                + header_dict.length() + 1;
    std::size_t padding_len = 64 - preamble_length % 64;
    std::string padding(padding_len, ' ');
    wr(header_dict.data(), header_dict.size());
    wr(padding.data(), padding.size());
    char newline = '\n';
    wr(&newline, 1);
    char val = has_dataset ? 1 : 0;
    wr(&val, 1);
    return buf;
}


struct RunSpan {
    uint64_t first_record;   // inclusive
    uint64_t last_record;    // exclusive
};

struct PartitionTask {
    uint32_t worker_id;
    uint32_t gid_start;        // inclusive
    uint32_t gid_end;          // exclusive
    uint64_t output_offset;    // byte offset in final output file
    std::vector<RunSpan> spans;  // per run file: which records belong to this partition
};


// Process one partition's k-way merge into the pre-allocated output file.
// Uses Vitter's Algorithm R: each gid keeps at most merge_degree unique
// neighbors with uniform M/n selection probability, no full candidate pool.
void process_partition(
    const PartitionTask &task,
    const std::vector<std::string> &run_files,
    int output_fd,
    uint32_t merge_degree,
    uint32_t constructed_deg,
    uint64_t per_run_buf_bytes,
    uint64_t output_buf_bytes) {

    const uint64_t record_words = 1ULL + constructed_deg;
    const uint64_t record_bytes = record_words * sizeof(uint32_t);
    const uint64_t run_header_bytes = sizeof(uint64_t) + sizeof(uint32_t);
    const size_t K = run_files.size();

    std::vector<cached_ifstream> readers(K);
    std::vector<uint64_t> records_remaining(K, 0);
    std::vector<std::vector<uint32_t>> current(K, std::vector<uint32_t>(record_words));

    struct HeapEntry { uint32_t gid; size_t run_id; };
    auto cmp = [](const HeapEntry &a, const HeapEntry &b) {
        return a.gid > b.gid;  // min-heap on gid
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, decltype(cmp)> heap(cmp);

    // Open readers, seek to partition start, prime heap.
    for (size_t i = 0; i < K; i++) {
        const RunSpan &span = task.spans[i];
        if (span.first_record >= span.last_record) continue;

        records_remaining[i] = span.last_record - span.first_record;
        uint64_t byte_offset = run_header_bytes + span.first_record * record_bytes;
        uint64_t span_bytes = records_remaining[i] * record_bytes;
        uint64_t cache_size = std::min(per_run_buf_bytes, span_bytes);
        if (cache_size == 0) cache_size = record_bytes;

        readers[i].open(run_files[i], cache_size);
        readers[i].seekg((std::streampos)byte_offset);

        readers[i].read((char *)current[i].data(), record_bytes);
        records_remaining[i]--;
        heap.push({current[i][0], i});
    }

    PwriteBufWriter writer(output_fd, task.output_offset, output_buf_bytes);

    // Per-gid state.
    std::random_device rd;
    std::mt19937 urng((uint32_t)(rd() ^ (task.worker_id * 0x9e3779b9u)));
    std::vector<uint32_t> reservoir;
    reservoir.reserve(merge_degree);
    std::unordered_set<uint32_t> seen;
    seen.reserve((size_t)constructed_deg * 4);
    uint64_t n_seen = 0;

    auto add_candidate = [&](uint32_t c) {
        if (!seen.insert(c).second) return;       // dedup
        n_seen++;
        if (n_seen <= (uint64_t)merge_degree) {
            reservoir.push_back(c);
        } else {
            // Vitter's R: keep with probability merge_degree / n_seen, replace
            // a uniform random slot. r in [0, n_seen) — r < merge_degree keeps.
            uint64_t r = (uint64_t)urng() % n_seen;
            if (r < (uint64_t)merge_degree) {
                reservoir[(size_t)r] = c;
            }
        }
    };

    auto write_slot = [&](uint32_t gid_for_msg) {
        if (reservoir.size() < (size_t)merge_degree) {
            std::stringstream ss;
            ss << "Stage2 worker " << task.worker_id << ": reservoir underflow "
               << "at gid=" << gid_for_msg << " (have=" << reservoir.size()
               << ", need=" << merge_degree << ")";
            throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        writer.append(reservoir.data(),
                      (size_t)merge_degree * sizeof(uint32_t));
        reservoir.clear();
        seen.clear();
        n_seen = 0;
    };

    uint32_t cur_id = task.gid_start;
    bool have_data = false;
    uint64_t flushed_count = 0;

    while (!heap.empty()) {
        HeapEntry top = heap.top();
        heap.pop();
        uint32_t gid = top.gid;
        size_t run_id = top.run_id;

        while (cur_id < gid) {
            if (!have_data) {
                std::stringstream ss;
                ss << "Stage2 worker " << task.worker_id
                   << ": no records for gid=" << cur_id
                   << " (next seen gid=" << gid << ")";
                throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            write_slot(cur_id);
            have_data = false;
            cur_id++;
            flushed_count++;
        }

        have_data = true;
        const uint32_t *rec = current[run_id].data();
        for (uint32_t j = 0; j < constructed_deg; j++) {
            add_candidate(rec[1 + j]);
        }

        if (records_remaining[run_id] > 0) {
            readers[run_id].read((char *)current[run_id].data(), record_bytes);
            records_remaining[run_id]--;
            heap.push({current[run_id][0], run_id});
        }
    }

    if (have_data) {
        write_slot(cur_id);
        have_data = false;
        cur_id++;
        flushed_count++;
    }

    if (flushed_count != (uint64_t)(task.gid_end - task.gid_start)) {
        std::stringstream ss;
        ss << "Stage2 worker " << task.worker_id << ": wrote "
           << flushed_count << " slots, expected "
           << (task.gid_end - task.gid_start);
        throw diskann::ANNException(ss.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    writer.flush();
}


// Build P partition tasks: split [0, nnodes) into roughly equal gid ranges,
// then binary-search each run file for the byte boundaries of each partition.
std::vector<PartitionTask> build_partition_tasks(
    const std::vector<std::string> &run_files,
    uint32_t constructed_deg,
    uint64_t nnodes,
    uint64_t header_size,
    uint32_t merge_degree,
    uint32_t num_workers) {

    const uint64_t record_words = 1ULL + constructed_deg;
    const uint64_t record_bytes = record_words * sizeof(uint32_t);
    const size_t K = run_files.size();

    std::vector<int> fds(K, -1);
    std::vector<uint64_t> nrec_per_run(K, 0);

    auto close_all = [&]() {
        for (size_t i = 0; i < K; i++) if (fds[i] >= 0) ::close(fds[i]);
    };

    try {
        for (size_t i = 0; i < K; i++) {
            fds[i] = ::open(run_files[i].c_str(), O_RDONLY);
            if (fds[i] < 0) {
                throw diskann::ANNException(
                    "Failed to open run file: " + run_files[i],
                    -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            nrec_per_run[i] = read_run_file_header_pread(fds[i], constructed_deg, run_files[i]);
        }

        std::vector<PartitionTask> tasks(num_workers);

        // First pass: gid_start boundary per partition; gid_end of partition p
        // equals gid_start of partition p+1, so binary-search shared.
        std::vector<std::vector<uint64_t>> boundary_records(num_workers + 1,
            std::vector<uint64_t>(K, 0));
        for (uint32_t p = 0; p <= num_workers; p++) {
            uint64_t gid_b = (nnodes * (uint64_t)p) / num_workers;
            for (size_t i = 0; i < K; i++) {
                if (p == 0) {
                    boundary_records[p][i] = 0;
                } else if (p == num_workers) {
                    boundary_records[p][i] = nrec_per_run[i];
                } else {
                    boundary_records[p][i] = find_first_record_ge_gid(
                        fds[i], nrec_per_run[i], (uint32_t)gid_b, record_bytes);
                }
            }
        }

        for (uint32_t p = 0; p < num_workers; p++) {
            uint64_t gid_start = (nnodes * (uint64_t)p) / num_workers;
            uint64_t gid_end   = (nnodes * (uint64_t)(p + 1)) / num_workers;
            tasks[p].worker_id = p;
            tasks[p].gid_start = (uint32_t)gid_start;
            tasks[p].gid_end   = (uint32_t)gid_end;
            tasks[p].output_offset = header_size
                + gid_start * (uint64_t)merge_degree * sizeof(uint32_t);
            tasks[p].spans.resize(K);
            for (size_t i = 0; i < K; i++) {
                tasks[p].spans[i].first_record = boundary_records[p][i];
                tasks[p].spans[i].last_record  = boundary_records[p + 1][i];
            }
        }

        close_all();
        return tasks;
    } catch (...) {
        close_all();
        throw;
    }
}


// Per-worker memory estimate: K readers * per_run_buf + output_buf + scratch.
uint64_t estimate_stage2_per_worker_bytes(
    size_t K, uint64_t per_run_buf_bytes, uint64_t output_buf_bytes) {
    const uint64_t small_overhead = 16ULL * 1024 * 1024;  // heap, current[], seen, etc.
    return (uint64_t)K * per_run_buf_bytes + output_buf_bytes + small_overhead;
}


// ---- Auto-tune: derive max_threads / per_run_buf / output_buf from K +
// memory_budget. memory_budget is required (the build memory ceiling, passed
// in from the script). Any user-supplied value > 0 locks that quantity; 0 =
// auto.
//
// NUMA caveat: hardware_concurrency() does not distinguish NUMA nodes, and
// the worker scheduler / allocator here are not NUMA-aware. On multi-socket
// boxes, crossing socket boundaries adds ~30-100 ns per memory access, which
// can cap real speedup well below P. If you observe this, either:
//   * pin Stage 2 to one socket via `numactl --cpunodebind=N --membind=N`
//     before launching the merge binary, or
//   * pass `stage2_max_threads = cores_per_socket` to keep workers local.
// We deliberately don't do thread pinning inside the process because it would
// also affect the host threads RAFT/CUDA/MKL spawn elsewhere.

// Best-effort: is the device holding `path` rotational (HDD)?
// Returns false for SSD / NVMe / unknown — only true when we can positively
// confirm rotational=1 on the underlying block device.
bool is_rotational_disk(const std::string &path) {
    struct stat sb;
    if (::stat(path.c_str(), &sb) != 0) return false;
    auto try_read = [](const std::string &p) -> int {
        std::ifstream f(p);
        int v = -1;
        if (f) f >> v;
        return v;
    };
    char buf[256];
    unsigned int mj = major(sb.st_dev), mn = minor(sb.st_dev);
    // Try the device directly; partitions don't own queue/, the parent does.
    snprintf(buf, sizeof(buf), "/sys/dev/block/%u:%u/queue/rotational", mj, mn);
    int r = try_read(buf);
    if (r < 0) {
        snprintf(buf, sizeof(buf), "/sys/dev/block/%u:%u/../queue/rotational", mj, mn);
        r = try_read(buf);
    }
    return r > 0;
}

struct Stage2AutoConfig {
    uint64_t memory_budget_bytes;
    uint32_t max_threads;
    uint64_t per_run_buf_bytes;
    uint64_t output_buf_bytes;
    uint32_t P;
    bool is_rotational;
    bool budget_too_tight;   // even minimums don't fit max_threads workers
};

// Derive the three auto-fillable quantities (max_threads, per_run_buf,
// output_buf) from a fixed memory budget + K + host info. The budget itself
// is taken as-is (it's the build-time memory ceiling, which is the resource
// pool Stage 2 can reclaim).
Stage2AutoConfig auto_tune_stage2_params(
    size_t K,
    uint64_t memory_budget_bytes,
    uint32_t user_max_threads,
    uint64_t user_per_run_buf,
    uint64_t user_output_buf,
    uint64_t nnodes,
    const std::string &output_path) {

    Stage2AutoConfig cfg{};
    const uint64_t MB = 1ULL << 20;
    const uint64_t scratch = 16 * MB;
    cfg.memory_budget_bytes = memory_budget_bytes;
    cfg.is_rotational = is_rotational_disk(output_path);
    cfg.budget_too_tight = false;

    if (memory_budget_bytes == 0) {
        throw diskann::ANNException(
            "Stage 2 memory_budget is 0 — script must pass build memory ceiling.",
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // (1) max_threads
    if (user_max_threads > 0) {
        cfg.max_threads = user_max_threads;
    } else {
        cfg.max_threads = std::max(1u, (uint32_t)std::thread::hardware_concurrency());
    }
    if ((uint64_t)cfg.max_threads > nnodes) cfg.max_threads = (uint32_t)nnodes;
    if (cfg.max_threads == 0) cfg.max_threads = 1;

    // (2) buffer bounds depend on disk type
    uint64_t per_run_min, per_run_default, per_run_max;
    uint64_t out_min, out_default, out_max;
    if (cfg.is_rotational) {
        per_run_min     = 8  * MB;
        per_run_default = 16 * MB;
        per_run_max     = 64 * MB;
        out_min     = 32  * MB;
        out_default = 64  * MB;
        out_max     = 256 * MB;
    } else {
        per_run_min     = 1  * MB;
        per_run_default = 8  * MB;
        per_run_max     = 32 * MB;
        out_min     = 4   * MB;
        out_default = 32  * MB;
        out_max     = 128 * MB;
    }

    bool auto_per_run = (user_per_run_buf == 0);
    bool auto_output  = (user_output_buf  == 0);
    cfg.per_run_buf_bytes = auto_per_run ? per_run_default : user_per_run_buf;
    cfg.output_buf_bytes  = auto_output  ? out_default     : user_output_buf;

    auto per_worker_now = [&]() {
        return estimate_stage2_per_worker_bytes(
            K, cfg.per_run_buf_bytes, cfg.output_buf_bytes);
    };

    uint64_t target_workers = cfg.max_threads;
    uint64_t needed = target_workers * per_worker_now();

    if (needed <= memory_budget_bytes) {
        // Slack — grow auto fields (output_buf first since it's +1 share per
        // worker; per_run_buf is +K shares, so more expensive).
        uint64_t slack = memory_budget_bytes - needed;
        if (auto_output && cfg.output_buf_bytes < out_max) {
            uint64_t extra_each = std::min(slack / target_workers,
                                           out_max - cfg.output_buf_bytes);
            cfg.output_buf_bytes += extra_each;
            slack -= extra_each * target_workers;
        }
        if (auto_per_run && cfg.per_run_buf_bytes < per_run_max
                && slack >= (uint64_t)K * target_workers) {
            uint64_t extra_each = std::min(slack / target_workers / (uint64_t)K,
                                           per_run_max - cfg.per_run_buf_bytes);
            cfg.per_run_buf_bytes += extra_each;
        }
        cfg.P = (uint32_t)target_workers;
    } else {
        // Tight — shrink auto fields. Per_run_buf has K-fold leverage so shrink
        // it first.
        if (auto_per_run) {
            uint64_t budget_per_worker = memory_budget_bytes / target_workers;
            if (budget_per_worker > cfg.output_buf_bytes + scratch) {
                uint64_t for_runs = budget_per_worker - cfg.output_buf_bytes - scratch;
                uint64_t new_per_run = for_runs / (uint64_t)K;
                cfg.per_run_buf_bytes = std::max(per_run_min,
                                                 std::min(new_per_run, per_run_max));
            } else {
                cfg.per_run_buf_bytes = per_run_min;
            }
        }
        if (per_worker_now() * target_workers > memory_budget_bytes && auto_output) {
            uint64_t budget_per_worker = memory_budget_bytes / target_workers;
            uint64_t reader_bytes = (uint64_t)K * cfg.per_run_buf_bytes;
            if (budget_per_worker > reader_bytes + scratch) {
                uint64_t for_out = budget_per_worker - reader_bytes - scratch;
                cfg.output_buf_bytes = std::max(out_min,
                                                 std::min(for_out, out_max));
            } else {
                cfg.output_buf_bytes = out_min;
            }
        }
        // Even with minimums it might not fit max_threads — drop P.
        uint64_t pw = per_worker_now();
        if (pw * target_workers > memory_budget_bytes) {
            cfg.budget_too_tight = true;
            cfg.P = (uint32_t)std::max<uint64_t>(1, memory_budget_bytes / pw);
        } else {
            cfg.P = (uint32_t)target_workers;
        }
    }

    // K=1 edge case: parallel-by-gid-range still works (all workers open the
    // single run file at different byte offsets). Heap pop is trivial; not
    // worth disabling parallelism. Page cache is shared across workers on the
    // same file — that's a win on SSD, not contention.
    // No special path; just leave P alone.

    if ((uint64_t)cfg.P > nnodes) cfg.P = (uint32_t)nnodes;
    if (cfg.P == 0) cfg.P = 1;

    return cfg;
}


// Pick P respecting BOTH memory budget and CPU thread cap. Memory always wins.
// (Kept for reference; the parallel driver now goes through auto_tune_*.)
uint32_t decide_num_workers(
    size_t K, uint64_t memory_budget_bytes, uint32_t max_threads_cap,
    uint64_t per_run_buf_bytes, uint64_t output_buf_bytes, uint64_t nnodes) {

    uint64_t per_worker = estimate_stage2_per_worker_bytes(K, per_run_buf_bytes, output_buf_bytes);
    uint32_t p_mem = (per_worker == 0)
        ? 1u
        : (uint32_t)std::max<uint64_t>(1, memory_budget_bytes / per_worker);

    uint32_t p_cpu = max_threads_cap;
    if (p_cpu == 0) {
        p_cpu = std::max(1u, (uint32_t)std::thread::hardware_concurrency());
    }

    uint32_t p = std::min(p_mem, p_cpu);
    if ((uint64_t)p > nnodes) p = (uint32_t)nnodes;
    if (p == 0) p = 1;
    return p;
}


// Parallel Stage 2 driver. Layout written to output_index_file:
//   [CAGRA preamble (header_size bytes)]
//   [nnodes * merge_degree * 4 bytes of neighbor IDs, pwrite by workers]
//   [has_dataset trailer]
void stage2_kway_merge_parallel(
    const std::vector<std::string> &run_files,
    const std::string &output_index_file,
    uint32_t merge_degree,
    uint32_t constructed_deg,
    uint64_t nnodes,
    uint64_t memory_budget_bytes,
    uint32_t max_threads_cap,
    uint64_t per_run_buf_bytes,
    uint64_t output_buf_bytes) {

    if (run_files.empty()) {
        throw diskann::ANNException("Stage2: no run files", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
    }

    // -- Phase 1: master writes CAGRA preamble (creates/truncates the file).
    {
        cached_ofstream writer(output_index_file, 32ULL * 1024 * 1024);
        std::string dtype_string = "<f4";
        dtype_string.resize(4);
        writer.write(dtype_string.c_str(), 4);
        int version = 4;
        write_once<int>(writer, version);
        uint32_t rows = (uint32_t)nnodes;
        write_once<std::uint32_t>(writer, rows);
        uint32_t dim = 0;
        write_once<std::uint32_t>(writer, dim);
        write_once<std::uint32_t>(writer, merge_degree);
        unsigned short metric = 0;
        write_once<unsigned short>(writer, metric);
        write_header<uint32_t>(writer, false, rows, merge_degree);
        // dtor flushes + closes
    }

    // Header size = current file size on disk (cached_ofstream dtor flushed).
    uint64_t header_size = 0;
    {
        struct stat sb;
        if (::stat(output_index_file.c_str(), &sb) != 0) {
            throw diskann::ANNException(
                std::string("stat() failed on output file: ") + std::strerror(errno),
                -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        header_size = (uint64_t)sb.st_size;
    }

    // -- Phase 2: pre-allocate the neighbor data region (avoids inode lock
    // contention from multiple threads extending the file).
    uint64_t data_bytes = (uint64_t)nnodes * (uint64_t)merge_degree * sizeof(uint32_t);
    uint64_t data_end = header_size + data_bytes;

    int output_fd = ::open(output_index_file.c_str(), O_WRONLY);
    if (output_fd < 0) {
        throw diskann::ANNException(
            std::string("open() failed for pwrite: ") + std::strerror(errno),
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (::fallocate(output_fd, 0, (off_t)header_size, (off_t)data_bytes) != 0) {
        // Fallback to ftruncate (sparse extend) — older/odd filesystems.
        if (::ftruncate(output_fd, (off_t)data_end) != 0) {
            int saved = errno;
            ::close(output_fd);
            throw diskann::ANNException(
                std::string("Pre-allocation failed (fallocate+ftruncate): ")
                + std::strerror(saved),
                -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    // -- Phase 3: auto-tune the four knobs, then build partition tasks.
    size_t K = run_files.size();
    Stage2AutoConfig cfg = auto_tune_stage2_params(
        K, memory_budget_bytes, max_threads_cap,
        per_run_buf_bytes, output_buf_bytes,
        nnodes, output_index_file);

    uint64_t per_worker_est = estimate_stage2_per_worker_bytes(
        K, cfg.per_run_buf_bytes, cfg.output_buf_bytes);

    printf("[Stage2-parallel] auto-tune: K=%zu, P=%u, budget=%lu MB, "
           "per_run_buf=%lu MB, output_buf=%lu MB, "
           "disk=%s, per-worker~%lu MB, hw_concurrency=%u%s\n",
           K, cfg.P,
           (unsigned long)(cfg.memory_budget_bytes / (1024 * 1024)),
           (unsigned long)(cfg.per_run_buf_bytes / (1024 * 1024)),
           (unsigned long)(cfg.output_buf_bytes / (1024 * 1024)),
           cfg.is_rotational ? "HDD(rotational)" : "SSD/unknown",
           (unsigned long)(per_worker_est / (1024 * 1024)),
           (unsigned)std::thread::hardware_concurrency(),
           cfg.budget_too_tight ? " [budget tight: P dropped below max_threads]" : "");
    fflush(stdout);

    // Use the tuned values for the rest of Stage 2.
    uint32_t P = cfg.P;
    per_run_buf_bytes = cfg.per_run_buf_bytes;
    output_buf_bytes  = cfg.output_buf_bytes;

    std::vector<PartitionTask> tasks = build_partition_tasks(
        run_files, constructed_deg, nnodes, header_size, merge_degree, P);

    // -- Phase 4: launch P workers.
    std::vector<std::thread> workers;
    workers.reserve(P);
    std::atomic<int> failed_count(0);
    std::mutex err_mtx;
    std::string first_err;
    std::mutex log_mtx;

    for (uint32_t p = 0; p < P; p++) {
        workers.emplace_back([&, p]() {
            try {
                auto t0 = std::chrono::high_resolution_clock::now();
                process_partition(tasks[p], run_files, output_fd,
                                  merge_degree, constructed_deg,
                                  per_run_buf_bytes, output_buf_bytes);
                auto t1 = std::chrono::high_resolution_clock::now();
                long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::lock_guard<std::mutex> lk(log_mtx);
                printf("[Stage2-parallel] worker %u done in %lld ms "
                       "(gid [%u, %u), %u slots)\n",
                       p, ms, tasks[p].gid_start, tasks[p].gid_end,
                       tasks[p].gid_end - tasks[p].gid_start);
                fflush(stdout);
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lk(err_mtx);
                if (first_err.empty()) first_err = e.what();
                failed_count.fetch_add(1);
                std::lock_guard<std::mutex> lk2(log_mtx);
                fprintf(stderr, "[Stage2-parallel] worker %u FAILED: %s\n", p, e.what());
                fflush(stderr);
            }
        });
    }
    for (auto &t : workers) t.join();

    if (failed_count.load() > 0) {
        ::close(output_fd);
        throw diskann::ANNException(
            "Stage2 parallel failed: " + first_err,
            -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // -- Phase 5: master appends has_dataset trailer.
    std::vector<char> trailer = compose_has_dataset_trailer(false);
    {
        size_t total = 0;
        while (total < trailer.size()) {
            ssize_t w = ::pwrite(output_fd, trailer.data() + total,
                                 trailer.size() - total,
                                 (off_t)(data_end + total));
            if (w < 0) {
                if (errno == EINTR) continue;
                int saved = errno;
                ::close(output_fd);
                throw diskann::ANNException(
                    std::string("Failed writing trailer: ") + std::strerror(saved),
                    -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            total += (size_t)w;
        }
    }

    // Ensure data lands on disk before we return.
    if (::fsync(output_fd) != 0) {
        // Not fatal in many setups; warn instead of throw.
        fprintf(stderr, "[Stage2-parallel] fsync warning: %s\n", std::strerror(errno));
    }
    ::close(output_fd);
}

} // anonymous namespace


int scaleGANN_pipelined_merge(
    const std::string &base_folder,
    const std::string &index_name,
    const std::string &output_index_file,
    const std::string &tmp_dir,
    uint64_t nshards,
    uint32_t merge_degree,
    uint32_t constructed_deg,
    uint32_t max_merge_workers,
    uint64_t sort_buf_bytes,
    int poll_interval_ms,
    bool cleanup_runs,
    bool cleanup_shard_index,
    bool cleanup_partition_inputs,
    uint64_t stage2_memory_budget_bytes,
    uint32_t stage2_max_threads,
    uint64_t stage2_per_run_buf_bytes,
    uint64_t stage2_output_buf_bytes) {

    namespace fs = std::filesystem;
    fs::create_directories(tmp_dir);

    auto pipeline_start = std::chrono::high_resolution_clock::now();
    auto last_build_done = pipeline_start;

    std::vector<bool> dispatched(nshards, false);
    std::vector<std::vector<std::string>> shard_runs(nshards);
    std::atomic<uint32_t> max_gid(0);

    CountSemaphore slots((int)max_merge_workers);
    std::vector<std::thread> threads;
    threads.reserve(nshards);
    std::mutex log_mtx;

    uint32_t completed_dispatch = 0;
    while (completed_dispatch < nshards) {
        bool dispatched_anything = false;
        for (uint32_t s = 0; s < nshards; s++) {
            if (dispatched[s]) continue;
            std::string marker = base_folder + "/partition" + std::to_string(s) + "/build.done";
            if (!fs::exists(marker)) continue;

            dispatched[s] = true;
            completed_dispatch++;
            dispatched_anything = true;
            last_build_done = std::chrono::high_resolution_clock::now();

            slots.acquire();
            threads.emplace_back([&, s]() {
                try {
                    auto r = stage1_process_shard(s, base_folder, index_name, tmp_dir,
                                                  constructed_deg, sort_buf_bytes,
                                                  cleanup_shard_index,
                                                  cleanup_partition_inputs);
                    shard_runs[s] = std::move(r.run_files);
                    uint32_t lmax = r.local_max_gid;
                    uint32_t cur = max_gid.load();
                    while (lmax > cur && !max_gid.compare_exchange_weak(cur, lmax)) {}
                    {
                        std::lock_guard<std::mutex> lk(log_mtx);
                        printf("[Stage1] shard %u: size=%u, runs=%zu, took %lld ms\n",
                               s, r.shard_size, shard_runs[s].size(), r.duration_ms);
                        fflush(stdout);
                    }
                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lk(log_mtx);
                    fprintf(stderr, "[Stage1] shard %u failed: %s\n", s, e.what());
                    fflush(stderr);
                }
                slots.release();
            });
        }
        if (completed_dispatch < nshards && !dispatched_anything) {
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        }
    }

    for (auto &t : threads) t.join();
    auto stage1_done = std::chrono::high_resolution_clock::now();

    std::vector<std::string> all_runs;
    for (auto &sr : shard_runs) {
        for (auto &r : sr) all_runs.push_back(r);
    }

    uint64_t nnodes = (uint64_t)max_gid.load() + 1;
    {
        std::lock_guard<std::mutex> lk(log_mtx);
        printf("[Stage2] Starting k-way merge: nnodes=%lu, run files=%zu\n",
               nnodes, all_runs.size());
        fflush(stdout);
    }

    stage2_kway_merge_parallel(all_runs, output_index_file, merge_degree,
                               constructed_deg, nnodes,
                               stage2_memory_budget_bytes,
                               stage2_max_threads,
                               stage2_per_run_buf_bytes,
                               stage2_output_buf_bytes);

    auto pipeline_end = std::chrono::high_resolution_clock::now();

    if (cleanup_runs) {
        for (auto &r : all_runs) {
            std::error_code ec;
            fs::remove(r, ec);
        }
    }

    auto ms = [](auto d){ return std::chrono::duration_cast<std::chrono::milliseconds>(d).count(); };
    long long t_build  = ms(last_build_done - pipeline_start);
    long long t_stage1 = ms(stage1_done     - pipeline_start);
    long long t_stage2 = ms(pipeline_end    - stage1_done);
    long long t_total  = ms(pipeline_end    - pipeline_start);
    long long t_extra  = ms(pipeline_end    - last_build_done);

    printf("\n=== Pipelined Merge Timing ===\n");
    printf("build wall-clock           (start -> last build.done): %lld ms\n", t_build);
    printf("stage1 wall-clock          (start -> all workers done): %lld ms\n", t_stage1);
    printf("stage2 wall-clock          (k-way merge)              : %lld ms\n", t_stage2);
    printf("total wall-clock                                       : %lld ms\n", t_total);
    printf("extra (= total - build, NOT overlapped)                : %lld ms\n", t_extra);

    return 0;
}
