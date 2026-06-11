#include <iostream>
#include <sys/stat.h>
#include <cassert>
#include <filesystem>
#include <type_traits>
#include <stdexcept>
#include <fstream>
#include <omp.h>
#include <filesystem>
#include <chrono>

#include "datasetIO.hpp"


std::size_t cleanup_partition_dir(const std::string& base_folder) {
    namespace fs = std::filesystem;
    // Nothing to clean if the folder doesn't exist yet (first run).
    if (!fs::exists(base_folder) || !fs::is_directory(base_folder)) {
        return 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::size_t removed = 0;
    try {
        for (const auto& entry : fs::directory_iterator(base_folder)) {
            const auto& path = entry.path();
            const std::string name = path.filename().string();
            // Match "partition*" subdirs (partition0, partition1, ...) and the
            // centroids.bin file written by scaleGANN_partitions_with_ram_budget.
            // Anything else in base_folder is left untouched (user's own files).
            bool is_partition_dir = entry.is_directory() && name.rfind("partition", 0) == 0;
            bool is_centroids = !entry.is_directory() && name == "centroids.bin";
            if (is_partition_dir || is_centroids) {
                removed += fs::remove_all(path);
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[cleanup] warning while removing old partition data: "
                  << e.what() << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (removed > 0) {
        std::cout << "[cleanup] removed " << removed
                  << " old partition entries from " << base_folder
                  << " in " << ms
                  << " ms (preprocessing, NOT included in partition timing)"
                  << std::endl;
    }
    return removed;
}


uint32_t suffixToType(std::string file){
    uint32_t dotPos = file.find_last_of('.');
    if (dotPos == std::string::npos) {
        throw std::invalid_argument("File does not have a valid suffix.");
    }

    std::string suffix = file.substr(dotPos + 1);

    if (suffix == "fbin" || suffix=="fvecs") {
        return 0;
    } else if (suffix == "ibin" || suffix=="ivecs") {
        return 1;
    } else if (suffix == "u8bin" || suffix=="bvecs") {
        return 2;
    // } else if (suffix == "i8bin") {
    //     return 3;
    } else {
        throw std::invalid_argument("Unknown file suffix: " + suffix);
    }
}

template <typename T>
std::string typeToSuffix(){
    if constexpr (std::is_same_v<T, float>) {
        return ".fbin";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return ".ibin";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return ".u8bin";
    } else {
        throw std::runtime_error("When handling IO, unrecogonized data type.");
    }
}


template <typename T>
void arrayToVector(T* arr, std::vector<std::vector<T>>& vec){
    uint32_t npts = vec.size();
    uint32_t ndim = vec[0].size();

    for(uint32_t i = 0; i < npts; i++){
        for(uint32_t j = 0; j < ndim; j++){
            vec[i][j] = arr[i*ndim + j];
            assert((std::is_same<T, float>::value) && "T must be float type");
        }
    }
}


void readMetadata(const std::string& filename, uint32_t* header){
  FILE* fp_{nullptr};
  fp_ = fopen(filename.c_str(), "r");
  if (!fp_) { throw std::runtime_error("open file failed: " + filename); }
  if (fread(header, sizeof(uint32_t), 2, fp_) != 2) {
    throw std::runtime_error("read header (rows & dims) of file failed: " + filename);
  }
  fclose(fp_);
}

void readMetadataOneDimension(const std::string& filename, uint32_t* header){
    FILE* fp_{nullptr};
    fp_ = fopen(filename.c_str(), "r");
    if (!fp_) { throw std::runtime_error("open file failed: " + filename); }
    if (fread(header, sizeof(uint32_t), 1, fp_) != 1) {
      throw std::runtime_error("read header (rows & dims) of file failed: " + filename);
    }
    fclose(fp_);
  }


template <typename T>
void readFile(const std::string& filename, std::vector<std::vector<T>>& data) {
    FILE* fp_{nullptr};
    uint32_t nrows_;
    uint32_t ndims_;
    size_t file_size;

    fp_ = fopen(filename.c_str(), "r");
    if (!fp_) { throw std::runtime_error("open file failed: " + filename); }

    struct stat statbuf;
    if (stat(filename.c_str(), &statbuf) != 0) { throw std::runtime_error("stat() failed: " + filename); }
    file_size = statbuf.st_size;

    uint32_t header[2];
    if (fread(header, sizeof(uint32_t), 2, fp_) != 2) {
      throw std::runtime_error("read header (rows & dims) of file failed: " + filename);
    }
    nrows_ = header[0];
    ndims_ = header[1];
    printf("Read file with shape -> rows and dims: %d %d\n", nrows_, ndims_);

    data.resize(nrows_);
    for(auto& row: data){
        row.resize(ndims_);
    }

    for(int i = 0; i < nrows_; i++){
        if (fread(data[i].data(), sizeof(T), ndims_, fp_) != ndims_) {
            throw std::runtime_error("fread() BinFile " + filename + " failed");
        }
    }
    fclose(fp_);
}


template <typename T>
void readFile_fromId(const std::string& filename, std::vector<std::vector<T>>& data){
    
}


template <typename T>
void readFileOneDimension(const std::string& filename, std::vector<T>& data) {
    FILE* fp_{nullptr};
    uint32_t nsize_;

    fp_ = fopen(filename.c_str(), "r");
    if (!fp_) { throw std::runtime_error("open file failed: " + filename); }

    uint32_t header[1];
    if (fread(header, sizeof(uint32_t), 1, fp_) != 1) {
      throw std::runtime_error("read header (vector size) of file failed: " + filename);
    }
    nsize_ = header[0];
    printf("Read file for one dimension vector of size: %d\n", nsize_);

    data.resize(nsize_);

    if (fread(data.data(), sizeof(T), nsize_, fp_) != nsize_) {
        throw std::runtime_error("fread() BinFile " + filename + " failed");
    }
    fclose(fp_);
}


template <typename T>
void readDatasetPartitions_fromId(const std::string& basePath, std::vector<std::vector<std::vector<T>>>& partitions){

}


template <typename T>
void readDatasetPartitions(const std::string& basePath, std::vector<std::vector<std::vector<T>>>& partitions){
    int x = 0;

    while (true) {
        std::string filePath = basePath + "/partition" + std::to_string(x) + "/data" + typeToSuffix<T>();

        if (!std::filesystem::exists(filePath)) {
            std::cout << "File " << filePath << " doesn't exist, and finish reading a new partition" << std::endl;
            break;
        }

        uint32_t header[2];
        readMetadata(filePath, header);
        uint32_t npts = header[0];
        uint32_t ndim = header[1];
        std::vector<std::vector<T>> vec(npts, std::vector<T>(ndim));
        readFile<T>(filePath, vec);
                
        partitions.push_back(vec);

        ++x;
    }
}


void readIdxMaps(const std::string idx_file, std::vector<std::vector<uint32_t>>& idx_map){
    if (!std::filesystem::exists(idx_file)) {
        std::cout << "File " << idx_file << " doesn't exist, and finish reading a new partition" << std::endl;
        return;
    }

    // uint32_t header[1];
    // readMetadataOneDimension(idx_file, header);
    // uint32_t nsize = header[0];
    // std::vector<uint32_t> idx_vec(nsize);

    std::vector<uint32_t> idx_vec(0);
    readFileOneDimension<uint32_t>(idx_file, idx_vec);            
    idx_map.push_back(idx_vec);
}


template <typename T>
void read_query(const std::string query_file,
    std::vector<std::vector<T>>& query){
        
    std::ifstream file(query_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open file " + query_file);
    }

    uint32_t num_queries = 0, dimension = 0;
    file.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dimension), sizeof(uint32_t));
    printf("There are %d queries, each has %d dimensions\n", num_queries, dimension);
    query.resize(num_queries);
    for(auto& row: query){
        row.resize(dimension);
    }

    uint32_t suffixType = suffixToType(query_file);
    if (suffixType == 0) { // "fbin"
        if constexpr (std::is_same_v<T, float>) {
            for (uint32_t i = 0; i < num_queries; ++i) {
                file.read(reinterpret_cast<char*>(query[i].data()), dimension * sizeof(float));
            }
        } else {
            throw std::runtime_error("Error: fbin file requires T to be float.");
        }
    } else if (suffixType == 1) { // "ibin"
        if constexpr (std::is_same_v<T, int32_t>) {
            for (uint32_t i = 0; i < num_queries; ++i) {
                file.read(reinterpret_cast<char*>(query[i].data()), dimension * sizeof(int32_t));
            }
        } else {
            throw std::runtime_error("Error: ibin file requires T to be int32_t.");
        }
    } else if (suffixType == 2) { // "u8bin"
        if constexpr (std::is_same_v<T, uint8_t>) {
            for (uint32_t i = 0; i < num_queries; ++i) {
                file.read(reinterpret_cast<char*>(query[i].data()), dimension * sizeof(uint8_t));
            }
        } else {
            throw std::runtime_error("Error: u8bin file requires T to be uint8_t.");
        }
    } else {
        throw std::invalid_argument("Unknown file suffix: " + suffixType);
    }

}


void read_groundTruth(const std::string truth_file,
    std::vector<std::vector<uint32_t>>& groundTruth){

    std::ifstream file(truth_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open file " + truth_file);
    }

    uint32_t num_queries = 0, num_truth = 0;
    file.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_truth), sizeof(uint32_t));
    printf("There are %d queries, each has %d truth values\n", num_queries, num_truth);
    groundTruth.resize(num_queries);
    for(auto& row: groundTruth){
        row.resize(num_truth);
    }

    for (uint32_t i = 0; i < num_queries; ++i) {
        file.read(reinterpret_cast<char*>(groundTruth[i].data()), num_truth * sizeof(int32_t));
    }

}




template <typename T>
void writeDatasetPartitions(const std::string& basePath, const std::vector<std::vector<std::vector<T>>>& partitions){
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < partitions.size(); ++i) {
        std::string dirPath = basePath + "/partition" + std::to_string(i);
        if (!std::filesystem::exists(dirPath)) {
            if (std::filesystem::create_directories(dirPath)) {
                std::cout << "Dir doesn't exists but success to create such dir: " << dirPath << std::endl;
            } else {
                std::cerr << "Dir doesn't exists and fail to create such dir: " << dirPath << std::endl;
            }
        }
        std::string filePath = dirPath + "/data" + typeToSuffix<T>();
        FILE* file = fopen(filePath.c_str(), "wb");
        if (!file) {
            perror(("Error opening file: " + filePath).c_str());
            continue;
        }

        const auto& matrix = partitions[i];
        uint32_t rows = matrix.size();
        uint32_t cols = rows > 0 ? matrix[0].size() : 0;

        fwrite(&rows, sizeof(uint32_t), 1, file);
        fwrite(&cols, sizeof(uint32_t), 1, file);

        for (const auto& row : matrix) {
            if (row.size() != cols) {
                std::cerr << "Error: Inconsistent row size in matrix " << i << std::endl;
                break;
            }
            fwrite(row.data(), sizeof(T), row.size(), file);
        }

        fclose(file);
        std::cout << "Successfully wrote to " << filePath << std::endl;
    }
}


void writeIdxMaps(const std::string& basePath, const std::vector<std::vector<uint32_t>>& idx_map){
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < idx_map.size(); ++i) {
        std::string dirPath = basePath + "/partition" + std::to_string(i);
        if (!std::filesystem::exists(dirPath)) {
            if (std::filesystem::create_directories(dirPath)) {
                std::cout << "Dir doesn't exists but success to create such dir: " << dirPath << std::endl;
            } else {
                std::cerr << "Dir doesn't exists and fail to create such dir: " << dirPath << std::endl;
            }
        }
        std::string filePath = dirPath + "/idmap.ibin";
        FILE* file = fopen(filePath.c_str(), "wb");
        if (!file) {
            perror(("Error opening file: " + filePath).c_str());
            continue;
        }

        const std::vector<uint32_t>& matrix = idx_map[i];
        uint32_t size = matrix.size();

        fwrite(&size, sizeof(uint32_t), 1, file);


        fwrite(matrix.data(), sizeof(uint32_t), size, file);


        fclose(file);
        std::cout << "Successfully wrote to " << filePath << std::endl;
    }
}



template std::string typeToSuffix<float>();
template std::string typeToSuffix<uint32_t>();
template std::string typeToSuffix<uint8_t>();

template void arrayToVector<float>(float* arr, std::vector<std::vector<float>>& vec);
template void arrayToVector<uint32_t>(uint32_t* arr, std::vector<std::vector<uint32_t>>& vec);
template void arrayToVector<uint8_t>(uint8_t* arr, std::vector<std::vector<uint8_t>>& vec);

template void readFile<float>(const std::string& filename, std::vector<std::vector<float>>& data);
template void readFile<uint32_t>(const std::string& filename, std::vector<std::vector<uint32_t>>& data);
template void readFile<uint8_t>(const std::string& filename, std::vector<std::vector<uint8_t>>& data);

template void readFileOneDimension<float>(const std::string& filename, std::vector<float>& data);
template void readFileOneDimension<uint32_t>(const std::string& filename, std::vector<uint32_t>& data);
template void readFileOneDimension<uint8_t>(const std::string& filename, std::vector<uint8_t>& data);


template void readDatasetPartitions<float>(const std::string& basePath, std::vector<std::vector<std::vector<float>>>& partitions);
template void readDatasetPartitions<uint32_t>(const std::string& basePath, std::vector<std::vector<std::vector<uint32_t>>>& partitions);
template void readDatasetPartitions<uint8_t>(const std::string& basePath, std::vector<std::vector<std::vector<uint8_t>>>& partitions);

template void read_query<float>(const std::string query_file, std::vector<std::vector<float>>& query);
template void read_query<uint32_t>(const std::string query_file, std::vector<std::vector<uint32_t>>& query);
template void read_query<uint8_t>(const std::string query_file, std::vector<std::vector<uint8_t>>& query);


template void writeDatasetPartitions<float>(const std::string& basePath, const std::vector<std::vector<std::vector<float>>>& partitions);
template void writeDatasetPartitions<uint32_t>(const std::string& basePath, const std::vector<std::vector<std::vector<uint32_t>>>& partitions);
template void writeDatasetPartitions<uint8_t>(const std::string& basePath, const std::vector<std::vector<std::vector<uint8_t>>>& partitions);