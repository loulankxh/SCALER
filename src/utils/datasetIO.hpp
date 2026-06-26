#include <vector>
#include <string>
#include <cstddef>

uint32_t suffixToType(std::string file);

// Preprocessing helper: deletes leftover partition output from a previous run.
//   Scans base_folder and removes any "partition*" subdirectory and the
//   "centroids.bin" file. Should be called BEFORE the official timer starts —
//   self-prints its own duration but does not contribute to partition timing.
//   Returns the number of filesystem entries removed.
//
//   Why this matters: opening a multi-GB file with std::ios::trunc forces a
//   synchronous block-release + journal commit, and old cached pages compete
//   with new I/O for OS page cache. Unlinking via std::filesystem::remove_all
//   uses unlink() which is essentially async (kernel releases blocks lazily),
//   so cleanup costs are much smaller and don't pollute the timed run.
std::size_t cleanup_partition_dir(const std::string& base_folder);

template <typename T>
std::string typeToSuffix();

template <typename T>
void arrayToVector(T* arr, std::vector<std::vector<T>>& vec);

void readMetadata(const std::string& filename, uint32_t* header);

void readMetadataOneDimension(const std::string& filename, uint32_t* header);

template <typename T>
void readFile(const std::string& filename, std::vector<std::vector<T>>& data);

template <typename T>
void readFileOneDimension(const std::string& filename, std::vector<T>& data);

template <typename T>
void readDatasetPartitions(const std::string& basePath, std::vector<std::vector<std::vector<T>>>& partitions);

void readIdxMaps(const std::string idx_file, std::vector<std::vector<uint32_t>>& idx_map);

template <typename T>
void read_query(const std::string query_file,
    std::vector<std::vector<T>>& query);

void read_groundTruth(const std::string truth_file,
    std::vector<std::vector<uint32_t>>& groundTruth);



template <typename T>
void writeDatasetPartitions(const std::string& basePath, const std::vector<std::vector<std::vector<T>>>& partitions);

void writeIdxMaps(const std::string& basePath, const std::vector<std::vector<uint32_t>>& idx_map);