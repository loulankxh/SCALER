#include <string>
#include <cstdint>

int DiskANN_merge(const std::string &vamana_prefix, const std::string &index_name, const std::string &idmaps_prefix,
    const uint64_t nshards, uint32_t max_degree,
    const std::string &output_vamana, const std::string &medoids_file, bool use_filters,
    const std::string &labels_to_medoids_file);

int scaleGANN_merge(const std::string base_folder,
    const uint64_t nshards, uint32_t max_degree, uint32_t constructed_deg,
    const std::string output_index_file,
    const std::string index_name);

// Pipelined merge that overlaps with build.
// Watches for ${base_folder}/partition<i>/build.done markers; processes each
// shard's index file (Stage 1: sequential read + remap + external sort to runs)
// as soon as it appears, with at most max_merge_workers running concurrently.
// After all shards are done, runs a k-way merge (Stage 2) across all runs to
// produce the final merged index.
// Stage 2 is parallel: split [0, nnodes) into P workers; each does its own
// K-way merge over the corresponding gid range; each pwrites into a shared
// pre-allocated output file at its byte offset.
//
//   cleanup_shard_index: if true, delete each shard's index file
//     (partition<i>/index/<index_name>) as soon as its Stage 1 finishes. The
//     run files contain the same neighborhood data already remapped to global
//     ids, so Stage 2 never reads the shard index again.
//   cleanup_partition_inputs: if true, delete each shard's partition inputs
//     (partition<i>/data.<suffix>, partition<i>/idmap.ibin) as soon as its
//     Stage 1 finishes. The idmap is fully loaded into RAM before its Stage 1
//     starts; the data file is build-only and not read by the merge pipeline.
//   stage2_memory_budget_bytes: total RAM budget for Stage 2 (Stage 1 has
//     ended, so this can be the whole input-memory ceiling). Used to bound P.
//   stage2_max_threads: CPU-side cap on workers; 0 means hardware_concurrency().
//   stage2_per_run_buf_bytes: per-reader read cache (default 8 MB if 0).
//   stage2_output_buf_bytes: per-worker pwrite buffer (default 32 MB if 0).
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
    uint64_t stage2_output_buf_bytes);