#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Optimized CPU k-means (Lloyd) for scaleGANN.
//
// Drop-in replacement for DiskANN's `kmeans::run_lloyds`, with the same
// signature and identical mathematical semantics (FP results may differ by
// the last 1-2 ULPs due to parallel reduction order, which is also true for
// the original — k-means is robust to this).
//
// Three CPU-side optimizations vs DiskANN's reference implementation:
//
//   [FIX 1] No `#pragma omp critical` in the inverted-index update path.
//           The reference serializes 8.4M push_back calls into a critical
//           section, collapsing 80 threads down to 1. We use per-thread
//           local lists and merge after the parallel loop.
//
//   [FIX 2] Centroid update is parallelized over POINTS (not over the K
//           clusters). With K=10 and T=80 threads, the reference leaves
//           70 threads idle. We use per-thread (cluster, dim) accumulators
//           and reduce after the loop, keeping all threads busy.
//
//   [FIX 3] The two "broadcast" SGEMMs that initialize the distance matrix
//           (|x|^2 to each column, |c|^2 to each row) are replaced by a
//           single OMP loop — rank-1 broadcasts via SGEMM carry pure
//           dispatch overhead. The third SGEMM (the actual -2 X·C^T) is
//           kept since it is the genuine matrix-matrix work.
//
// What is NOT changed:
//   - Lloyd's algorithm itself
//   - Early-stop criterion (still 1e-5 relative residual change)
//   - The kmeans++ initialization (still call DiskANN's
//     `kmeans::kmeanspp_selecting_pivots` before this)

namespace kmeans_cpu {

// Same signature as DiskANN's `kmeans::run_lloyds`.
//   data: row-major num_points × dim, float
//   centers: row-major num_centers × dim, float — in/out (init must be filled
//            by caller, e.g. via kmeans++)
//   max_reps: max Lloyd iterations
//   closest_docs: optional output, num_centers vectors of point ids
//   closest_center: optional output, num_points uint32_t
// Returns the final residual (sum of squared distances to assigned center).
float run_lloyds_opt(float *data,
                     std::size_t num_points,
                     std::size_t dim,
                     float *centers,
                     std::size_t num_centers,
                     std::size_t max_reps,
                     std::vector<std::size_t> *closest_docs,
                     std::uint32_t *closest_center);

}  // namespace kmeans_cpu
