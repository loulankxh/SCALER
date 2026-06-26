#include "kmeans_cpu.h"

#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>

#include <omp.h>
#include <mkl.h>

// We reuse two utilities from DiskANN as-is (they are already efficient):
//   - math_utils::compute_vecs_l2sq  (computes ||v||^2 per row, OMP+MKL)
//   - math_utils::calc_distance      (scalar L2-sq distance, used for residual)
#include "../../DiskANN/include/math_utils.h"

namespace {

// -----------------------------------------------------------------------------
// [FIX 3] Build the squared-distance matrix without the two broadcast SGEMMs.
//   dist[i][j] = ||x_i||^2 + ||c_j||^2 - 2 * x_i · c_j
//
// Reference does this as 3 SGEMM calls; the first two are pure broadcasts
// (rank-1 outer products with vectors of ones) and carry MKL dispatch overhead.
// We replace them with one OMP loop, then keep the real X·C^T SGEMM.
// -----------------------------------------------------------------------------
void build_dist_matrix(const float *data, std::size_t num_points, std::size_t dim,
                       const float *centers, std::size_t num_centers,
                       const float *docs_l2sq, const float *centers_l2sq,
                       float *dist_matrix)
{
    // dist_matrix[i][j] = docs_l2sq[i] + centers_l2sq[j]
    // (Parallel over i; the inner num_centers loop is small and vectorizable.)
    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < (std::int64_t)num_points; i++) {
        float di = docs_l2sq[i];
        float *row = dist_matrix + i * num_centers;
        for (std::size_t j = 0; j < num_centers; j++) {
            row[j] = di + centers_l2sq[j];
        }
    }

    // dist_matrix += -2 * X * C^T   (beta=1, alpha=-2)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (MKL_INT)num_points, (MKL_INT)num_centers, (MKL_INT)dim,
                -2.0f,
                data,    (MKL_INT)dim,
                centers, (MKL_INT)dim,
                1.0f,
                dist_matrix, (MKL_INT)num_centers);
}

// -----------------------------------------------------------------------------
// [FIX 1] argmin + inverted-index without `#pragma omp critical`.
//   Each thread accumulates its own inverted-index lists; we merge after the
//   parallel loop. Uses static schedule so each thread covers a contiguous
//   range of i, which makes per-thread lists naturally sorted by point id.
// -----------------------------------------------------------------------------
void argmin_and_invert(const float *dist_matrix,
                       std::size_t num_points, std::size_t num_centers,
                       std::uint32_t *closest_center,
                       std::vector<std::size_t> *closest_docs /* may be NULL */)
{
    const int max_threads = omp_get_max_threads();

    // Per-thread local inverted-index lists.
    // Only allocate the outer vector if we actually need inverted index.
    std::vector<std::vector<std::vector<std::size_t>>> local_inv;
    if (closest_docs) {
        local_inv.assign(max_threads,
                         std::vector<std::vector<std::size_t>>(num_centers));
    }

    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < (std::int64_t)num_points; i++) {
        const float *row = dist_matrix + i * num_centers;
        float minv = std::numeric_limits<float>::max();
        std::uint32_t mini = 0;
        for (std::size_t j = 0; j < num_centers; j++) {
            if (row[j] < minv) { minv = row[j]; mini = (std::uint32_t)j; }
        }
        closest_center[i] = mini;
        if (closest_docs) {
            int tid = omp_get_thread_num();
            local_inv[tid][mini].push_back((std::size_t)i);
        }
    }

    if (!closest_docs) return;

    // Merge per-thread lists into the caller-provided closest_docs.
    // Parallelize over clusters so different shards don't contend.
    // First, clear the destination lists (caller may have stale data from
    // previous iter).
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t c = 0; c < (std::int64_t)num_centers; c++) {
        // Compute total size and reserve once to avoid repeated growth.
        std::size_t total = 0;
        for (int t = 0; t < max_threads; t++) total += local_inv[t][c].size();
        closest_docs[c].clear();
        closest_docs[c].reserve(total);
        for (int t = 0; t < max_threads; t++) {
            auto &src = local_inv[t][c];
            closest_docs[c].insert(closest_docs[c].end(), src.begin(), src.end());
        }
    }
}

// -----------------------------------------------------------------------------
// [FIX 2] Update centers with per-thread (cluster × dim) accumulators, then
//   reduce. The reference parallelizes over the K clusters, which leaves
//   T - K threads idle when K < T (e.g. K=10, T=80). We parallelize over the
//   num_points dimension instead.
//
// Behavioral parity with the original:
//   - Empty clusters (count == 0) → center is reset to all-zeros (matches the
//     reference, which does `memset(centers, 0)` first and never overwrites
//     empty clusters). We reproduce this by clearing centers up front and
//     skipping the divide for empty clusters.
// -----------------------------------------------------------------------------
void update_centers(const float *data, std::size_t num_points, std::size_t dim,
                    const std::uint32_t *closest_center,
                    float *centers, std::size_t num_centers)
{
    const int max_threads = omp_get_max_threads();
    // Per-thread sums, in doubles to match the reference's accumulation type.
    // Layout: tsum[t] is a flat (num_centers * dim) array.
    std::vector<std::vector<double>> tsum(max_threads,
                                          std::vector<double>(num_centers * dim, 0.0));
    std::vector<std::vector<std::size_t>> tcnt(max_threads,
                                                std::vector<std::size_t>(num_centers, 0));

    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < (std::int64_t)num_points; i++) {
        int tid = omp_get_thread_num();
        std::uint32_t c = closest_center[i];
        const float *x = data + (std::size_t)i * dim;
        double *s = tsum[tid].data() + (std::size_t)c * dim;
        for (std::size_t d = 0; d < dim; d++) s[d] += (double)x[d];
        tcnt[tid][c]++;
    }

    // Clear centers first — empty clusters end up at the origin, matching
    // the reference behavior (memset before per-cluster update).
    std::memset(centers, 0, sizeof(float) * num_centers * dim);

    #pragma omp parallel for schedule(static)
    for (std::int64_t c = 0; c < (std::int64_t)num_centers; c++) {
        std::size_t total = 0;
        for (int t = 0; t < max_threads; t++) total += tcnt[t][c];
        if (total == 0) continue;   // leave centers[c] == 0 (parity with original)
        float *out = centers + (std::size_t)c * dim;
        for (std::size_t d = 0; d < dim; d++) {
            double s = 0.0;
            for (int t = 0; t < max_threads; t++) s += tsum[t][c * dim + d];
            out[d] = (float)(s / (double)total);
        }
    }
}

// -----------------------------------------------------------------------------
// Residual = sum over points of squared distance to assigned center.
// Same chunked-with-padding pattern as the reference, just inlined here.
// -----------------------------------------------------------------------------
double compute_residual(const float *data, std::size_t num_points, std::size_t dim,
                        const float *centers,
                        const std::uint32_t *closest_center)
{
    constexpr std::size_t BUF_PAD = 32;
    constexpr std::size_t CHUNK_SIZE = 2 * 8192;
    std::size_t nchunks = (num_points + CHUNK_SIZE - 1) / CHUNK_SIZE;
    std::vector<float> residuals(nchunks * BUF_PAD, 0.0f);

    #pragma omp parallel for schedule(static, 32)
    for (std::int64_t chunk = 0; chunk < (std::int64_t)nchunks; chunk++) {
        std::size_t beg = chunk * CHUNK_SIZE;
        std::size_t end = std::min(num_points, beg + CHUNK_SIZE);
        float local = 0.0f;
        for (std::size_t d = beg; d < end; d++) {
            local += math_utils::calc_distance(
                const_cast<float *>(data + d * dim),
                const_cast<float *>(centers + (std::size_t)closest_center[d] * dim),
                dim);
        }
        residuals[chunk * BUF_PAD] = local;
    }

    double total = 0.0;
    for (std::size_t c = 0; c < nchunks; c++) total += residuals[c * BUF_PAD];
    return total;
}

// -----------------------------------------------------------------------------
// One Lloyd iteration: assignment + center update + residual.
// `docs_l2sq` is computed once outside and passed in (data never changes).
// `centers_l2sq` is computed here since centers change each iter.
// `dist_matrix` is a caller-provided scratch of num_points * num_centers floats.
// -----------------------------------------------------------------------------
float lloyds_iter(float *data, std::size_t num_points, std::size_t dim,
                  float *centers, std::size_t num_centers,
                  const float *docs_l2sq,
                  float *dist_matrix,
                  std::vector<std::size_t> *closest_docs,
                  std::uint32_t *closest_center)
{
    std::vector<float> centers_l2sq(num_centers);
    math_utils::compute_vecs_l2sq(centers_l2sq.data(), centers, num_centers, dim);

    // [FIX 3] dist matrix without broadcast SGEMMs.
    build_dist_matrix(data, num_points, dim, centers, num_centers,
                      docs_l2sq, centers_l2sq.data(), dist_matrix);

    // [FIX 1] argmin + inverted-index without critical.
    argmin_and_invert(dist_matrix, num_points, num_centers,
                      closest_center, closest_docs);

    // [FIX 2] centroid update parallel over points.
    update_centers(data, num_points, dim, closest_center, centers, num_centers);

    return (float)compute_residual(data, num_points, dim, centers, closest_center);
}

}  // anonymous namespace

namespace kmeans_cpu {

float run_lloyds_opt(float *data,
                     std::size_t num_points,
                     std::size_t dim,
                     float *centers,
                     std::size_t num_centers,
                     std::size_t max_reps,
                     std::vector<std::size_t> *closest_docs,
                     std::uint32_t *closest_center)
{
    // Caller may pass NULL for the result buffers — allocate locally then.
    bool owns_closest_docs = false;
    bool owns_closest_center = false;
    if (closest_docs == nullptr) {
        closest_docs = new std::vector<std::size_t>[num_centers];
        owns_closest_docs = true;
    }
    if (closest_center == nullptr) {
        closest_center = new std::uint32_t[num_points];
        owns_closest_center = true;
    }

    // Pre-compute data L2-sq once (data is invariant across iterations).
    std::vector<float> docs_l2sq(num_points);
    math_utils::compute_vecs_l2sq(docs_l2sq.data(), data, num_points, dim);

    // Scratch distance matrix, reused across iters. num_points × num_centers
    // floats. For 8.4M × 10 = 320 MB — sizable but matches DiskANN's footprint.
    std::vector<float> dist_matrix((std::size_t)num_points * num_centers);

    float residual = std::numeric_limits<float>::max();
    float old_residual;
    for (std::size_t it = 0; it < max_reps; it++) {
        old_residual = residual;
        residual = lloyds_iter(data, num_points, dim, centers, num_centers,
                               docs_l2sq.data(), dist_matrix.data(),
                               closest_docs, closest_center);
        if (it != 0 && ((old_residual - residual) / residual) < 0.00001f) {
            std::cout << "[kmeans_cpu] residual converged at iter " << it
                      << " (old=" << old_residual << ", new=" << residual
                      << ") — early termination" << std::endl;
            break;
        }
        if (residual < std::numeric_limits<float>::epsilon()) break;
    }

    if (owns_closest_docs)   delete[] closest_docs;
    if (owns_closest_center) delete[] closest_center;
    return residual;
}

}  // namespace kmeans_cpu
