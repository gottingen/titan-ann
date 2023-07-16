// Copyright 2023 The Turbo Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "tann_cli.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl/mkl.h>
#include <unordered_map>
#include "tann/tsl/robin_map.h"
#include "tann/tsl/robin_set.h"
#include "tann_cli.h"


namespace detail {
    void
    compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const uint64_t dim) {
        assert(points_l2sq != NULL);
#pragma omp parallel for schedule(static, 65536)
        for (int64_t d = 0; d < num_points; ++d)
            points_l2sq[d] = cblas_sdot((int64_t) dim, matrix + (ptrdiff_t) d * (ptrdiff_t) dim, 1,
                                        matrix + (ptrdiff_t) d * (ptrdiff_t) dim, 1);
    }

    void distsq_to_points(const size_t dim,
                          float *dist_matrix, // Col Major, cols are queries, rows are points
                          size_t npoints, const float *const points,
                          const float *const points_l2sq, // points in Col major
                          size_t nqueries, const float *const queries,
                          const float *const queries_l2sq, // queries in Col major
                          float *ones_vec)          // Scratchspace of num_data size and init to 1.0
    {
        bool ones_vec_alloc = false;
        if (ones_vec == NULL) {
            ones_vec = new float[nqueries > npoints ? nqueries : npoints];
            std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float) 1.0);
            ones_vec_alloc = true;
        }
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float) -2.0, points, dim, queries,
                    dim,
                    (float) 0.0, dist_matrix, npoints);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float) 1.0, points_l2sq, npoints,
                    ones_vec, nqueries, (float) 1.0, dist_matrix, npoints);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, npoints, nqueries, 1, (float) 1.0, ones_vec, npoints,
                    queries_l2sq, nqueries, (float) 1.0, dist_matrix, npoints);
        if (ones_vec_alloc)
            delete[] ones_vec;
    }

    void inner_prod_to_points(const size_t dim,
                              float *dist_matrix, // Col Major, cols are queries, rows are points
                              size_t npoints, const float *const points, size_t nqueries, const float *const queries,
                              float *ones_vec) // Scratchspace of num_data size and init to 1.0
    {
        bool ones_vec_alloc = false;
        if (ones_vec == NULL) {
            ones_vec = new float[nqueries > npoints ? nqueries : npoints];
            std::fill_n(ones_vec, nqueries > npoints ? nqueries : npoints, (float) 1.0);
            ones_vec_alloc = true;
        }
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, npoints, nqueries, dim, (float) -1.0, points, dim, queries,
                    dim,
                    (float) 0.0, dist_matrix, npoints);

        if (ones_vec_alloc)
            delete[] ones_vec;
    }

    void exact_knn(const size_t dim, const size_t k,
                   size_t *const closest_points,     // k * num_queries preallocated, col
            // major, queries columns
                   float *const dist_closest_points, // k * num_queries
            // preallocated, Dist to
            // corresponding closes_points
                   size_t npoints,
                   float *points_in, // points in Col major
                   size_t nqueries, float *queries_in,
                   tann::Metric metric) // queries in Col major
    {
        float *points_l2sq = new float[npoints];
        float *queries_l2sq = new float[nqueries];
        compute_l2sq(points_l2sq, points_in, npoints, dim);
        compute_l2sq(queries_l2sq, queries_in, nqueries, dim);

        float *points = points_in;
        float *queries = queries_in;

        if (metric == tann::Metric::COSINE) { // we convert cosine distance as
            // normalized L2 distnace
            points = new float[npoints * dim];
            queries = new float[nqueries * dim];
#pragma omp parallel for schedule(static, 4096)
            for (int64_t i = 0; i < (int64_t) npoints; i++) {
                float norm = std::sqrt(points_l2sq[i]);
                if (norm == 0) {
                    norm = std::numeric_limits<float>::epsilon();
                }
                for (uint32_t j = 0; j < dim; j++) {
                    points[i * dim + j] = points_in[i * dim + j] / norm;
                }
            }

#pragma omp parallel for schedule(static, 4096)
            for (int64_t i = 0; i < (int64_t) nqueries; i++) {
                float norm = std::sqrt(queries_l2sq[i]);
                if (norm == 0) {
                    norm = std::numeric_limits<float>::epsilon();
                }
                for (uint32_t j = 0; j < dim; j++) {
                    queries[i * dim + j] = queries_in[i * dim + j] / norm;
                }
            }
            // recalculate norms after normalizing, they should all be one.
            compute_l2sq(points_l2sq, points, npoints, dim);
            compute_l2sq(queries_l2sq, queries, nqueries, dim);
        }

        std::cout << "Going to compute " << k << " NNs for " << nqueries << " queries over " << npoints << " points in "
                  << dim << " dimensions using";
        if (metric == tann::Metric::INNER_PRODUCT)
            std::cout << " MIPS ";
        else if (metric == tann::Metric::COSINE)
            std::cout << " Cosine ";
        else
            std::cout << " L2 ";
        std::cout << "distance fn. " << std::endl;

        size_t q_batch_size = (1 << 9);
        float *dist_matrix = new float[(size_t) q_batch_size * (size_t) npoints];

        for (size_t b = 0; b < div_round_up(nqueries, q_batch_size); ++b) {
            int64_t q_b = b * q_batch_size;
            int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

            if (metric == tann::Metric::L2 || metric == tann::Metric::COSINE) {
                distsq_to_points(dim, dist_matrix, npoints, points, points_l2sq, q_e - q_b,
                                 queries + (ptrdiff_t) q_b * (ptrdiff_t) dim, queries_l2sq + q_b);
            } else {
                inner_prod_to_points(dim, dist_matrix, npoints, points, q_e - q_b,
                                     queries + (ptrdiff_t) q_b * (ptrdiff_t) dim);
            }
            std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;

#pragma omp parallel for schedule(dynamic, 16)
            for (long long q = q_b; q < q_e; q++) {
                maxPQIFCS point_dist;
                for (size_t p = 0; p < k; p++)
                    point_dist.emplace(p, dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints]);
                for (size_t p = k; p < npoints; p++) {
                    if (point_dist.top().second >
                        dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints])
                        point_dist.emplace(p, dist_matrix[(ptrdiff_t) p + (ptrdiff_t) (q - q_b) * (ptrdiff_t) npoints]);
                    if (point_dist.size() > k)
                        point_dist.pop();
                }
                for (ptrdiff_t l = 0; l < (ptrdiff_t) k; ++l) {
                    closest_points[(ptrdiff_t) (k - 1 - l) + (ptrdiff_t) q * (ptrdiff_t) k] = point_dist.top().first;
                    dist_closest_points[(ptrdiff_t) (k - 1 - l) +
                                        (ptrdiff_t) q * (ptrdiff_t) k] = point_dist.top().second;
                    point_dist.pop();
                }
                assert(std::is_sorted(dist_closest_points + (ptrdiff_t) q * (ptrdiff_t) k,
                                      dist_closest_points + (ptrdiff_t) (q + 1) * (ptrdiff_t) k));
            }
            std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
        }

        delete[] dist_matrix;

        delete[] points_l2sq;
        delete[] queries_l2sq;

        if (metric == tann::Metric::COSINE) {
            delete[] points;
            delete[] queries;
        }
    }

    void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim) {
        size_t read_blk_size = 64 * 1024 * 1024;
        tann::cached_ifstream reader(bin_file, read_blk_size);
        tann::cout << "Reading truthset file " << bin_file.c_str() << " ..." << std::endl;
        size_t actual_file_size = reader.get_file_size();

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (uint32_t) npts_i32;
        dim = (uint32_t) dim_i32;

        tann::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "... " << std::endl;

        int truthset_type = -1; // 1 means truthset has ids and distances, 2 means
        // only ids, -1 is error
        size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_with_dists)
            truthset_type = 1;

        size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_just_ids)
            truthset_type = 2;

        if (truthset_type == -1) {
            std::stringstream stream;
            stream << "Error. File size mismatch. File should have bin format, with "
                      "npts followed by ngt followed by npts*ngt ids and optionally "
                      "followed by npts*ngt distance values; actual size: "
                   << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
                   << expected_file_size_just_ids;
            tann::cout << stream.str();
            throw tann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        ids = new uint32_t[npts * dim];
        reader.read((char *) ids, npts * dim * sizeof(uint32_t));

        if (truthset_type == 1) {
            dists = new float[npts * dim];
            reader.read((char *) dists, npts * dim * sizeof(float));
        }
    }
}
namespace tann {
    bool verbose;
}