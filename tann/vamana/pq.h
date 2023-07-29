// Copyright 2023 The titan-search Authors.
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
#ifndef TANN_VAMANA_PQ_H_
#define TANN_VAMANA_PQ_H_

#include "tann/core/allocator.h"
#include "tann/vamana/constants.h"
#include <cstddef>
#include <cstdint>

namespace tann {
    class FixedChunkQuantizeTable {
        float *tables = nullptr; // pq_tables = float array of size [256 * ndims]
        uint64_t ndims = 0;      // ndims = true dimension of vectors
        uint64_t n_chunks = 0;
        bool use_rotation = false;
        uint32_t *chunk_offsets = nullptr;
        float *centroid = nullptr;
        float *tables_tr = nullptr; // same as pq_tables, but col-major
        float *rotmat_tr = nullptr;

    public:
        FixedChunkQuantizeTable();

        virtual ~FixedChunkQuantizeTable();

        void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks);

        uint32_t get_num_chunks();

        void preprocess_query(float *query_vec);

        // assumes pre-processed query
        void populate_chunk_distances(const float *query_vec, float *dist_vec);

        float l2_distance(const float *query_vec, uint8_t *base_vec);

        float inner_product(const float *query_vec, uint8_t *base_vec);

        // assumes no rotation is involved
        void inflate_vector(uint8_t *base_vec, float *out_vec);

        void populate_chunk_inner_products(const float *query_vec, float *dist_vec);
    };


    struct QuantizeScratch {
        float *aligned_pqtable_dist_scratch = nullptr; // MUST BE AT LEAST [256 * NCHUNKS]
        float *aligned_dist_scratch = nullptr;         // MUST BE AT LEAST tann MAX_DEGREE
        uint8_t *aligned_pq_coord_scratch = nullptr;   // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
        float *rotated_query = nullptr;
        float *aligned_query_float = nullptr;

        QuantizeScratch(size_t graph_degree, size_t aligned_dim) {
            tann::Allocator::alloc_aligned((void **) &aligned_pq_coord_scratch,
                                (size_t) graph_degree * (size_t) constants::kMaxQuantizeChunk * sizeof(uint8_t), 256);
            tann::Allocator::alloc_aligned((void **) &aligned_pqtable_dist_scratch, 256 * (size_t) constants::kMaxQuantizeChunk * sizeof(float),
                                256);
            tann::Allocator::alloc_aligned((void **) &aligned_dist_scratch, (size_t) graph_degree * sizeof(float), 256);
            tann::Allocator::alloc_aligned((void **) &aligned_query_float, aligned_dim * sizeof(float), 8 * sizeof(float));
            tann::Allocator::alloc_aligned((void **) &rotated_query, aligned_dim * sizeof(float), 8 * sizeof(float));

            memset(aligned_query_float, 0, aligned_dim * sizeof(float));
            memset(rotated_query, 0, aligned_dim * sizeof(float));
        }

        void set(size_t dim,  turbo::Span<uint8_t> data_point, const float norm = 1.0f) {
            auto query = to_span<float>(data_point);
            for (size_t d = 0; d < dim; ++d) {
                if (norm != 1.0f)
                    rotated_query[d] = aligned_query_float[d] = query[d] / norm;
                else
                    rotated_query[d] = aligned_query_float[d] = query[d];
            }
        }
    };

    class Quantize {
    public:
        static void
        aggregate_coords(const std::vector<unsigned> &ids, const uint8_t *all_coords, const uint64_t ndims,
                         uint8_t *out);

        static void pq_dist_lookup(const uint8_t *pq_ids, const size_t n_pts, const size_t pq_nchunks, const float *pq_dists,
                            std::vector<float> &dists_out);

        // Need to replace calls to these with calls to vector& based functions above
        static void
        aggregate_coords(const unsigned *ids, const uint64_t n_ids, const uint8_t *all_coords, const uint64_t ndims,
                         uint8_t *out);

        static void pq_dist_lookup(const uint8_t *pq_ids, const size_t n_pts, const size_t pq_nchunks, const float *pq_dists,
                            float *dists_out);

        TURBO_DLL static int generate_pq_pivots(const float *const train_data, size_t num_train, unsigned dim,
                                         unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps,
                                         std::string pq_pivots_path, bool make_zero_mean = false);

        TURBO_DLL static int generate_opq_pivots(const float *train_data, size_t num_train, unsigned dim, unsigned num_centers,
                                          unsigned num_pq_chunks, std::string opq_pivots_path,
                                          bool make_zero_mean = false);

        template<typename T>
        static int generate_pq_data_from_pivots(const std::string &data_file, unsigned num_centers, unsigned num_pq_chunks,
                                         const std::string &pq_pivots_path,
                                         const std::string &pq_compressed_vectors_path,
                                         bool use_opq = false);

        template<typename T>
        static void generate_disk_quantized_data(const std::string &data_file_to_use, const std::string &disk_pq_pivots_path,
                                          const std::string &disk_pq_compressed_vectors_path,
                                          const tann::MetricType compareMetric, const double p_val,
                                          size_t &disk_pq_dims);

        template<typename T>
        static void generate_quantized_data(const std::string &data_file_to_use, const std::string &pq_pivots_path,
                                     const std::string &pq_compressed_vectors_path,
                                     const tann::MetricType compareMetric,
                                     const double p_val, const uint64_t num_pq_chunks, const bool use_opq,
                                     const std::string &codebook_prefix = "");
    };
}
#endif  // TANN_VAMANA_PQ_H_
