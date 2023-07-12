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
#include "turbo/flags/flags.h"
#include "turbo/format/print.h"
#include "tann/vamana/filter_utils.h"
#include "tann/vamana/utils.h"

#ifndef TANN_TOOLS_TANN_CLI_H_
#define TANN_TOOLS_TANN_CLI_H_

#define PARTSIZE 10000000
#define ALIGNMENT 512

namespace detail {
    typedef tsl::robin_set<std::string> label_set;
    typedef std::string path;

    template<class T>
    T div_round_up(const T numerator, const T denominator) {
        return (numerator % denominator == 0) ? (numerator / denominator) : 1 + (numerator / denominator);
    }

    using pairIF = std::pair<size_t, float>;

    struct cmpmaxstruct {
        bool operator()(const pairIF &l, const pairIF &r) {
            return l.second < r.second;
        };
    };


    using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

    void compute_l2sq(float *const points_l2sq, const float *const matrix, const int64_t num_points, const uint64_t dim);

    void distsq_to_points(const size_t dim,
                          float *dist_matrix, // Col Major, cols are queries, rows are points
                          size_t npoints, const float *const points,
                          const float *const points_l2sq, // points in Col major
                          size_t nqueries, const float *const queries,
                          const float *const queries_l2sq, // queries in Col major
                          float *ones_vec = NULL);          // Scratchspace of num_data size and init to 1.0
    void inner_prod_to_points(const size_t dim,
                              float *dist_matrix, // Col Major, cols are queries, rows are points
                              size_t npoints, const float *const points, size_t nqueries, const float *const queries,
                              float *ones_vec = NULL); // Scratchspace of num_data size and init to 1.0

    void exact_knn(const size_t dim, const size_t k,
                   size_t *const closest_points,     // k * num_queries preallocated, col
            // major, queries columns
                   float *const dist_closest_points, // k * num_queries
            // preallocated, Dist to
            // corresponding closes_points
                   size_t npoints,
                   float *points_in, // points in Col major
                   size_t nqueries, float *queries_in,
                   tann::Metric metric = tann::Metric::L2); // queries in Col major

    template<class T>
    T *aligned_malloc(const size_t n, const size_t alignment) {
#ifdef _WINDOWS
        return (T *)_aligned_malloc(sizeof(T) * n, alignment);
#else
        return static_cast<T *>(aligned_alloc(alignment, sizeof(T) * n));
#endif
    }

    inline bool custom_dist(const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) {
        return a.second < b.second;
    }


    template<typename T>
    inline int get_num_parts(const char *filename) {
        std::ifstream reader;
        reader.exceptions(std::ios::failbit | std::ios::badbit);
        reader.open(filename, std::ios::binary);
        std::cout << "Reading bin file " << filename << " ...\n";
        int npts_i32, ndims_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &ndims_i32, sizeof(int));
        std::cout << "#pts = " << npts_i32 << ", #dims = " << ndims_i32 << std::endl;
        reader.close();
        uint32_t num_parts =
                (npts_i32 % PARTSIZE) == 0 ? npts_i32 / PARTSIZE : (uint32_t) std::floor(npts_i32 / PARTSIZE) + 1;
        std::cout << "Number of parts: " << num_parts << std::endl;
        return num_parts;
    }

    template<typename T>
    inline void
    load_bin_as_float(const char *filename, float *&data, size_t &npts_u64, size_t &ndims_u64, int part_num) {
        std::ifstream reader;
        reader.exceptions(std::ios::failbit | std::ios::badbit);
        reader.open(filename, std::ios::binary);
        std::cout << "Reading bin file " << filename << " ...\n";
        int npts_i32, ndims_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &ndims_i32, sizeof(int));
        uint64_t start_id = part_num * PARTSIZE;
        uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t) npts_i32);
        npts_u64 = end_id - start_id;
        ndims_u64 = (uint64_t) ndims_i32;
        std::cout << "#pts in part = " << npts_u64 << ", #dims = " << ndims_u64
                  << ", size = " << npts_u64 * ndims_u64 * sizeof(T) << "B" << std::endl;

        reader.seekg(start_id * ndims_u64 * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);
        T *data_T = new T[npts_u64 * ndims_u64];
        reader.read((char *) data_T, sizeof(T) * npts_u64 * ndims_u64);
        std::cout << "Finished reading part of the bin file." << std::endl;
        reader.close();
        data = aligned_malloc<float>(npts_u64 * ndims_u64, ALIGNMENT);
#pragma omp parallel for schedule(dynamic, 32768)
        for (int64_t i = 0; i < (int64_t) npts_u64; i++) {
            for (int64_t j = 0; j < (int64_t) ndims_u64; j++) {
                float cur_val_float = (float) data_T[i * ndims_u64 + j];
                std::memcpy((char *) (data + i * ndims_u64 + j), (char *) &cur_val_float, sizeof(float));
            }
        }
        delete[] data_T;
        std::cout << "Finished converting part data to float." << std::endl;
    }

    void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim);

}


namespace tann {

    extern bool verbose;

    void set_bin_to_tsv(turbo::App &app);
    void bin_to_tsv();

    void set_calc_recall(turbo::App &app);
    void calc_recall();

    void set_compute_groundtruth(turbo::App &app);
    void compute_groundtruth();

    void set_compute_groundtruth_filter(turbo::App &app);
    void compute_groundtruth_filter();

    void set_count_bfs_level(turbo::App &app);
    void count_bfs_level();

    void set_create_disk_layout(turbo::App &app);
    void create_disk_layout();

    void set_fbin_to_int8(turbo::App &app);
    void fbin_to_int8();

    void set_fvec_to_bin(turbo::App &app);
    void fvec_to_bin();

    void set_fvec_to_bvec(turbo::App &app);
    void fvec_to_bvec();

    void set_gen_randome_slice(turbo::App &app);
    void gen_randome_slice();

    void set_gen_pq(turbo::App &app);
    void gen_pq();

    void set_gen_label(turbo::App &app);
    void gen_label();

    void set_int8_to_float(turbo::App &app);
    void int8_to_float();

    void set_int8_to_float_scale(turbo::App &app);
    void int8_to_float_scale();

    void set_ivecs_to_bin(turbo::App &app);
    void ivecs_to_bin();

    void set_merge_shards(turbo::App &app);
    void merge_shards();

    void set_rand_data_gen(turbo::App &app);
    void rand_data_gen();

    void set_simulate_recall(turbo::App &app);
    void simulate_recall();

    void set_stats_label_data(turbo::App &app);
    void stats_label_data();

    void set_tsv_to_bin(turbo::App &app);
    void tsv_to_bin();

    void set_uint8_to_float(turbo::App &app);
    void uint8_to_float();

    void set_uint32_to_uint8(turbo::App &app);
    void uint32_to_uint8();
    void set_vector_analysis(turbo::App &app);
    void vector_analysis();
}
#endif  // TANN_TOOLS_TANN_CLI_H_
