// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Copyright 2023 The Tann Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once
#include "common_includes.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "turbo/platform/port.h"
#include "scratch.h"
#include "turbo/container/flat_hash_set.h"
#include "turbo/container/flat_hash_map.h"

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace tann {

  template<typename T, typename LabelT = uint32_t>
  class PQFlashIndex {
   public:
    TURBO_DLL PQFlashIndex(
        std::shared_ptr<AlignedFileReader> &fileReader,
        tann::Metric                     metric = tann::Metric::L2);
    TURBO_DLL ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
    TURBO_DLL int load(tann::MemoryMappedFiles &files,
                               uint32_t num_threads, const char *index_prefix);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    TURBO_DLL int load(uint32_t num_threads, const char *index_prefix);
#endif

#ifdef EXEC_ENV_OLS
    TURBO_DLL int load_from_separate_paths(
        tann::MemoryMappedFiles &files, uint32_t num_threads,
        const char *index_filepath, const char *pivots_filepath,
        const char *compressed_filepath);
#else
    TURBO_DLL int load_from_separate_paths(
        uint32_t num_threads, const char *index_filepath,
        const char *pivots_filepath, const char *compressed_filepath);
#endif

    TURBO_DLL void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
    TURBO_DLL void generate_cache_list_from_sample_queries(
        MemoryMappedFiles &files, std::string sample_bin, _u64 l_search,
        _u64 beamwidth, _u64 num_nodes_to_cache, uint32_t nthreads,
        std::vector<uint32_t> &node_list);
#else
    TURBO_DLL void generate_cache_list_from_sample_queries(
        std::string sample_bin, _u64 l_search, _u64 beamwidth,
        _u64 num_nodes_to_cache, uint32_t num_threads,
        std::vector<uint32_t> &node_list);
#endif

    TURBO_DLL void cache_bfs_levels(_u64 num_nodes_to_cache,
                                            std::vector<uint32_t> &node_list,
                                            const bool shuffle = false);

    TURBO_DLL void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width,
        const bool use_reorder_data = false, QueryStats *stats = nullptr);

    TURBO_DLL void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const bool use_filter,
        const LabelT &filter_label, const bool use_reorder_data = false,
        QueryStats *stats = nullptr);

    TURBO_DLL void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, QueryStats *stats = nullptr);

    TURBO_DLL void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const bool use_filter,
        const LabelT &filter_label, const _u32 io_limit,
        const bool use_reorder_data = false, QueryStats *stats = nullptr);

    TURBO_DLL LabelT
    get_converted_label(const std::string &filter_label);

    TURBO_DLL _u32 range_search(const T *query1, const double range,
                                        const _u64          min_l_search,
                                        const _u64          max_l_search,
                                        std::vector<_u64>  &indices,
                                        std::vector<float> &distances,
                                        const _u64          min_beam_width,
                                        QueryStats         *stats = nullptr);

    TURBO_DLL _u64 get_data_dim();

    std::shared_ptr<AlignedFileReader> &reader;

    TURBO_DLL tann::Metric get_metric();

   protected:
    TURBO_DLL void use_medoids_data_as_centroids();
    TURBO_DLL void setup_thread_data(_u64 nthreads,
                                             _u64 visited_reserve = 4096);

    TURBO_DLL void set_universal_label(const LabelT &label);

   private:
    TURBO_DLL inline bool point_has_label(_u32 point_id, _u32 label_id);
    std::unordered_map<std::string, LabelT> load_label_map(
        const std::string &map_file);
    TURBO_DLL void parse_label_file(const std::string &map_file,
                                            size_t            &num_pts_labels);
    TURBO_DLL void get_label_file_metadata(std::string map_file,
                                                   _u32       &num_pts,
                                                   _u32 &num_total_labels);
    TURBO_DLL inline int32_t get_filter_number(
        const LabelT &filter_label);

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1

    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // Data used for searching with re-order vectors
    _u64 ndims_reorder_vecs = 0, reorder_data_start_sector = 0,
         nvecs_per_sector = 0;

    tann::Metric metric = tann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float max_base_norm = 0.0f;

    // data info
    _u64 num_points = 0;
    _u64 num_frozen_points = 0;
    _u64 frozen_location = 0;
    _u64 data_dim = 0;
    _u64 disk_data_dim = 0;  // will be different from data_dim only if we use
                             // PQ for disk data (very large dimensionality)
    _u64 aligned_dim = 0;
    _u64 disk_bytes_per_point = 0;

    std::string                        disk_index_file;
    std::vector<std::pair<_u32, _u32>> node_visit_counter;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8              *data = nullptr;
    _u64              n_chunks;
    FixedChunkPQTable pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>>     dist_cmp;
    std::shared_ptr<Distance<float>> dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    bool              use_disk_index_pq = false;
    _u64              disk_pq_n_chunks = 0;
    FixedChunkPQTable disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *medoids = nullptr;
    // defaults to 1
    size_t num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *centroid_data = nullptr;

    // nhood_cache
    unsigned                                     *nhood_cache_buf = nullptr;
    turbo::flat_hash_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;

    // coord_cache
    T                        *coord_cache_buf = nullptr;
    turbo::flat_hash_map<_u32, T *> coord_cache;

    // thread-specific scratch
    ConcurrentQueue<SSDThreadData<T> *> thread_data;
    _u64                                max_nthreads;
    bool                                load_flag = false;
    bool                                count_visited_nodes = false;
    bool                                reorder_data_exists = false;
    _u64                                reoreder_data_offset = 0;

    // filter support
    _u32                                   *_pts_to_label_offsets = nullptr;
    _u32                                   *_pts_to_labels = nullptr;
    turbo::flat_hash_set<LabelT>                  _labels;
    std::unordered_map<LabelT, _u32>        _filter_to_medoid_id;
    bool                                    _use_universal_label;
    _u32                                    _universal_filter_num;
    std::vector<LabelT>                     _filter_list;
    turbo::flat_hash_set<_u32>                    _dummy_pts;
    turbo::flat_hash_set<_u32>                    _has_dummy_pts;
    turbo::flat_hash_map<_u32, _u32>              _dummy_to_real_map;
    turbo::flat_hash_map<_u32, std::vector<_u32>> _real_to_dummy_map;
    std::unordered_map<std::string, LabelT> _label_map;

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = SECTOR_LEN;
    char            *getHeaderBytes();
#endif
  };
}  // namespace tann
