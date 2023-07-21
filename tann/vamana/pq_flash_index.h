// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "tann/common/common_includes.h"

#include "tann/io/aligned_file_reader.h"
#include "tann/common/concurrent_queue.h"
#include "tann/vamana/neighbor.h"
#include "tann/vamana/parameters.h"
#include "tann/vamana/percentile_stats.h"
#include "tann/vamana/pq.h"
#include "tann/common/utils.h"
#include "turbo/platform/port.h"
#include "tann/vamana/scratch.h"
#include "turbo/container/flat_hash_map.h"
#include "turbo/container/flat_hash_set.h"

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace tann {

    template<typename T, typename LabelT = uint32_t>
    class PQFlashIndex {
    public:
        TURBO_DLL PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader,
                               tann::Metric metric = tann::Metric::L2);

        TURBO_DLL ~PQFlashIndex();

#ifdef EXEC_ENV_OLS
        TURBO_DLL int load(tann::MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix);
#else
        // load compressed data, and obtains the handle to the disk-resident index
        TURBO_DLL int load(uint32_t num_threads, const char *index_prefix);

#endif

#ifdef EXEC_ENV_OLS
        TURBO_DLL int load_from_separate_paths(tann::MemoryMappedFiles &files, uint32_t num_threads,
                                                       const char *index_filepath, const char *pivots_filepath,
                                                       const char *compressed_filepath);
#else
        TURBO_DLL int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                               const char *pivots_filepath, const char *compressed_filepath);

#endif

        TURBO_DLL void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
        TURBO_DLL void generate_cache_list_from_sample_queries(MemoryMappedFiles &files, std::string sample_bin,
                                                                       uint64_t l_search, uint64_t beamwidth,
                                                                       uint64_t num_nodes_to_cache, uint32_t nthreads,
                                                                       std::vector<uint32_t> &node_list);
#else
        TURBO_DLL void generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                               uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                               uint32_t num_threads,
                                                               std::vector<uint32_t> &node_list);

#endif

        TURBO_DLL void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                        const bool shuffle = false);

        TURBO_DLL void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                          uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                          const bool use_reorder_data = false, QueryStats *stats = nullptr);

        TURBO_DLL void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                          uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                          const bool use_filter, const LabelT &filter_label,
                                          const bool use_reorder_data = false, QueryStats *stats = nullptr);

        TURBO_DLL void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                          uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                          const uint32_t io_limit, const bool use_reorder_data = false,
                                          QueryStats *stats = nullptr);

        TURBO_DLL void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                          uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                          const bool use_filter, const LabelT &filter_label,
                                          const uint32_t io_limit, const bool use_reorder_data = false,
                                          QueryStats *stats = nullptr);

        TURBO_DLL LabelT get_converted_label(const std::string &filter_label);

        TURBO_DLL uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                        const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                        std::vector<float> &distances, const uint64_t min_beam_width,
                                        QueryStats *stats = nullptr);

        TURBO_DLL uint64_t get_data_dim();

        std::shared_ptr<AlignedFileReader> &reader;

        TURBO_DLL tann::Metric get_metric();

    protected:
        TURBO_DLL void use_medoids_data_as_centroids();

        TURBO_DLL void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);

        TURBO_DLL void set_universal_label(const LabelT &label);

    private:
        TURBO_DLL inline bool point_has_label(uint32_t point_id, uint32_t label_id);

        std::unordered_map<std::string, LabelT> load_label_map(const std::string &map_file);

        TURBO_DLL void parse_label_file(const std::string &map_file, size_t &num_pts_labels);

        TURBO_DLL void get_label_file_metadata(std::string map_file, uint32_t &num_pts, uint32_t &num_total_labels);

        TURBO_DLL inline int32_t get_filter_number(const LabelT &filter_label);

        // index info
        // nhood of node `i` is in sector: [i / nnodes_per_sector]
        // offset in sector: [(i % nnodes_per_sector) * max_node_len]
        // nnbrs of node `i`: *(unsigned*) (buf)
        // nbrs of node `i`: ((unsigned*)buf) + 1

        uint64_t max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

        // Data used for searching with re-order vectors
        uint64_t ndims_reorder_vecs = 0, reorder_data_start_sector = 0, nvecs_per_sector = 0;

        tann::Metric metric = tann::Metric::L2;

        // used only for inner product search to re-scale the result value
        // (due to the pre-processing of base during index build)
        float max_base_norm = 0.0f;

        // data info
        uint64_t num_points = 0;
        uint64_t num_frozen_points = 0;
        uint64_t frozen_location = 0;
        uint64_t data_dim = 0;
        uint64_t disk_data_dim = 0; // will be different from data_dim only if we use
        // PQ for disk data (very large dimensionality)
        uint64_t aligned_dim = 0;
        uint64_t disk_bytes_per_point = 0;

        std::string disk_index_file;
        std::vector<std::pair<uint32_t, uint32_t>> node_visit_counter;

        // PQ data
        // n_chunks = # of chunks ndims is split into
        // data: char * n_chunks
        // chunk_size = chunk size of each dimension chunk
        // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
        uint8_t *data = nullptr;
        uint64_t n_chunks;
        FixedChunkPQTable pq_table;

        // distance comparator
        std::shared_ptr<Distance<T>> dist_cmp;
        std::shared_ptr<Distance<float>> dist_cmp_float;

        // for very large datasets: we use PQ even for the disk resident index
        bool use_disk_index_pq = false;
        uint64_t disk_pq_n_chunks = 0;
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
        unsigned *nhood_cache_buf = nullptr;
        turbo::flat_hash_map<uint32_t, std::pair<uint32_t, uint32_t *>> nhood_cache;

        // coord_cache
        T *coord_cache_buf = nullptr;
        turbo::flat_hash_map<uint32_t, T *> coord_cache;

        // thread-specific scratch
        ConcurrentQueue<SSDThreadData<T> *> thread_data;
        uint64_t max_nthreads;
        bool load_flag = false;
        bool count_visited_nodes = false;
        bool reorder_data_exists = false;
        uint64_t reoreder_data_offset = 0;

        // filter support
        uint32_t *_pts_to_label_offsets = nullptr;
        uint32_t *_pts_to_labels = nullptr;
        turbo::flat_hash_set<LabelT> _labels;
        std::unordered_map<LabelT, uint32_t> _filter_to_medoid_id;
        bool _use_universal_label;
        uint32_t _universal_filter_num;
        std::vector<LabelT> _filter_list;
        turbo::flat_hash_set<uint32_t> _dummy_pts;
        turbo::flat_hash_set<uint32_t> _has_dummy_pts;
        turbo::flat_hash_map<uint32_t, uint32_t> _dummy_to_real_map;
        turbo::flat_hash_map<uint32_t, std::vector<uint32_t>> _real_to_dummy_map;
        std::unordered_map<std::string, LabelT> _label_map;

#ifdef EXEC_ENV_OLS
        // Set to a larger value than the actual header to accommodate
        // any additions we make to the header. This is an outer limit
        // on how big the header can be.
        static const int HEADER_SIZE = SECTOR_LEN;
        char *getHeaderBytes();
#endif
    };
} // namespace tann
