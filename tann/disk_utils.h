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

#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>

#ifdef __APPLE__
#else

#include <malloc.h>

#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else

#include <unistd.h>

typedef int FileHandle;
#endif

#include "cached_io.h"
#include "common_includes.h"

#include "utils.h"
#include "turbo/platform/port.h"

namespace tann {
    const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
    const double PQ_TRAINING_SET_FRACTION = 0.1;
    const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
    const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
    const uint32_t NUM_NODES_TO_CACHE = 250000;
    const uint32_t WARMUP_L = 20;
    const uint32_t NUM_KMEANS_REPS = 12;

    template<typename T, typename LabelT>
    class PQFlashIndex;

    TURBO_DLL double get_memory_budget(const std::string &mem_budget_str);

    TURBO_DLL double get_memory_budget(double search_ram_budget_in_gb);

    TURBO_DLL void add_new_file_to_single_index(std::string index_file,
                                                        std::string new_file);

    TURBO_DLL size_t calculate_num_pq_chunks(double final_index_ram_limit,
                                                     size_t points_num,
                                                     uint32_t dim);

    TURBO_DLL void read_idmap(const std::string &fname,
                                      std::vector<unsigned> &ivecs);

#ifdef EXEC_ENV_OLS
    template<typename T>
    TURBO_DLL T *load_warmup(MemoryMappedFiles &files,
                                     const std::string &cache_warmup_file,
                                     uint64_t &warmup_num, uint64_t warmup_dim,
                                     uint64_t warmup_aligned_dim);
#else

    template<typename T>
    TURBO_DLL T *load_warmup(const std::string &cache_warmup_file,
                                     uint64_t &warmup_num, uint64_t warmup_dim,
                                     uint64_t warmup_aligned_dim);

#endif

    TURBO_DLL int merge_shards(
            const std::string &vamana_prefix, const std::string &vamana_suffix,
            const std::string &idmaps_prefix, const std::string &idmaps_suffix,
            const _u64 nshards, unsigned max_degree, const std::string &output_vamana,
            const std::string &medoids_file, bool use_filters = false,
            const std::string &labels_to_medoids_file = std::string(""));

    TURBO_DLL void extract_shard_labels(
            const std::string &in_label_file, const std::string &shard_ids_bin,
            const std::string &shard_label_file);

    template<typename T>
    TURBO_DLL std::string preprocess_base_file(
            const std::string &infile, const std::string &indexPrefix,
            tann::Metric &distMetric);

    template<typename T, typename LabelT = uint32_t>
    TURBO_DLL int build_merged_vamana_index(
            std::string base_file, tann::Metric _compareMetric, unsigned L,
            unsigned R, double sampling_rate, double ram_budget,
            std::string mem_index_path, std::string medoids_file,
            std::string centroids_file, size_t build_pq_bytes, bool use_opq,
            bool use_filters = false, const std::string &label_file = std::string(""),
            const std::string &labels_to_medoids_file = std::string(""),
            const std::string &universal_label = "", const _u32 Lf = 0);

    template<typename T, typename LabelT>
    TURBO_DLL uint32_t optimize_beamwidth(
            std::unique_ptr<tann::PQFlashIndex<T, LabelT>> &_pFlashIndex,
            T *tuning_sample, _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim,
            uint32_t L, uint32_t nthreads, uint32_t start_bw = 2);

    template<typename T, typename LabelT = uint32_t>
    TURBO_DLL int build_disk_index(
            const char *dataFilePath, const char *indexFilePath,
            const char *indexBuildParameters, tann::Metric _compareMetric,
            bool use_opq = false, bool use_filters = false,
            const std::string &label_file =
            std::string(""),  // default is empty string for no label_file
            const std::string &universal_label = "", const _u32 filter_threshold = 0,
            const _u32 Lf = 0);  // default is empty string for no universal label

    template<typename T>
    TURBO_DLL void create_disk_layout(
            const std::string base_file, const std::string mem_index_file,
            const std::string output_file,
            const std::string reorder_data_file = std::string(""));

}  // namespace tann
