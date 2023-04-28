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

#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <set>
#include <boost/program_options.hpp>
#include <sstream>
#include "tann/index.h"
#include "tann/disk_utils.h"
#include "tann/math_utils.h"
#include "tann/memory_mapper.h"
#include "tann/pq_flash_index.h"
#include "tann/partition.h"
#include "tann/timer.h"

#ifndef _WINDOWS

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "tann/linux_aligned_file_reader.h"

#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

namespace po = boost::program_options;

#define WARMUP false

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
    std::stringstream ss;
    ss << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        ss << std::setw(8) << percentiles[s] << "%";
    }
    TURBO_LOG(INFO) << ss.str();
    std::stringstream sss;
    sss << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        sss << std::setw(9) << results[s];
    }
    TURBO_LOG(INFO) << sss.str();
}

template<typename T, typename LabelT = uint32_t>
int search_disk_index(tann::Metric &metric,
                      const std::string &index_path_prefix,
                      const std::string &query_file, std::string &gt_file,
                      const unsigned num_threads, const float search_range,
                      const unsigned beamwidth,
                      const unsigned num_nodes_to_cache,
                      const std::vector<unsigned> &Lvec) {
    std::string pq_prefix = index_path_prefix + "_pq";
    std::string disk_index_file = index_path_prefix + "_disk.index";
    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";
    std::stringstream sss;
    sss << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        sss << "beamwidth to be optimized for each L value" << std::endl;
    else
        sss << " beamwidth: " << beamwidth << std::endl;
    std::cout<<sss.str();
    // load query bin
    T *query = nullptr;
    std::vector<std::vector<_u32>> groundtruth_ids;
    size_t query_num, query_dim, query_aligned_dim, gt_num;
    tann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                              query_aligned_dim);

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && file_exists(gt_file)) {
        tann::load_range_truthset(
                gt_file, groundtruth_ids,
                gt_num);  // use for range search type of truthset
        //    tann::prune_truthset_for_range(gt_file, search_range,
        //    groundtruth_ids, gt_num); // use for traditional truthset
        if (gt_num != query_num) {
            TURBO_LOG(ERROR)<< "Error. Mismatch in number of queries and ground truth data";
            return -1;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new tann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<tann::PQFlashIndex<T, LabelT>> _pFlashIndex(
            new tann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0) {
        return res;
    }
    // cache bfs levels
    std::vector<uint32_t> node_list;
    TURBO_LOG(INFO)<< "Caching " << num_nodes_to_cache
               << " BFS nodes around medoid(s)";
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    //  _pFlashIndex->generate_cache_list_from_sample_queries(
    //      warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP) {
        if (file_exists(warmup_query_file)) {
            tann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                                      warmup_dim, warmup_aligned_dim);
        } else {
            warmup_num = (std::min)((_u32) 150000, (_u32) 15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            tann::alloc_aligned(((void **) &warmup),
                                warmup_num * warmup_aligned_dim * sizeof(T),
                                8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++) {
                for (uint32_t d = 0; d < warmup_dim; d++) {
                    warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
                }
            }
        }
        TURBO_LOG(INFO)<< "Warming up index... ";
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1,
                                             warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        TURBO_LOG(INFO) << "..done";
    }
    std::stringstream ss;
    ss.setf(std::ios_base::fixed, std::ios_base::floatfield);
    ss.precision(2);

    std::string recall_string = "Recall@rng=" + std::to_string(search_range);
    ss << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
               << std::setw(16) << "QPS" << std::setw(16) << "Mean Latency"
               << std::setw(16) << "99.9 Latency" << std::setw(16)
               << "Mean IOs" << std::setw(16) << "CPU (s)";
    if (calc_recall_flag) {
        ss << std::setw(16) << recall_string << std::endl;
    } else
        ss << std::endl;
    ss<< "=========================================================================================================="<< std::endl;
    std::cout<<ss.str();

    std::vector<std::vector<std::vector<uint32_t>>> query_result_ids(Lvec.size());

    uint32_t optimized_beamwidth = 2;
    uint32_t max_list_size = 10000;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        _u64 L = Lvec[test_id];

        if (beamwidth <= 0) {
            optimized_beamwidth =
                    optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
                                       warmup_aligned_dim, L, optimized_beamwidth);
        } else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].clear();
        query_result_ids[test_id].resize(query_num);

        tann::QueryStats *stats = new tann::QueryStats[query_num];

        auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
            std::vector<_u64> indices;
            std::vector<float> distances;
            _u32 res_count = _pFlashIndex->range_search(
                    query + (i * query_aligned_dim), search_range, L, max_list_size,
                    indices, distances, optimized_beamwidth, stats + i);
            query_result_ids[test_id][i].reserve(res_count);
            query_result_ids[test_id][i].resize(res_count);
            for (_u32 idx = 0; idx < res_count; idx++)
                query_result_ids[test_id][i][idx] = indices[idx];
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        auto qps = (1.0 * query_num) / (1.0 * diff.count());

        auto mean_latency = tann::get_mean_stats<float>(
                stats, query_num,
                [](const tann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = tann::get_percentile_stats<float>(
                stats, query_num, 0.999,
                [](const tann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = tann::get_mean_stats<unsigned>(
                stats, query_num,
                [](const tann::QueryStats &stats) { return stats.n_ios; });

        float mean_cpuus = tann::get_mean_stats<float>(
                stats, query_num,
                [](const tann::QueryStats &stats) { return stats.cpu_us; });

        float recall = 0;
        float ratio_of_sums = 0;
        if (calc_recall_flag) {
            recall = tann::calculate_range_search_recall(
                    query_num, groundtruth_ids, query_result_ids[test_id]);

            _u32 total_true_positive = 0;
            _u32 total_positive = 0;
            for (_u32 i = 0; i < query_num; i++) {
                total_true_positive += query_result_ids[test_id][i].size();
                total_positive += groundtruth_ids[i].size();
            }

            ratio_of_sums = (1.0 * total_true_positive) / (1.0 * total_positive);
        }
        std::stringstream ss;
        ss << std::setw(6) << L << std::setw(12) << optimized_beamwidth
                   << std::setw(16) << qps << std::setw(16) << mean_latency
                   << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                   << std::setw(16) << mean_cpuus;
        if (calc_recall_flag) {
            ss << std::setw(16) << recall << "," << ratio_of_sums
                       << std::endl;
        } else
            ss << std::endl;
    }

    ss << "Done searching. " << std::endl;
    std::cout<<ss.str();

    tann::aligned_free(query);
    if (warmup != nullptr)
        tann::aligned_free(warmup);
    return 0;
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix,
            query_file, gt_file;
    unsigned num_threads, W, num_nodes_to_cache;
    std::vector<unsigned> Lvec;
    float range;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type",
                           po::value<std::string>(&data_type)->required(),
                           "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/fast_l2>");
        desc.add_options()("index_path_prefix",
                           po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix to the index");
        desc.add_options()("query_file",
                           po::value<std::string>(&query_file)->required(),
                           "Query file in binary format");
        desc.add_options()(
                "gt_file",
                po::value<std::string>(&gt_file)->default_value(std::string("null")),
                "ground truth file for the queryset");
        desc.add_options()("range_threshold,K",
                           po::value<float>(&range)->required(),
                           "Number of neighbors to be returned");
        desc.add_options()("search_list,L",
                           po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                           "List of L values of search");
        desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                           "Beamwidth for search");
        desc.add_options()(
                "num_nodes_to_cache",
                po::value<uint32_t>(&num_nodes_to_cache)->default_value(100000),
                "Beamwidth for search");
        desc.add_options()(
                "num_threads,T",
                po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                "Number of threads used for building index (defaults to "
                "omp_get_num_procs())");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    tann::Metric metric;
    if (dist_fn == std::string("mips")) {
        metric = tann::Metric::INNER_PRODUCT;
    } else if (dist_fn == std::string("l2")) {
        metric = tann::Metric::L2;
    } else if (dist_fn == std::string("cosine")) {
        metric = tann::Metric::COSINE;
    } else {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) &&
        (metric == tann::Metric::INNER_PRODUCT)) {
        std::cout << "Currently support only floating point data for Inner Product."
                  << std::endl;
        return -1;
    }

    try {
        if (data_type == std::string("float"))
            return search_disk_index<float>(metric, index_path_prefix, query_file,
                                            gt_file, num_threads, range, W,
                                            num_nodes_to_cache, Lvec);
        else if (data_type == std::string("int8"))
            return search_disk_index<int8_t>(metric, index_path_prefix, query_file,
                                             gt_file, num_threads, range, W,
                                             num_nodes_to_cache, Lvec);
        else if (data_type == std::string("uint8"))
            return search_disk_index<uint8_t>(metric, index_path_prefix, query_file,
                                              gt_file, num_threads, range, W,
                                              num_nodes_to_cache, Lvec);
        else {
            std::cerr << "Unsupported data type. Use float or int8 or uint8"
                      << std::endl;
            return -1;
        }
    } catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        TURBO_LOG(ERROR) << "Index search failed." << std::endl;
        return -1;
    }
}
