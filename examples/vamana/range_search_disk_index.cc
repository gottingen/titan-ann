// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <set>
#include "turbo/flags/flags.h"

#include "tann/diskann/index.h"
#include "tann/diskann/disk_utils.h"
#include "tann/common/math_utils.h"
#include "tann/io/memory_mapper.h"
#include "tann/diskann/pq_flash_index.h"
#include "tann/diskann/partition.h"
#include "tann/diskann/timer.h"

#ifndef _WINDOWS

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "tann/io/linux_aligned_file_reader.h"

#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results) {
    tann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        tann::cout << std::setw(8) << percentiles[s] << "%";
    }
    tann::cout << std::endl;
    tann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        tann::cout << std::setw(9) << results[s];
    }
    tann::cout << std::endl;
}

template<typename T, typename LabelT = uint32_t>
int search_disk_index(tann::Metric &metric, const std::string &index_path_prefix, const std::string &query_file,
                      std::string &gt_file, const uint32_t num_threads, const float search_range,
                      const uint32_t beamwidth, const uint32_t num_nodes_to_cache, const std::vector<uint32_t> &Lvec) {
    std::string pq_prefix = index_path_prefix + "_pq";
    std::string disk_index_file = index_path_prefix + "_disk.index";
    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    tann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        tann::cout << "beamwidth to be optimized for each L value" << std::endl;
    else
        tann::cout << " beamwidth: " << beamwidth << std::endl;

    // load query bin
    T *query = nullptr;
    std::vector<std::vector<uint32_t>> groundtruth_ids;
    size_t query_num, query_dim, query_aligned_dim, gt_num;
    tann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && turbo::filesystem::exists(gt_file)) {
        tann::load_range_truthset(gt_file, groundtruth_ids,
                                  gt_num); // use for range search type of truthset
        //    tann::prune_truthset_for_range(gt_file, search_range,
        //    groundtruth_ids, gt_num); // use for traditional truthset
        if (gt_num != query_num) {
            tann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
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
    reader.reset(new tann::LinuxAlignedFileReader());
#endif

    std::unique_ptr<tann::PQFlashIndex<T, LabelT>> _pFlashIndex(
            new tann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0) {
        return res;
    }
    // cache bfs levels
    std::vector<uint32_t> node_list;
    tann::cout << "Caching " << num_nodes_to_cache << " BFS nodes around medoid(s)" << std::endl;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    //  _pFlashIndex->generate_cache_list_from_sample_queries(
    //      warmup_query_file, 15, 6, num_nodes_to_cache, num_threads,
    //      node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP) {
        if (turbo::filesystem::exists(warmup_query_file)) {
            tann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        } else {
            warmup_num = (std::min)((uint32_t) 150000, (uint32_t) 15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            tann::alloc_aligned(((void **) &warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
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
        tann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) warmup_num; i++) {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        tann::cout << "..done" << std::endl;
    }

    tann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    tann::cout.precision(2);

    std::string recall_string = "Recall@rng=" + std::to_string(search_range);
    tann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
               << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
               << "CPU (s)";
    if (calc_recall_flag) {
        tann::cout << std::setw(16) << recall_string << std::endl;
    } else
        tann::cout << std::endl;
    tann::cout << "==============================================================="
                  "==========================================="
               << std::endl;

    std::vector<std::vector<std::vector<uint32_t>>> query_result_ids(Lvec.size());

    uint32_t optimized_beamwidth = 2;
    uint32_t max_list_size = 10000;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];

        if (beamwidth <= 0) {
            optimized_beamwidth =
                    optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        } else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].clear();
        query_result_ids[test_id].resize(query_num);

        tann::QueryStats *stats = new tann::QueryStats[query_num];

        auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) query_num; i++) {
            std::vector<uint64_t> indices;
            std::vector<float> distances;
            uint32_t res_count =
                    _pFlashIndex->range_search(query + (i * query_aligned_dim), search_range, L, max_list_size, indices,
                                               distances, optimized_beamwidth, stats + i);
            query_result_ids[test_id][i].reserve(res_count);
            query_result_ids[test_id][i].resize(res_count);
            for (uint32_t idx = 0; idx < res_count; idx++)
                query_result_ids[test_id][i][idx] = (uint32_t) indices[idx];
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        auto qps = (1.0 * query_num) / (1.0 * diff.count());

        auto mean_latency = tann::get_mean_stats<float>(
                stats, query_num, [](const tann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = tann::get_percentile_stats<float>(
                stats, query_num, 0.999, [](const tann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = tann::get_mean_stats<uint32_t>(stats, query_num,
                                                       [](const tann::QueryStats &stats) { return stats.n_ios; });

        double mean_cpuus = tann::get_mean_stats<float>(
                stats, query_num, [](const tann::QueryStats &stats) { return stats.cpu_us; });

        double recall = 0;
        double ratio_of_sums = 0;
        if (calc_recall_flag) {
            recall =
                    tann::calculate_range_search_recall((uint32_t) query_num, groundtruth_ids,
                                                        query_result_ids[test_id]);

            uint32_t total_true_positive = 0;
            uint32_t total_positive = 0;
            for (uint32_t i = 0; i < query_num; i++) {
                total_true_positive += (uint32_t) query_result_ids[test_id][i].size();
                total_positive += (uint32_t) groundtruth_ids[i].size();
            }

            ratio_of_sums = (1.0 * total_true_positive) / (1.0 * total_positive);
        }

        tann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                   << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                   << std::setw(16) << mean_cpuus;
        if (calc_recall_flag) {
            tann::cout << std::setw(16) << recall << "," << ratio_of_sums << std::endl;
        } else
            tann::cout << std::endl;
    }

    tann::cout << "Done searching. " << std::endl;

    tann::aligned_free(query);
    if (warmup != nullptr)
        tann::aligned_free(warmup);
    return 0;
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file;
    uint32_t num_threads, W, num_nodes_to_cache;
    std::vector<uint32_t> Lvec;
    float range;

    turbo::App app("range_search_disk_index");
    try {
        app.add_option("-t, --data_type", data_type, "data type <int8/uint8/float>")->required();
        app.add_option("-F, --dist_fn", dist_fn,
                       "distance function <l2/mips/fast_l2>")->required();
        app.add_option("-x --index_path_prefix", index_path_prefix,
                       "Path prefix to the index")->required();
        app.add_option("-q, --query_file", query_file,
                       "Query file in binary format")->required();
        app.add_option("-g --gt_file", gt_file,
                       "ground truth file for the queryset")->default_val(std::string("null"));
        app.add_option("--range_threshold,-K", range,
                       "Number of neighbors to be returned")->required();
        app.add_option("--search_list,-L", Lvec,
                       "List of L values of search");
        app.add_option("--beamwidth,-W", W, "Beamwidth for search")->default_val(2);
        app.add_option("num_nodes_to_cache", num_nodes_to_cache,
                       "Beamwidth for search")->default_val(100000);
        app.add_option("--num_threads,-T", num_threads,
                       "Number of threads used for building index (defaults to "
                       "omp_get_num_procs())")->default_val(omp_get_num_procs());
        TURBO_FLAGS_PARSE(app, argc, argv);
    }
    catch (const std::exception &ex) {
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

    if ((data_type != std::string("float")) && (metric == tann::Metric::INNER_PRODUCT)) {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    try {
        if (data_type == std::string("float"))
            return search_disk_index<float>(metric, index_path_prefix, query_file, gt_file, num_threads, range, W,
                                            num_nodes_to_cache, Lvec);
        else if (data_type == std::string("int8"))
            return search_disk_index<int8_t>(metric, index_path_prefix, query_file, gt_file, num_threads, range, W,
                                             num_nodes_to_cache, Lvec);
        else if (data_type == std::string("uint8"))
            return search_disk_index<uint8_t>(metric, index_path_prefix, query_file, gt_file, num_threads, range, W,
                                              num_nodes_to_cache, Lvec);
        else {
            std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        tann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
