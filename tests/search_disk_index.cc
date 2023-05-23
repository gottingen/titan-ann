// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "tann/common_includes.h"
#include <boost/program_options.hpp>

#include "tann/index.h"
#include "tann/disk_utils.h"
#include "tann/math_utils.h"
#include "tann/memory_mapper.h"
#include "tann/partition.h"
#include "tann/pq_flash_index.h"
#include "tann/timer.h"
#include "tann/percentile_stats.h"

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

#define WARMUP false

namespace po = boost::program_options;

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
int search_disk_index(tann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false) {
    tann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        tann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        tann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        tann::cout << "." << std::endl;
    else
        tann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    tann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty()) {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num) {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file)) {
        tann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            tann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
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
    tann::cout << "Caching " << num_nodes_to_cache << " BFS nodes around medoid(s)" << std::endl;
    //_pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    if (num_nodes_to_cache > 0)
        _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache, num_threads,
                                                              node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP) {
        if (file_exists(warmup_query_file)) {
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

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    tann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
               << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
               << "CPU (s)";
    if (calc_recall_flag) {
        tann::cout << std::setw(16) << recall_string << std::endl;
    } else
        tann::cout << std::endl;
    tann::cout << "==============================================================="
                  "======================================================="
               << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];

        if (L < recall_at) {
            tann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0) {
            tann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                    optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        } else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new tann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) query_num; i++) {
            if (!filtered_search) {
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i);
            } else {
                LabelT label_for_search;
                if (query_filters.size() == 1) { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                } else { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                        query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                        query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true,
                        label_for_search,
                        use_reorder_data, stats + i);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        tann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                query_num, recall_at);

        auto mean_latency = tann::get_mean_stats<float>(
                stats, query_num, [](const tann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = tann::get_percentile_stats<float>(
                stats, query_num, 0.999, [](const tann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = tann::get_mean_stats<uint32_t>(stats, query_num,
                                                       [](const tann::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = tann::get_mean_stats<float>(stats, query_num,
                                                      [](const tann::QueryStats &stats) { return stats.cpu_us; });

        double recall = 0;
        if (calc_recall_flag) {
            recall = tann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                            query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        tann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                   << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                   << std::setw(16) << mean_cpuus;
        if (calc_recall_flag) {
            tann::cout << std::setw(16) << recall << std::endl;
        } else
            tann::cout << std::endl;
        delete[] stats;
    }

    tann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L: Lvec) {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        tann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        tann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }

    tann::aligned_free(query);
    if (warmup != nullptr)
        tann::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, filter_label,
            label_type, query_filters_file;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                           "distance function <l2/mips/fast_l2>");
        desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                           "Path prefix to the index");
        desc.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                           "Path prefix for saving results of the queries");
        desc.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                           "Query file in binary format");
        desc.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                           "ground truth file for the queryset");
        desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(), "Number of neighbors to be returned");
        desc.add_options()("search_list,L", po::value<std::vector<uint32_t>>(&Lvec)->multitoken(),
                           "List of L values of search");
        desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                           "Beamwidth for search. Set 0 to optimize internally.");
        desc.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                           "Beamwidth for search");
        desc.add_options()("search_io_limit",
                           po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
                           "Max #IOs for search");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");
        desc.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                           "Include full precision data in the index. Use only in "
                           "conjuction with compressed data on SSD.");
        desc.add_options()("filter_label", po::value<std::string>(&filter_label)->default_value(std::string("")),
                           "Filter Label for Filtered Search");
        desc.add_options()("query_filters_file",
                           po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                           "Filter file for Queries for Filtered Search ");
        desc.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                           "Storage type of Labels <uint/ushort>, default value is uint which "
                           "will consume memory 4 bytes per filter");
        desc.add_options()("fail_if_recall_below", po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                           "If set to a value >0 and <100%, program returns -1 if best recall "
                           "found is below this threshold. ");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
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

    if (use_reorder_data && data_type != std::string("float")) {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "") {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "") {
        query_filters.push_back(filter_label);
    } else if (query_filters_file != "") {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try {
        if (!query_filters.empty() && label_type == "ushort") {
            if (data_type == std::string("float"))
                return search_disk_index<float, uint16_t>(
                        metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                        num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters,
                        use_reorder_data);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                        metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                        num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters,
                        use_reorder_data);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                        metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                        num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters,
                        use_reorder_data);
            else {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        } else {
            if (data_type == std::string("float"))
                return search_disk_index<float>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                fail_if_recall_below, query_filters, use_reorder_data);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data);
            else {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        tann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}