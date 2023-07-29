// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <cstring>

#ifndef _WINDOWS

#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#endif

#include "tann/diskann/index.h"
#include "tann/io/memory_mapper.h"
#include "tann/common/utils.h"
#include "turbo/flags/flags.h"

template<typename T, typename LabelT = uint32_t>
int search_memory_index(tann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::string> &query_filters, const float fail_if_recall_below) {
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    tann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    // Check for ground truth
    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && turbo::filesystem::exists(truthset_file)) {
        tann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    } else {
        tann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

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

    using TagT = uint32_t;
    const bool concurrent = false, pq_dist_build = false, use_opq = false;
    const size_t num_pq_chunks = 0;
    using IndexType = tann::Index<T, TagT, LabelT>;
    const size_t num_frozen_pts = IndexType::get_graph_num_frozen_points(index_path);
    IndexType index(metric, query_dim, 0, dynamic, tags, concurrent, pq_dist_build, num_pq_chunks, use_opq,
                    num_frozen_pts);
    std::cout << "Index class instantiated" << std::endl;
    index.load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;
    if (metric == tann::FAST_L2)
        index.optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags) {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    } else {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag) {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++) {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    if (not tags) {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags) {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];
        if (L < recall_at) {
            tann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search) {
                LabelT filter_label_as_num;
                if (query_filters.size() == 1) {
                    filter_label_as_num = index.get_converted_label(query_filters[0]);
                } else {
                    filter_label_as_num = index.get_converted_label(query_filters[i]);
                }
                auto retval = index.search_with_filters(query + i * query_aligned_dim, filter_label_as_num, recall_at,
                                                        L, query_result_ids[test_id].data() + i * recall_at,
                                                        query_result_dists[test_id].data() + i * recall_at);
                cmp_stats[i] = retval.second;
            } else if (metric == tann::FAST_L2) {
                index.search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
                                                   query_result_ids[test_id].data() + i * recall_at);
            } else if (tags) {
                index.search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                       query_result_tags.data() + i * recall_at, nullptr, res);
                for (int64_t r = 0; r < (int64_t) recall_at; r++) {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            } else {
                cmp_stats[i] = index
                        .search(query + i * query_aligned_dim, recall_at, L,
                                query_result_ids[test_id].data() + i * recall_at)
                        .second;
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float) (diff.count() * 1000000);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag) {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++) {
                recalls.push_back(tann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                                         query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
                std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float) query_num;

        if (tags) {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float) mean_latency
                      << std::setw(15) << (float) latency_stats[(uint64_t) (0.999 * query_num)];
        } else {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float) mean_latency << std::setw(15)
                      << (float) latency_stats[(uint64_t) (0.999 * query_num)];
        }
        for (double recall: recalls) {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L: Lvec) {
        if (L < recall_at) {
            tann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path = result_path_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        tann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);
        test_id++;
    }

    tann::aligned_free(query);

    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_file, filter_label, label_type,
            query_filters_file;
    uint32_t num_threads, K;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread;
    float fail_if_recall_below = 0.0f;

    turbo::App app;

    app.add_option("-t, --data_type", data_type, "data type <int8/uint8/float>")->required();
    app.add_option("-F, --dist_fn", dist_fn,
                   "distance function <l2/mips/fast_l2/cosine>")->required();
    app.add_option("-x, --index_path_prefix", index_path_prefix,
                   "Path prefix to the index")->required();
    app.add_option("-P, --result_path", result_path,
                   "Path prefix for saving results of the queries")->required();
    app.add_option("-q, --query_file", query_file,
                   "Query file in binary format")->required();
    app.add_option("-l, --filter_label", filter_label,
                   "Filter Label for Filtered Search")->default_val(std::string(""));
    app.add_option("--query_filters_file",
                   query_filters_file,
                   "Filter file for Queries for Filtered Search ")->default_val(std::string(""));
    app.add_option("-z, --label_type", label_type,
                   "Storage type of Labels <uint/ushort>, default value is uint which "
                   "will consume memory 4 bytes per filter")->default_val("uint");
    app.add_option("-g, --gt_file", gt_file,
                   "ground truth file for the queryset")->default_val(std::string("null"));
    app.add_option("--recall_at,-K", K, "Number of neighbors to be returned")->required();
    app.add_option("--print_all_recalls", print_all_recalls,
                   "Print recalls at all positions, from 1 up to specified "
                   "recall_at value");
    app.add_option("--search_list,-L", Lvec, "List of L values of search")->required();
    app.add_option("--num_threads,-T", num_threads,
                   "Number of threads used for building index (defaults to "
                   "omp_get_num_procs())")->default_val(omp_get_num_procs());
    app.add_option("--dynamic", dynamic,
                   "Whether the index is dynamic. Default false.")->default_val(false);
    app.add_option("--tags", tags,
                   "Whether to search with tags. Default false.")->default_val(false);
    app.add_option("--qps_per_thread", show_qps_per_thread,
                   "Print overall QPS divided by the number of threads in "
                   "the output table");
    app.add_option("--fail_if_recall_below", fail_if_recall_below,
                   "If set to a value >0 and <100%, program returns -1 if best recall "
                   "found is below this threshold. ")->default_val(0.0f);

    TURBO_FLAGS_PARSE(app, argc, argv);

    tann::Metric metric;
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float"))) {
        metric = tann::Metric::INNER_PRODUCT;
    } else if (dist_fn == std::string("l2")) {
        metric = tann::Metric::L2;
    } else if (dist_fn == std::string("cosine")) {
        metric = tann::Metric::COSINE;
    } else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float"))) {
        metric = tann::Metric::FAST_L2;
    } else {
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

    if (dynamic && not tags) {
        std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
        return -1;
    }

    if (fail_if_recall_below < 0.0 || fail_if_recall_below >= 100.0) {
        std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
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
            if (data_type == std::string("int8")) {
                return search_memory_index<int8_t, uint16_t>(
                        metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K,
                        print_all_recalls,
                        Lvec, dynamic, tags, show_qps_per_thread, query_filters, fail_if_recall_below);
            } else if (data_type == std::string("uint8")) {
                return search_memory_index<uint8_t, uint16_t>(
                        metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K,
                        print_all_recalls,
                        Lvec, dynamic, tags, show_qps_per_thread, query_filters, fail_if_recall_below);
            } else if (data_type == std::string("float")) {
                return search_memory_index<float, uint16_t>(metric, index_path_prefix, result_path, query_file,
                                                            gt_file,
                                                            num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                            show_qps_per_thread, query_filters,
                                                            fail_if_recall_below);
            } else {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        } else {
            if (data_type == std::string("int8")) {
                return search_memory_index<int8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                   num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                   show_qps_per_thread, query_filters, fail_if_recall_below);
            } else if (data_type == std::string("uint8")) {
                return search_memory_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                    num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                    show_qps_per_thread, query_filters, fail_if_recall_below);
            } else if (data_type == std::string("float")) {
                return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                  show_qps_per_thread, query_filters, fail_if_recall_below);
            } else {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        tann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
