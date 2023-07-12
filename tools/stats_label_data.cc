// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include <string.h>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <set>
#include "tann_cli.h"
#include "tann/common/utils.h"

#ifndef _WINDOWS

#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#else
#include <Windows.h>
#endif

namespace detail {
    void stats_analysis(const std::string labels_file, std::string univeral_label, uint32_t density = 10) {
        std::string token, line;
        std::ifstream labels_stream(labels_file);
        std::unordered_map<std::string, uint32_t> label_counts;
        std::string label_with_max_points;
        uint32_t max_points = 0;
        long long sum = 0;
        long long point_cnt = 0;
        float avg_labels_per_pt, mean_label_size;

        std::vector<uint32_t> labels_per_point;
        uint32_t dense_pts = 0;
        if (labels_stream.is_open()) {
            while (getline(labels_stream, line)) {
                point_cnt++;
                std::stringstream iss(line);
                uint32_t lbl_cnt = 0;
                while (getline(iss, token, ',')) {
                    lbl_cnt++;
                    token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                    token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                    if (label_counts.find(token) == label_counts.end())
                        label_counts[token] = 0;
                    label_counts[token]++;
                }
                if (lbl_cnt >= density) {
                    dense_pts++;
                }
                labels_per_point.emplace_back(lbl_cnt);
            }
        }

        std::cout << "fraction of dense points with >= " << density
                  << " labels = " << (float) dense_pts / (float) labels_per_point.size() << std::endl;
        std::sort(labels_per_point.begin(), labels_per_point.end());

        std::vector<std::pair<std::string, uint32_t>> label_count_vec;

        for (auto it = label_counts.begin(); it != label_counts.end(); it++) {
            auto &lbl = *it;
            label_count_vec.emplace_back(std::make_pair(lbl.first, lbl.second));
            if (lbl.second > max_points) {
                max_points = lbl.second;
                label_with_max_points = lbl.first;
            }
            sum += lbl.second;
        }

        sort(label_count_vec.begin(), label_count_vec.end(),
             [](const std::pair<std::string, uint32_t> &lhs, const std::pair<std::string, uint32_t> &rhs) {
                 return lhs.second < rhs.second;
             });

        for (float p = 0; p < 1; p += 0.05) {
            std::cout << "Percentile " << (100 * p) << "\t"
                      << label_count_vec[(size_t) (p * label_count_vec.size())].first
                      << " with count=" << label_count_vec[(size_t) (p * label_count_vec.size())].second << std::endl;
        }

        std::cout << "Most common label "
                  << "\t" << label_count_vec[label_count_vec.size() - 1].first
                  << " with count=" << label_count_vec[label_count_vec.size() - 1].second << std::endl;
        if (label_count_vec.size() > 1)
            std::cout << "Second common label "
                      << "\t" << label_count_vec[label_count_vec.size() - 2].first
                      << " with count=" << label_count_vec[label_count_vec.size() - 2].second << std::endl;
        if (label_count_vec.size() > 2)
            std::cout << "Third common label "
                      << "\t" << label_count_vec[label_count_vec.size() - 3].first
                      << " with count=" << label_count_vec[label_count_vec.size() - 3].second << std::endl;
        avg_labels_per_pt = sum / (float) point_cnt;
        mean_label_size = sum / (float) label_counts.size();
        std::cout << "Total number of points = " << point_cnt << ", number of labels = " << label_counts.size()
                  << std::endl;
        std::cout << "Average number of labels per point = " << avg_labels_per_pt << std::endl;
        std::cout << "Mean label size excluding 0 = " << mean_label_size << std::endl;
        std::cout << "Most popular label is " << label_with_max_points << " with " << max_points << " pts" << std::endl;
    }
    std::string labels_file, universal_label;
    uint32_t density;
}  // namespace detail

namespace tann {
    void set_stats_label_data(turbo::App &app) {
        auto *sub = app.add_subcommand("stats_label_data", "stats_label_data");
        sub->add_option("-l, --labels_file", detail::labels_file,
                           "path to labels data file.")->required();
        sub->add_option("-u, --universal_label", detail::universal_label,
                           "Universal label used in labels file.")->required();
        sub->add_option("-d, --density", detail::density,
                           "Number of labels each point in labels file, defaults to 1")->default_val(1);
        sub->callback([]() {
            stats_label_data();
        });
    }
    void stats_label_data() {
        detail::stats_analysis(detail::labels_file, detail::universal_label, detail::density);
    }
}  // namespace tann
