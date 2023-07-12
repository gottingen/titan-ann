// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "tann/vamana/utils.h"
#include "tann/vamana/disk_utils.h"
#include "tann_cli.h"

namespace detail {
    static std::string input;
    static std::string output;
    static uint32_t  recall_at;
}

namespace tann {
    void set_calc_recall(turbo::App &app) {
        auto *cal_recall =app.add_subcommand("cal_recall", "calculate_recall");
        cal_recall->add_option("-i, --input",detail::input, "input file")->required();
        cal_recall->add_option("-o, --output",detail::output, "output file")->required();
        cal_recall->add_option("-r, --recall",detail::recall_at, "recall")->required();
        cal_recall->callback([](){
            calc_recall();
        });
    }
    void calc_recall() {
        uint32_t *gold_std = NULL;
        float *gs_dist = nullptr;
        uint32_t *our_results = NULL;
        float *or_dist = nullptr;
        size_t points_num, points_num_gs, points_num_or;
        size_t dim_gs;
        size_t dim_or;
        tann::load_truthset(detail::input, gold_std, gs_dist, points_num_gs, dim_gs);
        tann::load_truthset(detail::output, our_results, or_dist, points_num_or, dim_or);

        if (points_num_gs != points_num_or) {
            std::cout << "Error. Number of queries mismatch in ground truth and "
                         "our results"
                      << std::endl;
            return ;
        }
        points_num = points_num_gs;

        if ((dim_or < detail::recall_at) || (detail::recall_at > dim_gs)) {
            std::cout << "ground truth has size " << dim_gs << "; our set has " << dim_or
                      << " points. Asking for recall "
                      << detail::recall_at << std::endl;
        }
        std::cout << "Calculating recall@" << detail::recall_at << std::endl;
        double recall_val = tann::calculate_recall((uint32_t) points_num, gold_std, gs_dist, (uint32_t) dim_gs,
                                                   our_results, (uint32_t) dim_or, (uint32_t) detail::recall_at);

        //  double avg_recall = (recall*1.0)/(points_num*1.0);
        std::cout << "Avg. recall@" << detail::recall_at << " is " << recall_val << "\n";
    }
}