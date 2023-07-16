// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include "tann_cli.h"

namespace detail {
    inline float aggregate_recall(const uint32_t k_aggr, const uint32_t k, const uint32_t npart, uint32_t *count,
                                  const std::vector<float> &recalls) {
        float found = 0;
        for (uint32_t i = 0; i < npart; ++i) {
            size_t max_found = std::min(count[i], k);
            found += recalls[max_found - 1] * max_found;
        }
        return found / (float) k_aggr;
    }

    void simulate(const uint32_t k_aggr, const uint32_t k, const uint32_t npart, const uint32_t nsim,
                  const std::vector<float> &recalls) {
        std::random_device r;
        std::default_random_engine randeng(r());
        std::uniform_int_distribution<int> uniform_dist(0, npart - 1);

        uint32_t *count = new uint32_t[npart];
        double aggr_recall = 0;

        for (uint32_t i = 0; i < nsim; ++i) {
            for (uint32_t p = 0; p < npart; ++p) {
                count[p] = 0;
            }
            for (uint32_t t = 0; t < k_aggr; ++t) {
                count[uniform_dist(randeng)]++;
            }
            aggr_recall += aggregate_recall(k_aggr, k, npart, count, recalls);
        }

        std::cout << "Aggregate recall is " << aggr_recall / (double) nsim << std::endl;
        delete[] count;
    }

    static uint32_t k_aggr;
    static uint32_t k;
    static uint32_t npart ;
    static uint32_t nsim;
    static std::vector<float> recalls;
}

namespace tann {
    void set_simulate_recall(turbo::App &app) {
        auto *sub = app.add_subcommand("simulate_recall", "simulate_recall");
        sub->add_option("-a, --k_aggr", detail::k_aggr, "output")->required();
        sub->add_option("-k, --ktop", detail::k, "data_type")->required();
        sub->add_option("-n, --npart", detail::npart, "Dimensoinality of the vector")->required(true);
        sub->add_option("-s, --nsim", detail::nsim, "Number of vectors")->required(true);
        sub->add_option("-r, --recall", detail::recalls, "Norm of the vectors")->required(true);
        sub->callback([]() {
            simulate_recall();
        });
    }
    void simulate_recall() {

        if (detail::recalls.size() != detail::k) {
            std::cerr << "Please input k numbers for recall@1, recall@2 .. recall@k" << std::endl;
        }
        if (detail::k_aggr > detail::npart * detail::k) {
            std::cerr << "k_aggr must be <= k * npart" << std::endl;
            exit(-1);
        }
        if (detail::nsim <= detail::npart * detail::k_aggr) {
            std::cerr << "Choose nsim > npart*k_aggr" << std::endl;
            exit(-1);
        }

        detail::simulate(detail::k_aggr, detail::k, detail::npart, detail::nsim, detail::recalls);

        return ;
    }
}