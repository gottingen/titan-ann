// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <random>
#include <math.h>
#include <cmath>
#include "tann/vamana/utils.h"
#include "tann_cli.h"

namespace detail {

    class ZipfDistribution {
    public:
        ZipfDistribution(uint64_t num_points, uint32_t num_labels)
                : num_labels(num_labels), num_points(num_points),
                  uniform_zero_to_one(std::uniform_real_distribution<>(0.0, 1.0)) {
        }

        std::unordered_map<uint32_t, uint32_t> createDistributionMap() {
            std::unordered_map<uint32_t, uint32_t> map;
            uint32_t primary_label_freq = (uint32_t) ceil(num_points * distribution_factor);
            for (uint32_t i{1}; i < num_labels + 1; i++) {
                map[i] = (uint32_t) ceil(primary_label_freq / i);
            }
            return map;
        }

        int writeDistribution(std::ofstream &outfile) {
            auto distribution_map = createDistributionMap();
            for (uint32_t i{0}; i < num_points; i++) {
                bool label_written = false;
                for (auto it = distribution_map.cbegin(); it != distribution_map.cend(); it++) {
                    auto label_selection_probability = std::bernoulli_distribution(
                            distribution_factor / (double) it->first);
                    if (label_selection_probability(rand_engine) && distribution_map[it->first] > 0) {
                        if (label_written) {
                            outfile << ',';
                        }
                        outfile << it->first;
                        label_written = true;
                        // remove label from map if we have used all labels
                        distribution_map[it->first] -= 1;
                    }
                }
                if (!label_written) {
                    outfile << 0;
                }
                if (i < num_points - 1) {
                    outfile << '\n';
                }
            }
            return 0;
        }

        int writeDistribution(std::string filename) {
            std::ofstream outfile(filename);
            if (!outfile.is_open()) {
                std::cerr << "Error: could not open output file " << filename << '\n';
                return -1;
            }
            writeDistribution(outfile);
            outfile.close();
        }

    private:
        const uint32_t num_labels;
        const uint64_t num_points;
        const double distribution_factor = 0.7;
        std::knuth_b rand_engine;
        const std::uniform_real_distribution<double> uniform_zero_to_one;
    };

    static std::string output;
    static uint32_t num_labels;
    static uint64_t num_points;
    static std::string distribution_type;
}
namespace tann {
    void set_gen_label(turbo::App &app) {
        auto *sub = app.add_subcommand("gen_label", "gen_random_slice");
        sub->add_option("-o, --output", detail::output, "path to output")->required();
        sub->add_option("-n, --num_labels", detail::num_labels,
                        "Number of unique labels, up to 5000")->required();
        sub->add_option("-N, --num_points", detail::num_points,
                        "Number of points in dataset")->required();
        sub->add_option("-D, --distribution_type",
                        detail::distribution_type,
                        "Distribution function for labels <random/zipf/one_per_point> defaults "
                        "to random")->default_val("random");

        sub->callback([]() {
            gen_label();
        });
    }

    void gen_label() {

        if (detail::num_labels > 5000) {
            std::cerr << "Error: num_labels must be 5000 or less" << '\n';
            return;
        }

        if (detail::num_points <= 0) {
            std::cerr << "Error: num_points must be greater than 0" << '\n';
            return;
        }

        std::cout << "Generating synthetic labels for " << detail::num_points << " points with " << detail::num_labels
                  << " unique labels"
                  << '\n';

        try {
            std::ofstream outfile(detail::output);
            if (!outfile.is_open()) {
                std::cerr << "Error: could not open output file " << detail::output << '\n';
                return;
            }

            if (detail::distribution_type == "zipf") {
                detail::ZipfDistribution zipf(detail::num_points, detail::num_labels);
                zipf.writeDistribution(outfile);
            } else if (detail::distribution_type == "random") {
                for (size_t i = 0; i < detail::num_points; i++) {
                    bool label_written = false;
                    for (size_t j = 1; j <= detail::num_labels; j++) {
                        // 50% chance to assign each label
                        if (rand() > (RAND_MAX / 2)) {
                            if (label_written) {
                                outfile << ',';
                            }
                            outfile << j;
                            label_written = true;
                        }
                    }
                    if (!label_written) {
                        outfile << 0;
                    }
                    if (i < detail::num_points - 1) {
                        outfile << '\n';
                    }
                }
            } else if (detail::distribution_type == "one_per_point") {
                std::random_device rd;                                // obtain a random number from hardware
                std::mt19937 gen(rd());                               // seed the generator
                std::uniform_int_distribution<> distr(0, detail::num_labels); // define the range

                for (size_t i = 0; i < detail::num_points; i++) {
                    outfile << distr(gen);
                    if (i != detail::num_points - 1)
                        outfile << '\n';
                }
            }
            if (outfile.is_open()) {
                outfile.close();
            }

            std::cout << "Labels written to " << detail::output << '\n';
        }
        catch (const std::exception &ex) {
            std::cerr << "Label generation failed: " << ex.what() << '\n';
            return;
        }

        return;
    }
}