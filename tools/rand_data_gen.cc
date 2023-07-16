// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <cstdlib>
#include <random>
#include <cmath>
#include "turbo/flags/flags.h"
#include "tann_cli.h"
#include "tann/common/utils.h"

namespace detail {
    int block_write_float(std::ofstream &writer, size_t ndims, size_t npts, float norm) {
        auto vec = new float[ndims];

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> normal_rand{0, 1};

        for (size_t i = 0; i < npts; i++) {
            float sum = 0;
            for (size_t d = 0; d < ndims; ++d)
                vec[d] = (float) normal_rand(gen);
            for (size_t d = 0; d < ndims; ++d)
                sum += vec[d] * vec[d];
            for (size_t d = 0; d < ndims; ++d)
                vec[d] = vec[d] * norm / std::sqrt(sum);

            writer.write((char *) vec, ndims * sizeof(float));
        }

        delete[] vec;
        return 0;
    }

    int block_write_int8(std::ofstream &writer, size_t ndims, size_t npts, float norm) {
        auto vec = new float[ndims];
        auto vec_T = new int8_t[ndims];

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> normal_rand{0, 1};

        for (size_t i = 0; i < npts; i++) {
            float sum = 0;
            for (size_t d = 0; d < ndims; ++d)
                vec[d] = (float) normal_rand(gen);
            for (size_t d = 0; d < ndims; ++d)
                sum += vec[d] * vec[d];
            for (size_t d = 0; d < ndims; ++d)
                vec[d] = vec[d] * norm / std::sqrt(sum);

            for (size_t d = 0; d < ndims; ++d) {
                vec_T[d] = (int8_t) std::round(vec[d]);
            }

            writer.write((char *) vec_T, ndims * sizeof(int8_t));
        }

        delete[] vec;
        delete[] vec_T;
        return 0;
    }

    int block_write_uint8(std::ofstream &writer, size_t ndims, size_t npts, float norm) {
        auto vec = new float[ndims];
        auto vec_T = new int8_t[ndims];

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<> normal_rand{0, 1};

        for (size_t i = 0; i < npts; i++) {
            float sum = 0;
            for (size_t d = 0; d < ndims; ++d)
                vec[d] = (float) normal_rand(gen);
            for (size_t d = 0; d < ndims; ++d)
                sum += vec[d] * vec[d];
            for (size_t d = 0; d < ndims; ++d)
                vec[d] = vec[d] * norm / std::sqrt(sum);

            for (size_t d = 0; d < ndims; ++d) {
                vec_T[d] = 128 + (int8_t) std::round(vec[d]);
            }

            writer.write((char *) vec_T, ndims * sizeof(uint8_t));
        }

        delete[] vec;
        delete[] vec_T;
        return 0;
    }

    static std::string data_type;
    static std::string output;
    static size_t ndims;
    static size_t npts;
    static float norm;
}
namespace tann {
    void set_rand_data_gen(turbo::App &app) {
        auto *sub = app.add_subcommand("rand_data_gen", "rand_data_gen");
        sub->add_option("-o, --output", detail::output, "output")->required();
        sub->add_option("-t, --data_type", detail::data_type, "data_type")->required();
        sub->add_option("-D, --ndims", detail::ndims, "Dimensoinality of the vector")->required(true);
        sub->add_option("-N, --npts", detail::npts, "Number of vectors")->required(true);
        sub->add_option("-m, --norm", detail::norm, "Norm of the vectors")->required(true);
        sub->callback([]() {
            rand_data_gen();
        });
    }
    void rand_data_gen();
    void rand_data_gen() {


        if (detail::data_type != std::string("float") && detail::data_type != std::string("int8") &&
                detail::data_type != std::string("uint8")) {
            std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
            return ;
        }

        if (detail::norm <= 0.0) {
            std::cerr << "Error: Norm must be a positive number" << std::endl;
            return ;
        }

        if (detail::data_type == std::string("int8") || detail::data_type == std::string("uint8")) {
            if (detail::norm > 127) {
                std::cerr << "Error: for int8/uint8 datatypes, L2 norm can not be "
                             "greater "
                             "than 127"
                          << std::endl;
                return ;
            }
        }

        try {
            std::ofstream writer;
            writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            writer.open(detail::output, std::ios::binary);
            auto npts_u32 = (uint32_t) detail::npts;
            auto ndims_u32 = (uint32_t) detail::ndims;
            writer.write((char *) &npts_u32, sizeof(uint32_t));
            writer.write((char *) &ndims_u32, sizeof(uint32_t));

            size_t blk_size = 131072;
            size_t nblks = ROUND_UP(detail::npts, blk_size) / blk_size;
            std::cout << "# blks: " << nblks << std::endl;

            int ret = 0;
            for (size_t i = 0; i < nblks; i++) {
                size_t cblk_size = std::min(detail::npts - i * blk_size, blk_size);
                if (detail::data_type == std::string("float")) {
                    ret = detail::block_write_float(writer, detail::ndims, cblk_size, detail::norm);
                } else if (detail::data_type == std::string("int8")) {
                    ret = detail::block_write_int8(writer, detail::ndims, cblk_size, detail::norm);
                } else if (detail::data_type == std::string("uint8")) {
                    ret = detail::block_write_uint8(writer, detail::ndims, cblk_size, detail::norm);
                }
                if (ret == 0)
                    std::cout << "Block #" << i << " written" << std::endl;
                else {
                    writer.close();
                    std::cout << "failed to write" << std::endl;
                    return ;
                }
            }
            writer.close();
        }
        catch (const std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            tann::cerr << "Index build failed." << std::endl;
            return ;
        }

        return;
    }
}  // namespace tann
