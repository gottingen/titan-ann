// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include "tann/diskann/partition.h"
#include "tann/common/utils.h"
#include "tann_cli.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

namespace detail {
    template<typename T>
    int aux_main(const std::string & base_file, const std::string &output_prefix, float sampling_rate) {
        tann::gen_random_slice<T>(base_file, output_prefix, sampling_rate);
        return 0;
    }

    static std::string base_file;
    static std::string output;
    static float rate;
    static std::string data_type;
}  // namespace detail

namespace tann {
    void set_gen_randome_slice(turbo::App &app) {
        auto *sub = app.add_subcommand("gen_random_slice", "gen_random_slice");
        sub->add_option("-o, --output", detail::output, "path to output")->required();
        sub->add_option("-b,--base_file", detail::base_file, "base file")->required();
        sub->add_option("-R, --rate", detail::rate, "rate")->required();
        sub->add_option("-t, --data_type", detail::data_type, "data_type")->required();
        sub->callback([]() {
            gen_randome_slice();
        });
    }

    void gen_randome_slice() {

        if (detail::data_type == std::string("float")) {
            detail::aux_main<float>(detail::base_file, detail::output, detail::rate);
        } else if (detail::data_type == std::string("int8")) {
            detail::aux_main<int8_t>(detail::base_file, detail::output, detail::rate);
        } else if (detail::data_type== std::string("uint8")) {
            detail::aux_main<uint8_t>(detail::base_file, detail::output, detail::rate);
        } else
            std::cout << "Unsupported type. Use float/int8/uint8." << std::endl;
        return ;
    }
}  // namespace tann
