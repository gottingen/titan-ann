// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "tann/vamana/utils.h"
#include "tann_cli.h"

namespace detail {
    static std::string input;
    static std::string output;
}
namespace tann {
    void set_uint8_to_float(turbo::App &app) {
        auto *sub = app.add_subcommand("uint8_to_float", "uint8_to_float");
        sub->add_option("-i, --input", detail::input, "input data path")->required();
        sub->add_option("-o, --output", detail::output, "path to output")->required();
        sub->callback([]() {
            uint8_to_float();
        });
    }
    void uint8_to_float() {
        uint8_t *input;
        size_t npts, nd;
        tann::load_bin<uint8_t>(detail::input, input, npts, nd);
        float *output = new float[npts * nd];
        tann::convert_types<uint8_t, float>(input, output, npts, nd);
        tann::save_bin<float>(detail::output, output, npts, nd);
        delete[] output;
        delete[] input;
    }
}
