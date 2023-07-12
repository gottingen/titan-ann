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
    void set_uint32_to_uint8(turbo::App &app) {
        auto *sub = app.add_subcommand("uint32_to_uint8", "uint32_to_uint8");
        sub->add_option("-i, --input", detail::input, "input data path")->required();
        sub->add_option("-o, --output", detail::output, "path to output")->required();
        sub->callback([]() {
            uint32_to_uint8();
        });
    }
    void uint32_to_uint8(){
        uint32_t *input;
        size_t npts, nd;
        tann::load_bin<uint32_t>(detail::input, input, npts, nd);
        uint8_t *output = new uint8_t[npts * nd];
        tann::convert_types<uint32_t, uint8_t>(input, output, npts, nd);
        tann::save_bin<uint8_t>(detail::output, output, npts, nd);
        delete[] output;
        delete[] input;
    }
}