// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "tann/common/utils.h"
#include "tann/vamana/disk_utils.h"
#include "tann/io/cached_io.h"
#include "tann_cli.h"

namespace detail {
    template<typename T>
    int create_disk_layout(const std::string &base_file,const std::string &vamana_file, const std::string &output_file) {
        tann::create_disk_layout<T>(base_file, vamana_file, output_file);
        return 0;
    }
    static std::string data_type;
    static std::string base_file;
    static std::string vamana_file;
    static std::string output_file;
}  // namespace detail
namespace tann {
    void set_create_disk_layout(turbo::App &app) {
        auto *sub = app.add_subcommand("create_disk_layout", "count bfs level");
        sub->add_option("-t, --data_type", detail::data_type, "data type <int8/uint8/float>")->required();
        sub->add_option("-b, --base_file", detail::base_file, "distance function <l2/mips>")->required();
        sub->add_option("-v, --vamana", detail::vamana_file,"File containing the base vectors in binary format")->required();
        sub->add_option("-o, --output", detail::output_file,"File containing the query vectors in binary format")->required();
        sub->callback([]() {
            create_disk_layout();
        });
    }
    void create_disk_layout() {
        int ret_val = -1;
        if (std::string(detail::data_type) == std::string("float"))
            ret_val = detail::create_disk_layout<float>(detail::base_file, detail::vamana_file, detail::output_file);
        else if (std::string(detail::data_type) == std::string("int8"))
            ret_val = detail::create_disk_layout<int8_t>(detail::base_file, detail::vamana_file, detail::output_file);
        else if (std::string(detail::data_type) == std::string("uint8"))
            ret_val = detail::create_disk_layout<uint8_t>(detail::base_file, detail::vamana_file, detail::output_file);
        else {
            std::cout << "unsupported type. use int8/uint8/float " << std::endl;
            ret_val = -2;
        }
        return;
    }
}  // namespace tann