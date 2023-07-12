// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>

#ifndef _WINDOWS

#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#endif

#include "tann/vamana/utils.h"
#include "tann/vamana/index.h"
#include "tann/vamana/memory_mapper.h"
#include "tann_cli.h"

namespace detail {
    template<typename T>
    void bfs_count(const std::string &index_path, uint32_t data_dims) {
        using TagT = uint32_t;
        using LabelT = uint32_t;
        tann::Index<T, TagT, LabelT> index(tann::Metric::L2, data_dims, 0, false, false);
        std::cout << "Index class instantiated" << std::endl;
        index.load(index_path.c_str(), 1, 100);
        std::cout << "Index loaded" << std::endl;
        index.count_nodes_at_bfs_levels();
    }

    static std::string data_type;
    static std::string index_path_prefix;
    static std::uint32_t data_dims;
}  // namespace detail

namespace tann {

    void set_count_bfs_level(turbo::App &app) {
        auto *sub = app.add_subcommand("count_bfs_level", "count bfs level");
        sub->add_option("-t, data_type", detail::data_type, "data type <int8/uint8/float>")->required();
        sub->add_option("-p, --index_path_prefix", detail::index_path_prefix,
                        "Path prefix to the index")->required();
        sub->add_option("-d, --data_dims", detail::data_dims, "Dimensionality of the data")->required();
        sub->callback([]() {
            count_bfs_level();
        });
    }

    void count_bfs_level() {
        try {
            if (detail::data_type == std::string("int8"))
                detail::bfs_count<int8_t>(detail::index_path_prefix, detail::data_dims);
            else if (detail::data_type == std::string("uint8"))
                detail::bfs_count<uint8_t>(detail::index_path_prefix, detail::data_dims);
            if (detail::data_type == std::string("float"))
                detail::bfs_count<float>(detail::index_path_prefix, detail::data_dims);
        }
        catch (std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            tann::cerr << "Index BFS failed." << std::endl;
            return ;
        }
    }
}  // namespace tann
