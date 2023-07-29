// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "tann/diskann/disk_utils.h"
#include "tann/io/cached_io.h"
#include "tann/common/utils.h"
#include "tann_cli.h"

namespace detail {

    static std::string vamana_prefix;
    static std::string vamana_suffix;
    static std::string idmaps_prefix;
    static std::string idmaps_suffix;
    static uint64_t nshards;
    static uint32_t max_degree;

    static std::string output_index;
    static std::string output_medoids;


}
namespace tann {
    void set_merge_shards(turbo::App &app) {
        auto *sub = app.add_subcommand("merge_shards", "merge_shards");
        sub->add_option("-p, --vamana_prefix", detail::vamana_prefix, "input data path")->required();
        sub->add_option("-s, --vamana_suffix", detail::vamana_suffix, "path to output")->required();
        sub->add_option("-i, --idmaps_prefix", detail::idmaps_prefix, "path to output")->required();
        sub->add_option("-I, --idmaps_suffix", detail::idmaps_suffix, "path to output")->required();
        sub->add_option("-n, --nshards", detail::nshards, "path to output")->required();
        sub->add_option("-m, --max_degree", detail::max_degree, "path to output")->required();
        sub->add_option("-o, --output_medoids", detail::output_medoids, "path to output")->required();
        sub->add_option("-x, --output_index", detail::output_index, "path to output")->required();
        sub->callback([]() {
            merge_shards();
        });
    }

    void merge_shards() {

        tann::merge_shards(detail::vamana_prefix, detail::vamana_suffix, detail::idmaps_prefix, detail::idmaps_suffix,
                           detail::nshards, detail::max_degree,
                           detail::output_index, detail::output_medoids);
    }
}  // namespace tann
