// Copyright 2023 The Turbo Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "turbo/flags/flags.h"
#include "turbo/format/print.h"
#include "tann_cli.h"

int main(int argc, char **argv) {
    turbo::App app{"elastic ann search client"};
    app.callback([&app] {
        if (app.get_subcommands().empty()) {
            turbo::Println("{}", app.help());
        }
    });
    app.add_flag("-v, --verbose", tann::verbose,
                 "verbose detail message")->default_val(false);
    // Call the setup functions for the subcommands.
    // They are kept alive by a shared pointer in the
    // lambda function
    tann::set_bin_to_tsv(app);
    tann::set_calc_recall(app);
    tann::set_compute_groundtruth(app);
    tann::set_compute_groundtruth_filter(app);
    tann::set_count_bfs_level(app);
    tann::set_create_disk_layout(app);
    tann::set_fbin_to_int8(app);
    tann::set_fvec_to_bin(app);
    tann::set_fvec_to_bvec(app);
    tann::set_gen_randome_slice(app);
    tann::set_gen_pq(app);
    tann::set_gen_label(app);
    tann::set_int8_to_float(app);
    tann::set_int8_to_float_scale(app);
    tann::set_ivecs_to_bin(app);
    tann::set_merge_shards(app);
    tann::set_rand_data_gen(app);
    tann::set_simulate_recall(app);
    tann::set_stats_label_data(app);
    tann::set_tsv_to_bin(app);
    tann::set_uint8_to_float(app);
    tann::set_uint32_to_uint8(app);
    tann::set_vector_analysis(app);
    // More setup if needed, i.e., other subcommands etc.

    TURBO_FLAGS_PARSE(app, argc, argv);

    return 0;
}