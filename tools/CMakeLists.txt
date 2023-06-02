#
# Copyright 2023 The Tann Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

carbin_cc_binary(
        NAMESPACE tann
        NAME
        fvecs_to_bin
        SOURCES
        "fvecs_to_bin.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        fvecs_to_bvecs
        SOURCES
        "fvecs_to_bvecs.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        rand_data_gen
        SOURCES
        "rand_data_gen.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        float_bin_to_int8
        SOURCES
        "float_bin_to_int8.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        ivecs_to_bin
        SOURCES
        "ivecs_to_bin.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        count_bfs_levels
        SOURCES
        "count_bfs_levels.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        tsv_to_bin
        SOURCES
        "tsv_to_bin.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        bin_to_tsv
        SOURCES
        "bin_to_tsv.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        int8_to_float
        SOURCES
        "int8_to_float.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        int8_to_float_scale
        SOURCES
        "int8_to_float_scale.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        create_disk_layout
        SOURCES
        "create_disk_layout.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        compute_groundtruth
        SOURCES
        "compute_groundtruth.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        calculate_recall
        SOURCES
        "calculate_recall.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)
carbin_cc_binary(
        NAMESPACE tann
        NAME
        generate_synthetic_labels
        SOURCES
        "generate_synthetic_labels.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        gen_random_slice
        SOURCES
        "gen_random_slice.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        generate_pq
        SOURCES
        "generate_pq.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        partition_with_ram_budget
        SOURCES
        "partition_with_ram_budget.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        simulate_aggregate_recall
        SOURCES
        "simulate_aggregate_recall.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        stats_label_data
        SOURCES
        "stats_label_data.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        uint8_to_float
        SOURCES
        "uint8_to_float.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)


carbin_cc_binary(
        NAMESPACE tann
        NAME
        uint32_to_uint8
        SOURCES
        "uint32_to_uint8.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        vector_analysis
        SOURCES
        "vector_analysis.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        merge_shards
        SOURCES
        "merge_shards.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        partition_data
        SOURCES
        "partition_data.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::graph
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
        PUBLIC
)