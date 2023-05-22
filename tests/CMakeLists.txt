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

find_package(Boost REQUIRED)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        build_disk_index
        SOURCES
        "build_disk_index.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        build_memory_index
        SOURCES
        "build_memory_index.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        build_stitched_index
        SOURCES
        "build_stitched_index.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        range_search_disk_index
        SOURCES
        "range_search_disk_index.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        search_disk_index
        SOURCES
        "search_disk_index.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        search_memory_index
        SOURCES
        "search_memory_index.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        test_insert_deletes_consolidate
        SOURCES
        "test_insert_deletes_consolidate.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        test_streaming_scenario
        SOURCES
        "test_streaming_scenario.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        fvecs_to_bin
        SOURCES
        "utils/fvecs_to_bin.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        fvecs_to_bvecs
        SOURCES
        "utils/fvecs_to_bvecs.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        rand_data_gen
        SOURCES
        "utils/rand_data_gen.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        float_bin_to_int8
        SOURCES
        "utils/float_bin_to_int8.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        ivecs_to_bin
        SOURCES
        "utils/ivecs_to_bin.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        count_bfs_levels
        SOURCES
        "utils/count_bfs_levels.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        tsv_to_bin
        SOURCES
        "utils/tsv_to_bin.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        bin_to_tsv
        SOURCES
        "utils/bin_to_tsv.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        int8_to_float
        SOURCES
        "utils/int8_to_float.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        int8_to_float_scale
        SOURCES
        "utils/int8_to_float_scale.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        create_disk_layout
        SOURCES
        "utils/create_disk_layout.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        compute_groundtruth
        SOURCES
        "utils/compute_groundtruth.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        calculate_recall
        SOURCES
        "utils/calculate_recall.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)
carbin_cc_binary(
        NAMESPACE tann
        NAME
        generate_synthetic_labels
        SOURCES
        "utils/generate_synthetic_labels.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        gen_random_slice
        SOURCES
        "utils/gen_random_slice.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        generate_pq
        SOURCES
        "utils/generate_pq.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        partition_with_ram_budget
        SOURCES
        "utils/partition_with_ram_budget.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        simulate_aggregate_recall
        SOURCES
        "utils/simulate_aggregate_recall.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        stats_label_data
        SOURCES
        "utils/stats_label_data.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        uint8_to_float
        SOURCES
        "utils/uint8_to_float.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)


carbin_cc_binary(
        NAMESPACE tann
        NAME
        uint32_to_uint8
        SOURCES
        "utils/uint32_to_uint8.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        vector_analysis
        SOURCES
        "utils/vector_analysis.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        merge_shards
        SOURCES
        "utils/merge_shards.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        partition_data
        SOURCES
        "utils/partition_data.cc"
        COPTS
        ${TURBO_DEFAULT_COPTS}
        ${TURBO_RANDOM_RANDEN_COPTS}
        ${TURBO_SIMD_FLAGS}
        DEPS
        tann::tann
        ${TURBO_DYLINK}
        ${MKL_LIBRARIES}
        boost_program_options
)