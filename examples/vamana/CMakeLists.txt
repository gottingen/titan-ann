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
        build_disk_index
        SOURCES
        "build_disk_index.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        build_memory_index
        SOURCES
        "build_memory_index.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        build_stitched_index
        SOURCES
        "build_stitched_index.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        range_search_disk_index
        SOURCES
        "range_search_disk_index.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        search_disk_index
        SOURCES
        "search_disk_index.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        search_memory_index
        SOURCES
        "search_memory_index.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        test_insert_deletes_consolidate
        SOURCES
        "test_insert_deletes_consolidate.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_binary(
        NAMESPACE tann
        NAME
        test_streaming_scenario
        SOURCES
        "test_streaming_scenario.cc"
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

