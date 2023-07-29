#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###########################################################################
# below are examples, after initialization, modify it to your own.
###########################################################################
###carbin_example
file(GLOB_RECURSE COMMON_SRC common/*.cc)
file(GLOB_RECURSE CORE_SRC core/*.cc)
file(GLOB_RECURSE STORE_SRC store/*.cc)
file(GLOB_RECURSE IO_SRC io/*.cc)
file(GLOB_RECURSE DISTANCE_SRC distance/*.cc)
file(GLOB_RECURSE CORE_SRC core/*.cc)
file(GLOB_RECURSE FLAT_SRC flat/*.cc)
file(GLOB_RECURSE DS_SRC datasets/*.cc)
file(GLOB_RECURSE HNSW_SRC hnsw/*.cc)

set(TANN_LIB_SRC
        ${COMMON_SRC}
        ${DISKANN_SRC}
        ${STORE_SRC}
        ${IO_SRC}
        ${DISTANCE_SRC}
        ${CORE_SRC}
        ${FLAT_SRC}
        ${DS_SRC}
        ${CORE_SRC}
        ${HNSW_SRC}
        )
add_definitions(
        -DUSE_AVX2
)
carbin_cc_library(
        NAMESPACE tann
        NAME tann
        SOURCES
        ${TANN_LIB_SRC}
        COPTS
        ${CARBIN_CXX_OPTIONS}
        DEPS
        ${CARBIN_DEPS_LINK}
        PUBLIC
)
