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

file(GLOB_RECURSE CORE_SRC "core/*.cc")
file(GLOB_RECURSE UTILITY_SRC "utility/*.cc")
file(GLOB_RECURSE IO_SRC "io/*.cc")
file(GLOB_RECURSE QUANTIZER_SRC "quantizer/*.cc")
file(GLOB_RECURSE VECTOR_SRC "vector/*.cc")
file(GLOB_RECURSE GRAPH_SRC "graph/*.cc")
file(GLOB_RECURSE DISTANCE_SRC "distance/*.cc")
set(LIB_SRC
        ${CORE_SRC}
        ${UTILITY_SRC}
        ${IO_SRC}
        ${QUANTIZER_SRC}
        ${VECTOR_SRC}
        ${GRAPH_SRC}
        ${DISTANCE_SRC}
        )
carbin_cc_library(
        NAMESPACE tann
        NAME tann
        SOURCES
        ${LIB_SRC}
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-Wno-pessimizing-move"
        "-Wno-delete-non-virtual-dtor"
        "-Wno-reorder"
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo
        PUBLIC
)
