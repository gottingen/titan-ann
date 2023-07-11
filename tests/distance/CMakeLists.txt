#
# Copyright 2023 The titan-search Authors.
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

message(STATUS ${CARBIN_SIMD_FLAGS})
carbin_cc_test(
        NAME
        distance_test
        SOURCES
        distance_test.cc
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-Wno-unused-but-set-variable"
        "-Wno-unused-variable"
        DEPS
        ${CARBIN_DEPS_LINK}
        turbo
        tann::ngt
)