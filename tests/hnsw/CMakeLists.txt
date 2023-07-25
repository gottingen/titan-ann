# Copyright 2023 The titan-search Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

carbin_cc_test(
        NAME
        add_remove_test
        SOURCES
        add_remove_test.cc
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-ggdb3"
        "-g"
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_test(
        NAME
        leveled_graph_test
        SOURCES
        leveled_graph_test.cc
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-ggdb3"
        "-g"
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_test(
        NAME
        multi_thread_replace_test
        SOURCES
        multi_thread_replace_test.cc
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-ggdb3"
        "-g"
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

carbin_cc_test(
        NAME
        search_closer_first_test
        SOURCES
        search_closer_first_test.cc
        COPTS
        ${CARBIN_CXX_OPTIONS}
        "-ggdb3"
        "-g"
        DEPS
        tann::tann
        ${CARBIN_DEPS_LINK}
)

