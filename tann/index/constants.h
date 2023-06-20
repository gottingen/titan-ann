// Copyright 2023 The titan-search Authors.
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


#ifndef TANN_INDEX_CONSTANTS_H_
#define TANN_INDEX_CONSTANTS_H_

#include <cstddef>

namespace tann {

    constexpr size_t kDefaultMaxItem = 1000000;
    namespace hnsw {
        constexpr size_t DEFAULT_M = 16;
        constexpr size_t DEFAULT_EF_CONSTRUCTION = 200;
        constexpr size_t DEFAULT_EF = 50;
        constexpr size_t DEFAULT_MAX_ELEMENTS = 50000;
        constexpr size_t DEFAULT_RANDOM_SEED = 100;

    }

}  // namespace tann
#endif  // TANN_INDEX_CONSTANTS_H_
