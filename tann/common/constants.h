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

#ifndef TANN_COMMON_CONSTANTS_H_
#define TANN_COMMON_CONSTANTS_H_

#include <cstddef>

namespace tann::constants {

    /// for common
    static constexpr size_t kMaxElements = 100000;
    static constexpr size_t kBatchSize = 256;
    /// for hnsw
    static constexpr size_t kHnswM = 16;
    static constexpr size_t kHnswEf = 50;
    static constexpr size_t kHnswEfConstruction = 200;
    static constexpr size_t kHnswRandomSeed = 100;
}  // namespace tann::constants

#endif  // TANN_COMMON_CONSTANTS_H_
