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
#ifndef TANN_INDEX_OPTION_H_
#define TANN_INDEX_OPTION_H_

#include <string>
#include <cstddef>
#include "tann/index/metric.h"

namespace tann {

    struct IndexOption {
        struct HnswOption {
            size_t m;
            size_t ef_construction;
            size_t ef;
            size_t max_elements;
        };
        HnswOption hnsw;
        MetricType metric;
        size_t dimension;
        std::string index_name;
    };
}  // namespace tann

#endif  // TANN_INDEX_OPTION_H_
