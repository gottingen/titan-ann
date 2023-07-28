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
#ifndef TANN_CORE_INDEX_OPTION_H_
#define TANN_CORE_INDEX_OPTION_H_

#include "tann/core/types.h"

namespace tann {

    struct IndexOption {
        // common option
        DataType data_type{DataType::DT_NONE};
        MetricType metric{MetricType::UNDEFINED};
        EngineType engine_type{EngineType::ENGINE_HNSW};
        size_t dimension{0};
        size_t batch_size{constants::kBatchSize};
        size_t max_elements{constants::kMaxElements};
        size_t number_thread{4};
        bool enable_replace_vacant{true};
    };

    struct HnswIndexOption : public IndexOption {
        size_t m{constants::kHnswM};
        size_t ef_construction{constants::kHnswEfConstruction};
        size_t ef{constants::kHnswEf};
        size_t random_seed{constants::kHnswRandomSeed};
    };

}  // namespace tann

#endif  // TANN_CORE_INDEX_OPTION_H_
