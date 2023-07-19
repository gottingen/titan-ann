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
#ifndef TANN_CORE_TYPES_H_
#define TANN_CORE_TYPES_H_

#include <cstdint>
#include <cstddef>
#include "turbo/log/logging.h"
namespace tann {
    typedef uint32_t location_t;
    typedef size_t label_type;
    typedef double distance_type;

    enum class EngineType {
        ENGINE_FLAT,
        ENGINE_PQ,
        ENGINE_VAMANA,
        ENGINE_HNSW,
        ENGINE_SPTAG
    };

    enum class DataType {
        DT_NONE = 0,
        DT_UINT8,
        DT_FLOAT16,
        DT_FLOAT
    };

    inline size_t data_type_size(DataType dt) {
        switch (dt) {
            case DataType::DT_UINT8:
                return 1;
            case DataType::DT_FLOAT16:
                return 2;
            case DataType::DT_FLOAT:
                return 4;
            default:
                break;
        }
        TLOG_CHECK(false, "unknown type");
        return 0;
    }

    enum MetricType {
        UNDEFINED = 0,
        METRIC_L1,
        METRIC_L2,
        METRIC_IP,
        METRIC_HAMMING,
        METRIC_JACCARD,
        METRIC_COSINE,
        METRIC_ANGLE,
        METRIC_NORMALIZED_COSINE,
        METRIC_NORMALIZED_ANGLE,
        METRIC_NORMALIZED_L2,
        METRIC_POINCARE,
        METRIC_LORENTZ,
    };

    class BaseFilterFunctor {
    public:
        virtual ~BaseFilterFunctor() = default;

        virtual bool operator()(label_type id) { return true; }
    };

} // namespace tann
#endif  // TANN_CORE_TYPES_H_
