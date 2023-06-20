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


#ifndef TANN_INDEX_DISTANCE_H_
#define TANN_INDEX_DISTANCE_H_

#include "turbo/platform/port.h"
#include "tann/index/metric.h"
#include <cstddef>

namespace tann {

    template<typename T>
    struct Distance {
    public:
        explicit Distance(MetricType type) : metric(type) {}

        virtual ~Distance() = default;
        // distance comparison function
        TURBO_DLL virtual float compare(T *x, T *y, size_t len) const = 0;

        TURBO_DLL [[nodiscard]] virtual bool need_normalization() const noexcept { return false; }

        TURBO_DLL virtual void normalization(const T *query, size_t len, T *dst) const = 0;

        MetricType metric;
        size_t alignment_factor = 8;

    };
}  // namespace tann

#endif  // TANN_INDEX_DISTANCE_H_
