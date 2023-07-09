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


#ifndef TANN_DISTANCE_DISTANCE_H_
#define TANN_DISTANCE_DISTANCE_H_

#include "turbo/meta/span.h"
#include "tann/index/option.h"
#include "turbo/platform/port.h"
#include "turbo/log/logging.h"
#include "turbo/simd/simd.h"
#include "tann/common/half.hpp"
#include <vector>

namespace tann {

    template<typename T>
    struct Distance {
    public:
        virtual ~Distance() = default;
        // distance comparison function
        TURBO_DLL [[nodiscard]] virtual float compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const = 0;

        TURBO_DLL [[nodiscard]] virtual bool need_normalization() const noexcept = 0;

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const = 0;
    };

    template<typename T>
    struct DistanceL1 : public Distance<T> {
    public:
        // distance comparison function
        TURBO_DLL [[nodiscard]] float compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const override {
            TLOG_CHECK(x.size() == y.size(), "x and y must be equal");
            using b_type = turbo::simd::batch<T>;
            std::size_t inc = b_type::size;
            std::size_t size = x.size();
            std::size_t vec_size = size - size % inc;
            float sum = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type bx = turbo::simd::load_aligned(x.data() + i);
                b_type by = turbo::simd::load_aligned(y.data() + i);
                auto c = turbo::simd::abs(bx - by);
                sum += turbo::simd::reduce_add(c);
            }
            for (std::size_t i = vec_size; i < size; ++i) {
                sum += std::abs(x[i] - y[i]);
            }
            return sum;

        }

        TURBO_DLL [[nodiscard]] virtual bool need_normalization() const noexcept override {
            return false;
        }

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const override {
            TLOG_CHECK(false, "not impl");
        }
    };
}  // namespace tann

#endif  // TANN_DISTANCE_DISTANCE_H_
