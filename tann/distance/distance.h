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
#include "turbo/simd/memory/alignment.h"
#include "turbo/base/bits.h"
#include "tann/common/half.hpp"
#include <vector>

namespace tann {

    template<class T, class Tag, class Arch = turbo::simd::default_arch>
    struct Distance {
    public:
        virtual ~Distance() = default;
        // distance comparison function
        TURBO_DLL [[nodiscard]] virtual double compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const = 0;

        TURBO_DLL [[nodiscard]] virtual bool need_normalization() const noexcept = 0;

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const = 0;
    };

    template<class T, class Tag, class Arch = turbo::simd::default_arch>
    struct DistanceL1 : public Distance<T, Tag, Arch> {
    public:
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const override {
            TLOG_CHECK(x.size() == y.size(), "x and y must be equal");
            using b_type = turbo::simd::batch<T, Arch>;
            std::size_t inc = b_type::size;
            std::size_t size = x.size();
            std::size_t vec_size = size - size % inc;
            double sum = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type bx = turbo::simd::load<Arch>(&x[i], Tag());
                b_type by = turbo::simd::load<Arch>(&y[i], Tag());
                auto c = turbo::simd::abs(bx - by);
                sum += turbo::simd::reduce_add(c);
            }
            for (std::size_t i = vec_size; i < size; ++i) {
                sum += std::abs(x[i] - y[i]);
                TLOG_INFO(1);
            }
            TLOG_INFO(1);
            return sum;

        }

        TURBO_DLL [[nodiscard]] bool need_normalization() const noexcept override {
            return false;
        }

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const override {
            TLOG_CHECK(false, "not impl");
        }
    };

    template<class T, class Tag, class Arch = turbo::simd::default_arch>
    struct DistanceL2 : public Distance<T, Tag, Arch> {
    public:
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const override {
            TLOG_CHECK(x.size() == y.size(), "x and y must be equal");
            using b_type = turbo::simd::batch<T, Arch>;
            std::size_t inc = b_type::size;
            std::size_t size = x.size();
            std::size_t vec_size = size - size % inc;
            auto vsum = b_type::broadcast(0.0);
            double sum = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type bx = turbo::simd::load<Arch>(&x[i], Tag());
                b_type by = turbo::simd::load<Arch>(&y[i], Tag());
                auto c = bx - by;
                vsum += turbo::simd::mul(c, c);
            }
            for (size_t i = 0; i < inc; ++i) {
                sum += vsum.get(i);
            }
            for (std::size_t i = vec_size; i < size; ++i) {
                sum += (x[i] - y[i]) * (x[i] - y[i]);
            }
            return std::sqrt(sum);
        }

        TURBO_DLL [[nodiscard]] bool need_normalization() const noexcept override {
            return false;
        }

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const override {
            TLOG_CHECK(false, "not impl");
        }
    };

    template<class T, class Tag, class Arch = turbo::simd::default_arch>
    struct DistanceHamming : public Distance<T, Tag, Arch> {
    public:
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const override {
            size_t mem_size = x.size() * sizeof(T);
            TLOG_CHECK(x.size() == y.size(), "x and y must be equal");
            TLOG_CHECK(mem_size % sizeof(uint64_t) == 0);
            size_t usize = mem_size / sizeof(uint64_t);
            uint64_t *px = reinterpret_cast<uint64_t *>(x.data());
            uint64_t *py = reinterpret_cast<uint64_t *>(y.data());
            size_t sum = 0.0;
            for (size_t i = 0; i < usize; ++i) {
                sum += turbo::popcount(*(px + i) ^ *(py + i));
            }
            return static_cast<double>(sum);
        }

        TURBO_DLL [[nodiscard]] bool need_normalization() const noexcept override {
            return false;
        }

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const override {
            TLOG_CHECK(false, "not impl");
        }
    };

    template<class T, class Tag, class Arch = turbo::simd::default_arch>
    struct DistanceJaccard : public Distance<T, Tag, Arch> {
    public:
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const override {
            size_t mem_size = x.size() * sizeof(T);
            TLOG_CHECK(x.size() == y.size(), "x and y must be equal");
            TLOG_CHECK(mem_size % sizeof(uint64_t) == 0, "total memory size must be aligned sizeof(uint64-t)");
            size_t usize = mem_size / sizeof(uint64_t);
            auto *px = reinterpret_cast<uint64_t *>(x.data());
            auto *py = reinterpret_cast<uint64_t *>(y.data());
            size_t sum = 0;
            size_t sum_de = 0;
            for (size_t i = 0; i < usize; ++i) {
                sum += turbo::popcount(*(px + i) & *(py + i));
                sum_de += turbo::popcount(*(px + i) | *(py + i));
            }
            return 1.0 - static_cast<double>(sum) / static_cast<double>(sum_de);
        }

        TURBO_DLL [[nodiscard]] bool need_normalization() const noexcept override {
            return false;
        }

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const override {
            TLOG_CHECK(false, "not impl");
        }
    };

    template<class T, class Tag, class Arch = turbo::simd::default_arch>
    struct DistanceCosine : public Distance<T, Tag, Arch> {
    public:
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const override {
            TLOG_CHECK(x.size() == y.size(), "x and y must be equal");
            using b_type = turbo::simd::batch<T, Arch>;

            std::size_t inc = b_type::size;
            std::size_t size = x.size();

            std::size_t vec_size = size - size % inc;
            auto sum_bx = b_type::broadcast(0.0);
            auto sum_by = b_type::broadcast(0.0);
            auto sum_b = b_type::broadcast(0.0);
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type bx = turbo::simd::load<Arch>(&x[i], Tag());
                b_type by = turbo::simd::load<Arch>(&y[i], Tag());
                sum_bx += bx * bx;
                sum_by += by * by;
                sum_b += turbo::simd::mul(bx, by);
            }
            double sum_x = 0;
            double sum_y = 0;
            double sum = 0;
            for (size_t i = vec_size; i < size; ++i) {
                sum_x += x[i] * x[i];
                sum_x += y[i] * y[i];
                sum += x[i] * y[i];
            }
            for (size_t i = 0; i < inc; i++) {
                sum_x += sum_bx.get(i);
                sum_y += sum_by.get(i);
                sum += sum_b.get(i);
            }
            double cosine = sum / std::sqrt(sum_x * sum_y);
            return cosine;
        }

        TURBO_DLL [[nodiscard]] bool need_normalization() const noexcept override {
            return false;
        }

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const override {
            TLOG_CHECK(false, "not impl");
        }
    };

    template<class T, class Tag, class Arch = turbo::simd::default_arch>
    struct DistanceNormalizedCosine : public Distance<T, Tag, Arch> {
    public:
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(const turbo::Span<T> &x, const turbo::Span<T> &y) const override {
            TLOG_CHECK(x.size() == y.size(), "x and y must be equal");
            using b_type = turbo::simd::batch<T, Arch>;

            std::size_t inc = b_type::size;
            std::size_t size = x.size();

            std::size_t vec_size = size - size % inc;
            auto sum_b = b_type::broadcast(0.0);
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type bx = turbo::simd::load<Arch>(&x[i], Tag());
                b_type by = turbo::simd::load<Arch>(&y[i], Tag());
                sum_b += turbo::simd::mul(bx, by);
            }
            double sum = 0;
            for (size_t i = vec_size; i < size; ++i) {
                sum += x[i] * y[i];
            }
            for (size_t i = 0; i < inc; i++) {
                sum += sum_b.get(i);
            }
            return sum;
        }

        TURBO_DLL [[nodiscard]] bool need_normalization() const noexcept override {
            return true;
        }

        TURBO_DLL virtual void normalization(const turbo::Span<T> &x, turbo::Span<T> &des) const override {
            TLOG_CHECK(x.size() == des.size(), "x and y must be equal");
            using b_type = turbo::simd::batch<T, Arch>;

            std::size_t inc = b_type::size;
            std::size_t size = x.size();

            std::size_t vec_size = size - size % inc;
            auto sum_bx = b_type::broadcast(0.0);
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type bx = turbo::simd::load<Arch>(&x[i], Tag());
                sum_bx += bx * bx;
            }
            double sum_x = 0;
            for (size_t i = vec_size; i < size; ++i) {
                sum_x += x[i] * x[i];
            }
            for (size_t i = 0; i < inc; i++) {
                sum_x += sum_bx.get(i);
            }

            double normal = std::sqrt(sum_x);
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type bx = turbo::simd::load<Arch>(&x[i], Tag());
                turbo::simd::store_unaligned(&des[i], bx / normal);
            }
            for (size_t i = vec_size; i < size; ++i) {
                des[i] += x[i] / normal;
            }
        }
    };
}  // namespace tann

#endif  // TANN_DISTANCE_DISTANCE_H_
