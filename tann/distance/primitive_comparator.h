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

#ifndef TANN_DISTANCE_PRIMITIVE_COMPARATOR_H_
#define TANN_DISTANCE_PRIMITIVE_COMPARATOR_H_

#include "tann/common/config.h"
#include "turbo/simd/simd.h"
#include "turbo/meta/span.h"
#include "turbo/log/logging.h"
#include "turbo/base/bits.h"

namespace tann {

    class PrimComparator {
    public:
        static double absolute(double v) { return fabs(v); }

        static double absolute(float16 v) { return fabs(v); }

        static int absolute(int v) { return abs(v); }

        static long absolute(long v) { return abs(v); }

        // l1 comparators
        template<typename T, typename COMPARE_TYPE>
        static double simple_compare_l1(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            const T *last = a.data() + a.size();
            const T *lastgroup = last - 3;
            const T *pa = a.data();
            const T *pb = b.data();
            COMPARE_TYPE diff0, diff1, diff2, diff3;
            double d = 0.0;
            while (pa < lastgroup) {
                diff0 = (COMPARE_TYPE) (pa[0] - pb[0]);
                diff1 = (COMPARE_TYPE) (pa[1] - pb[1]);
                diff2 = (COMPARE_TYPE) (pa[2] - pb[2]);
                diff3 = (COMPARE_TYPE) (pa[3] - pb[3]);
                d += absolute(diff0) + absolute(diff1) + absolute(diff2) + absolute(diff3);
                pa += 4;
                pb += 4;
            }
            while (pa < last) {
                diff0 = (COMPARE_TYPE) *pa++ - (COMPARE_TYPE) *pb++;
                d += absolute(diff0);
            }
            return d;
        }

        inline static double
        simple_compare_l1(const turbo::Span<uint8_t> &a, const turbo::Span<uint8_t> &b) {
            return simple_compare_l1<uint8_t, int>(a, b);
        }

        inline static double simple_compare_l1(const turbo::Span<float> &a, const turbo::Span<float> &b) {
            return simple_compare_l1<float, double>(a, b);
        }

        inline static double simple_compare_l1(const turbo::Span<float16> &a, turbo::Span<float16> &b) {
            return simple_compare_l1<float16, double>(a, b);
        }

        template<typename T>
        inline static double compare_l1(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
            constexpr bool is_float_type = std::is_floating_point_v<T>;
            bool is_aligned = turbo::simd::is_aligned(static_cast<const T *>(a.data())) &&
                              turbo::simd::is_aligned(static_cast<const T *>(b.data()));
            TLOG_CHECK(is_aligned, "the memory must be aligned");
            std::size_t inc = b_type::size;
            std::size_t size = a.size();
            // size for which the vectorization is possible
            std::size_t vec_size = size - size % inc;
            double sum = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type avec = b_type::load(&a[i], turbo::simd::aligned_mode());
                b_type bvec = b_type::load(&b[i], turbo::simd::aligned_mode());
                if constexpr (is_float_type) {
                    sum += turbo::simd::reduce_add(turbo::simd::fabs(avec - bvec));
                } else {
                    sum += turbo::simd::reduce_add(turbo::simd::abs(avec - bvec));
                }
            }
            for (std::size_t i = vec_size; i < size; ++i) {
                sum += absolute(a[i] - b[i]);
            }
            return sum;
        }

        // l2 comparators
        template<typename T, typename COMPARE_TYPE>
        inline static double simple_compare_l2(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            const T *pa = a.data();
            const T *pb = b.data();
            const T *last = pa + a.size();
            const T *lastgroup = last - 3;
            COMPARE_TYPE diff0, diff1, diff2, diff3;
            double d = 0.0;
            while (pa < lastgroup) {
                diff0 = static_cast<COMPARE_TYPE>(pa[0] - pb[0]);
                diff1 = static_cast<COMPARE_TYPE>(pa[1] - pb[1]);
                diff2 = static_cast<COMPARE_TYPE>(pa[2] - pb[2]);
                diff3 = static_cast<COMPARE_TYPE>(pa[3] - pb[3]);
                d += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                pa += 4;
                pb += 4;
            }
            while (pa < last) {
                diff0 = static_cast<COMPARE_TYPE>(*pa++ - *pb++);
                d += diff0 * diff0;
            }
            return sqrt(static_cast<double>(d));
        }

        inline static double
        simple_compare_l2(const turbo::Span<uint8_t> &a, const turbo::Span<uint8_t> &b) {
            return simple_compare_l2<uint8_t, int>(a, b);
        }

        inline static double simple_compare_l2(const turbo::Span<float> &a, const turbo::Span<float> &b) {
            return simple_compare_l2<float, double>(a, b);
        }

        inline static double
        simple_compare_l2(const turbo::Span<float16> &a, const turbo::Span<float16> &b) {
            return simple_compare_l2<float16, double>(a, b);
        }

        template<typename T>
        inline static double compare_l2(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
            bool is_aligned = turbo::simd::is_aligned(static_cast<const T *>(a.data())) &&
                              turbo::simd::is_aligned(static_cast<const T *>(b.data()));
            TLOG_CHECK(is_aligned, "the memory must be aligned");
            static_assert(sizeof(T) >= 4, "sizeof(T) >= 4");
            std::size_t inc = b_type::size;
            std::size_t size = a.size();
            // size for which the vectorization is possible
            std::size_t vec_size = size - size % inc;
            b_type sum_v = b_type::broadcast(0.0);
            double sum = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type avec = b_type::load(&a[i], turbo::simd::aligned_mode());
                b_type bvec = b_type::load(&b[i], turbo::simd::aligned_mode());
                auto diff = avec - bvec;
                sum_v += turbo::simd::mul(diff, diff);
            }
            sum += turbo::simd::reduce_add(sum_v);
            for (std::size_t i = vec_size; i < size; ++i) {
                auto df = a[i] - b[i];
                sum += df * df;
            }
            return sqrt(sum);
        }
        /////
        /// hamming distance

        template<typename T>
        inline static double simple_compare_hamming(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            static_assert(std::is_integral_v<T>, "must be integer typer");
            auto *last = reinterpret_cast<const uint32_t *>(a.data() + a.size());
            auto *uinta = reinterpret_cast<const uint32_t *>(a.data());
            auto *uintb = reinterpret_cast<const uint32_t *>(b.data());
            size_t count = 0;
            while (uinta < last) {
                count += turbo::popcount(*uinta++ ^ *uintb++);
            }

            return static_cast<double>(count);
        }

        template<size_t N>
        inline static double popcount(const turbo::simd::batch<uint64_t, turbo::simd::default_arch> &a) {
            TLOG_CHECK(false, "not impl");
            return 0;
        }

        template<typename T>
        inline static double compare_hamming(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
            using cb_type = turbo::simd::batch<uint64_t, turbo::simd::default_arch>;
            static_assert(std::is_integral_v<T>, "must be integer typer");
            static_assert(std::is_unsigned_v<T>, "must be integer typer");
            bool is_aligned = turbo::simd::is_aligned(static_cast<const T *>(a.data())) &&
                              turbo::simd::is_aligned(static_cast<const T *>(b.data()));
            TLOG_CHECK(is_aligned, "the memory must be aligned");
            std::size_t inc = b_type::size;
            std::size_t size = a.size();
            // size for which the vectorization is possible
            std::size_t vec_size = size - size % inc;
            constexpr size_t neles = turbo::simd::default_arch::alignment() / sizeof(uint64_t);
            double sum = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                cb_type avec = cb_type::load(reinterpret_cast<const uint64_t *>(&a[i]), turbo::simd::aligned_mode());
                cb_type bvec = cb_type::load(reinterpret_cast<const uint64_t *>(&b[i]), turbo::simd::aligned_mode());
                sum += popcount<neles>(avec ^ bvec);

            }
            for (std::size_t i = vec_size; i < size; i += sizeof(uint64_t)) {
                sum += turbo::popcount(
                        *reinterpret_cast<const uint64_t *>(&a[i]) ^ *reinterpret_cast<const uint64_t *>(&b[i]));
            }
            return sum;
        }

        ////////////
        /// compare jaccard
        template<typename T>
        inline static double simple_compare_jaccard(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            auto *last = reinterpret_cast<const uint32_t *>(a.data() + a.size());

            auto *uinta = reinterpret_cast<const uint32_t *>(a.data());
            auto *uintb = reinterpret_cast<const uint32_t *>(b.data());
            size_t count = 0;
            size_t countDe = 0;
            while (uinta < last) {
                count += turbo::popcount(*uinta & *uintb);
                countDe += turbo::popcount(*uinta++ | *uintb++);
                count += turbo::popcount(*uinta & *uintb);
                countDe += turbo::popcount(*uinta++ | *uintb++);
            }
            return 1.0 - static_cast<double>(count) / static_cast<double>(countDe);
        }

        template<typename T>
        inline static double compare_jaccard(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
            using cb_type = turbo::simd::batch<uint64_t, turbo::simd::default_arch>;
            static_assert(std::is_integral_v<T>, "must be integer typer");
            static_assert(std::is_unsigned_v<T>, "must be integer typer");
            bool is_aligned = turbo::simd::is_aligned(static_cast<const T *>(a.data())) &&
                              turbo::simd::is_aligned(static_cast<const T *>(b.data()));
            TLOG_CHECK(is_aligned, "the memory must be aligned");
            std::size_t inc = b_type::size;
            std::size_t size = a.size();
            // size for which the vectorization is possible
            std::size_t vec_size = size - size % inc;
            constexpr size_t neles = turbo::simd::default_arch::alignment() / sizeof(uint64_t);
            double sum = 0.0;
            double sum_de = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                cb_type avec = cb_type::load(reinterpret_cast<const uint64_t *>(&a[i]), turbo::simd::aligned_mode());
                cb_type bvec = cb_type::load(reinterpret_cast<const uint64_t *>(&b[i]), turbo::simd::aligned_mode());

                sum += popcount<neles>(avec & bvec);
                sum_de += popcount<neles>(avec | bvec);

            }
            for (std::size_t i = vec_size; i < size; i += sizeof(uint64_t)) {
                sum += turbo::popcount(
                        *reinterpret_cast<const uint64_t *>(&a[i]) & *reinterpret_cast<const uint64_t *>(&b[i]));
                sum_de += turbo::popcount(
                        *reinterpret_cast<const uint64_t *>(&a[i]) | *reinterpret_cast<const uint64_t *>(&b[i]));
            }
            return 1.0 - sum / sum_de;
        }

        template<typename T>
        inline static double simple_compare_cosine(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double normA = 0.0;
            double normB = 0.0;
            double sum = 0.0;
            auto size = a.size();
            for (size_t loc = 0; loc < size; loc++) {
                normA += static_cast<double>(a[loc]) * static_cast<double>(a[loc]);
                normB += static_cast<double>(b[loc]) * static_cast<double>(b[loc]);
                sum += static_cast<double>(a[loc]) * static_cast<double>(b[loc]);
            }

            double cosine = sum / sqrt(normA * normB);

            return cosine;
        }

        ////////////////////////////
        /// compare cosine
        /// When the type is less than 4 bytes, numerical overflow often occurs. In this case,
        /// use the general method to do norm, and the simd method is not applicable
        template<typename T>
        inline static double compare_cosine(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
            bool is_aligned = turbo::simd::is_aligned(static_cast<const T *>(a.data())) &&
                              turbo::simd::is_aligned(static_cast<const T *>(b.data()));
            TLOG_CHECK(is_aligned, "the memory must be aligned");
            static_assert(sizeof(T) >= 4, "sizeof(T) >= 4");
            std::size_t inc = b_type::size;
            std::size_t size = a.size();
            // size for which the vectorization is possible
            std::size_t vec_size = size - size % inc;
            b_type sum_v = b_type::broadcast(0.0);
            b_type sum_av = b_type::broadcast(0.0);
            b_type sum_bv = b_type::broadcast(0.0);
            double sum = 0.0;
            double norm_a = 0.0;
            double norm_b = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type avec = b_type::load(&a[i], turbo::simd::aligned_mode());
                b_type bvec = b_type::load(&b[i], turbo::simd::aligned_mode());
                sum_av += avec * avec;
                sum_bv += bvec * bvec;
                sum_v += avec * bvec;
            }
            sum += turbo::simd::reduce_add(sum_v);
            norm_a += turbo::simd::reduce_add(sum_av);
            norm_b += turbo::simd::reduce_add(sum_bv);
            for (std::size_t i = vec_size; i < size; ++i) {
                norm_a += a[i] * a[i];
                norm_a += b[i] * b[i];
                sum += a[i] * b[i];
            }
            double cosine = sum / sqrt(norm_a * norm_b);
            return cosine;
        }

        //////////////////////////////
        /// angle distance
        template<typename T>
        inline static double compare_angle(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double cosine = compare_cosine(a, b);
            if (cosine >= 1.0) {
                return 0.0;
            } else if (cosine <= -1.0) {
                return acos(-1.0);
            } else {
                return acos(cosine);
            }
        }

        template<typename T>
        inline static double
        simple_compare_inner_product(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double sum = 0.0;
            auto size = a.size();
            for (size_t loc = 0; loc < size; loc++) {
                sum += static_cast<float>(a[loc]) * static_cast<float>(b[loc]);
            }
            return sum;
        }

        template<typename T>
        inline static double compare_inner_product(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
            bool is_aligned = turbo::simd::is_aligned(static_cast<const T *>(a.data())) &&
                              turbo::simd::is_aligned(static_cast<const T *>(b.data()));
            TLOG_CHECK(is_aligned, "the memory must be aligned");
            static_assert(sizeof(T) >= 4, "sizeof(T) >= 4");
            std::size_t inc = b_type::size;
            std::size_t size = a.size();
            // size for which the vectorization is possible
            std::size_t vec_size = size - size % inc;
            b_type sum_v = b_type::broadcast(0.0);
            double sum = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type avec = b_type::load(&a[i], turbo::simd::aligned_mode());
                b_type bvec = b_type::load(&b[i], turbo::simd::aligned_mode());
                sum_v += avec * bvec;
            }
            sum += turbo::simd::reduce_add(sum_v);
            for (std::size_t i = vec_size; i < size; ++i) {
                sum += a[i] * b[i];
            }
            return sum;
        }

        //////////////////////////////////////////////
        /// vector that should be normalized
        template<typename T>
        inline static double
        compare_normalized_cosine(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            // auto v = 1.0 - compare_inner_product(a, b);
            // return v < 0.0 ? -v : v;
            return compare_inner_product(a, b);
        }

        //////////////////////////////////////////////
        /// vector that should be normalized
        template<typename T>
        inline static double compare_normalized_angle(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double cosine = compare_inner_product(a, b);
            if (cosine >= 1.0) {
                return 0.0;
            } else if (cosine <= -1.0) {
                return acos(-1.0);
            } else {
                return acos(cosine);
            }
        }

        //////////////////////////////////////////////
        /// vector that should be normalized
        template<typename T>
        inline static double compare_normalized_l2(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double v = 2.0 - 2.0 * compare_inner_product(a, b);
            if (v < 0.0) {
                return 0.0;
            } else {
                return sqrt(v);
            }
        }

        //////////////////////////////////////////////
        /// Poincare distance
        /// for that norm(a) and norm(b) all < 1, so it must be float or float16
        template<typename T>
        inline static double simple_compare_poincare(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double sum_a = 0.0;
            double sum_b = 0.0;
            double c2 = simple_compare_l2(a, b);
            for (size_t i = 0; i < a.size(); i++) {
                sum_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                sum_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }
            return std::acosh(1 + 2.0 * c2 * c2 / (1.0 - sum_a) / (1.0 - sum_b));
        }

        template<typename T>
        inline static double compare_poincare(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
            bool is_aligned = turbo::simd::is_aligned(static_cast<const T *>(a.data())) &&
                              turbo::simd::is_aligned(static_cast<const T *>(b.data()));
            TLOG_CHECK(is_aligned, "the memory must be aligned");
            static_assert(sizeof(T) >= 4, "sizeof(T) >= 4");
            std::size_t inc = b_type::size;
            std::size_t size = a.size();
            // size for which the vectorization is possible
            std::size_t vec_size = size - size % inc;
            b_type sum_v = b_type::broadcast(0.0);
            b_type sum_av = b_type::broadcast(0.0);
            b_type sum_bv = b_type::broadcast(0.0);
            double sum = 0.0;
            double sum_a = 0.0;
            double sum_b = 0.0;
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type avec = b_type::load(&a[i], turbo::simd::aligned_mode());
                b_type bvec = b_type::load(&b[i], turbo::simd::aligned_mode());
                auto diff = avec - bvec;
                sum_v += turbo::simd::mul(diff, diff);
                sum_av += avec * avec;
                sum_bv += bvec * bvec;
            }
            sum += turbo::simd::reduce_add(sum_v);
            sum_a += turbo::simd::reduce_add(sum_av);
            sum_b += turbo::simd::reduce_add(sum_bv);
            for (std::size_t i = vec_size; i < size; ++i) {
                auto df = a[i] - b[i];
                sum += df * df;
                sum_a += a[i] * a[i];
                sum_b += b[i] * b[i];
            }
            return std::acosh(1 + 2.0 * sum / (1.0 - sum_a) / (1.0 - sum_b));
        }

        template<typename T>
        inline static double simple_compare_lorentz(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double sum = static_cast<double>(a[0]) * static_cast<double>(b[0]);
            for (size_t i = 1; i < a.size(); i++) {
                sum -= static_cast<double>(a[i]) * static_cast<double>(b[i]);
            }
            return std::acosh(sum);
        }

        template<typename T>
        inline static double compare_lorentz(const turbo::Span<T> &a, const turbo::Span<T> &b) {
            double sum = static_cast<double>(a[0]) * static_cast<double>(b[0]);
            auto ip = compare_inner_product(a, b);
            sum = sum * 2 - ip;
            return std::acosh(sum);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////
    // specializations
    /// l1 comparator
    template<>
    inline double
    PrimComparator::compare_l1<float16>(const turbo::Span<float16> &a, const turbo::Span<float16> &b) {
        __m256 sum = _mm256_setzero_ps();
        const float16 *last = a.data() + a.size();
        const float16 *pa = a.data();
        const float16 *pb = b.data();
        const float16 *lastgroup = last - 7;
        while (pa < lastgroup) {
            __m256 x1 = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pa))),
                                      _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pb))));
            const __m256 mask = _mm256_set1_ps(-0.0f);
            __m256 v = _mm256_andnot_ps(mask, x1);
            sum = _mm256_add_ps(sum, v);
            pa += 8;
            pb += 8;
        }
        __attribute__((aligned(32))) float f[8];
        _mm256_store_ps(f, sum);
        double s = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
        while (pa < last) {
            double d = fabs(*pa++ - *pb++);
            s += d;
        }
        return s;
    }

    template<>
    inline double PrimComparator::compare_l1<unsigned char>(const turbo::Span<unsigned char> &a,
                                                            const turbo::Span<unsigned char> &b) {
        __m128 sum = _mm_setzero_ps();
        const unsigned char *pa = a.data();
        const unsigned char *pb = b.data();
        const unsigned char *last = pa + a.size();
        const unsigned char *lastgroup = last - 7;
        const __m128i zero = _mm_setzero_si128();
        while (pa < lastgroup) {
            __m128i x1 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i const *) pa));
            __m128i x2 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i const *) pb));
            x1 = _mm_subs_epi16(x1, x2);
            x1 = _mm_sign_epi16(x1, x1);
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpacklo_epi16(x1, zero)));
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpackhi_epi16(x1, zero)));
            pa += 8;
            pb += 8;
        }
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum);
        double s = f[0] + f[1] + f[2] + f[3];
        while (pa < last) {
            double d = fabs(static_cast<double>(*pa++) - static_cast<double>(*pb++));
            s += d;
        }
        return s;
    }

    // l2 comparator
    template<>
    inline double
    PrimComparator::compare_l2<float16>(const turbo::Span<float16> &a, const turbo::Span<float16> &b) {
        const float16 *pa = a.data();
        const float16 *pb = b.data();
        const float16 *last = pa + a.size();
#if TURBO_WITH_AVX512F
        __m512 sum512 = _mm512_setzero_ps();
            while (pa < last) {
          __m512 v = _mm512_sub_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa))),
                       _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb))));
          sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v, v));
          pa += 16;
          pb += 16;
            }

            __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#elif TURBO_WITH_AVX2
        __m256 sum256 = _mm256_setzero_ps();
        __m256 v;
        while (pa < last) {
            v = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pa))),
                              _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pb))));
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v, v));
            pa += 8;
            pb += 8;
            v = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pa))),
                              _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pb))));
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v, v));
            pa += 8;
            pb += 8;
        }
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#else
        __m128 sum128 = _mm_setzero_ps();
            __m128 v;
            while (pa < last) {
          __m128i va = _mm_load_si128(reinterpret_cast<const __m128i*>(pa));
          __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(pb));
          v = _mm_sub_ps(_mm_cvtph_ps(va), _mm_cvtph_ps(vb));
          sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
          va = _mm_srli_si128(va, 8);
          vb = _mm_srli_si128(vb, 8);
          v = _mm_sub_ps(_mm_cvtph_ps(va), _mm_cvtph_ps(vb));
          sum128 = _mm_add_ps(sum128, _mm_mul_ps(v, v));
              pa += 8;
              pb += 8;
            }
#endif

        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum128);
        double s = f[0] + f[1] + f[2] + f[3];
        return sqrt(s);
    }

    template<>
    inline double PrimComparator::compare_l2<unsigned char>(const turbo::Span<unsigned char> &a,
                                                            const turbo::Span<unsigned char> &b) {
        __m128 sum = _mm_setzero_ps();
        const unsigned char *pa = a.data();
        const unsigned char *pb = b.data();
        const unsigned char *last = pa + a.size();
        const unsigned char *lastgroup = last - 7;
        const __m128i zero = _mm_setzero_si128();
        while (pa < lastgroup) {
            __m128i x1 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i const *) pa));
            __m128i x2 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i const *) pb));
            x1 = _mm_subs_epi16(x1, x2);
            __m128i v = _mm_mullo_epi16(x1, x1);
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpacklo_epi16(v, zero)));
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, zero)));
            pa += 8;
            pb += 8;
        }
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum);
        double s = f[0] + f[1] + f[2] + f[3];
        while (pa < last) {
            int d = (int) *pa++ - (int) *pb++;
            s += d * d;
        }
        return sqrt(s);
    }

    ////////////////////////
    /// popcount
    // avx2
    template<>
    inline double PrimComparator::popcount<4>(const turbo::simd::batch<uint64_t, turbo::simd::default_arch> &a) {
        return turbo::popcount(a.get(0)) + turbo::popcount(a.get(1)) + turbo::popcount(a.get(2)) +
               turbo::popcount(a.get(3));
    }

    //////
    // avx512
    template<>
    inline double PrimComparator::popcount<8>(const turbo::simd::batch<uint64_t, turbo::simd::default_arch> &a) {
        return turbo::popcount(a.get(0)) + turbo::popcount(a.get(1)) + turbo::popcount(a.get(2)) +
               turbo::popcount(a.get(3)) + turbo::popcount(a.get(4)) + turbo::popcount(a.get(5)) +
               turbo::popcount(a.get(7)) + turbo::popcount(a.get(7));
    }

    ////////////////////////
    /// comparator cosine for float16

    template<>
    inline double PrimComparator::compare_cosine<float16>(const turbo::Span<float16> &a,
                                                          const turbo::Span<float16> &b) {

        const float16 *last = a.data() + a.size();
        const float16 *pa = a.data();
        const float16 *pb = b.data();
#if TURBO_WITH_AVX512F
        __m512 normA = _mm512_setzero_ps();
            __m512 normB = _mm512_setzero_ps();
            __m512 sum = _mm512_setzero_ps();
            while (pa < last) {
          __m512 am = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa)));
          __m512 bm = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb)));
          normA = _mm512_add_ps(normA, _mm512_mul_ps(am, am));
          normB = _mm512_add_ps(normB, _mm512_mul_ps(bm, bm));
          sum = _mm512_add_ps(sum, _mm512_mul_ps(am, bm));
          pa += 16;
          pb += 16;
            }
            __m256 am256 = _mm256_add_ps(_mm512_extractf32x8_ps(normA, 0), _mm512_extractf32x8_ps(normA, 1));
            __m256 bm256 = _mm256_add_ps(_mm512_extractf32x8_ps(normB, 0), _mm512_extractf32x8_ps(normB, 1));
            __m256 s256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
            __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(am256, 0), _mm256_extractf128_ps(am256, 1));
            __m128 bm128 = _mm_add_ps(_mm256_extractf128_ps(bm256, 0), _mm256_extractf128_ps(bm256, 1));
            __m128 s128 = _mm_add_ps(_mm256_extractf128_ps(s256, 0), _mm256_extractf128_ps(s256, 1));
#elif TURBO_WITH_AVX2
        __m256 normA = _mm256_setzero_ps();
        __m256 normB = _mm256_setzero_ps();
        __m256 sum = _mm256_setzero_ps();
        __m256 am, bm;
        while (pa < last) {
            am = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pa)));
            bm = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pb)));
            normA = _mm256_add_ps(normA, _mm256_mul_ps(am, am));
            normB = _mm256_add_ps(normB, _mm256_mul_ps(bm, bm));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(am, bm));
            pa += 8;
            pb += 8;
        }
        __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(normA, 0), _mm256_extractf128_ps(normA, 1));
        __m128 bm128 = _mm_add_ps(_mm256_extractf128_ps(normB, 0), _mm256_extractf128_ps(normB, 1));
        __m128 s128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
#else
        __m128 am128 = _mm_setzero_ps();
        __m128 bm128 = _mm_setzero_ps();
        __m128 s128 = _mm_setzero_ps();
        __m128 am, bm;
        while (pa < last) {
            __m128i va = _mm_load_si128(reinterpret_cast<const __m128i *>(pa));
            __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i *>(pb));
            am = _mm_cvtph_ps(va);
            bm = _mm_cvtph_ps(vb);
            am128 = _mm_add_ps(am128, _mm_mul_ps(am, am));
            bm128 = _mm_add_ps(bm128, _mm_mul_ps(bm, bm));
            s128 = _mm_add_ps(s128, _mm_mul_ps(am, bm));
            va = _mm_srli_si128(va, 8);
            vb = _mm_srli_si128(vb, 8);
            am = _mm_cvtph_ps(va);
            bm = _mm_cvtph_ps(vb);
            am128 = _mm_add_ps(am128, _mm_mul_ps(am, am));
            bm128 = _mm_add_ps(bm128, _mm_mul_ps(bm, bm));
            s128 = _mm_add_ps(s128, _mm_mul_ps(am, bm));
            pa += 8;
            pb += 8;
        }

#endif

        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, am128);
        double na = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, bm128);
        double nb = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, s128);
        double s = f[0] + f[1] + f[2] + f[3];

        double cosine = s / sqrt(na * nb);
        return cosine;
    }

    template<>
    inline double PrimComparator::compare_cosine<uint8_t>(const turbo::Span<uint8_t> &a,
                                                          const turbo::Span<uint8_t> &b) {
        return simple_compare_cosine(a, b);
    }

    template<>
    inline double PrimComparator::compare_cosine<int16_t>(const turbo::Span<int16_t> &a,
                                                          const turbo::Span<int16_t> &b) {
        return simple_compare_cosine(a, b);
    }

    template<>
    inline double PrimComparator::compare_cosine<uint16_t>(const turbo::Span<uint16_t> &a,
                                                           const turbo::Span<uint16_t> &b) {
        return simple_compare_cosine(a, b);
    }

    /////////////////////////////////
    /// inner product
    template<>
    inline double PrimComparator::compare_inner_product<float16>(const turbo::Span<float16> &a,
                                                                 const turbo::Span<float16> &b) {
        const float16 *pa = a.data();
        const float16 *pb = b.data();
        const float16 *last = pa + a.size();
#if TURBO_WITH_AVX512F
        __m512 sum512 = _mm512_setzero_ps();
            while (pa < last) {
          sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa))),
                                   _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb)))));

          pa += 16;
          pb += 16;
            }
            __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#elif TURBO_WITH_AVX2
        __m256 sum256 = _mm256_setzero_ps();
        while (pa < last) {
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pa))),
                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pb)))));
            pa += 8;
            pb += 8;
        }
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#else
        __m128 sum128 = _mm_setzero_ps();
        while (pa < last) {
            __m128i va = _mm_load_si128(reinterpret_cast<const __m128i*>(pa));
            __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(pb));
            sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_cvtph_ps(va), _mm_cvtph_ps(vb)));
            va = _mm_srli_si128(va, 8);
            vb = _mm_srli_si128(vb, 8);
            sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_cvtph_ps(va), _mm_cvtph_ps(vb)));
            pa += 8;
            pb += 8;
        }
#endif
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum128);
        double s = static_cast<double>(f[0]) + static_cast<double>(f[1]) + static_cast<double>(f[2]) +
                   static_cast<double>(f[3]);
        return s;
    }

    template<>
    inline double PrimComparator::compare_inner_product<uint8_t>(const turbo::Span<uint8_t> &a,
                                                                 const turbo::Span<uint8_t> &b) {
        return simple_compare_inner_product(a, b);
    }

    template<>
    inline double PrimComparator::compare_inner_product<uint16_t>(const turbo::Span<uint16_t> &a,
                                                                  const turbo::Span<uint16_t> &b) {
        return simple_compare_inner_product(a, b);
    }

    template<>
    inline double PrimComparator::compare_inner_product<int16_t>(const turbo::Span<int16_t> &a,
                                                                 const turbo::Span<int16_t> &b) {
        return simple_compare_inner_product(a, b);
    }

    template<>
    inline double PrimComparator::compare_poincare<float16>(const turbo::Span<float16> &a,
                                                            const turbo::Span<float16> &b) {
        const float16 *last = a.data() + a.size();
        const float16 *pa = a.data();
        const float16 *pb = b.data();
#if TURBO_WITH_AVX512F
        __m512 normA = _mm512_setzero_ps();
        __m512 normB = _mm512_setzero_ps();
        __m512 sum = _mm512_setzero_ps();
        while (pa < last) {
            __m512 am = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(pa)));
            __m512 bm = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(pb)));
            normA = _mm512_add_ps(normA, _mm512_mul_ps(am, am));
            normB = _mm512_add_ps(normB, _mm512_mul_ps(bm, bm));
            __m512 v = _mm512_sub_ps(am, bm);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(v, v));
            pa += 16;
            pb += 16;
        }
        __m256 am256 = _mm256_add_ps(_mm512_extractf32x8_ps(normA, 0), _mm512_extractf32x8_ps(normA, 1));
        __m256 bm256 = _mm256_add_ps(_mm512_extractf32x8_ps(normB, 0), _mm512_extractf32x8_ps(normB, 1));
        __m256 s256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
        __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(am256, 0), _mm256_extractf128_ps(am256, 1));
        __m128 bm128 = _mm_add_ps(_mm256_extractf128_ps(bm256, 0), _mm256_extractf128_ps(bm256, 1));
        __m128 s128 = _mm_add_ps(_mm256_extractf128_ps(s256, 0), _mm256_extractf128_ps(s256, 1));
#elif TURBO_WITH_AVX2
        __m256 normA = _mm256_setzero_ps();
        __m256 normB = _mm256_setzero_ps();
        __m256 sum = _mm256_setzero_ps();
        __m256 am, bm;
        while (pa < last) {
            am = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pa)));
            bm = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pb)));
            normA = _mm256_add_ps(normA, _mm256_mul_ps(am, am));
            normB = _mm256_add_ps(normB, _mm256_mul_ps(bm, bm));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_sub_ps(am, bm), _mm256_sub_ps(am, bm)));
            pa += 8;
            pb += 8;
        }
        __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(normA, 0), _mm256_extractf128_ps(normA, 1));
        __m128 bm128 = _mm_add_ps(_mm256_extractf128_ps(normB, 0), _mm256_extractf128_ps(normB, 1));
        __m128 s128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
#else
        __m128 am128 = _mm_setzero_ps();
        __m128 bm128 = _mm_setzero_ps();
        __m128 s128 = _mm_setzero_ps();
        __m128 am, bm;
        while (pa < last) {
            __m128i va = _mm_load_si128(reinterpret_cast<const __m128i *>(pa));
            __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i *>(pb));
            am = _mm_cvtph_ps(va);
            bm = _mm_cvtph_ps(vb);
            am128 = _mm_add_ps(am128, _mm_mul_ps(am, am));
            bm128 = _mm_add_ps(bm128, _mm_mul_ps(bm, bm));
            __m128 diff = _mm_sub_ps(am, bm);
            s128 = _mm_add_ps(s128, _mm_mul_ps(diff, diff));
            va = _mm_srli_si128(va, 8);
            vb = _mm_srli_si128(vb, 8);
            am = _mm_cvtph_ps(va);
            bm = _mm_cvtph_ps(vb);
            am128 = _mm_add_ps(am128, _mm_mul_ps(am, am));
            bm128 = _mm_add_ps(bm128, _mm_mul_ps(bm, bm));
            s128 = _mm_add_ps(s128, _mm_mul_ps(am, bm));
            pa += 8;
            pb += 8;
        }

#endif

        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, am128);
        double na = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, bm128);
        double nb = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, s128);
        double s = f[0] + f[1] + f[2] + f[3];

        return std::acosh(1 + 2.0 * s / (1.0 - na) / (1.0 - nb));
    }
}  // namespace tann

#endif  // TANN_DISTANCE_PRIMITIVE_COMPARATOR_H_
