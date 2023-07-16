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
#ifndef TANN_DISTANCE_UTILITY_H_
#define TANN_DISTANCE_UTILITY_H_

#include "turbo/container/array_view.h"
#include "turbo/simd/simd.h"
#include "tann/common/config.h"
#include "turbo/format/print.h"

namespace tann {

    // When the type is less than 4 bytes, numerical overflow often occurs. In this case,
    // use the general method to do norm, and the simd method is not applicable
    template<typename T>
    inline float get_l2_norm_simple(const turbo::array_view<T> &arr) {
        float sum = 0.0f;
        auto dim = arr.size();
        for (uint32_t i = 0; i < dim; i++) {
            sum += static_cast<float>(arr[i]) * static_cast<float>(arr[i]);
        }
        return std::sqrt(sum);
    }

    template<typename T>
    inline float get_l2_norm(const turbo::array_view<T> &arr) {
        static_assert(sizeof(T) >= 4, "sizeof T > 4");
        using b_type = turbo::simd::batch<T, turbo::simd::default_arch>;
        std::size_t inc = b_type::size;
        std::size_t size = arr.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        b_type sum_array = b_type::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&arr[i], turbo::simd::aligned_mode());
            sum_array += avec * avec;
        }
        auto sum = turbo::simd::reduce_add(sum_array);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += arr[i] * arr[i];
        }
        return sqrt(sum);
    }

    template<>
    inline float get_l2_norm<float16>(const turbo::array_view<float16> &arr) {

        const float16 *last = arr.data() + arr.size();
        const float16 *pa = arr.data();
#if TURBO_WITH_AVX512F
        __m512 normA = _mm512_setzero_ps();
        while (pa < last) {
            __m512 am = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(pa)));
            normA = _mm512_add_ps(normA, _mm512_mul_ps(am, am));
            pa += 16;
        }
        __m256 am256 = _mm256_add_ps(_mm512_extractf32x8_ps(normA, 0), _mm512_extractf32x8_ps(normA, 1));
        __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(am256, 0), _mm256_extractf128_ps(am256, 1));
#elif TURBO_WITH_AVX2
        __m256 normA = _mm256_setzero_ps();
        __m256 am;
        while (pa < last) {
            am = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pa)));
            normA = _mm256_add_ps(normA, _mm256_mul_ps(am, am));
            pa += 8;
        }
        __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(normA, 0), _mm256_extractf128_ps(normA, 1));
#endif

        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, am128);
        double na = f[0] + f[1] + f[2] + f[3];
        return sqrt(na);
    }

    template<>
    inline float get_l2_norm<int8_t>(const turbo::array_view<int8_t> &arr) {
        return get_l2_norm_simple(arr);
    }

    template<>
    inline float get_l2_norm<uint8_t>(const turbo::array_view<uint8_t> &arr) {
        return get_l2_norm_simple(arr);
    }

    template<>
    inline float get_l2_norm<uint16_t>(const turbo::array_view<uint16_t> &arr) {
        return get_l2_norm_simple(arr);
    }

    template<>
    inline float get_l2_norm<int16_t>(const turbo::array_view<int16_t> &arr) {
        return get_l2_norm_simple(arr);
    }

    inline void l2_norm(turbo::array_view<float> &arr) {
        auto norm = get_l2_norm(arr);
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        std::size_t inc = b_type::size;
        std::size_t size = arr.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&arr[i], turbo::simd::aligned_mode());
            avec /=  norm;
            avec.store(&arr[i], turbo::simd::aligned_mode());
        }
        for (std::size_t i = vec_size; i < size; ++i) {
            arr[i] /= norm;
        }
    }

    inline void l2_norm(turbo::array_view<float16> &arr) {
        auto norm = get_l2_norm(arr);
        for (std::size_t i = 0; i < arr.size(); ++i) {
            arr[i] = arr[i] / norm;
        }
    }


}  // namespace tann

#endif  // TANN_DISTANCE_UTILITY_H_
