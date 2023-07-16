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

#include "turbo/simd/simd.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "tann/distance/utility.h"
#include "tann/distance/primitive_distance.h"
#include <vector>
#include "turbo/format/print.h"
#include "tann/common/utils.h"
#include "test_util.h"
#include "turbo/meta/type_traits.h"


TEST_CASE_TEMPLATE("int8 norm", T, TEST_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceL1<T>::alignment_bytes;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    a.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i);
    }

    turbo::array_view<T> b(a.data(), a.size(), false);

    auto n1 = tann::get_l2_norm<T>(b);
    auto  n2 = tann::get_l2_norm_simple(b);
    turbo::Println("type: {} get_l2_norm:{} get_norm:{}", turbo::type_info_of<T>()->name(), n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);
}

struct check_supported
{
    template <class Arch>
    void operator()(Arch) const
    {
        turbo::Println("arch name:{}", Arch::name());
        turbo::Println("alignment:{}", Arch::alignment());
        turbo::Println("requires_alignment:{}", turbo::simd::default_arch::requires_alignment());
    }
};

TEST_CASE_TEMPLATE("alignment", T, TEST_TYPES) {
    turbo::simd::supported_architectures::for_each(check_supported());

}

TEST_CASE_TEMPLATE("l1 distance", T, TEST_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceL1<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceL1<T> pdl1;
    auto n1 = pdl1.compare(ax, bx);
    CHECK_EQ(n1, 872);
    //turbo::Println("l1:{}", n1);
}

TEST_CASE_TEMPLATE("l2 distance", T, TEST_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceL2<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceL2<T> pdl2;
    auto n1 = pdl2.compare(ax, bx);
    turbo::Println("l1:{}", n1);
    CHECK_LT(n1 - 95.163, 0.001);

}

TEST_CASE_TEMPLATE("hamming distance", T, TEST_HM_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceHamming<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceHamming<T> pdh;
    auto n1 = pdh.compare(ax, bx);
    auto n2 = tann::PrimComparator::simple_compare_hamming<T>(ax, bx);
    turbo::Println("hamming:{} {}", n1, n2);
    CHECK_EQ(n1, n2);

}

TEST_CASE_TEMPLATE("jaccard distance", T, TEST_HM_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceJaccard<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceJaccard<T> pdj;
    auto n1 = pdj.compare(ax, bx);
    auto n2 = tann::PrimComparator::simple_compare_jaccard<T>(ax, bx);
    turbo::Println("hamming:{} {}", n1, n2);
    CHECK_LT(n1 - n2, 0.001);

}

TEST_CASE_TEMPLATE("cosine distance", T, TEST_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceCosine<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceCosine<T> pdc;
    auto n1 = pdc.compare(ax, bx);
    auto n2 = tann::PrimComparator::simple_compare_cosine<T>(ax, bx);
    turbo::Println("cosine:{} {}", n1, n2);
    CHECK_LT(n1 - n2, 0.001);

}

TEST_CASE_TEMPLATE("angle distance", T, TEST_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceAngle<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceAngle<T> pda;
    auto n1 = pda.compare(ax, bx);
    auto n2 = tann::PrimComparator::compare_angle<T>(ax, bx);
    turbo::Println("angle:{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("ip distance", T, TEST_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceAngle<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceIP<T> pdi;
    auto n1 = pdi.compare(ax, bx);
    auto n2 = tann::PrimComparator::compare_inner_product<T>(ax, bx);
    turbo::Println("ip:{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("normalized cosine distance", T, TEST_NORM_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceAngle<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceCosine<T> pdc;
    auto n2 = pdc.compare(ax, bx);
    tann::PrimDistanceNormalizedCosine<T> pdnc;
    pdnc.preprocess_base_points(ax);
    pdnc.preprocess_base_points(bx);
    auto n1 = pdnc.compare(ax, bx);
    turbo::Println("nor cosine :{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("normalized angle distance", T, TEST_NORM_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceAngle<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceAngle<T> pda;
    auto n2 = pda.compare(ax, bx);
    tann::PrimDistanceNormalizedAngle<T> pdna;
    pdna.preprocess_base_points(ax);
    pdna.preprocess_base_points(bx);
    auto n1 = pdna.compare(ax, bx);
    turbo::Println("nor angle :{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("lorentz distance", T, TEST_NORM_TYPES) {
    constexpr std::size_t al = tann::PrimDistanceAngle<T>::alignment_bytes;
    turbo::Println("al:{}", al);
    std::vector<T, turbo::simd::aligned_allocator<T, al>> a;
    std::vector<T, turbo::simd::aligned_allocator<T, al>> b;
    a.reserve(128);
    b.reserve(128);
    for(int i = 0; i< 128; i++) {
        a.emplace_back(i%20);
        b.emplace_back((128-i)%10);
    }

    a[0] += 99;
    b[0] += 98;
    turbo::array_view<T> ax(a.data(), a.size(), false);
    turbo::array_view<T> bx(b.data(), a.size(), false);
    tann::PrimDistanceLorentz<T> pdlz;
    auto n2 = pdlz.compare(ax, bx);
    auto n1 = tann::PrimComparator::simple_compare_lorentz(ax, bx);
    turbo::Println("lorentz :{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}