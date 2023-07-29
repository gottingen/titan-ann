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
#include "test_util.h"
#include "turbo/meta/type_traits.h"
#include "tann/core/allocator.h"

TEST_CASE_TEMPLATE("int8 norm", T, TEST_TYPES) {
    tann::AlignedQuery<T> a;
    a.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i);
    }

    auto b = tann::to_span<T>(a);

    auto n1 = tann::get_l2_norm<T>(b);
    auto n2 = tann::get_l2_norm_simple(b);
    turbo::Println("type: {} get_l2_norm:{} get_norm:{}", turbo::type_info_of<T>()->name(), n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);
}

TEST_CASE_TEMPLATE("l1 distance", T, TEST_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto ax = tann::to_span<T>(a);
    auto bx = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::simple_compare_l1<T, double>(ax, bx);
    auto n2 = tann::PrimComparator::compare_l1(ax, bx);
    CHECK_EQ(n1, 872);
    CHECK_EQ(n1, n2);
    turbo::Println("l1 n1:{} n2:{}", n1, n2);
}

TEST_CASE_TEMPLATE("l2 distance", T, TEST_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }
    auto ax = tann::to_span<T>(a);
    auto bx = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::compare_l2(ax, bx);
    auto n2 = tann::PrimComparator::simple_compare_l2<T, double>(ax, bx);
    turbo::Println("l2 n1:{} n2:{}", n1, n2);
    CHECK_LT(fabs(n1 - 95.163), 0.001);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("hamming distance", T, TEST_HM_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto ax = tann::to_span<T>(a);
    auto bx = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::compare_hamming<T>(ax, bx);
    auto n2 = tann::PrimComparator::simple_compare_hamming<T>(ax, bx);
    turbo::Println("hamming:{} {}", n1, n2);
    CHECK_EQ(n1, n2);

}

TEST_CASE_TEMPLATE("jaccard distance", T, TEST_HM_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto ay = tann::to_span<T>(a);
    auto by = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::compare_jaccard<T>(ay, by);
    auto n2 = tann::PrimComparator::simple_compare_jaccard<T>(ay, by);
    turbo::Println("hamming:{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("cosine distance", T, TEST_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto ay = tann::to_span<T>(a);
    auto by = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::compare_cosine<T>(ay, by);
    auto n2 = tann::PrimComparator::simple_compare_cosine<T>(ay, by);
    turbo::Println("cosine:{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("angle distance", T, TEST_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto ay = tann::to_span<T>(a);
    auto by = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::compare_angle<T>(ay, by);
    auto n2 = tann::PrimComparator::compare_angle<T>(ay, by);
    turbo::Println("angle:{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("ip distance", T, TEST_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto ay = tann::to_span<T>(a);
    auto by = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::compare_inner_product<T>(ay, by);
    auto n2 = tann::PrimComparator::simple_compare_inner_product<T>(ay, by);
    turbo::Println("ip:{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("normalized cosine distance", T, TEST_NORM_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto ay = tann::to_span<T>(a);
    auto by = tann::to_span<T>(b);
    tann::l2_norm(ay);
    tann::l2_norm(by);
    auto n1 = tann::PrimComparator::compare_normalized_cosine<T>(ay, by);
    auto n2 = tann::PrimComparator::compare_normalized_cosine<T>(ay, by);
    turbo::Println("nor cosine :{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("normalized angle distance", T, TEST_NORM_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }
    auto ay = tann::to_span<T>(a);
    auto by = tann::to_span<T>(b);
    tann::l2_norm(ay);
    tann::l2_norm(by);
    auto n1 = tann::PrimComparator::compare_normalized_angle<T>(ay, by);
    auto n2 = tann::PrimComparator::compare_normalized_angle<T>(ay, by);
    turbo::Println("nor angle :{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("poincare distance", T, TEST_NORM_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    auto at = tann::to_span<T>(a);
    auto bt = tann::to_span<T>(b);
    auto an = tann::get_l2_norm(at);
    auto bn = tann::get_l2_norm(bt);
    // make norm < 1
    for (int i = 0; i < 128; i++) {
        at[i] /= an * 1.2;
        bt[i] /= bn * 1.1;
    }
    auto n1 = tann::PrimComparator::compare_poincare<T>(at, bt);
    auto n2 = tann::PrimComparator::simple_compare_poincare<T>(at, bt);
    turbo::Println("poincare :{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}

TEST_CASE_TEMPLATE("lorentz distance", T, TEST_NORM_TYPES) {
    tann::AlignedQuery<T> a;
    tann::AlignedQuery<T> b;
    a.reserve(128);
    b.reserve(128);
    for (int i = 0; i < 128; i++) {
        a.emplace_back(i % 20);
        b.emplace_back((128 - i) % 10);
    }

    a[0] += 99;
    b[0] += 98;
    auto ay = tann::to_span<T>(a);
    auto by = tann::to_span<T>(b);
    auto n1 = tann::PrimComparator::simple_compare_lorentz(ay, by);
    auto n2 = tann::PrimComparator::compare_lorentz(ay, by);
    turbo::Println("lorentz :{} {}", n1, n2);
    CHECK_LT(fabs(n1 - n2), 0.001);

}