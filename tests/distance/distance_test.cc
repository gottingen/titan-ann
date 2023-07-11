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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "tann/distance/distance.h"
#include "turbo/log/logging.h"
#include "tann/comparator/primitive_comparator.h"
#include "turbo/simd/memory/aligned_allocator.h"

class DistanceTest {
public:
    DistanceTest() {
        a = std::vector<float, turbo::simd::aligned_allocator<float, 512>>{1.0, 2.0, 4.0, 4.6, 1.0, 2.0, 4.0,
                                                                                      4.6, 1.0, 2.0, 4.0, 4.6, 1.0, 2.0,
                                                                                      4.0, 4.6};
        b = std::vector<float, turbo::simd::aligned_allocator<float, 512>>{4.0, 5.2, 4.0, 2.6, 4.0, 5.2, 4.0,
                                                                                      2.6, 4.0, 5.2, 4.0, 2.6, 4.0, 5.2,
                                                                                      4.0, 2.6};
        da = std::vector<double, turbo::simd::aligned_allocator<double, 512>>{1.0, 2.0, 4.0, 4.6, 1.0, 2.0, 4.0,
                                                                           4.6, 1.0, 2.0, 4.0, 4.6, 1.0, 2.0,
                                                                           4.0, 4.6};
        db = std::vector<double, turbo::simd::aligned_allocator<double, 512>>{4.0, 5.2, 4.0, 2.6, 4.0, 5.2, 4.0,
                                                                           2.6, 4.0, 5.2, 4.0, 2.6, 4.0, 5.2,
                                                                           4.0, 2.6};
        ia = std::vector<uint64_t>{10, 20, 40, 46, 10, 20, 40, 46, 10, 20, 40, 46, 10, 20, 40, 46};
        ib = std::vector<uint64_t>{40, 52, 40, 26, 40, 52, 40, 26, 40, 52, 40, 26, 40, 52, 40, 26};
    }

    ~DistanceTest() {

    }

    std::vector<float, turbo::simd::aligned_allocator<float, 512>> a;
    std::vector<float, turbo::simd::aligned_allocator<float, 512>> b;

    std::vector<double, turbo::simd::aligned_allocator<double, 512>> da;
    std::vector<double, turbo::simd::aligned_allocator<double, 512>> db;

    std::vector<uint64_t> ia;
    std::vector<uint64_t> ib;
};


TEST_CASE_FIXTURE(DistanceTest, "l1_float") {
    TLOG_INFO("1");
    turbo::Span<float> sa(a);
    turbo::Span<float> sb(b);
    tann::DistanceL1<float, turbo::simd::aligned_mode> dl1;
    auto s1 = dl1.compare(sa, sb);
    TLOG_INFO("2");

    double s2 = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        s2 += std::abs(sa[i] - sb[i]);
    }
    TLOG_INFO("s1:{} s2: {}", s1, s2);
    CHECK_LT(std::abs(s1 - s2), 0.001);
}


TEST_CASE_FIXTURE(DistanceTest, "l1_double") {

    std::vector<double> c{1.0, 2.0, 4.0, 4.6};
    std::vector<double> d{4.0, 5.2, 4.0, 2.6};
    turbo::Span<double> sa(c);
    turbo::Span<double> sb(d);
    tann::DistanceL1<double, turbo::simd::unaligned_mode> dl1;
    auto s1 = dl1.compare(sa, sb);

    double s2 = 0.0;
    for (int i = 0; i < sa.size(); ++i) {
        s2 += std::abs(sa[i] - sb[i]);
    }
    TLOG_INFO("s1:{} s2: {}", s1, s2);
    CHECK_LT(std::abs(s1 - s2), 0.001);
}

TEST_CASE_FIXTURE(DistanceTest, "l2_float") {
    turbo::Span<float> sa(a);
    turbo::Span<float> sb(b);
    tann::DistanceL2<float, turbo::simd::aligned_mode> dl2;
    auto s1 = dl2.compare(sa, sb);

    double s2 = 0.0;
    for (int i = 0; i < sa.size(); ++i) {
        s2 += (sa[i] - sb[i]) * (sa[i] - sb[i]);
    }
    s2 = std::sqrt(s2);
    auto s3 = tann::PrimitiveComparator::L2Float().compare(a.data(), b.data(), a.size());
    TLOG_INFO("s1:{} s2: {} s3: {}", s1, s2, s3);
    CHECK_LT(std::abs(s1 - s2), 0.001);
}

TEST_CASE_FIXTURE(DistanceTest, "hamming_uint64") {

    turbo::Span<uint64_t> sa(ia);
    turbo::Span<uint64_t> sb(ib);
    tann::DistanceHamming<uint64_t, turbo::simd::aligned_mode> dh;
    auto s1 = dh.compare(sa, sb);
    auto s3 = tann::PrimitiveComparator::compareHammingDistance(ia.data(), ib.data(), ia.size());
    TLOG_INFO("s1:{} s3: {}", s1, s3);
    CHECK_LT(std::abs(s1 - s3), 0.001);
}

TEST_CASE_FIXTURE(DistanceTest, "hamming_uint32") {

    turbo::Span<uint32_t> sa(reinterpret_cast<uint32_t *>(a.data()),
                             a.size() * sizeof(decltype(a)::value_type) / sizeof(uint32_t));
    turbo::Span<uint32_t> sb(reinterpret_cast<uint32_t *>(b.data()),
                             b.size() * sizeof(decltype(b)::value_type) / sizeof(uint32_t));
    tann::DistanceHamming<uint32_t, turbo::simd::aligned_mode> dh;
    auto s1 = dh.compare(sa, sb);

    auto s3 = tann::PrimitiveComparator::compareHammingDistance(a.data(), b.data(), a.size());
    TLOG_INFO("s1:{} s3: {}", s1, s3);
    CHECK_LT(std::abs(s1 - s3), 0.001);
}

TEST_CASE_FIXTURE(DistanceTest, "hamming_float") {

    turbo::Span<float> sa(a);
    turbo::Span<float> sb(b);
    tann::DistanceHamming<float, turbo::simd::aligned_mode> dh;
    auto s1 = dh.compare(sa, sb);

    auto s3 = tann::PrimitiveComparator::compareHammingDistance(a.data(), b.data(), a.size());
    TLOG_INFO("s1:{} s3: {}", s1, s3);
    CHECK_LT(std::abs(s1 - 128), 0.001);
}

TEST_CASE_FIXTURE(DistanceTest, "jaccard_float") {

    turbo::Span<float> sa(a);
    turbo::Span<float> sb(b);
    tann::DistanceJaccard<float, turbo::simd::aligned_mode> dh;
    auto s1 = dh.compare(sa, sb);

    auto s3 = tann::PrimitiveComparator::compareJaccardDistance(a.data(), b.data(), a.size());
    TLOG_INFO("s1:{} s3: {}", s1, s3);
    CHECK_LT(std::abs(s1 - s3), 0.001);
}

TEST_CASE_FIXTURE(DistanceTest, "cosine_float") {

    turbo::Span<float> sa(a);
    turbo::Span<float> sb(b);
    tann::DistanceCosine<float, turbo::simd::aligned_mode> dc;
    auto s1 = dc.compare(sa, sb);

    auto s3 = tann::PrimitiveComparator::compareCosine(a.data(), b.data(), a.size());
    TLOG_INFO("s1:{} s3: {}", s1, s3);
    CHECK_LT(std::abs(s1 - s3), 0.001);
}

TEST_CASE_FIXTURE(DistanceTest, "cosine_double") {

    turbo::Span<double> sa(da);
    turbo::Span<double> sb(db);
    tann::DistanceCosine<double, turbo::simd::aligned_mode> dc;
    auto s1 = dc.compare(sa, sb);

    auto s3 = tann::PrimitiveComparator::compareCosine(a.data(), b.data(), a.size());
    TLOG_INFO("s1:{} s3: {}", s1, s3);
    CHECK_LT(std::abs(s1 - s3), 0.001);
}
/*
TEST_CASE_FIXTURE(DistanceTest, "cosine_int64") {

    turbo::Span<uint64_t> sa(ia);
    turbo::Span<uint64_t> sb(ib);
    tann::DistanceCosine<uint64_t, turbo::simd::aligned_mode> dc;
    auto s1 = dc.compare(sa, sb);

    auto s3 = tann::PrimitiveComparator::compareCosine(ia.data(), ib.data(), ia.size());
    TLOG_INFO("s1:{} s3: {}", s1, s3);
    CHECK_LT(std::abs(s1 - s3), 0.001);
}
*/
TEST_CASE_FIXTURE(DistanceTest, "nor_cosine_float") {

    turbo::Span<float> sa(a);
    turbo::Span<float> sb(b);
    tann::DistanceNormalizedCosine<float, turbo::simd::aligned_mode> dnc;
    std::vector<float> na(a.size());
    std::vector<float> nb(a.size());
    turbo::Span<float> sna(na);
    turbo::Span<float> snb(nb);
    dnc.normalization(sa, sna);
    dnc.normalization(sb, snb);
    auto s1 = dnc.compare(sna, snb);

    tann::DistanceCosine<float, turbo::simd::aligned_mode> dc;
    auto s2 = dc.compare(sna, snb);
    auto s3 = dc.compare(sa, sb);
    TLOG_INFO("s1:{} s2:{} s3: {}", s1, s2, s3);
    CHECK_LT(std::abs(s1 - s3), 0.001);
}

