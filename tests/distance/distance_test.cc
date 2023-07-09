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
#include "tann/common/half.hpp"
TEST_CASE("l1, float") {

    std::vector<float> a{1.0, 2.0, 4.0, 4.6};
    std::vector<float> b{4.0, 5.2, 4.0, 2.6};
    turbo::Span<float> sa(a);
    turbo::Span<float> sb(b);
    tann::DistanceL1<float> dl1;
    auto s1 = dl1.compare(sa, sb);

    float s2 = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        s2 += std::abs(sa[i] - sb[i]);
    }
    TLOG_INFO("s1:{} s2: {}", s1, s2);
    CHECK_LT(std::abs(s1 - s2),  0.001);
}

TEST_CASE("l1, double") {

    std::vector<double> a{1.0, 2.0, 4.0, 4.6};
    std::vector<double> b{4.0, 5.2, 4.0, 2.6};
    turbo::Span<double> sa(a);
    turbo::Span<double> sb(b);
    tann::DistanceL1<double> dl1;
    auto s1 = dl1.compare(sa, sb);

    float s2 = 0.0;
    for (int i = 0; i < sa.size(); ++i) {
        s2 += std::abs(sa[i] - sb[i]);
    }
    TLOG_INFO("s1:{} s2: {}", s1, s2);
    CHECK_LT(std::abs(s1 - s2),  0.001);
}
