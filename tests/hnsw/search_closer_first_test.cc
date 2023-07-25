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
#include "hnsw_test_fixture.h"
#include <assert.h>

#include <vector>
#include <iostream>
#include "tann/hnsw/hnsw_index.h"
#include <thread>
#include <chrono>

namespace {

    TEST_CASE_FIXTURE(HnswIndexFilterFixture, "closer") {

        TLOG_INFO("insert data");
        tann::WriteOption op;
        for (size_t i = 0; i < n; ++i) {
            auto r1 = findex.add_vector(op, turbo::Span<uint8_t>(reinterpret_cast<uint8_t *>(data.data() + d * i),
                                                                 d * sizeof(float)), i);
            CHECK_EQ(r1.ok(), true);
            r1 = hindex.add_vector(op, turbo::Span<uint8_t>(reinterpret_cast<uint8_t *>(data.data() + d * i),
                                                            d * sizeof(float)), i);
            CHECK_EQ(r1.ok(), true);
        }

        TLOG_INFO("search data");
        // test searchKnnCloserFirst of BruteforceSearch
        for (size_t j = 0; j < nq; ++j) {
            auto *p = reinterpret_cast<uint8_t *>(query.data() + j * d);
            tann::QueryContext query_h(turbo::Span<uint8_t>(p, d * sizeof(float)));
            query_h.k = k;
            tann::QueryContext query_f(turbo::Span<uint8_t>(p, d * sizeof(float)));
            query_f.k = k;
            auto r1 = hindex.search_vector(&query_h);
            auto r2 = findex.search_vector(&query_f);
            CHECK_EQ(r1.ok(), true);
            CHECK_EQ(r2.ok(), true);
            auto gd = query_h.results;
            auto res = query_f.results;
            assert(gd.size() == res.size());
            size_t t = gd.size();
            for (size_t i = 0; i < res.size(); i++) {
                CHECK_EQ(gd[i].second, res[i].second);
            }
        }
    }

}  // namespace

