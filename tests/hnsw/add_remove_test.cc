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
#include "hnsw_test_fixture.h"
#include "doctest/doctest.h"
#include <thread>
#include <chrono>


TEST_CASE_FIXTURE(HnswIndexTestFixture, "add and remove") {

    // generate random labels to delete them from index
    int iter = 0;
    while (iter < max_elements) {

        try {
            auto r = index->add_vector(wop, turbo::Span<uint8_t>((uint8_t *) (batch1.get() + d * iter), d * sizeof(float)),
                                       iter);
            if (!r.ok()) {
                turbo::Println(r.status().ToString());
            }
        } catch (std::exception &e) {
            turbo::Println(e.what());
            FAIL(e.what());
        }

        iter += 1;
    }
    CHECK_EQ(max_elements, index->size());
    CHECK_EQ(0, index->remove_size());
    // delete half random elements of batch1 data
    for (int i = 0; i < num_elements; i++) {
        auto r = index->remove_vector(rand_labels[i]);
        if (!r.ok()) {
            turbo::Println(r.ToString());
        }
    }
    CHECK_EQ(num_elements, index->remove_size());
}
