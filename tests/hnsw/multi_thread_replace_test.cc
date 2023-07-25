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

#include "hnsw_test_fixture.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"

#include "tann/hnsw/hnsw_index.h"
#include <thread>
#include <chrono>


TEST_CASE_FIXTURE(HnswIndexTestFixture, "add and remove") {
    std::cout << "Running multithread load test" << std::endl;

    int iter = 0;
    while (iter < 200) {
        index.reset(new tann::HnswIndex);
        auto rs = index->initialize(option);

        // add batch1 data
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
            auto r = index->add_vector(wop, turbo::Span<uint8_t>((uint8_t *)(batch1.get() + d * row), d * sizeof(float )), row);
            CHECK_EQ(r.ok(), true);
        });

        // delete half random elements of batch1 data
        for (int i = 0; i < num_elements; i++) {
            auto r = index->remove_vector(rand_labels[i]);
            CHECK_EQ(r.ok(), true);
        }

        // replace deleted elements with batch2 data
        ParallelFor(0, num_elements, num_threads, [&](size_t row, size_t threadId) {
            int label = rand_labels[row] + max_elements;
            auto r= index->add_vector(wop1, turbo::Span<uint8_t>((uint8_t *)(batch2.get() + d * row), d * sizeof(float )), label);
            if(!r.ok()) {
                turbo::Println(r.ToString());
            }
            CHECK_EQ(r.ok(), true);
        });

        iter += 1;
        index.reset();
    }
    
    std::cout << "Finish" << std::endl;
}
