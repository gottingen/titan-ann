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

#include "tann/hnsw/hnsw_index.h"
#include <thread>
#include <chrono>


class HnswIndexTestFixture {
public:
    HnswIndexTestFixture() {
        rng.seed(47);
        auto ptr = new tann::HnswIndexOption;
        ptr->data_type = tann::DataType::DT_FLOAT;
        ptr->dimension = 16;
        ptr->metric = tann::METRIC_L2;
        option.reset(ptr);

        batch1.reset(new float[d * max_elements]);
        for (int i = 0; i < d * max_elements; i++) {
            batch1[i] = distrib_real(rng);
        }

        rand_labels.resize(max_elements);
        for (int i = 0; i < max_elements; i++) {
            rand_labels[i] = i;
        }
        std::shuffle(rand_labels.begin(), rand_labels.end(), rng);
        index.reset(new tann::HnswIndex);
        wop.replace_deleted = false;
        wop1.replace_deleted = true;

        auto rs = index->initialize(option.get());
        assert(rs.ok());

    }

    ~HnswIndexTestFixture() {

    }

    int d = 16;
    int num_elements = 1000;
    int max_elements = 2 * num_elements;
    std::mt19937 rng;
    std::unique_ptr<tann::IndexOption> option;
    std::uniform_real_distribution<> distrib_real;
    std::unique_ptr<float[]> batch1;
    std::vector<int> rand_labels;
    std::unique_ptr<tann::HnswIndex> index;
    tann::WriteOption wop;
    tann::WriteOption wop1;
};

TEST_CASE_FIXTURE(HnswIndexTestFixture, "add and remove") {
    std::cout << "Running multithread load test" << std::endl;

    // generate random labels to delete them from index
    int iter = 0;
    while (iter < max_elements) {

        try {
            auto r = index->add_vector(wop, turbo::Span<uint8_t>((uint8_t *) (batch1.get() + d * iter), d * sizeof(float)),
                                       iter);
            if (!r.ok()) {
                turbo::Println(r.ToString());
                FAIL(r.ToString());
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
