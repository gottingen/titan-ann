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

#ifndef TANN_HNSW_TEST_FIXTURE_H
#define TANN_HNSW_TEST_FIXTURE_H

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

#endif //TANN_HNSW_TEST_FIXTURE_H
