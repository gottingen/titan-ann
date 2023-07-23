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


#include "tann/hnsw/hnsw_index.h"
#include <thread>
#include <chrono>

int main() {
    std::cout << "Running multithread load test" << std::endl;
    int d = 16;
    int num_elements = 1000;
    int max_elements = 2 * num_elements;
    int num_threads = 2;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;

    auto ptr = new tann::HnswIndexOption;
    ptr->data_type = tann::DataType::DT_FLOAT;
    ptr->dimension = 16;
    ptr->metric = tann::METRIC_L2;
    std::unique_ptr<tann::IndexOption> option(ptr);
    // generate batch1 and batch2 data
    float *batch1 = new float[d * max_elements];
    for (int i = 0; i < d * max_elements; i++) {
        batch1[i] = distrib_real(rng);
    }
    float *batch2 = new float[d * num_elements];
    for (int i = 0; i < d * num_elements; i++) {
        batch2[i] = distrib_real(rng);
    }

    // generate random labels to delete them from index
    std::vector<int> rand_labels(max_elements);
    for (int i = 0; i < max_elements; i++) {
        rand_labels[i] = i;
    }
    std::shuffle(rand_labels.begin(), rand_labels.end(), rng);

    int iter = 0;
    tann::WriteOption wop;
    tann::WriteOption wop1;
    wop.replace_deleted = false;
    wop1.replace_deleted = true;
    auto *index = new tann::HnswIndex;
    auto rs = index->initialize(std::move(option));
    if (!rs.ok()) {
        turbo::Println(rs.ToString());
        return 0;
    }
    while (iter < max_elements) {

        try {
            auto r = index->add_vector(wop, turbo::Span<uint8_t>((uint8_t *) (batch1 + d * iter), d * sizeof(float)),
                                       iter);
            if (!r.ok()) {
                turbo::Println(r.ToString());
                return 0;
            }
        } catch (std::exception &e) {
            turbo::Println(e.what());
            return 0;
        }

        iter += 1;
    }
    turbo::Println("index remove size:{}", index->remove_size());
    // delete half random elements of batch1 data
    for (int i = 0; i < num_elements; i++) {
        auto r = index->remove_vector(rand_labels[i]);
        if (!r.ok()) {
            turbo::Println(r.ToString());
            return 0;
        }
    }
    turbo::Println("index remove size:{}", index->remove_size());
    delete index;
    std::cout << "Finish" << std::endl;

    delete[] batch1;
    delete[] batch2;
    return 0;
}
