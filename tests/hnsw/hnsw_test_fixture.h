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
#include "tann/flat/flat_index.h"
#include "doctest/doctest.h"
#include <thread>
#include <chrono>

using tann::label_type;

class HnswIndexTestFixture {
public:
    HnswIndexTestFixture() {
        rng.seed(47);
        hnsw_option.data_type = tann::DataType::DT_FLOAT;
        hnsw_option.dimension = 16;
        hnsw_option.metric = tann::METRIC_L2;
        option = reinterpret_cast<tann::IndexOption *>(&hnsw_option);

        batch1.reset(new float[d * max_elements]);
        for (int i = 0; i < d * max_elements; i++) {
            batch1[i] = distrib_real(rng);
        }

        batch2.reset(new float[d * max_elements]);
        for (int i = 0; i < d * max_elements; i++) {
            batch2[i] = distrib_real(rng);
        }

        rand_labels.resize(max_elements);
        for (int i = 0; i < max_elements; i++) {
            rand_labels[i] = i;
        }
        std::shuffle(rand_labels.begin(), rand_labels.end(), rng);

        wop.replace_deleted = false;
        wop1.replace_deleted = true;

        index.reset(new tann::HnswIndex);
        auto rs = index->initialize(option);
        assert(rs.ok());

    }

    ~HnswIndexTestFixture() {

    }

    int d = 16;
    int num_elements = 1000;
    int num_threads = 4;
    int max_elements = 2 * num_elements;
    std::mt19937 rng;
    tann::IndexOption *option;
    tann::HnswIndexOption hnsw_option;
    std::uniform_real_distribution<> distrib_real;
    std::unique_ptr<float[]> batch1;
    std::unique_ptr<float[]> batch2;
    std::vector<int> rand_labels;
    std::unique_ptr<tann::HnswIndex> index;
    tann::WriteOption wop;
    tann::WriteOption wop1;
};

class HnswIndexFilterFixture {
public:
    HnswIndexFilterFixture() {
        data.resize(n * d);
        query.resize(nq * d);
        rng.seed(47);
        std::uniform_real_distribution<> distrib;

        for (label_type i = 0; i < n * d; ++i) {
            data[i] = distrib(rng);
        }
        for (label_type i = 0; i < nq * d; ++i) {
            query[i] = distrib(rng);
        }
        hnsw_option.data_type = tann::DataType::DT_FLOAT;
        hnsw_option.dimension = 16;
        hnsw_option.metric = tann::METRIC_L2;
        option = reinterpret_cast<tann::IndexOption *>(&hnsw_option);
        auto rs = hindex.initialize(option);
        CHECK_EQ(rs.ok(), true);

        rs = findex.initialize(option);
        CHECK_EQ(rs.ok(), true);
    }

    std::vector<float> data;
    std::vector<float> query;

    std::mt19937 rng;
    int d = 16;
    label_type n = 100;
    label_type nq = 10;
    size_t k = 10;
    tann::IndexOption *option;
    tann::HnswIndexOption hnsw_option;
    tann::HnswIndex hindex;
    tann::FlatIndex findex;
};

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread: threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


#endif //TANN_HNSW_TEST_FIXTURE_H
