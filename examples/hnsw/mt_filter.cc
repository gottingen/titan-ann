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


#include "tann/core/index_core.h"
#include "turbo/flags/flags.h"
#include <thread>
#include <random>


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
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


// Filter that allows labels divisible by divisor
class PickDivisibleIds : public tann::BaseFilterFunctor {
    unsigned int divisor = 1;
public:
    PickDivisibleIds(unsigned int divisor) : divisor(divisor) {
        assert(divisor != 0);
    }

    bool operator()(tann::label_type label_id) {
        return label_id % divisor == 0;
    }
};


int main(int argc, char **argv) {
    turbo::App app;
    int num_threads;
    app.add_option("-t, --thread", num_threads, "Number of threads for operations with index")->default_val(20);
    TURBO_FLAGS_PARSE(app, argc, argv);
    // Initing index
    tann::IndexOption option;
    tann::HnswIndexOption hnsw_option;
    option.data_type = tann::DataType::DT_FLOAT;
    option.dimension = 16;
    option.metric = tann::METRIC_L2;
    option.engine_type = tann::EngineType::ENGINE_HNSW;
    option.max_elements = 10000;
    hnsw_option.ef_construction = 200;
    hnsw_option.m = 16;

    std::unique_ptr<tann::IndexCore> index = std::make_unique<tann::IndexCore>();
    auto r = index->initialize(option, hnsw_option);
    if (!r.ok()) {
        turbo::Println(r.ToString());
    }
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float *data = new float[option.dimension * option.max_elements];
    for (int i = 0; i < option.dimension * option.max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    ParallelFor(0, option.max_elements, num_threads, [&](size_t row, size_t threadId) {
        tann::WriteOption wop;
        auto ra = index->add_vector(wop, turbo::Span<uint8_t>((uint8_t *) (data + option.dimension * row),
                                                              option.dimension * sizeof(float)), row);
        if (!ra.ok()) {
            turbo::Println("{}", ra.status().ToString());
        }
    });

    turbo::Println("build index done");
    // Create filter that allows only even labels
    PickDivisibleIds pickIdsDivisibleByTwo(2);

    // Query the elements for themselves with filter and check returned labels

    int k = 10;
    std::vector<tann::label_type> neighbors(option.max_elements * k);
    /*
    for (int row = 0; row < option->max_elements; ++row) {
        tann::QueryContext query(turbo::Span<uint8_t>(reinterpret_cast<uint8_t *>(data + option->dimension * row),
                                                      option->dimension * sizeof(float)));
        query.k = k;
        query.is_allowed = &pickIdsDivisibleByTwo;
        auto rs = index->search_vector(&query);
        auto &result = query.results;
        turbo::Println("search vector {}", row);
        for (int i = 0; i < result.size(); i++) {
            neighbors[row * k + i] = result[i].second;
        }
    }*/

    ParallelFor(0, option.max_elements, num_threads, [&](size_t row, size_t threadId) {
        tann::SearchContext query(turbo::Span<uint8_t>(reinterpret_cast<uint8_t*>(data + option.dimension * row), option.dimension * sizeof(float )));
        query.k = k;
        query.is_allowed = &pickIdsDivisibleByTwo;
        tann::SearchResult res;
        auto rs = index->search_vector(&query, res);
        auto &result = res.results;
        turbo::Println("results: {}",result.size());
        for (int i = 0; i < result.size(); i++) {
            turbo::Println("results: {}",turbo::FormatRange("{}", result, ", "));
            neighbors[row * k + i] = result[i].second;
        }
    });
    turbo::Println("search index done");
    for (tann::label_type label: neighbors) {
        if (label % 2 == 1) std::cout << "Error: found odd label\n";
    }
    turbo::Println("check search index done");
    delete[] data;
    return 0;
}

