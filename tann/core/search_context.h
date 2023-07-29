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
#ifndef TANN_CORE_SEARCH_CONTEXT_H_
#define TANN_CORE_SEARCH_CONTEXT_H_

#include <cstddef>
#include <queue>
#include <vector>
#include "tann/core/types.h"
#include "tann/core/allocator.h"

namespace tann {

    struct SearchContext {
        explicit SearchContext(turbo::Span<uint8_t> query) {
            original_query = to_span<uint8_t>(query);
        }
        explicit SearchContext(turbo::Span<float16> query) {
            original_query = to_span<uint8_t>(query);
        }

        explicit SearchContext(turbo::Span<float> query) {
            original_query = to_span<uint8_t>(query);
        }

        explicit SearchContext(const std::vector<uint8_t> &query) {
            original_query = to_span<uint8_t>(query);
        }

        explicit SearchContext(const std::vector<float16> &query) {
            original_query = to_span<uint8_t>(query);
        }

        explicit SearchContext(const std::vector<float> &query) {
            original_query = to_span<uint8_t>(query);
        }
    public:
        std::size_t k{0};
        std::size_t search_list{0};
        BaseFilterFunctor *is_allowed{nullptr};
        bool get_raw_vector{false};
        bool is_normalized{false};
        bool desc{false};
        turbo::Span<uint8_t> original_query;

    };
    struct SearchResult {
        std::vector<std::pair<distance_type, label_type>> results;
        std::vector<std::vector<uint8_t>> vectors;
        int64_t cost_ns{0};
    };

    struct InsertResult {
        int64_t cost_ns{0};
    };
}  // namespace

#endif  // TANN_CORE_SEARCH_CONTEXT_H_
