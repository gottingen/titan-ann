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
#ifndef TANN_CORE_QUERY_CONTEXT_H_
#define TANN_CORE_QUERY_CONTEXT_H_

#include <cstddef>
#include <queue>
#include <vector>
#include "tann/core/types.h"
#include "tann/core/allocator.h"

namespace tann {

    [[maybe_unused]] inline turbo::Span<uint8_t> make_aligned_query(turbo::Span<uint8_t> q, AlignedQuery<uint8_t>&raw_mem) {
        raw_mem.resize(q.size());
        std::memcpy(raw_mem.data(), q.data(), q.size());
        return to_span<uint8_t>(raw_mem);
    }

    [[maybe_unused]] inline turbo::Span<uint8_t> make_aligned_query(turbo::Span<float16> q, AlignedQuery<uint8_t>&raw_mem) {
        raw_mem.resize(raw_mem.size());
        auto rq = to_span<uint8_t, float16>(q);
        std::memcpy(raw_mem.data(), rq.data(), rq.size());
        return to_span<uint8_t>(raw_mem);
    }

    [[maybe_unused]] inline turbo::Span<uint8_t> make_aligned_query(turbo::Span<float> q, AlignedQuery<uint8_t>&raw_mem) {
        raw_mem.resize(raw_mem.size());
        auto rq = to_span<uint8_t, float>(q);
        std::memcpy(raw_mem.data(), rq.data(), rq.size());
        return to_span<uint8_t>(raw_mem);
    }

    [[maybe_unused]] inline turbo::Span<uint8_t> make_aligned_query(const AlignedQuery<uint8_t> &q, AlignedQuery<uint8_t>&raw_mem) {
        raw_mem.resize(q.size());
        std::memcpy(raw_mem.data(), q.data(), q.size());
        return to_span<uint8_t>(raw_mem);
    }

    [[maybe_unused]] inline turbo::Span<uint8_t> make_aligned_query(const AlignedQuery<float16> &q, AlignedQuery<uint8_t>&raw_mem) {
        raw_mem.resize(raw_mem.size());
        auto rq = to_span<uint8_t, float16>(q);
        std::memcpy(raw_mem.data(), rq.data(), rq.size());
        return to_span<uint8_t>(raw_mem);
    }

    [[maybe_unused]] inline turbo::Span<uint8_t> make_aligned_query(const AlignedQuery<float> &q, AlignedQuery<uint8_t>&raw_mem) {
        raw_mem.resize(raw_mem.size());
        auto rq = to_span<uint8_t, float>(q);
        std::memcpy(raw_mem.data(), rq.data(), rq.size());
        return to_span<uint8_t>(raw_mem);
    }


    struct QueryContext {
        explicit QueryContext(turbo::Span<uint8_t> query) {
            make_aligned_query(query, raw_query);
        }
        explicit QueryContext(turbo::Span<float16> query) {
            make_aligned_query(query, raw_query);
        }

        explicit QueryContext(turbo::Span<float> query) {
            make_aligned_query(query, raw_query);
        }

        explicit QueryContext(const AlignedQuery<uint8_t> &query) {
            make_aligned_query(query, raw_query);
        }

        explicit QueryContext(const AlignedQuery<float16> &query) {
            make_aligned_query(query, raw_query);
        }

        explicit QueryContext(const AlignedQuery<float> &query) {
            make_aligned_query(query, raw_query);
        }
    public:
        std::size_t k{0};
        std::vector<std::pair<distance_type, label_type>> results;
        BaseFilterFunctor *is_allowed{nullptr};
        std::vector<std::vector<uint8_t>> vectors;
        bool get_raw_vector{false};
        bool desc{false};
        AlignedQuery<uint8_t> raw_query;

    };
}  // namespace

namespace fmt {
    template <> struct formatter<std::pair<tann::distance_type, tann::label_type>>: formatter<float> {
        // parse is inherited from formatter<float>.

        auto format(std::pair<tann::distance_type, tann::label_type> c, format_context& ctx) const {
            format_to(ctx.out(), "dist:{} label:{}", c.first, c.second);
            return ctx.out();
        }
    };
}
#endif  // TANN_CORE_QUERY_CONTEXT_H_
