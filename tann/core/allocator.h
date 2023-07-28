//
// Created by jeff on 23-7-19.
//

#ifndef TANN_CORE_ALLOCATOR_H_
#define TANN_CORE_ALLOCATOR_H_

#include "turbo/simd/simd.h"
#include "turbo/meta/span.h"
#include "tann/core/types.h"
#include <vector>

namespace tann {
    class Allocator {
    public:
        static constexpr std::size_t requires_alignment = turbo::simd::default_arch::requires_alignment();
        // If an algorithm has a requirement that some data be aligned to a certain
        // boundary it can use this function to indicate that requirement. Currently,
        // we are setting it to  Arch::alignment() for bytes alignment, and the dim alignment is set
        // to alignment_bytes/sizeof(T)
        static constexpr std::size_t alignment_bytes = turbo::simd::default_arch::alignment();

        typedef turbo::simd::aligned_allocator<uint8_t, alignment_bytes> allocator_type;
        static allocator_type alloc;
    };

    ///////////////////////
    /// for query alloc
    template<typename T>
    using AlignedQuery =  std::vector<T, turbo::simd::aligned_allocator<T, Allocator::alignment_bytes>>;


    template<typename T, typename U>
    inline turbo::Span<T> to_span(const AlignedQuery<U> &query) {
        return turbo::Span<T>(const_cast<T*>(reinterpret_cast<const T*>(query.data())), query.size() * sizeof(U)/sizeof(T));
    }

    template<typename T, typename U>
    inline turbo::Span<T> to_span(const std::vector<U> &query) {
        return turbo::Span<T>(const_cast<T*>(reinterpret_cast<const T*>(query.data())), query.size() * sizeof(U)/sizeof(T));
    }

    template<typename T, typename U>
    inline turbo::Span<T> to_span(const turbo::Span<U> &query) {
        return turbo::Span<T>(reinterpret_cast<T*>(query.data()), query.size()* sizeof(U)/sizeof(T));
    }

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

}
#endif  // TANN_CORE_ALLOCATOR_H_
