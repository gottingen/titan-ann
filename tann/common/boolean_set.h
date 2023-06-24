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
#ifndef TANN_COMMON_BOOLEAN_SET_H_
#define TANN_COMMON_BOOLEAN_SET_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tann {
    // BooleanSet has been already optimized.
    class BooleanSet {
    public:
        BooleanSet(size_t s) {
            size = (s >> 6) + 1; // 2^6=64
            size = ((size >> 2) << 2) + 4;
            bitvec.resize(size);
        }

        inline uint64_t getBitString(size_t i) { return (uint64_t) 1 << (i & (64 - 1)); }

        inline uint64_t &getEntry(size_t i) { return bitvec[i >> 6]; }

        inline bool operator[](size_t i) {
            return (getEntry(i) & getBitString(i)) != 0;
        }

        inline void set(size_t i) {
            getEntry(i) |= getBitString(i);
        }

        inline void insert(size_t i) { set(i); }

        inline void reset(size_t i) {
            getEntry(i) &= ~getBitString(i);
        }

        std::vector<uint64_t> bitvec;
        uint64_t size;
    };

}  // namespace tann
#endif  // TANN_COMMON_BOOLEAN_SET_H_
