// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Copyright 2023 The Tann Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <memory>
#include <type_traits>
#include <vector>
#include "turbo/container/dynamic_bitset.h"

namespace tann {
    // A map whose key is a natural number (from 0 onwards) and maps to a value.
    // Made as both memory and performance efficient map for scenario such as
    // TANN location-to-tag map. There, the pool of numbers is consecutive from
    // zero to some max value, and it's expected that most if not all keys from 0
    // up to some current maximum will be present in the map. The memory usage of
    // the map is determined by the largest inserted key since it uses vector as a
    // backing store and bitset for presence indication.
    //
    // Thread-safety: this class is not thread-safe in general.
    // Exception: multiple read-only operations are safe on the object only if
    // there are no writers to it in parallel.
    template<typename Key, typename Value>
    class natural_number_map {
    public:
        static_assert(std::is_trivial<Key>::value, "Key must be a trivial type");
        // Some of the class member prototypes are done with this assumption to
        // minimize verbosity since it's the only use case.
        static_assert(std::is_trivial<Value>::value,
                      "Value must be a trivial type");

        // Represents a reference to a element in the map. Used while iterating
        // over map entries.
        struct position {
            size_t _key;
            // The number of keys that were enumerated when iterating through the map
            // so far. Used to early-terminate enumeration when ithere
            // are no more entries in the map.
            size_t _keys_already_enumerated;

            // Returns whether it's valid to access the element at this position in
            // the map.
            bool is_valid() const;
        };

        natural_number_map();

        void reserve(size_t count);

        size_t size() const;

        void set(Key key, Value value);

        void erase(Key key);

        bool contains(Key key) const;

        bool try_get(Key key, Value &value) const;

        // Returns the value at the specified position. Prerequisite: position is
        // valid.
        Value get(const position &pos) const;

        // Finds the first element in the map, if any. Invalidated by changes in the
        // map.
        position find_first() const;

        // Finds the next element in the map after the specified position.
        // Invalidated by changes in the map.
        position find_next(const position &after_position) const;

        void clear();

    private:
        // Number of entries in the map. Not the same as size() of the
        // _values_vector below.
        size_t _size;

        // Array of values. The key is the index of the value.
        std::vector<Value> _values_vector;

        // Values that are in the set have the corresponding bit index set
        // to 1.
        //
        std::unique_ptr<turbo::dynamic_bitset<>> _values_bitset;
    };
}  // namespace tann
