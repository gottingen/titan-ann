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

#include "boost_dynamic_bitset_fwd.h"

namespace tann {
  // A set of natural numbers (from 0 onwards). Made for scenario where the
  // pool of numbers is consecutive from zero to some max value and very
  // efficient methods for "add to set", "get any value from set", "is in set"
  // are needed. The memory usage of the set is determined by the largest
  // number of inserted entries (uses a vector as a backing store) as well as
  // the largest value to be placed in it (uses bitset as well).
  //
  // Thread-safety: this class is not thread-safe in general.
  // Exception: multiple read-only operations (e.g. is_in_set, empty, size) are
  // safe on the object only if there are no writers to it in parallel.
  template<typename T>
  class natural_number_set {
   public:
    static_assert(std::is_trivial<T>::value,
                  "Identifier must be a trivial type");

    natural_number_set();

    bool   is_empty() const;
    void   reserve(size_t count);
    void   insert(T id);
    T      pop_any();
    void   clear();
    size_t size() const;
    bool   is_in_set(T id) const;

   private:
    // Values that are currently in set.
    std::vector<T> _values_vector;

    // Values that are in the set have the corresponding bit index set
    // to 1.
    //
    // Use a pointer here to allow for forward declaration of dynamic_bitset
    // in public headers to avoid making boost a dependency for clients
    // of TANN.
    std::unique_ptr<boost::dynamic_bitset<>> _values_bitset;
  };
}  // namespace tann
