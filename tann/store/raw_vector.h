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


#ifndef TANN_STORE_RAW_VECTOR_H_
#define TANN_STORE_RAW_VECTOR_H_

#include <vector>
namespace tann {
/*
    template<typename T, size_t ND, size_t RS = 128>
    class RawVector{
    public:
        RawVector() = default;

        size_t size() const {}

        bool empty() const;

        void resize(size_t n);

        auto get_vector(size_t index) {
            return flare::make_view(_regions[0], 2, flare::all());
        }

        template<typename E>
        size_t add_vector(const flare::expression<E> &v) {
            _regions[0](1) = v();
        }

    private:
        std::vector<flare::array<T>> _regions;
        size_t _size{0};
        size_t _dimensions = ND;
        size_t _region_size = RS;
    };*/
}
#endif  // TANN_STORE_RAW_VECTOR_H_
