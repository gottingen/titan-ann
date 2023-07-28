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
#ifndef TANN_CORE_NEIHBOR_QUEUE_H_
#define TANN_CORE_NEIHBOR_QUEUE_H_

#include <cstddef>
#include <mutex>
#include <vector>
#include <cstring>
#include "tann/core/types.h"

namespace tann {

    struct NeighborEntity {
        label_type label{constants::kUnknownLabel};
        location_t lid{constants::kUnknownLocation};
        double distance{0.0};
        bool expanded{false};

        NeighborEntity() = default;

        NeighborEntity(double d, location_t o) : lid{o}, distance{d} {

        }

        NeighborEntity(double d, label_type l, location_t o) : label{l}, lid{o}, distance{d} {
        }

        inline bool operator<(const NeighborEntity &other) const {
            return distance < other.distance || (distance == other.distance && label < other.label);
        }

        inline bool operator==(const NeighborEntity &other) const {
            return (label == other.label);
        }
    };

    // Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
    //            the first Neighbor which is unexpanded.
    class NeighborQueue {
    public:
        NeighborQueue() : _size(0), _capacity(0), _cur(0) {
        }

        explicit NeighborQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1) {
        }

        const NeighborEntity &top() const {
            return _data[_size - 1];
        }

        void pop() {
            _size--;
        }

        void insert(double dist, location_t lid) {
            NeighborEntity nbr(dist, lid);
            insert(nbr);
        }

        // Inserts the item ordered into the set up to the sets capacity.
        // The item will be dropped if it is the same id as an exiting
        // set item or it has a greated distance than the final
        // item in the set. The set cursor that is used to pop() the
        // next item will be set to the lowest index of an uncheck item
        void insert(const NeighborEntity &nbr) {
            if (_size == _capacity && _data[_size - 1] < nbr) {
                return;
            }

            size_t lo = 0, hi = _size;
            while (lo < hi) {
                size_t mid = (lo + hi) >> 1;
                if (nbr < _data[mid]) {
                    hi = mid;
                    // Make sure the same id isn't inserted into the set
                } else if (_data[mid].lid == nbr.lid) {
                    return;
                } else {
                    lo = mid + 1;
                }
            }

            if (lo < _capacity) {
                std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(NeighborEntity));
            }
            _data[lo] = {nbr.distance, nbr.label, nbr.lid};
            if (_size < _capacity) {
                _size++;
            }
            if (lo < _cur) {
                _cur = lo;
            }
        }

        NeighborEntity closest_unexpanded() {
            _data[_cur].expanded = true;
            size_t pre = _cur;
            while (_cur < _size && _data[_cur].expanded) {
                _cur++;
            }
            return _data[pre];
        }

        bool has_unexpanded_node() const {
            return _cur < _size;
        }

        size_t size() const {
            return _size;
        }

        bool empty() const {
            return _size == 0;
        }

        size_t capacity() const {
            return _capacity;
        }

        void reserve(size_t capacity) {
            if (capacity + 1 > _data.size()) {
                _data.resize(capacity + 1);
            }
            _capacity = capacity;
        }

        NeighborEntity &operator[](size_t i) {
            return _data[i];
        }

        NeighborEntity operator[](size_t i) const {
            return _data[i];
        }

        void clear() {
            _size = 0;
            _cur = 0;
            _capacity = 0;
        }

        void swap(NeighborQueue & rhs) {
            std::swap(_size, rhs._size);
            std::swap(_capacity, rhs._capacity);
            std::swap(_cur, rhs._cur);
            std::swap(_data, rhs._data);
        }

    private:
        size_t _size, _capacity, _cur;
        std::vector<NeighborEntity> _data;
    };

} // namespace tann
namespace fmt {
    template<>
    struct formatter<tann::NeighborEntity> : formatter<double> {
        // parse is inherited from formatter<float>.

        auto format(const tann::NeighborEntity &c, format_context &ctx) const {
            format_to(ctx.out(), "distance:{} label:{} location:{}", c.distance, c.label, c.lid);
            return ctx.out();
        }
    };
}
#endif  // TANN_CORE_NEIHBOR_QUEUE_H_
