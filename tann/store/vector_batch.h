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


#ifndef TANN_STORE_VECTOR_BATCH_H_
#define TANN_STORE_VECTOR_BATCH_H_

#include "tann/core/vector_space.h"
#include "tann/core/allocator.h"

namespace tann {

    class VectorBatch {
    public:
        VectorBatch() = default;

        ~VectorBatch() {
            if (_data) {
                _vp->free_vector(_data, _capacity);
                _data = nullptr;
            }
        }
        VectorBatch(VectorBatch&& rhs) noexcept{
            _vp = rhs._vp;
            _ndim = rhs._ndim;
            _capacity = rhs._capacity;
            _data = rhs._data;
            rhs._vp = nullptr;
            rhs._ndim = 0;
            rhs._data = nullptr;
            rhs._capacity = 0;
        }
        VectorBatch& operator=(VectorBatch&& rhs) noexcept {
            _vp = rhs._vp;
            _ndim = rhs._ndim;
            _capacity = rhs._capacity;
            _data = rhs._data;
            rhs._vp = nullptr;
            rhs._ndim = 0;
            rhs._data = nullptr;
            rhs._capacity = 0;
            return *this;
        }

        [[nodiscard]] turbo::Status init(VectorSpace *vp, std::size_t n) {
            _vp = vp;
            try {
                _data = _vp->alloc_vector(n);
            } catch (std::exception &e) {
                return turbo::UnavailableError(e.what());
            }
            _ndim = 0;
            _capacity = n;
            return turbo::OkStatus();
        }

        [[nodiscard]] bool is_full() const {
            return _ndim == _capacity;
        }

        [[nodiscard]] bool is_empty() const {
            return _ndim == 0;
        }

        [[nodiscard]] std::size_t size() const {
            return _ndim;
        }

        [[nodiscard]] turbo::Span<uint8_t> at(std::size_t i) const {
            TLOG_CHECK(i < _ndim, "overflow");
            return turbo::Span<uint8_t>{_data + i * _vp->vector_byte_size, _vp->vector_byte_size};
        }

        [[nodiscard]] turbo::Span<uint8_t> at(std::size_t i) {
            TLOG_CHECK(i < _ndim, "overflow");
            return turbo::Span<uint8_t>{_data + i * _vp->vector_byte_size, _vp->vector_byte_size};
        }

        [[nodiscard]] turbo::Span<uint8_t> operator[](std::size_t i) const {
            TLOG_CHECK(i < _ndim, "overflow");
            return turbo::Span<uint8_t>{_data + i * _vp->vector_byte_size, _vp->vector_byte_size};
        }

        [[nodiscard]] turbo::Span<uint8_t> operator[](std::size_t i) {
            TLOG_CHECK(i < _ndim, "overflow");
            return turbo::Span<uint8_t>{_data + i * _vp->vector_byte_size, _vp->vector_byte_size};
        }

        std::size_t add_vector(const turbo::Span<uint8_t> &vector) {
            auto i = _ndim++;
            TLOG_CHECK(_ndim < _capacity);
            TLOG_CHECK(vector.size() == _vp->vector_byte_size);
            std::memcpy(vector.data(), _data + i * _vp->vector_byte_size, vector.size());
            return i;
        }

        std::size_t add_vector(const turbo::Span<uint8_t> &vector, std::size_t nvec) {
            auto i = _ndim;
            _ndim += nvec;
            TLOG_CHECK(_ndim < _capacity);
            TLOG_CHECK(vector.size() == _vp->vector_byte_size * nvec);
            std::memcpy(vector.data(), _data + i * _vp->vector_byte_size, vector.size());
            return i;
        }

        std::size_t add_vector(uint8_t *vector, std::size_t nvec) {
            auto i = _ndim;
            _ndim += nvec;
            TLOG_CHECK(_ndim < _capacity);
            std::memcpy(vector, _data + i * _vp->vector_byte_size, nvec * _vp->vector_byte_size);
            return i;
        }

        void resize(std::size_t n) {
            TLOG_CHECK(n <= _capacity);
            _ndim = n;
        }


        void set_vector(const std::size_t i, const turbo::Span<uint8_t> &vector) {
            TLOG_CHECK(i < _ndim);
            TLOG_CHECK(vector.size() == _vp->vector_byte_size);
            std::memcpy(vector.data(), _data + i * _vp->vector_byte_size, vector.size());
        }

        void set_vector(std::size_t i, uint8_t *vector, std::size_t nvec) {
            TLOG_CHECK(i + nvec < _ndim);
            std::memcpy(vector, _data + i * _vp->vector_byte_size, nvec * _vp->vector_byte_size);
        }

        void clear_vector(std::size_t i) {
            TLOG_CHECK(i < _ndim);
            std::memset(_data + i * _vp->vector_byte_size, 0, _vp->vector_byte_size);
        }

        void clear_vector(std::size_t start, std::size_t end) {
            TLOG_CHECK(start < _ndim);
            TLOG_CHECK(end < _ndim);
            std::memset(_data + start * _vp->vector_byte_size, 0, (end - start) * _vp->vector_byte_size);
        }

        void clear_vector() {
            std::memset(_data, 0, _capacity * _vp->vector_byte_size);
        }

        void clear() {
            clear_vector();
            _ndim = 0;
        }

        void remove_vector(std::size_t n) {
            if (n < _ndim) {
                _ndim -= n;
            } else {
                _ndim = 0;
            }

        }

        [[nodiscard]] std::size_t available() const {
            return _capacity - _ndim;
        }

        [[nodiscard]] std::size_t capacity() const {
            return _capacity;
        }

    private:
        /// no lint
        TURBO_NON_COPYABLE(VectorBatch);

    private:
        VectorSpace *_vp{nullptr};
        std::size_t _ndim{0};
        std::size_t _capacity{0};
        uint8_t *_data{nullptr};
    };

}  // namespace tann

#endif  // TANN_STORE_VECTOR_BATCH_H_
