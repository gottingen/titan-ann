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


#ifndef TANN_STORE_VECTOR_SET_H_
#define TANN_STORE_VECTOR_SET_H_

#include <vector>
#include <string_view>
#include "tann/core/vector_space.h"
#include "tann/store/vector_batch.h"
#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"

namespace tann {

    class VectorSet {
    public:
        VectorSet() = default;

        ~VectorSet() = default;

        turbo::Status init(VectorSpace *vp, std::size_t batch_size);

        [[nodiscard]] const VectorSpace *get_vector_space() const;

        [[nodiscard]] const std::vector<VectorBatch> &vector_batch() const;

        [[nodiscard]] std::vector<VectorBatch> &vector_batch();
        [[nodiscard]] std::size_t get_batch_size() const;

        void set_vector(std::size_t i, turbo::Span<uint8_t> vector);

        [[nodiscard]] turbo::Span<uint8_t> get_vector(std::size_t i) const;

        void copy_vector(std::size_t i, turbo::Span<uint8_t> &des) const;

        [[nodiscard]] double get_distance(std::size_t l1, std::size_t l2) const;

        [[nodiscard]] double get_distance(turbo::Span<uint8_t> vector, std::size_t l1) const;

        void get_distance(turbo::Span<uint8_t> vector, turbo::Span<std::size_t> ls,
                          turbo::Span<double> ds) const;

        void move_vector(std::size_t from, std::size_t to);

        void move_vector(std::size_t from, std::size_t to, std::size_t nvec);

        std::size_t add_vector(const turbo::Span<uint8_t> &vector);

        std::size_t prefer_add_vector(std::size_t n = 1);

        [[nodiscard]] std::size_t size() const;

        [[nodiscard]] std::size_t capacity() const;

        [[nodiscard]] std::size_t available() const;

        void reserve(std::size_t n);

        void shrink();

        void pop_back(std::size_t n = 1);

        void resize(std::size_t n);

        turbo::Status load(std::string_view path);
        turbo::Status save(std::string_view path);

        turbo::Status load(turbo::SequentialReadFile *file);
        turbo::Status save(turbo::SequentialWriteFile *file);
    private:
        void expend();

        VectorSpace *_vs{nullptr};
        std::size_t _batch_size{0};
        std::size_t _size{0};
        std::vector<VectorBatch> _data;
    };
}  // namespace tann

#endif  // TANN_STORE_VECTOR_SET_H_
