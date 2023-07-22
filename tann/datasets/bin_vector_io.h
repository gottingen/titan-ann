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

#ifndef TANN_DATASETS_BIN_VECTOR_IO_H_
#define TANN_DATASETS_BIN_VECTOR_IO_H_

#include "tann/core/vector_set_io.h"

namespace tann {

    class BinaryVectorSetReader : public VectorSetReader {
    public:
        BinaryVectorSetReader() = default;

        ~BinaryVectorSetReader() override = default;

        turbo::Status init(turbo::SequentialReadFile *file, SerializeOption option) override;

        turbo::Status load(VectorSet &dst) override;

        turbo::Status read_vector(turbo::Span<uint8_t> *vector) override;

        turbo::Status read_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size) override;

        std::size_t has_read() const override {
            return _has_read;
        }

    private:
        turbo::SequentialReadFile *_file{nullptr};
        SerializeOption _option;
        size_t _nvecs{0};
        size_t _ndims{0};
        size_t _element_size{0};
        size_t _vector_bytes{0};
        size_t _has_read{0};
    };

    class BinaryVectorSetWriter : public VectorSetWriter {
    public:
        BinaryVectorSetWriter() = default;

        ~BinaryVectorSetWriter() override = default;

        turbo::Status init(turbo::SequentialWriteFile *file, SerializeOption option) override;

        turbo::Status save(VectorSet &dst) override;

        turbo::Status write_vector(turbo::Span<uint8_t> *vector) override;

        turbo::Status write_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size) override;

        std::size_t has_write() const override {
            return _has_write;
        }

    private:
        turbo::SequentialWriteFile *_file{nullptr};
        SerializeOption _option;
        size_t _nvecs{0};
        size_t _ndims{0};
        size_t _element_size{0};
        size_t _vector_bytes{0};
        size_t _has_write{0};
    };
}  // namespace tann
#endif  // TANN_DATASETS_BIN_VECTOR_IO_H_
