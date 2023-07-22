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

#ifndef TANN_CORE_VECTOR_SET_IO_H_
#define TANN_CORE_VECTOR_SET_IO_H_

#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"
#include "tann/core/serialize_option.h"
#include "tann/store/vector_set.h"

namespace tann {

    class VectorSetReader {
    public:

        virtual ~VectorSetReader() = default;

        virtual turbo::Status init(turbo::SequentialReadFile *file, SerializeOption option) = 0;

        virtual turbo::Status load(VectorSet &dst) = 0;

        virtual turbo::Status read_vector(turbo::Span<uint8_t> *vector)  = 0;

        virtual turbo::Status read_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size)  = 0;

        virtual std::size_t has_read() const = 0;
    };

    class VectorSetWriter {
    public:

        virtual ~VectorSetWriter() = default;

        virtual turbo::Status init(turbo::SequentialWriteFile *file, SerializeOption option) = 0;

        virtual turbo::Status save(VectorSet &dst) = 0;

        virtual turbo::Status write_vector(turbo::Span<uint8_t> *vector)  = 0;

        virtual turbo::Status write_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size)  = 0;

        virtual std::size_t has_write() const = 0;
    };

}  // namespace tann

#endif  // TANN_CORE_VECTOR_SET_IO_H_
