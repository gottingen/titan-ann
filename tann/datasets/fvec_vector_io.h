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

#ifndef TANN_DATASETS_FVEC_VECTOR_IO_H_
#define TANN_DATASETS_FVEC_VECTOR_IO_H_

#include "tann/core/vector_set_io.h"

namespace tann {

    class FvecVectorSetReader : public VectorSetReader {
    public:
        FvecVectorSetReader() = default;

        ~FvecVectorSetReader() override = default;

        turbo::Status load(VectorSet &dst) override;

        turbo::Status read_vector(turbo::Span<uint8_t> *vector) override;

        turbo::ResultStatus<std::size_t> read_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size) override;
    private:
        turbo::Status init() override;
    };

    class FvecVectorSetWriter : public VectorSetWriter {
    public:
        FvecVectorSetWriter() = default;

        ~FvecVectorSetWriter() override = default;

        turbo::Status save(VectorSet &dst) override;

        turbo::Status write_vector(turbo::Span<uint8_t> *vector) override;

        turbo::Status write_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size) override;

    private:
        turbo::Status init() override;
    };
}  // namespace tann
#endif  // TANN_DATASETS_FVEC_VECTOR_IO_H_
