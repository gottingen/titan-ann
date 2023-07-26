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
#ifndef TANN_QUANTIZER_QUANTIZER_INTERFACE_H_
#define TANN_QUANTIZER_QUANTIZER_INTERFACE_H_

#include <cstdint>
#include "tann/core/types.h"
#include "turbo/base/status.h"
#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"

namespace tann {
    class QuantizerInterface {
    public:
        virtual void QuantizeVector(const void* vec, std::uint8_t* vecout, bool ADC = true) const = 0;

        virtual size_t QuantizeSize() const = 0;

        virtual void ReconstructVector(const std::uint8_t* qvec, void* vecout) const = 0;

        virtual size_t ReconstructSize() const = 0;

        virtual size_t ReconstructDim() const = 0;

        virtual std::uint64_t BufferSize() const = 0;

        virtual turbo::Status save(turbo::SequentialWriteFile *p_out) const = 0;

        virtual turbo::Status load(turbo::SequentialReadFile *p_in) = 0;

        virtual bool GetEnableADC() const = 0;

        virtual void SetEnableADC(bool enableADC) = 0;

        virtual int GetBase() const = 0;

        virtual float* GetL2DistanceTables() = 0;

        template<typename T>
        T* GetCodebooks();
    };
}
#endif  // TANN_QUANTIZER_QUANTIZER_INTERFACE_H_
