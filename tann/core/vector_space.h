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
#ifndef TANN_CORE_VECTOR_SPACE_H_
#define TANN_CORE_VECTOR_SPACE_H_

#include "tann/core/types.h"
#include "tann/distance/distance_factory.h"
#include "tann/core/allocator.h"
#include <memory>

namespace tann {

    template<DataType dt>
    struct data_type_traits {
    };

    template<>
    struct data_type_traits<DataType::DT_UINT8> {
        using value_type = uint8_t;
    };

    template<>
    struct data_type_traits<DataType::DT_FLOAT16> {
        using value_type = float16;
    };

    template<>
    struct data_type_traits<DataType::DT_FLOAT> {
        using value_type = float16;
    };

    class VectorSpace {
    public:
        turbo::Status init(size_t dim, MetricType m, DataType dt) {
            metric_type = m;
            data_type = dt;
            auto r = DistanceFactory::create_distance_factor(metric_type, data_type);
            if (!r.ok()) {
                return r.status();
            }
            distance_factor.reset(r.value());
            dimension = dim;
            type_size = data_type_size(data_type);
            vector_byte_size = dimension * type_size;
            alignment_dim = Allocator::alignment_bytes / type_size;
            arch_name = turbo::simd::default_arch::name();
            return turbo::OkStatus();
        }

        uint8_t *alloc_vector(std::size_t n) {
            return Allocator::alloc.allocate(n * vector_byte_size);
        }

        void free_vector(uint8_t *ptr, std::size_t n) {
            Allocator::alloc.deallocate(ptr, n);
        }

    public:
        std::unique_ptr<DistanceBase> distance_factor;
        size_t alignment_dim{0};
        size_t dimension{0};
        size_t type_size{0};
        size_t vector_byte_size{0};
        DataType data_type{DataType::DT_NONE};
        std::string arch_name;
        MetricType metric_type;
    };
}  // namespace tann

#endif  // TANN_CORE_VECTOR_SPACE_H_
