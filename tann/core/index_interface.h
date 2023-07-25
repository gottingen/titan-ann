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
#ifndef TANN_CORE_INDEX_INTERFACE_H_
#define TANN_CORE_INDEX_INTERFACE_H_

#include "tann/core/query_context.h"
#include "tann/core/vector_space.h"
#include "tann/core/types.h"
#include "tann/core/index_option.h"
#include "tann/core/serialize_option.h"

namespace tann {

    class IndexInterface {
    public:
        virtual ~IndexInterface() = default;

        //////////////////////////////////////////
        // Let the engine use lazy initialization and aggregate the initialization
        // parameters into the index configuration, which will be more convenient when
        // using the factory class method
        [[nodiscard]] virtual turbo::Status initialize(const IndexOption *option) = 0;

        [[nodiscard]] virtual  turbo::Status add_vector(const WriteOption &option, turbo::Span<uint8_t> vec, const label_type &label) = 0;

        [[nodiscard]] virtual  turbo::Status remove_vector(const label_type &label) = 0;

        [[nodiscard]] virtual  turbo::Status search_vector(QueryContext *qctx) = 0;

        [[nodiscard]] virtual turbo::Status save_index(const std::string &path, const SerializeOption &option) = 0;

        [[nodiscard]] virtual turbo::Status load_index(const std::string &path, const SerializeOption &option) = 0;

        [[nodiscard]] virtual std::size_t size() const = 0;

        [[nodiscard]] virtual std::size_t dimension() const = 0;

        [[nodiscard]] virtual bool dynamic() const = 0;

        [[nodiscard]] virtual bool need_model() const = 0;

    };
}  // namespace tann

#endif  // TANN_CORE_INDEX_INTERFACE_H_
