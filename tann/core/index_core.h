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
#ifndef TANN_CORE_INDEX_CORE_H_
#define TANN_CORE_INDEX_CORE_H_

#include <any>
#include "tann/core/search_context.h"
#include "tann/core/vector_space.h"
#include "tann/core/types.h"
#include "tann/core/engine.h"
#include "tann/core/index_option.h"
#include "tann/core/serialize_option.h"
#include "tann/store/mem_vector_store.h"

namespace tann {

    class IndexCore {
    public:
        IndexCore() = default;

        virtual ~IndexCore() = default;

        //////////////////////////////////////////
        // Let the engine use lazy initialization and aggregate the initialization
        // parameters into the index configuration, which will be more convenient when
        // using the factory class method
        [[nodiscard]] turbo::Status initialize(const IndexOption &option, const std::any &core_option);

        [[nodiscard]] turbo::ResultStatus<InsertResult>
        add_vector(const WriteOption &option, turbo::Span<uint8_t> data_point, const label_type &label);

        [[nodiscard]] turbo::Status remove_vector(const label_type &label);

        [[nodiscard]] virtual turbo::Status search_vector(SearchContext *qctx, SearchResult &result);

        [[nodiscard]] virtual turbo::Status save_index(const std::string &path, const SerializeOption &option);

        [[nodiscard]] virtual turbo::Status load_index(const std::string &path, const SerializeOption &option);

        [[nodiscard]] virtual std::size_t size() const;

        [[nodiscard]] virtual std::size_t dimension() const;

        [[nodiscard]] virtual bool support_dynamic() const;

        [[nodiscard]] virtual bool need_model() const;

        [[nodiscard]] virtual EngineType engine_type() const;

        [[nodiscard]] virtual size_t remove_size() const;

    private:
        VectorSpace _vector_space;
        IndexOption _base_option;
        MemVectorStore _data_store;
        std::unique_ptr<Engine> _engine;
        ConcurrentQueue<WorkSpace *> _ws_pool;
        bool is_initial{false};

    };
}  // namespace tann

#endif  // TANN_CORE_INDEX_INTERFACE_H_
