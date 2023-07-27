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

#ifndef TANN_CORE_ENGINE_H_
#define TANN_CORE_ENGINE_H_

#include <any>
#include "tann/core/types.h"
#include "turbo/base/status.h"
#include "tann/core/index_option.h"
#include "tann/core/query_context.h"
#include "tann/store/mem_vector_store.h"
#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"

namespace tann {
    class Engine {
    public:
        virtual ~Engine() = default;
        virtual turbo::Status initialize(const std::any& option, VectorSpace *vs, MemVectorStore*store) = 0;
        virtual turbo::Status add_vector(const WriteOption &options,location_t lid, bool ever_added) = 0;

        virtual turbo::Status remove_vector(location_t lid) = 0;

        virtual turbo::Status search_vector(QueryContext *qctx) = 0;

        virtual turbo::Status save(turbo::SequentialWriteFile *file) = 0;
        virtual turbo::Status load(turbo::SequentialReadFile *file) = 0;
    };

    inline Engine* create_index_core(EngineType type) {
        switch (type) {
            case EngineType::ENGINE_FLAT:
                return nullptr;
            case EngineType::ENGINE_HNSW:
                return nullptr;
        }
        return nullptr;
    }
}
#endif  // TANN_CORE_ENGINE_H_
