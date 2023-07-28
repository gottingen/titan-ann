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
#include "tann/core/worker_space.h"
#include "tann/store/mem_vector_store.h"
#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"

#include "turbo/base/turbo_error.h"

namespace tann {
    class Engine {
    public:
        virtual ~Engine() = default;

        virtual turbo::Status initialize(const std::any &option, MemVectorStore *store) = 0;

        virtual turbo::Status add_vector(WorkSpace*ws, location_t lid) = 0;

        virtual turbo::Status remove_vector(location_t lid) = 0;

        virtual WorkSpace* make_workspace() = 0;

        virtual void setup_workspace(WorkSpace*ws) = 0;

        virtual turbo::Status search_vector(WorkSpace *ws) = 0;

        virtual turbo::Status save(turbo::SequentialWriteFile *file) = 0;

        virtual turbo::Status load(turbo::SequentialReadFile *file) = 0;

        virtual bool support_dynamic() const = 0;

        virtual bool need_model() const = 0;
    };

    Engine *create_index_core(EngineType type, const std::any &option);

    extern int EngineRegisterInternal(EngineType type, std::function<Engine *(const std::any &option)> creator);
}  // namespace tann

template<tann::EngineType engine>
class EngineRegisterHelper {
};

#define ENGINE_REGISTER(engine_type, creator)                   \
    const int TURBO_MAYBE_UNUSED TURBO_CONCAT(engine_type_dummy_, __LINE__) =              \
        ::tann::EngineRegisterInternal((engine_type), (creator)); \
    template <> class EngineRegisterHelper<(tann::EngineType)(engine_type)> {};
#endif  // TANN_CORE_ENGINE_H_
