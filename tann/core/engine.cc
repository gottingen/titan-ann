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
#include "tann/core/engine.h"
#include "tann/hnsw/hnsw_engine.h"
#include "tann/flat/flat_engine.h"

namespace tann {

    static std::function<Engine*(const std::any& option)> factory[256];

    int EngineRegisterInternal(EngineType type, std::function<Engine*(const std::any& option)> func) {
        auto idx = static_cast<int>(type);
        factory[idx] = std::move(func);
        return 0;
    }

    Engine *create_index_core(EngineType type, const std::any &option) {
        /*
        auto idx = static_cast<int>(type);
        if(factory[idx]) {
            return factory[idx](option);
        }
        return nullptr;
         */
         if(type == EngineType::ENGINE_HNSW) {
             return new HnswEngine();
         } else if(type == EngineType::ENGINE_FLAT) {
             return new FlatEngine();
         }
         return nullptr;
    }
}  // namespace tann
