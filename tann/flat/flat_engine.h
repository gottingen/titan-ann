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
#ifndef TANN_FLAT_FLAT_ENGINE_H_
#define TANN_FLAT_FLAT_ENGINE_H_

#include "tann/core/engine.h"

namespace tann {

    class FlatWorkSpace : public WorkSpace {
    public:
        FlatWorkSpace() = default;
    private:
        void clear_sub() override {

        }
    };
    class FlatEngine : public Engine {

    public:
        ~FlatEngine() override = default;

        turbo::Status initialize(const std::any &option, MemVectorStore *store) override;

        turbo::Status add_vector(WorkSpace*ws, location_t lid) override;

        WorkSpace* make_workspace() override;

        void setup_workspace(WorkSpace*ws) override {

        }

        turbo::Status remove_vector(location_t lid) override;

        turbo::Status search_vector(WorkSpace *ws) override;

        turbo::Status save(turbo::SequentialWriteFile *file) override;

        turbo::Status load(turbo::SequentialReadFile *file) override;

        bool support_dynamic() const override {
            return true;
        }

        bool need_model() const override {
            return false;
        }
    private:
        HnswIndexOption _option;
        MemVectorStore *_data_store;
    };
}

#endif  // TANN_FLAT_FLAT_ENGINE_H_
