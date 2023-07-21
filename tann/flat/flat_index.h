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
#ifndef TANN_FLAT_FLAT_INDEX_H_
#define TANN_FLAT_FLAT_INDEX_H_

#include <memory>
#include <shared_mutex>
#include <vector>
#include <queue>
#include "tann/core/types.h"
#include "turbo/container/flat_hash_map.h"
#include "tann/distance/distance_factory.h"
#include "tann/store/vector_set.h"
#include "turbo/container/flat_hash_map.h"
#include "tann/common/natural_number_set.h"
#include "tann/core/query_context.h"
#include "tann/core/index_option.h"
#include "tann/core/index_interface.h"

namespace tann {

    class FlatIndex : public IndexInterface {
    public:
        FlatIndex() = default;

        ~FlatIndex() override = default;

        [[nodiscard]] turbo::Status init(IndexOption option) override;

        void add_vector(turbo::Span<uint8_t> vec, const label_type &label) override;

        void remove_vector(const label_type &label) override;

        [[nodiscard]] turbo::Status search_vector(QueryContext *qctx) override;

        [[nodiscard]] turbo::Status save_index(const std::string &path, const IOOption &option) override;

        [[nodiscard]] turbo::Status load_index(const std::string &path, const IOOption &option) override;

        [[nodiscard]] std::size_t size() const override {
            return _data.size();
        }

        [[nodiscard]] std::size_t dimension() const  override {
            return _vs.dimension;
        }

        [[nodiscard]] bool dynamic() const override {
            return true;
        }

        [[nodiscard]] bool need_model() const override {
            return false;
        }

    private:
        VectorSpace _vs;
        std::shared_mutex _mutex;
        VectorSet _data TURBO_GUARDED_BY(_mutex);
        std::vector<label_type> _index_to_label;
        turbo::flat_hash_map<label_type, std::size_t> _label_map TURBO_GUARDED_BY(_mutex);
    };
}  // namespace tann

#endif  // TANN_FLAT_FLAT_INDEX_H_
