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
#include "tann/flat/flat_engine.h"

namespace tann {

    turbo::Status FlatEngine::initialize(const std::any &option, MemVectorStore *store) {
        _data_store = store;
        return turbo::OkStatus();
    }
    WorkSpace* FlatEngine::make_workspace() {
        return reinterpret_cast<WorkSpace*>(new FlatWorkSpace());
    }
    turbo::Status FlatEngine::add_vector(WorkSpace*ws, location_t lid) {
        return turbo::OkStatus();
    }

    turbo::Status FlatEngine::remove_vector(location_t lid) {
        return turbo::OkStatus();
    }

    turbo::Status FlatEngine::search_vector(WorkSpace *ws) {
        //// check ok, start to do search work
        auto query = to_span<uint8_t>(ws->query_view);
        auto data_size = _data_store->current_index();
        auto k = ws->search_context->k;
        auto is_allow_func = ws->search_context->is_allowed;
        auto first_travel_size = std::min(data_size, k);
        label_type label;
        auto &topk_results = ws->best_l_nodes;

        for (location_t i = 0; i < first_travel_size; i++) {
            if(_data_store->is_deleted(i)) {
                continue;
            }
            label = _data_store->get_label(i).value();
            if(is_allow_func && !(*is_allow_func)(label)) {
                continue;
            }
            auto d = _data_store->get_distance(query,i);
            topk_results.insert({label, i, d});
        }

        distance_type lastdist = topk_results.empty() ? std::numeric_limits<distance_type>::max() : topk_results.top().distance;
        for (size_t i = k; i < data_size; i++) {
            if(_data_store->is_deleted(i)) {
                continue;
            }
            label = _data_store->get_label(i).value();
            if(is_allow_func && !(*is_allow_func)(label)) {
                continue;
            }
            auto d = _data_store->get_distance(query,i);
            if(d < lastdist) {
                topk_results.insert({label, static_cast<location_t>(i), d});
                if(!topk_results.empty()) {
                    lastdist = topk_results.top().distance;
                }
            }
        }
        return turbo::OkStatus();
    }

    turbo::Status FlatEngine::save(turbo::SequentialWriteFile *file) {
        return turbo::OkStatus();
    }

    turbo::Status FlatEngine::load(turbo::SequentialReadFile *file) {
        return turbo::OkStatus();
    }

}  // namespace tann
