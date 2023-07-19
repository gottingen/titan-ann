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

#include "tann/flat/flat_index.h"

namespace tann {

    turbo::Status FlatIndex::init(IndexOption option) {
        auto r= _vs.init(option.dimension, option.metric, option.data_type);
        if(!r.ok()) {
            return r;
        }
        r = _data.init(&_vs, 256);
        return r;
    }

    void FlatIndex::add_vector(turbo::Span<uint8_t> vec, const label_type &label) {
        std::unique_lock l(_mutex);
        auto it = _label_map.find(label);
        std::size_t idx;
        if (it != _label_map.end()) {
            // modify
            idx = it->second;
            _data.set_vector(idx, vec);
            return;
        }
        idx = _data.prefer_add_vector();
        if (_index_to_label.size() <= idx) {
            _index_to_label.resize(idx + 1);
        }
        _label_map[label] = idx;
        _index_to_label[idx] = label;
        _data.set_vector(idx, vec);
        // idx is the last one
    }

    void FlatIndex::remove_vector(const label_type &label) {
        std::unique_lock l(_mutex);
        auto it = _label_map.find(label);
        if (it == _label_map.end()) {
            return;
        }
        auto idx = it->second;
        _label_map.erase(label);
        auto last_idx = _index_to_label.size() - 1;
        auto &last_label = _index_to_label[last_idx];
        _data.move_vector(last_idx, idx);
        _label_map[last_label] = idx;
        _data.pop_back();
        _index_to_label.resize(last_idx);
    }

    turbo::Status FlatIndex::search_vector(QueryContext *qctx) {
        if(!qctx) {
            return turbo::InvalidArgumentError("qctx is nullptr");
        }

        if(qctx->raw_query.size() != _vs.vector_byte_size) {
            return turbo::InvalidArgumentError("qctx query not set currently");
        }

        if(qctx->k == 0) {
            return turbo::InvalidArgumentError("qctx k not set to 0, no need to search anything");
        }

        //// check ok, start to do search work

        std::shared_lock l(_mutex);
        auto query = to_span<uint8_t>(qctx->raw_query);
        auto data_size = _data.size();
        auto first_travel_size = std::min(data_size, qctx->k);
        label_type label;
        std::priority_queue<std::pair<distance_type , label_type>> topResults;
        for (size_t i = 0; i < first_travel_size; i++) {
            label = _index_to_label[i];
            if(qctx->is_allowed && !(*qctx->is_allowed)(label)) {
                continue;
            }
            auto d = _data.get_distance(query,i);
            topResults.emplace(d, label);
        }

        distance_type lastdist = topResults.empty() ? std::numeric_limits<distance_type>::max() : topResults.top().first;
        for (size_t i = qctx->k; i < data_size; i++) {
            label = _index_to_label[i];
            if(qctx->is_allowed && !(*qctx->is_allowed)(label)) {
                continue;
            }
            auto d = _data.get_distance(query,i);
            if(d < lastdist) {
                topResults.emplace(d, label);
                if(topResults.size() > qctx->k) {
                    topResults.pop();
                }
                if(!topResults.empty()) {
                    lastdist = topResults.top().first;
                }
            }
        }
        qctx->results.reserve(topResults.size());
        while (!topResults.empty()) {
            auto it = topResults.top();
            qctx->results.emplace_back(it.first, it.second);
        }
        if(qctx->get_raw_vector) {
            qctx->vectors.resize(qctx->results.size());
            for(size_t i = 0; i < qctx->results.size(); ++i) {
                std::vector<uint8_t> tmp;
                tmp.resize(_vs.vector_byte_size);
                auto sp = to_span<uint8_t>(tmp);
                _data.copy_vector(_label_map[qctx->results[i].second], sp);
                qctx->vectors[i] = std::move(tmp);
            }
        }

        return turbo::OkStatus();
    }

}  // namespace tann
