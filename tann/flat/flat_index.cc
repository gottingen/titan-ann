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

    turbo::Status FlatIndex::initialize(const IndexOption *option) {
        _option = *option;
        auto r= _vs.init(option->dimension, option->metric, option->data_type);
        if(!r.ok()) {
            return r;
        }
        r = _data.init(&_vs, option->batch_size);
        _index_to_label.resize(option->max_elements);
        return r;
    }

    turbo::Status FlatIndex::add_vector(const WriteOption &option, turbo::Span<uint8_t> query, const label_type &label) {
        if(option.replace_deleted) {
            if(_data.size() >= _option.max_elements) {
                return turbo::ResourceExhaustedError("full");
            }
        } else {
            if(_data.current_index()>= _option.max_elements) {
                return turbo::ResourceExhaustedError("full");
            }
        }
        TURBO_UNUSED(option);
        turbo::Span<uint8_t> v = query;
        if(_vs.distance_factor->preprocessing_required()) {
            AlignedQuery<uint8_t> avec;
            make_aligned_query(query, avec);
            v = to_span<uint8_t>(avec);
            _vs.distance_factor->preprocess_base_points(v, 1);
        }
        std::unique_lock l(_mutex);
        auto it = _label_map.find(label);
        location_t idx;
        if (it != _label_map.end()) {
            // update
            idx = it->second;
            _data.set_vector(idx, v);
            return turbo::OkStatus();
        }
        bool has_vacant = false;
        if(option.replace_deleted) {
            has_vacant = _data.get_vacant(idx);
        }
        if(!has_vacant) {
            idx = _data.prefer_add_vector();
        }
        _label_map[label] = idx;
        _index_to_label[idx] = label;
        _data.set_vector(idx, v);
        // idx is the last one
        return turbo::OkStatus();
    }

    turbo::Status FlatIndex::remove_vector(const label_type &label) {
        std::unique_lock l(_mutex);
        auto it = _label_map.find(label);
        if (it == _label_map.end()) {
            return turbo::OkStatus();
        }
        auto idx = it->second;
        _data.mark_deleted(idx);
        _label_map.erase(label);
        return turbo::OkStatus();
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
            if(_data.is_deleted(i)) {
                continue;
            }
            label = _index_to_label[i];
            if(qctx->is_allowed && !(*qctx->is_allowed)(label)) {
                continue;
            }
            auto d = _data.get_distance(query,i);
            topResults.emplace(d, label);
        }

        distance_type lastdist = topResults.empty() ? std::numeric_limits<distance_type>::max() : topResults.top().first;
        for (size_t i = qctx->k; i < data_size; i++) {
            if(_data.is_deleted(i)) {
                continue;
            }
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
            topResults.pop();
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

    turbo::Status FlatIndex::save_index(const std::string &path, const SerializeOption &option)  {
        return turbo::OkStatus();
    }

    turbo::Status FlatIndex::load_index(const std::string &path, const SerializeOption &option) {
        return turbo::OkStatus();
    }
}  // namespace tann
