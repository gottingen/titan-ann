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
#include "tann/core/index_core.h"
#include "tann/core/vector_store_option.h"

namespace tann {

    [[nodiscard]] turbo::Status IndexCore::initialize(const IndexOption &option, std::any &core_option) {
        _base_option = option;
        auto r = _vector_space.init(_base_option.dimension, _base_option.metric, _base_option.data_type);
        if (!r.ok()) {
            return r;
        }
        VectorStoreOption store_option;
        store_option.batch_size = _base_option.batch_size;
        store_option.max_elements = _base_option.max_elements;
        store_option.enable_replace_vacant = _base_option.enable_replace_vacant;
        r = _data_store.initialize(&_vector_space, store_option);
        if (!r.ok()) {
            return r;
        }
        _engine.reset(create_index_core(_base_option.engine_type));
        r = _engine->initialize(core_option, &_vector_space, &_data_store);
        if (!r.ok()) {
            return r;
        }
        return turbo::OkStatus();
    }

    [[nodiscard]] turbo::Status
    IndexCore::add_vector(const WriteOption &option, turbo::Span<uint8_t> data_point, const label_type &label) {
        turbo::Span<uint8_t> vector_data = to_span<uint8_t>(data_point);
        AlignedQuery<uint8_t> aligned_vector;
        if(!option.is_normalized && _vector_space.distance_factor->preprocessing_required()) {
            vector_data = make_aligned_query(vector_data, aligned_vector);
            _vector_space.distance_factor->preprocess_base_points(vector_data, _vector_space.dimension);
        }
        // lock label for below operation
        LabelLockGuard label_guard(&_data_store, label);
        // guard for vector data write
        UpdateLockGuard write_guard(&_data_store);
        location_t  lid;
        // try get vacant
        bool is_vacant = false;
        if(option.replace_deleted) {
            lid = _data_store.get_vacant(label).value_or(constants::kUnknownLocation);
            is_vacant = true;
        }
        if(lid == constants::kUnknownLocation) {
            auto rv = _data_store.prefer_add_vector(label);
            if(!rv.ok()) {
                return rv.status();
            }
            lid = rv.value();
        }
        // set data to data store
        _data_store.set_vector(lid, vector_data);

        auto r= _engine->add_vector(option, lid, is_vacant);
        if(!r.ok()) {
            return r;
        }
        return turbo::OkStatus();
    }

    turbo::Status IndexCore::remove_vector(const label_type &label) {
        LabelLockGuard label_guard(&_data_store, label);
        auto rs = _data_store.remove_vector(label);
        if(!rs.ok()) {
            return rs.status();
        }
        auto lid = rs.value();
        auto s = _engine->remove_vector(lid);
        return s;
    }

    turbo::Status IndexCore::search_vector(QueryContext *qctx) {
        // guard for vector data update
        UpdateSharedLockGuard write_guard(&_data_store);
       return _engine->search_vector(qctx);
    }

    turbo::Status IndexCore::save_index(const std::string &path, const SerializeOption &option) {
        turbo::SequentialWriteFile file;
        auto r = file.open(path);
        if(!r.ok()) {
            return r;
        }

        // save mem engine
        r = _engine->save(&file);
        if(!r.ok()) {
            return r;
        }
        r = _data_store.save(&file);
        if(!r.ok()) {
            return r;
        }
        return turbo::OkStatus();
    }

    turbo::Status IndexCore::load_index(const std::string &path, const SerializeOption &option)  {
        turbo::SequentialReadFile file;
        auto r = file.open(path);
        if(!r.ok()) {
            return r;
        }

        // save mem engine
        r = _engine->load(&file);
        if(!r.ok()) {
            return r;
        }
        r = _data_store.load(&file);
        if(!r.ok()) {
            return r;
        }
        return turbo::OkStatus();
    }
}  // namespace tann
