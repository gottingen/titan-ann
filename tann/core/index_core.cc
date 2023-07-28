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

    [[nodiscard]] turbo::Status IndexCore::initialize(const IndexOption &option, const std::any &core_option) {
        TLOG_INFO("initialize index core");
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
        TLOG_INFO(" data_store initialize done");
        auto ptr = create_index_core(_base_option.engine_type, core_option);
        TLOG_INFO(" engine create done {}", turbo::Ptr(ptr));
        _engine.reset(ptr);
        if(!_engine) {
            return turbo::InvalidArgumentError("can not create index by input param");
        }
        TLOG_INFO(" engine create done");
        r = _engine->initialize(option, core_option, &_data_store);
        TLOG_INFO(" engine initialize done");
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO(" engine initialize done");
        for(size_t i = 0; i < option.number_thread; i++) {
            auto ws = _engine->make_workspace();
            if(!ws) {
                return turbo::ResourceExhaustedError("no memory");
            }
            _ws_pool.push(ws);
        }
        TLOG_INFO("work space pool size:{}", _ws_pool.size());
        is_initial = true;
        return turbo::OkStatus();
    }

    turbo::ResultStatus<InsertResult>
    IndexCore::add_vector(const WriteOption &option, turbo::Span<uint8_t> data_point, const label_type &label) {
        assert(is_initial);
        WorkSpaceGuard guard(_ws_pool);
        auto ws = guard.work_space();
        ws->set_up(option, data_point);

        if(!option.is_normalized && _vector_space.distance_factor->preprocessing_required()) {
            _vector_space.distance_factor->preprocess_base_points(ws->query_view, _vector_space.dimension);
        }
        // lock label for below operation
        LabelLockGuard label_guard(&_data_store, label);
        // guard for vector data write
        UpdateLockGuard write_guard(&_data_store);
        location_t  lid = constants::kUnknownLocation;
        // try get vacant
        bool is_vacant = false;
        if(option.replace_deleted) {
            lid = _data_store.get_vacant(label).value_or(constants::kUnknownLocation);
            is_vacant = true;
        }
        if(lid == constants::kUnknownLocation) {
            is_vacant = false;
            auto rv = _data_store.prefer_add_vector(label);
            if(!rv.ok()) {
                return rv.status();
            }
            lid = rv.value();
        }
        // set data to data store
        _data_store.set_vector(lid, ws->query_view);
        ws->is_update = is_vacant;
        auto r= _engine->add_vector(ws, lid);
        if(!r.ok()) {
            return r;
        }

        return InsertResult{ws->timer.elapsed_nano()};
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

    turbo::Status IndexCore::search_vector(SearchContext *sc, SearchResult &result) {
        // guard for vector data update
        WorkSpaceGuard guard(_ws_pool);
        auto ws = guard.work_space();
        ws->set_up(sc);
        if(!sc->is_normalized && _vector_space.distance_factor->preprocessing_required()) {
            _vector_space.distance_factor->preprocess_base_points(ws->query_view, _vector_space.dimension);
        }
        _engine->setup_workspace(ws);
        SearchResult results;
        UpdateSharedLockGuard write_guard(&_data_store);
       auto r = _engine->search_vector(ws);
       if(!r.ok()) {
           return r;
       }

        auto rsize = ws->best_l_nodes.size();
        for (int i = 0; i < rsize; ++i) {
            results.results.emplace_back(ws->best_l_nodes[i].distance, ws->best_l_nodes[i].label);
        }
        if(sc->get_raw_vector) {
            results.vectors.resize(rsize);
            for(size_t i = 0; i < rsize; ++i) {
                std::vector<uint8_t> tmp;
                tmp.resize(_vector_space.vector_byte_size);
                auto sp = to_span<uint8_t>(tmp);
                _data_store.copy_vector(ws->best_l_nodes[i].lid, sp);
                results.vectors[i] = std::move(tmp);
            }
        }
        results.cost_ns = ws->timer.elapsed_nano();
        return turbo::OkStatus();
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

    std::size_t IndexCore::size() const {
        return _data_store.size();
    }

    std::size_t IndexCore::dimension() const {
        return _vector_space.dimension;
    }

    bool IndexCore::support_dynamic() const {
        return _engine->support_dynamic();
    }

    bool IndexCore::need_model() const {
        return _engine->need_model();
    }

    EngineType IndexCore::engine_type() const {
        return _base_option.engine_type;
    }
    size_t IndexCore::remove_size() const {
        return _data_store.deleted_size();
    }
}  // namespace tann
