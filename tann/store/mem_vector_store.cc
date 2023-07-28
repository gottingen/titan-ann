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

#include "tann/store/mem_vector_store.h"
#include "turbo/log/logging.h"
#include "tann/common/ann_exception.h"
#include "tann/datasets/bin_vector_io.h"
#include "tann/io/utility.h"
#include "turbo/times/stop_watcher.h"

namespace tann {

    turbo::Status MemVectorStore::initialize(VectorSpace *vp, VectorStoreOption op) {
        _vs = vp;
        _option = op;
        _lid_to_label.resize(_option.max_elements, constants::kUnknownLabel);
        reserve_impl(_option.max_elements);
        _is_available = true;
        return turbo::OkStatus();
    }

    void MemVectorStore::reset_max_elements(uint32_t max_size) {
        TLOG_CHECK(_is_available, "should init be using");
        std::unique_lock<std::shared_mutex> lm(_meta_lock);
        TLOG_CHECK(_option.max_elements < max_size);
        _lid_to_label.resize(max_size, constants::kUnknownLabel);
        _option.max_elements = max_size;
    }

    const VectorSpace *MemVectorStore::get_vector_space() const {
        TLOG_CHECK(_is_available, "should init be using");
        return _vs;
    }

    const std::vector<VectorBatch> &MemVectorStore::vector_batch() const {
        TLOG_CHECK(_is_available, "should init be using");
        return _data;
    }

    std::vector<VectorBatch> &MemVectorStore::vector_batch() {
        TLOG_CHECK(_is_available, "should init be using");
        return _data;
    }

    location_t MemVectorStore::get_batch_size() const {
        return _option.batch_size;
    }

    void MemVectorStore::set_vector(const location_t i, turbo::Span<uint8_t> vector) {
        //std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(i < _current_idx.load(), "vector set size {}, but set the vector {}, overflow!", _current_idx.load(), i);
        auto bi = i / _option.batch_size;
        auto si = i % _option.batch_size;
        _data[bi].set_vector(si, vector);
    }


    turbo::Span<uint8_t> MemVectorStore::get_vector(location_t i) const {
        //std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(i < _current_idx, "vector set size {}, but get the vector {}, overflow!", _current_idx.load(), i);
        return get_vector_internal(i);
    }

    turbo::Span<uint8_t> MemVectorStore::get_vector_internal(location_t i) const {
        auto bi = i / _option.batch_size;
        auto si = i % _option.batch_size;
        return _data[bi].at(si);
    }

    void MemVectorStore::copy_vector(location_t i, turbo::Span<uint8_t> &des) const {
        //std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        auto ref = get_vector_internal(i);
        TLOG_CHECK(des.size() >= ref.size());
        std::memcpy(des.data(), ref.data(), ref.size());
    }

    double MemVectorStore::get_distance(location_t l1, location_t l2) const {
        //std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(l1 < _current_idx, "overflow");
        TLOG_CHECK(l2 < _current_idx, "overflow");
        //TLOG_INFO("compare {} {}", l1, l2);
        auto v1 = get_vector_internal(l1);
        auto v2 = get_vector_internal(l2);
        return _vs->distance_factor->compare(v1, v2);
    }

    double MemVectorStore::get_distance(turbo::Span<uint8_t> query, location_t l1) const {
        //std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(l1 < _current_idx, "should init be using");
        auto v1 = get_vector_internal(l1);
        return _vs->distance_factor->compare(v1, query);
    }

    void MemVectorStore::get_distance(turbo::Span<uint8_t> query, turbo::Span<std::size_t> ls,
                                      turbo::Span<double> ds) const {
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(ls.size() <= ds.size());
        for (size_t i = 0; i < ls.size(); ++i) {
            ds[i] = get_distance(query, ls[i]);
        }
    }

    void MemVectorStore::move_vector(location_t from, location_t to) {
        //std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(from < _current_idx, "overflow");
        TLOG_CHECK(to < _current_idx, "overflow");
        auto vf = get_vector_internal(from);
        auto vt = get_vector_internal(to);
        std::memcpy(vt.data(), vf.data(), vf.size());
    }


    turbo::ResultStatus<location_t> MemVectorStore::add_vector(label_type label, const turbo::Span<uint8_t> &query) {
        TLOG_CHECK(_is_available, "should init be using");
        auto r = get_vacant(label);
        if (!r.ok()) {
            r = prefer_add_vector(label);
        }
        if (!r.ok()) {
            return r.status();
        }
        auto lid = r.value();
        set_vector(lid, query);
        return lid;
    }

    void MemVectorStore::enable_vacant() {
        TLOG_CHECK(_is_available, "should init be using");
        _option.enable_replace_vacant = true;
    }

    void MemVectorStore::disable_vacant() {
        TLOG_CHECK(_is_available, "should init be using");
        _option.enable_replace_vacant = false;
    }

    turbo::ResultStatus<location_t> MemVectorStore::prefer_add_vector(label_type label) {
        std::unique_lock<std::shared_mutex> lock(_label_map_lock);
        //std::unique_lock<std::shared_mutex> ld(_data_lock);
        std::unique_lock<std::shared_mutex> lm(_meta_lock);
        TLOG_CHECK(_is_available, "should init be using");
        if (_current_idx >= _option.max_elements) {
            return turbo::ResourceExhaustedError("no space");
        }

        auto itr = _label_map.find(label);
        if (itr != _label_map.end()) {
            return turbo::AlreadyExistsError("");
        }
        auto lid = _current_idx.load();
        _label_map[label] = lid;
        _lid_to_label[lid] = label;
        auto new_size = _current_idx + 1;
        resize_impl(new_size);
        _current_idx = new_size;
        return lid;
    }

    turbo::ResultStatus<location_t> MemVectorStore::remove_vector(label_type label) {
        std::unique_lock<std::shared_mutex> lock(_meta_lock);
        std::unique_lock<std::shared_mutex> label_lock(_label_map_lock);
        TLOG_CHECK(_is_available, "should init be using");
        auto itr = _label_map.find(label);
        if (itr == _label_map.end()) {
            return turbo::NotFoundError("delete label not found");
        }
        auto lid = itr->second;
        _lid_to_label[lid] = constants::kUnknownLabel;
        _deleted_map.add(lid);
        ++_deleted_size;
        return lid;
    }

    std::size_t MemVectorStore::size() const {
        TLOG_CHECK(_is_available, "should init be using");
        return _current_idx - _deleted_size;
    }

    [[nodiscard]] std::size_t MemVectorStore::deleted_size() const {
        return _deleted_size;
    }

    [[nodiscard]] std::size_t MemVectorStore::current_index() const {
        TLOG_CHECK(_is_available, "should init be using");
        return _current_idx;
    }

    std::size_t MemVectorStore::capacity() const {
        //std::shared_lock<std::shared_mutex> l(_data_lock);
        return capacity_impl();
    }

    [[nodiscard]] std::size_t MemVectorStore::capacity_impl() const {
        auto size = _data.size() * _option.batch_size;
        return size > _option.max_elements ? _option.max_elements : size;
    }

    std::size_t MemVectorStore::available() const {
        //std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        return capacity_impl() - _current_idx;
    }

    void MemVectorStore::reserve(std::size_t n) {
        //std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        reserve_impl(n);
    }

    void MemVectorStore::reserve_impl(std::size_t n) {
        if (n > _option.max_elements) {
            n = _option.max_elements;
        }
        while (_data.size() * _option.batch_size < n) {
            expend();
        }
    }

    void MemVectorStore::expend() {
        tann::VectorBatch vb;
        auto r = vb.init(_vs->vector_byte_size, _option.batch_size);
        //auto r = _data.back().init(_vs, _option.batch_size);
        if (!r.ok()) {
            throw ANNException("no memory", -1);
        }
        _data.push_back(std::move(vb));
    }

    void MemVectorStore::shrink() {
        //std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        while (!_data.empty() && _data.back().is_empty()) {
            _data.pop_back();
        }
    }

    void MemVectorStore::pop_back(std::size_t n) {
        //std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(n < _current_idx);
        resize_impl(_current_idx - n);
    }

    void MemVectorStore::resize(std::size_t n) {
        //std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_is_available, "should init be using");
        resize_impl(n);
    }

    void MemVectorStore::resize_impl(std::size_t n) {
        if (n == _current_idx) {
            return;
        }
        if (n < _current_idx) {
            std::size_t need_to_pop = _current_idx - n;
            for (auto idx = _current_idx / _option.batch_size; need_to_pop > 0; idx--) {
                auto bs = _data[idx].size();
                if (bs >= need_to_pop) {
                    _data[idx].resize(bs - need_to_pop);
                    need_to_pop = 0;
                } else {
                    _data[idx].resize(0);
                    need_to_pop -= bs;
                }
            }
        } else {
            reserve_impl(n);
            std::size_t need_to_expand = n - _current_idx;
            for (auto idx = _current_idx / _option.batch_size; need_to_expand > 0; idx++) {
                auto ba = _data[idx].available();
                if (ba >= need_to_expand) {
                    auto nsize = _data[idx].size() + need_to_expand;
                    _data[idx].resize(nsize);
                    need_to_expand = 0;
                } else {
                    _data[idx].resize(_option.batch_size);
                    need_to_expand -= ba;
                }
            }
        }
        _current_idx = n;
    }

    turbo::Status MemVectorStore::load(std::string_view path) {
        turbo::SequentialReadFile file;
        auto r = file.open(path);
        if (!r.ok()) {
            return r;
        }
        return load(&file);
    }

    turbo::Status MemVectorStore::save(std::string_view path) {
        turbo::SequentialWriteFile file;
        auto r = file.open(path);
        if (!r.ok()) {
            return r;
        }
        return save(&file);
    }

    turbo::Status MemVectorStore::load(turbo::SequentialReadFile *file) {
        //std::unique_lock<std::shared_mutex> ld(_data_lock);
        std::unique_lock<std::shared_mutex> lm(_meta_lock);
        turbo::StopWatcher watcher("vector set deserialize");
        TLOG_CHECK(_is_available, "should init be using");
        _is_available = false;
        TLOG_INFO("deserialize vector set start");
        auto r = read_binary_pod(*file, _current_idx);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("deserialize vector size: {}", _current_idx.load());
        r = read_binary_pod(*file, _deleted_size);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("deserialize vector deleted size: {}", _deleted_size.load());
        std::vector<char> bs;
        r = read_binary_vector(*file, bs);
        if (!r.ok()) {
            return r;
        }
        _deleted_map = bluebird::Bitmap::read(bs.data());
        TLOG_INFO("deserialize vector deleted map, size: {}", bs.size());
        r = read_binary_vector(*file, _lid_to_label);
        if (!r.ok()) {
            return r;
        }
        tann::SerializeOption rop;
        rop.n_vectors = _current_idx;
        rop.dimension = _vs->dimension;
        rop.data_type = _vs->data_type;
        tann::BinaryVectorSetReader reader;
        r = reader.initialize(file, rop);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("deserialize vector reader initialize ok");
        resize_impl(_current_idx);
        for (size_t i = 0; i < _data.size(); ++i) {
            auto span = _data[i].to_span();
            auto rs = reader.read_batch(span, _data[i].size());
            if (!rs.ok()) {
                return rs.status();
            }
            if (rs.value() != _data[i].size()) {
                return turbo::DataLossError("vector loss");
            }
        }
        for (size_t i = 0; i < _current_idx; i++) {
            auto lb = _lid_to_label[i];
            if (lb != constants::kUnknownLabel) {
                _label_map[lb] = i;
            }
        }
        _is_available = true;
        TLOG_INFO("deserialize done, cost: {}ms", turbo::ToDoubleMilliseconds(watcher.elapsed()));
        return turbo::OkStatus();
    }

    turbo::Status MemVectorStore::save(turbo::SequentialWriteFile *file) {
        //std::unique_lock<std::shared_mutex> ld(_data_lock);
        std::unique_lock<std::shared_mutex> lm(_meta_lock);
        turbo::StopWatcher watcher("vector set serialize");
        TLOG_INFO("serialize vector set start");
        auto r = write_binary_pod(*file, _current_idx);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("serialize current idx: {}", _current_idx.load());
        r = write_binary_pod(*file, _deleted_size);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("serialize deleted size: {}", _deleted_size.load());
        size_t bm_size = _deleted_map.getSizeInBytes();
        std::vector<char> buf(bm_size);
        _deleted_map.write(buf.data());
        TLOG_INFO("serialize deleted map size: {}", bm_size);
        r = write_binary_vector(*file, buf);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("serialize location to label");
        r = write_binary_vector(*file, _lid_to_label);
        if (!r.ok()) {
            return r;
        }
        tann::SerializeOption rop;
        rop.n_vectors = _current_idx;
        rop.dimension = _vs->dimension;
        rop.data_type = _vs->data_type;
        tann::BinaryVectorSetWriter writer;
        r = writer.initialize(file, rop);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("serialize datasets writer ok");
        for (size_t i = 0; i < _data.size(); ++i) {
            auto span = _data[i].to_span();
            r = writer.write_batch(span, _data[i].size());
            if (!r.ok()) {
                return r;
            }
        }
        if (writer.has_write() != _current_idx) {
            return turbo::InternalError("bad vector number");
        }
        TLOG_INFO("serialize done, cost: {}ms", turbo::ToDoubleMilliseconds(watcher.elapsed()));
        return turbo::OkStatus();
    }

    turbo::ResultStatus<label_type> MemVectorStore::get_label(location_t loc) {
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(loc < _current_idx);
        // this always call after is_deleted, do not need to check
        //std::shared_lock<std::shared_mutex> lock(_meta_lock);
        //if(_lid_to_label[loc] == constants::kUnknownLabel) {
        //    return turbo::NotFoundError("label not exists");
        //}
        return _lid_to_label[loc];
    }

    [[nodiscard]] bool MemVectorStore::exists_label(label_type label) const {
        TLOG_CHECK(_is_available, "should init be using");
        std::shared_lock<std::shared_mutex> lock(_label_map_lock);
        auto itr = _label_map.find(label);
        if (itr == _label_map.end()) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool MemVectorStore::is_deleted(location_t loc) const {
        std::shared_lock<std::shared_mutex> lock(_meta_lock);
        TLOG_CHECK(_is_available, "should init be using");
        TLOG_CHECK(loc < _current_idx, "overflow");
        return _lid_to_label[loc] == constants::kUnknownLabel;
    }


    [[nodiscard]] turbo::ResultStatus<location_t> MemVectorStore::get_vacant(label_type label) {
        TLOG_CHECK(_is_available, "should init be using");
        if (!_option.enable_replace_vacant) {
            return turbo::UnavailableError("config not allow using vacant");
        }
        location_t lid;

        std::unique_lock<std::shared_mutex> lock(_meta_lock);
        std::unique_lock<std::shared_mutex> label_lock(_label_map_lock);

        if (_deleted_map.isEmpty()) {
            return turbo::ResourceExhaustedError("no vacant to use");
        }

        lid = _deleted_map.minimum();
        auto itr = _label_map.find(label);
        if (itr != _label_map.end()) {
            return turbo::AlreadyExistsError("label :{} already in store", label);
        }
        _deleted_map.remove(lid);
        _lid_to_label[lid] = label;
        --_deleted_size;
        _label_map[label] = lid;
        return lid;
    }
}  // namespace tann
