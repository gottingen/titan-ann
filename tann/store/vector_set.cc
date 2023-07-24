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

#include "tann/store/vector_set.h"
#include "turbo/log/logging.h"
#include "tann/common/ann_exception.h"
#include "tann/datasets/bin_vector_io.h"
#include "tann/io/utility.h"
#include "turbo/times/stop_watcher.h"

namespace tann {

    turbo::Status VectorSet::init(VectorSpace *vp, std::size_t n) {
        _vs = vp;
        _batch_size = n;
        return turbo::OkStatus();
    }

    const VectorSpace *VectorSet::get_vector_space() const {
        TLOG_CHECK(_vs, "should init be using");
        return _vs;
    }

    const std::vector<VectorBatch> &VectorSet::vector_batch() const {
        TLOG_CHECK(_vs, "should init be using");
        return _data;
    }

    std::vector<VectorBatch> &VectorSet::vector_batch() {
        TLOG_CHECK(_vs, "should init be using");
        return _data;
    }

    std::size_t VectorSet::get_batch_size() const {
        return _batch_size;
    }

    void VectorSet::set_vector(const std::size_t i, turbo::Span<uint8_t> vector) {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(i < _current_idx, "vector set size {}, but set the vector {}, overflow!", _current_idx, i);
        auto bi = i / _batch_size;
        auto si = i % _batch_size;
        _data[bi].set_vector(si, vector);
    }


    turbo::Span<uint8_t> VectorSet::get_vector(const std::size_t i) const {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(i < _current_idx, "vector set size {}, but get the vector {}, overflow!", _current_idx, i);
        auto bi = i / _batch_size;
        auto si = i % _batch_size;
        return _data[bi].at(si);
    }

    void VectorSet::copy_vector(std::size_t i, turbo::Span<uint8_t> &des) const {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        auto ref = get_vector(i);
        TLOG_CHECK(des.size() >= ref.size());
        std::memcpy(des.data(), ref.data(), ref.size());
    }

    double VectorSet::get_distance(std::size_t l1, std::size_t l2) const {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(l1 < _current_idx, "overflow");
        TLOG_CHECK(l2 < _current_idx, "overflow");
        //TLOG_INFO("compare {} {}", l1, l2);
        auto v1 = get_vector(l1);
        auto v2 = get_vector(l2);
        return _vs->distance_factor->compare(v1, v2);
    }

    double VectorSet::get_distance(turbo::Span<uint8_t> query, std::size_t l1) const {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(l1 < _current_idx, "should init be using");
        auto v1 = get_vector(l1);
        return _vs->distance_factor->compare(v1, query);
    }

    void VectorSet::get_distance(turbo::Span<uint8_t> query, turbo::Span<std::size_t> ls,
                                 turbo::Span<double> ds) const {
        TLOG_CHECK(ls.size() <= ds.size());
        for (size_t i = 0; i < ls.size(); ++i) {
            ds[i] = get_distance(query, ls[i]);
        }
    }

    void VectorSet::move_vector(std::size_t from, std::size_t to) {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(from < _current_idx, "overflow");
        TLOG_CHECK(to < _current_idx, "overflow");
        auto vf = get_vector(from);
        auto vt = get_vector(to);
        std::memcpy(vt.data(), vf.data(), vf.size());
    }

    void VectorSet::move_vector(std::size_t from, std::size_t to, std::size_t nvec) {
        TLOG_CHECK(_vs, "should init be using");
        for (int i = 0; i < nvec; ++i) {
            move_vector(from + i, to + i);
        }
    }

    std::size_t VectorSet::add_vector(const turbo::Span<uint8_t> &query) {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        auto bi = _current_idx / _batch_size;
        if (_current_idx >= capacity()) {
            expend();
        }
        _data[bi].add_vector(query);
        auto index = _current_idx;
        ++_current_idx;
        return index;
    }

    std::size_t VectorSet::prefer_add_vector(std::size_t nvec) {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        auto ret = _current_idx;
        auto new_size = _current_idx + nvec;

        while (_data.size() * _batch_size < new_size) {
            expend();
        }

        std::size_t need_to_expand = nvec;
        for (auto idx = _current_idx / _batch_size; need_to_expand > 0; idx++) {
            auto ba = _data[idx].available();
            if (ba >= need_to_expand) {
                auto nsize = _data[idx].size() + need_to_expand;
                _data[idx].resize(nsize);
                need_to_expand = 0;
            } else {
                _data[idx].resize(_batch_size);
                need_to_expand -= ba;
            }
        }
        _current_idx = new_size;
        return ret;
    }

    std::size_t VectorSet::size() const {
        TLOG_CHECK(_vs, "should init be using");
        return _current_idx - _deleted_size;
    }

    [[nodiscard]] std::size_t VectorSet::deleted_size() const {
        return _deleted_size;
    }

    [[nodiscard]] std::size_t VectorSet::current_index() const {
        TLOG_CHECK(_vs, "should init be using");
        return _current_idx;
    }

    std::size_t VectorSet::capacity() const {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        return _data.size() * _batch_size;
    }

    std::size_t VectorSet::available() const {
        std::shared_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        return _data.size() * _batch_size - _current_idx;
    }

    void VectorSet::reserve(std::size_t n) {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        while (_data.size() * _batch_size < n) {
            expend();
        }
    }

    void VectorSet::expend() {
        tann::VectorBatch vb;
        auto r = vb.init(_vs->vector_byte_size, _batch_size);
        //auto r = _data.back().init(_vs, _batch_size);
        if (!r.ok()) {
            throw ANNException("no memory", -1);
        }
        _data.push_back(std::move(vb));
    }

    void VectorSet::shrink() {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        while (!_data.empty() && _data.back().is_empty()) {
            _data.pop_back();
        }
    }

    void VectorSet::pop_back(std::size_t n) {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(n < _current_idx);
        resize(_current_idx - n);
    }

    void VectorSet::resize(std::size_t n) {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        TLOG_CHECK(_vs, "should init be using");
        if (n == _current_idx) {
            return;
        }
        if (n < _current_idx) {
            std::size_t need_to_pop = _current_idx - n;
            for (auto idx = _current_idx / _batch_size; need_to_pop > 0; idx--) {
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
            reserve(n);
            std::size_t need_to_expand = n - _current_idx;
            for (auto idx = _current_idx / _batch_size; need_to_expand > 0; idx++) {
                auto ba = _data[idx].available();
                if (ba >= need_to_expand) {
                    auto nsize = _data[idx].size() + need_to_expand;
                    _data[idx].resize(nsize);
                    need_to_expand = 0;
                } else {
                    _data[idx].resize(_batch_size);
                    need_to_expand -= ba;
                }
            }
        }
        _current_idx = n;
    }

    turbo::Status VectorSet::load(std::string_view path) {
        turbo::SequentialReadFile file;
        auto r = file.open(path);
        if (!r.ok()) {
            return r;
        }
        return load(&file);
    }

    turbo::Status VectorSet::save(std::string_view path) {
        turbo::SequentialWriteFile file;
        auto r = file.open(path);
        if (!r.ok()) {
            return r;
        }
        return save(&file);
    }

    turbo::Status VectorSet::load(turbo::SequentialReadFile *file) {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        turbo::StopWatcher watcher("vector set deserialize");
        TLOG_INFO("deserialize vector set start");
        auto r = read_binary_pod(*file, _current_idx);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("deserialize vector size: {}", _current_idx);
        r = read_binary_pod(*file, _deleted_size);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("deserialize vector deleted size: {}", _deleted_size);
        std::vector<char> bs;
        r = read_binary_vector(*file, bs);
        if (!r.ok()) {
            return r;
        }
        _deleted_map = bluebird::Bitmap::read(bs.data());
        TLOG_INFO("deserialize vector deleted map, size: {}", bs.size());
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
        resize(_current_idx);
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
        TLOG_INFO("deserialize done, cost: {}ms", turbo::ToDoubleMilliseconds(watcher.elapsed()));
        return turbo::OkStatus();
    }

    turbo::Status VectorSet::save(turbo::SequentialWriteFile *file) {
        std::unique_lock<std::shared_mutex> l(_data_lock);
        turbo::StopWatcher watcher("vector set serialize");
        TLOG_INFO("serialize vector set start");
        auto r = write_binary_pod(*file, _current_idx);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("serialize current idx: {}", _current_idx);
        r = write_binary_pod(*file, _deleted_size);
        if (!r.ok()) {
            return r;
        }
        TLOG_INFO("serialize deleted size: {}", _deleted_size);
        size_t bm_size = _deleted_map.getSizeInBytes();
        std::vector<char> buf(bm_size);
        _deleted_map.write(buf.data());
        TLOG_INFO("serialize deleted map size: {}", bm_size);
        r = write_binary_vector(*file, buf);
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
}  // namespace tann
