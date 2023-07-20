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
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(i < _size, "vector set size {}, but set the vector {}, overflow!", _size, i);
        auto bi = i / _batch_size;
        auto si = i % _batch_size;
        _data[bi].set_vector(si, vector);
    }


    turbo::Span<uint8_t> VectorSet::get_vector(const std::size_t i) const {
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(i < _size, "vector set size {}, but get the vector {}, overflow!", _size, i);
        auto bi = i / _batch_size;
        auto si = i % _batch_size;
        return _data[bi].at(si);
    }

    void VectorSet::copy_vector(std::size_t i, turbo::Span<uint8_t> &des) const {
        TLOG_CHECK(_vs, "should init be using");
        auto ref = get_vector(i);
        TLOG_CHECK(des.size() >= ref.size());
        std::memcpy(des.data(), ref.data(), ref.size());
    }

    double VectorSet::get_distance(std::size_t l1, std::size_t l2) const {
        TLOG_CHECK(_vs, "should init be using");
        auto v1 = get_vector(l1);
        auto v2 = get_vector(l2);
        return _vs->distance_factor->compare(v1, v2);
    }

    double VectorSet::get_distance(turbo::Span<uint8_t> query, std::size_t l1) const {
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
        TLOG_CHECK(_vs, "should init be using");
        auto bi = _size / _batch_size;
        if (_size >= capacity()) {
            expend();
        }
        _data[bi].add_vector(query);
        auto index = _size;
        ++_size;
        return index;
    }

    std::size_t VectorSet::prefer_add_vector(std::size_t n) {
        TLOG_CHECK(_vs, "should init be using");
        auto idx = _size;
        auto n_size = _size + n;
        resize(n_size);
        return idx;
    }

    std::size_t VectorSet::size() const {
        TLOG_CHECK(_vs, "should init be using");
        return _size;
    }

    std::size_t VectorSet::capacity() const {
        TLOG_CHECK(_vs, "should init be using");
        return _data.size() * _batch_size;
    }

    std::size_t VectorSet::available() const {
        TLOG_CHECK(_vs, "should init be using");
        return _data.size() * _batch_size - _size;
    }

    void VectorSet::reserve(std::size_t n) {
        TLOG_CHECK(_vs, "should init be using");
        while (capacity() < n) {
            expend();
        }
    }
    void VectorSet::expend() {
        tann::VectorBatch vb;
        auto r = vb.init(_vs, _batch_size);
        //auto r = _data.back().init(_vs, _batch_size);
        if(!r.ok()) {
            throw ANNException("no memory", -1);
        }
        _data.push_back(std::move(vb));
    }
    void VectorSet::shrink() {
        TLOG_CHECK(_vs, "should init be using");
        while(!_data.empty() && _data.back().is_empty()) {
            _data.pop_back();
        }
    }

    void VectorSet::pop_back(std::size_t n) {
        TLOG_CHECK(_vs, "should init be using");
        TLOG_CHECK(n < _size);
        resize(_size - n);
    }

    void VectorSet::resize(std::size_t n) {
        TLOG_CHECK(_vs, "should init be using");
        if(n ==  _size) {
            return;
        }
        if(n < _size) {
            std::size_t need_to_pop = _size - n;
            for(auto idx = _size/_batch_size; need_to_pop > 0; idx--) {
                auto bs = _data[idx].size();
                if(bs >= need_to_pop) {
                    _data[idx].resize(bs - need_to_pop);
                    need_to_pop = 0;
                } else {
                    _data[idx].resize(0);
                    need_to_pop -= bs;
                }
            }
        } else {
            reserve(n);
            std::size_t need_to_expand = n - _size;
            for(auto idx = _size/_batch_size; need_to_expand > 0; idx++) {
                auto ba = _data[idx].available();
                if(ba >= need_to_expand) {
                    auto nsize = _data[idx].size() + need_to_expand;
                    _data[idx].resize(nsize);
                    need_to_expand = 0;
                } else {
                    _data[idx].resize(_batch_size);
                    need_to_expand -= ba;
                }
            }
        }
        _size = n;
    }
}  // namespace tann
