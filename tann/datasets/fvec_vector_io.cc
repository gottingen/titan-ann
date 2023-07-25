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

#include "tann/datasets/fvec_vector_io.h"

namespace tann {

    turbo::Status FvecVectorSetReader::init() {
        uint32_t ndims_u32;
        auto r = _file->read(&ndims_u32, sizeof(ndims_u32));
        if (!r.ok()) {
            return r.status();
        }
        _ndims = ndims_u32;
        if (_ndims != _option.dimension) {
            return turbo::UnavailableError("bad format, option dimension: {} dimension read form file {}, not the same",
                                           _option.dimension, _ndims);
        }
        return turbo::OkStatus();
    }

    turbo::Status FvecVectorSetReader::load(VectorSet &dst) {
        std::vector<uint8_t> raw_mem;
        raw_mem.resize(_vector_bytes);
        auto span = to_span<uint8_t>(raw_mem);
        turbo::Status vst = turbo::OkStatus();
        while (_file->is_eof() && vst.ok()) {
            vst = read_vector(span);
            dst.add_vector(span);
        }
        if(!turbo::IsReachFileEnd(vst)) {
            return vst;
        }
        return turbo::OkStatus();
    }

    turbo::Status FvecVectorSetReader::read_vector(turbo::Span<uint8_t> &vector) {
        if (_file->is_eof()) {
            return turbo::ReachFileEnd("");
        }
        TLOG_CHECK(_vector_bytes <= vector.size(), "not enough space to read vector");
        auto r = _file->read(vector.data(), _vector_bytes);
        if (!r.ok()) {
            return r.status();
        }
        ++_has_read;
        if (!_file->is_eof()) {
            r = _file->skip(sizeof(uint32_t));
            TURBO_UNUSED(r);
        }
        return turbo::OkStatus();
    }

    turbo::ResultStatus<std::size_t> FvecVectorSetReader::read_batch(turbo::Span<uint8_t> &vector, std::size_t batch_size) {
        std::size_t i = 0;
        for (; i < batch_size; i++) {
            turbo::Span<uint8_t> v = turbo::Span<uint8_t>(vector.data() + i * _vector_bytes, _vector_bytes);
            auto r = read_vector(v);
            if (!r.ok()) {
                if(turbo::IsReachFileEnd(r)) {
                    break;
                }
                return r;
            }
        }
        return i;
    }

    turbo::Status FvecVectorSetWriter::init() {
        return turbo::OkStatus();
    }

    turbo::Status FvecVectorSetWriter::save(VectorSet &dst) {
        auto & bs = dst.vector_batch();
        for(auto & be : bs) {
            for(size_t i =0; i < be.size(); i++) {
                auto span = be.at(i);
                auto r = write_vector(span);
                if(!r.ok()) {
                    return r;
                }
            }
        }
        return turbo::OkStatus();
    }

    turbo::Status FvecVectorSetWriter::write_vector(turbo::Span<uint8_t> vector) {
        TLOG_CHECK(_vector_bytes <= vector.size(), "not enough space to read vector");
        uint32_t ndims = _option.dimension;
        auto r = _file->write(reinterpret_cast<const char *>(&ndims), sizeof(ndims));
        if (!r.ok()) {
            return r;
        }
        r = _file->write(reinterpret_cast<const char *>(vector.data()), _vector_bytes);
        if (!r.ok()) {
            return r;
        }
        ++_has_write;
        return turbo::OkStatus();
    }

    turbo::Status FvecVectorSetWriter::write_batch(turbo::Span<uint8_t> vector, std::size_t batch_size) {
        for(std::size_t i = 0; i < batch_size; i++) {
            turbo::Span<uint8_t> v = turbo::Span<uint8_t>(vector.data() + i * _vector_bytes, _vector_bytes);
            auto r = write_vector(v);
            if(!r.ok()) {
                return r;
            }
        }
        return turbo::OkStatus();
    }
}  // namespace tann
