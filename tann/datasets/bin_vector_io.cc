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

#include "tann/datasets/bin_vector_io.h"

namespace tann {

    turbo::Status BinaryVectorSetReader::init() {
        uint32_t nvec;
        uint32_t dim;
        auto r = _file->read(&nvec, sizeof(nvec));
        if (!r.ok()) {
            return r.status();
        }
        r = _file->read(&dim, sizeof(dim));
        if (!r.ok()) {
            return r.status();
        }
        _nvecs = nvec;
        _ndims = dim;
        if (_ndims != _option.dimension) {
            return turbo::UnavailableError("bad format, option dimension: {} dimension read form file {}, not the same",
                                           _option.dimension, _ndims);
        }
        return turbo::OkStatus();
    }


    turbo::Status BinaryVectorSetReader::read_vector(turbo::Span<uint8_t> &vector) {
        TLOG_CHECK(_vector_bytes <= vector.size(), "not enough space to read vector");
        TLOG_CHECK(_vector_bytes > 0, "vector bytes size not set");
        auto r = _file->read(vector.data(), _vector_bytes);
        if (!r.ok()) {
            return r.status();
        }
        ++_has_read;
        return turbo::OkStatus();
    }

    turbo::ResultStatus<std::size_t>
    BinaryVectorSetReader::read_batch(turbo::Span<uint8_t> &vector, std::size_t batch_size) {
        TLOG_CHECK(_vector_bytes * batch_size <= vector.size(), "not enough space to read vector");
        auto r = _file->read(vector.data(), _vector_bytes * batch_size);
        if (!r.ok()) {
            return r.status();
        }
        auto n = r.value() / _vector_bytes;
        _has_read += n;
        return n;
    }

    turbo::Status BinaryVectorSetWriter::init() {
        uint32_t nvec = _option.n_vectors;
        uint32_t dim = _option.dimension;
        auto r = _file->write(reinterpret_cast<const char *>(&nvec), sizeof(nvec));
        if (!r.ok()) {
            return r;
        }

        r = _file->write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        if (!r.ok()) {
            return r;
        }
        return turbo::OkStatus();
    }

    turbo::Status BinaryVectorSetWriter::write_vector(turbo::Span<uint8_t> vector) {
        if(_has_write >= _option.n_vectors) {
            return turbo::OutOfRangeError("read the max vector size");
        }
        TLOG_CHECK(_vector_bytes <= vector.size(), "not enough space to read vector");
        auto r = _file->write(reinterpret_cast<const char *>(vector.data()), _vector_bytes);
        if (!r.ok()) {
            return r;
        }
        ++_has_write;
        return turbo::OkStatus();
    }

    turbo::Status BinaryVectorSetWriter::write_batch(turbo::Span<uint8_t> vector, std::size_t batch_size) {
        if(_has_write + batch_size > _option.n_vectors) {
            return turbo::OutOfRangeError("read the max vector size");
        }
        TLOG_CHECK(_vector_bytes * batch_size <= vector.size(), "not enough space to read vector");
        auto r = _file->write(reinterpret_cast<const char *>(vector.data()), _vector_bytes * batch_size);
        if (!r.ok()) {
            return r;
        }
        _has_write += batch_size;
        return turbo::OkStatus();
    }
}  // namespace tann
