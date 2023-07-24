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
#include "tann/datasets/tsv_vector_io.h"
#include "turbo/strings/str_split.h"
#include "turbo/strings/str_trim.h"
#include "turbo/strings/numbers.h"
namespace tann {

    namespace detail {

        template <typename T>
        turbo::Status convert_to_vector(const std::string&line, std::size_t ndim, turbo::Span<uint8_t> *vector) {
            auto span = to_span<T>(*vector);
            auto sv = turbo::TrimRight(line);
            std::vector<std::string_view> elems = turbo::StrSplit(sv, '\t', turbo::SkipEmpty());
            if(elems.size() != ndim) {
                return turbo::DataLossError("bad format");
            }
            constexpr bool is_inttype = std::is_integral_v<T>;
            for(size_t i = 0; i < ndim; ++i) {
                if constexpr (is_inttype) {
                    turbo::ResultStatus<int64_t> rs = turbo::Atoi<int64_t>(elems[i]);
                    if(!rs.ok()) {
                        return rs.status();
                    }
                    span[i] = static_cast<T>(rs.value());
                } else {
                    turbo::ResultStatus<float> rs = turbo::Atof(elems[i]);
                    if(!rs.ok()) {
                        return rs.status();
                    }
                    span[i] = static_cast<T>(rs.value());
                }
            }
            return turbo::OkStatus();
        }
        template  turbo::Status convert_to_vector<uint8_t>(const std::string&line, std::size_t ndim, turbo::Span<uint8_t> *vector);
        template  turbo::Status convert_to_vector<float16>(const std::string&line, std::size_t ndim, turbo::Span<uint8_t> *vector);
        template  turbo::Status convert_to_vector<float>(const std::string&line, std::size_t ndim, turbo::Span<uint8_t> *vector);

        template <typename T>
       turbo::Status convert_to_string(std::size_t ndim, turbo::Span<uint8_t> *vector, std::string *result) {
            auto span = to_span<T>(*vector);
            if(span.size() != ndim) {
                return turbo::DataLossError("bad format");
            }
            *result = turbo::FormatRange("{}", span.begin(), span.end(), "\t");
            return turbo::OkStatus();
        }

        template turbo::Status convert_to_string<uint8_t>(std::size_t ndim, turbo::Span<uint8_t> *vector, std::string *result);
        template turbo::Status convert_to_string<float16>(std::size_t ndim, turbo::Span<uint8_t> *vector, std::string *result);
        template turbo::Status convert_to_string<float>(std::size_t ndim, turbo::Span<uint8_t> *vector, std::string *result);
    }  // namespace detail
    turbo::Status TsvVectorSetReader::init() {
        return turbo::OkStatus();
    }

    turbo::Status TsvVectorSetReader::load(VectorSet &dst) {
        std::vector<uint8_t> raw_mem;
        raw_mem.resize(_vector_bytes);
        auto span = to_span<uint8_t>(raw_mem);
        turbo::Status fst = turbo::OkStatus();
        turbo::Status vst = turbo::OkStatus();
        while(_file->is_eof() && vst.ok()) {
            vst = read_vector(&span);
            dst.add_vector(span);
        }
        if(!turbo::IsReachFileEnd(vst)) {
            return vst;
        }
        return turbo::OkStatus();
    }

    turbo::Status TsvVectorSetReader::read_vector(turbo::Span<uint8_t> *vector) {
        static constexpr size_t KTmpSize = 4096;
        std::string line;
        line.reserve(KTmpSize);
        bool find_eol = false;
        bool feof = false;
        while (!find_eol) {
            if(_cache_buf.empty()) {
                if(_file->is_eof()) {
                    feof = true;
                    break;
                }
                std::string temp;
                auto r = _file->read(&temp, KTmpSize);
                if (!r.ok()) {
                    if(turbo::IsReachFileEnd(r.status())) {
                        feof = true;
                    } else {
                        return r.status();
                    }
                }
                _cache_buf.append(temp);
            }
            auto it = _cache_buf.char_begin();
            size_t n = 0;
            while(it != _cache_buf.char_end() && !find_eol) {
                if ((*it =='\n') && !line.empty()) {
                    find_eol = true;
                } else {
                    line.append(1, *it);
                }
                ++n;
                ++it;
            }
            _cache_buf.remove_prefix(n);
        }
        if(line.empty() && feof) {
            return turbo::ReachFileEnd("file reach eof");
        }
        ++_has_read;
        if(_option.data_type == DataType::DT_UINT8) {
            return detail::convert_to_vector<uint8_t>(line, _option.dimension, vector);
        } else if(_option.data_type == DataType::DT_FLOAT16) {
            return detail::convert_to_vector<float16>(line, _option.dimension, vector);
        } else if(_option.data_type == DataType::DT_FLOAT) {
            return detail::convert_to_vector<float>(line, _option.dimension, vector);
        }
        return turbo::InvalidArgumentError("data type parameter error {}", turbo::nameof_enum(_option.data_type));
    }

    turbo::ResultStatus<std::size_t> TsvVectorSetReader::read_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size) {
        std::size_t i = 0;
        for(; i < batch_size; i++) {
            turbo::Span<uint8_t> v = turbo::Span<uint8_t>(vector->data() + i * _vector_bytes, _vector_bytes);
            auto r = read_vector(&v);
            if(!r.ok()) {
                return r;
            }
        }
        return i;
    }


    turbo::Status TsvVectorSetWriter::init(){
        return turbo::OkStatus();
    }

    turbo::Status TsvVectorSetWriter::save(VectorSet &dst) {
        auto & bs = dst.vector_batch();
        for(auto & be : bs) {
            for(size_t i =0; i < be.size(); i++) {
                auto span = be.at(i);
                auto r = write_vector(&span);
                if(!r.ok()) {
                    return r;
                }
            }
        }
        return turbo::OkStatus();
    }

    turbo::Status TsvVectorSetWriter::write_vector(turbo::Span<uint8_t> *vector) {
        turbo::Status r;
        std::string line;
        if(_option.data_type == DataType::DT_UINT8) {
            r = detail::convert_to_string<uint8_t>(_option.dimension, vector, &line);
        } else if(_option.data_type == DataType::DT_FLOAT16) {
            r = detail::convert_to_string<float16>(_option.dimension, vector, &line);
        } else if(_option.data_type == DataType::DT_FLOAT) {
            r = detail::convert_to_string<float>(_option.dimension, vector, &line);
        } else {
            return turbo::InvalidArgumentError("data type parameter error {}", turbo::nameof_enum(_option.data_type));
        }
        if(!r.ok()) {
            return r;
        }
        line += "\n";
        r = _file->write(line);
        ++ _has_write;
        return r;
    }

    turbo::Status TsvVectorSetWriter::write_batch(turbo::Span<uint8_t> *vector, std::size_t batch_size) {
        for(std::size_t i = 0; i < batch_size; i++) {
            turbo::Span<uint8_t> v = turbo::Span<uint8_t>(vector->data() + i * _vector_bytes, _vector_bytes);
            auto r = write_vector(&v);
            if(!r.ok()) {
                return r;
            }
        }
        return turbo::OkStatus();
    }

}  // namespace tann
