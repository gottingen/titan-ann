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

#ifndef TANN_IO_UTILITY_H_
#define TANN_IO_UTILITY_H_

#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"

namespace tann {
    template<typename T>
    [[nodiscard]] static inline turbo::Status write_binary_pod(turbo::SequentialWriteFile &out, const T &podRef) {
        return out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    [[nodiscard]] static inline  turbo::Status read_binary_pod(turbo::SequentialReadFile &in, T &podRef) {
        auto r = in.read((char *) &podRef, sizeof(T));
        if(!r.ok()) {
            return r.status();
        }
        if(r.value() != sizeof(T)) {
            return turbo::DataLossError("not enough data");
        }
        return turbo::OkStatus();
    }

    template<typename T>
    [[nodiscard]] static inline turbo::Status write_binary_vector(turbo::SequentialWriteFile &out, const std::vector<T> &list) {
        size_t s = list.size();
        auto r = write_binary_pod(out, s);
        if(!r.ok()) {
            return r;
        }
        return out.write((char *) list.data(), sizeof(T) * s);
    }

    template<typename T>
    [[nodiscard]] static inline  turbo::Status read_binary_vector(turbo::SequentialReadFile &in, std::vector<T> &list) {
        size_t s;
        auto r = read_binary_pod(in, s);
        if(!r.ok()) {
            return r;
        }
        list.resize(s);
        auto rs = in.read(list.data(), sizeof(T) * s);
        if(!rs.ok()) {
            return rs.status();
        }
        if(rs.value() != sizeof(T) * s) {
            return turbo::DataLossError("not enough data");
        }
        return turbo::OkStatus();
    }

}  // namespace tann

#endif  // TANN_IO_UTILITY_H_
