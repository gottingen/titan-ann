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

#ifndef TANN_COMMON_COMPACT_STRING_H_
#define TANN_COMMON_COMPACT_STRING_H_

#include <string_view>
#include <string>
#include <cstring>
#include "tann/common/exception.h"

namespace tann {
    class CompactString {
    public:
        CompactString() : vector(0) {}

        CompactString(const CompactString &v) : vector(0) { *this = v; }

        ~CompactString() { clear(); }

        void clear() {
            if (vector != 0) {
                delete[] vector;
            }
            vector = 0;
        }

        CompactString &operator=(const std::string &v) { return *this = v.c_str(); }

        CompactString &operator=(const CompactString &v) { return *this = v.vector; }

        CompactString &operator=(const char *str) {
            if (str == 0 || strlen(str) == 0) {
                clear();
                return *this;
            }
            if (size() != strlen(str)) {
                clear();
                vector = new char[strlen(str) + 1];
            }
            strcpy(vector, str);
            return *this;
        }

        char &at(size_t idx) const {
            if (idx >= size()) {
                TANN_THROW("CompactString: beyond the range");
            }
            return vector[idx];
        }

        char *c_str() { return vector; }

        size_t size() const {
            if (vector == 0) {
                return 0;
            } else {
                return (size_t) strlen(vector);
            }
        }

        char *vector;
    };
}  // namespace tann
#endif  // TANN_COMMON_COMPACT_STRING_H_
