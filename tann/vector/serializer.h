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

#ifndef TANN_VECTOR_SERIALIZER_H_
#define TANN_VECTOR_SERIALIZER_H_

#include "tann/vector/compact_vector.h"

namespace tann {

    namespace Serializer {
        static inline void read(std::istream &is, uint8_t *v, size_t s) {
            is.read((char *) v, s);
        }

        static inline void write(std::ostream &os, const uint8_t *v, size_t s) {
            os.write((const char *) v, s);
        }

        template<typename TYPE>
        void write(std::ostream &os, const TYPE v) {
            os.write((const char *) &v, sizeof(TYPE));
        }

        template<typename TYPE>
        void writeAsText(std::ostream &os, const TYPE v) {
            if (typeid(TYPE) == typeid(unsigned char)) {
                os << (int) v;
            } else {
                os << v;
            }
        }

        template<typename TYPE>
        void read(std::istream &is, TYPE &v) {
            is.read((char *) &v, sizeof(TYPE));
        }

        template<typename TYPE>
        void readAsText(std::istream &is, TYPE &v) {
            if (typeid(TYPE) == typeid(unsigned char)) {
                unsigned int tmp;
                is >> tmp;
                if (tmp > 255) {
                    TLOG_ERROR("Error! Invalid. {}", tmp);
                }
                v = (TYPE) tmp;
            } else {
                is >> v;
            }
        }

        template<typename TYPE>
        void write(std::ostream &os, const std::vector<TYPE> &v) {
            unsigned int s = v.size();
            write(os, s);
            for (unsigned int i = 0; i < s; i++) {
                write(os, v[i]);
            }
        }

        template<typename TYPE>
        void writeAsText(std::ostream &os, const std::vector<TYPE> &v) {
            unsigned int s = v.size();
            os << s << " ";
            for (unsigned int i = 0; i < s; i++) {
                writeAsText(os, v[i]);
                os << " ";
            }
        }

        template<typename TYPE>
        void write(std::ostream &os, const CompactVector<TYPE> &v) {
            unsigned int s = v.size();
            write(os, s);
            for (unsigned int i = 0; i < s; i++) {
                write(os, v[i]);
            }
        }

        template<typename TYPE>
        void writeAsText(std::ostream &os, const CompactVector<TYPE> &v) {
            unsigned int s = v.size();
            for (unsigned int i = 0; i < s; i++) {
                writeAsText(os, v[i]);
                os << " ";
            }
        }

        template<typename TYPE>
        void writeAsText(std::ostream &os, TYPE *v, size_t s) {
            os << s << " ";
            for (unsigned int i = 0; i < s; i++) {
                writeAsText(os, v[i]);
                os << " ";
            }
        }

        template<typename TYPE>
        void read(std::istream &is, std::vector<TYPE> &v) {
            v.clear();
            unsigned int s;
            read(is, s);
            v.reserve(s);
            for (unsigned int i = 0; i < s; i++) {
                TYPE val;
                read(is, val);
                v.push_back(val);
            }
        }

        template<typename TYPE>
        void readAsText(std::istream &is, std::vector<TYPE> &v) {
            v.clear();
            unsigned int s;
            is >> s;
            for (unsigned int i = 0; i < s; i++) {
                TYPE val;
                Serializer::readAsText(is, val);
                v.push_back(val);
            }
        }


        template<typename TYPE>
        void read(std::istream &is, CompactVector<TYPE> &v) {
            v.clear();
            unsigned int s;
            read(is, s);
            v.reserve(s);
            for (unsigned int i = 0; i < s; i++) {
                TYPE val;
                read(is, val);
                v.push_back(val);
            }
        }

        template<typename TYPE>
        void readAsText(std::istream &is, CompactVector<TYPE> &v) {
            v.clear();
            unsigned int s;
            is >> s;
            for (unsigned int i = 0; i < s; i++) {
                TYPE val;
                Serializer::readAsText(is, val);
                v.push_back(val);
            }
        }

        template<typename TYPE>
        void readAsText(std::istream &is, TYPE *v, size_t s) {
            unsigned int size;
            is >> size;
            if (s != size) {
                TLOG_ERROR("readAsText: something wrong. {}:{}", size, s);
                return;
            }
            for (unsigned int i = 0; i < s; i++) {
                TYPE val;
                Serializer::readAsText(is, val);
                v[i] = val;
            }
        }


    } // namespace Serialize

}  // namespace tann

#endif  // TANN_VECTOR_SERIALIZER_H_
