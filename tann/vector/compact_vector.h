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
#ifndef TANN_VECTOR_COMPACT_VECTOR_H_
#define TANN_VECTOR_COMPACT_VECTOR_H_

#include <cstddef>
#include <cstdint>
#include "tann/common/exception.h"
#include "turbo/format/str_format.h"
#include "turbo/log/logging.h"

namespace tann {
    template<class TYPE>
    class CompactVector {
    public:
        typedef TYPE *iterator;

        CompactVector() : vector(0), vectorSize(0), allocatedSize(0) {}

        virtual ~CompactVector() { clear(); }

        void clear() {
            if (vector != 0) {
                delete[] vector;
            }
            vector = 0;
            vectorSize = 0;
            allocatedSize = 0;
        }

        TYPE &front() { return vector[0]; }

        TYPE &back() { return vector[vectorSize - 1]; }

        bool empty() { return vector == 0; }

        iterator begin() { return &(vector[0]); }

        iterator end() { return begin() + vectorSize; }

        TYPE &operator[](size_t idx) const { return vector[idx]; }

        CompactVector &operator=(CompactVector<TYPE> &v) {
            assert((vectorSize == v.vectorSize) || (vectorSize == 0));
            if (vectorSize == v.vectorSize) {
                for (size_t i = 0; i < vectorSize; i++) {
                    vector[i] = v[i];
                }
                return *this;
            } else {
                reserve(v.vectorSize);
                assert(allocatedSize >= v.vectorSize);
                for (size_t i = 0; i < v.vectorSize; i++) {
                    push_back(v.at(i));
                }
                vectorSize = v.vectorSize;
                assert(vectorSize == v.vectorSize);
            }
            return *this;
        }

        TYPE &at(size_t idx) const {
            if (idx >= vectorSize) {
                std::stringstream msg;
                msg << "CompactVector: beyond the range. " << idx << ":" << vectorSize;
                TANN_THROW(msg);
            }
            return vector[idx];
        }

        iterator erase(iterator b, iterator e) {
            iterator ret;
            e = end() < e ? end() : e;
            for (iterator i = b; i < e; i++) {
                ret = erase(i);
            }
            return ret;
        }

        iterator erase(iterator i) {
            iterator back = i;
            vectorSize--;
            iterator e = end();
            for (; i < e; i++) {
                *i = *(i + 1);
            }
            return ++back;
        }

        void pop_back() {
            if (vectorSize > 0) {
                vectorSize--;
            }
        }

        iterator insert(iterator &i, const TYPE &data) {
            if (size() == 0) {
                push_back(data);
                return end();
            }
            off_t oft = i - begin();
            extend();
            i = begin() + oft;
            iterator b = begin();
            for (iterator ci = end(); ci > i && ci != b; ci--) {
                *ci = *(ci - 1);
            }
            *i = data;
            vectorSize++;
            return i + 1;
        }

        void push_back(const TYPE &data) {
            extend();
            vector[vectorSize] = data;
            vectorSize++;
        }

        void reserve(size_t s) {
            if (s <= allocatedSize) {
                return;
            } else {
                TYPE *newptr = new TYPE[s];
                TYPE *dstptr = newptr;
                TYPE *srcptr = vector;
                TYPE *endptr = srcptr + vectorSize;
                while (srcptr < endptr) {
                    *dstptr++ = *srcptr;
                    (*srcptr).~TYPE();
                    srcptr++;
                }
                allocatedSize = s;
                if (vector != 0) {
                    delete[] vector;
                }
                vector = newptr;
            }
        }

        void resize(size_t s, TYPE v = TYPE()) {
            if (s > allocatedSize) {
                size_t asize = allocatedSize == 0 ? 1 : allocatedSize;
                while (asize < s) {
                    asize <<= 1;
                }
                reserve(asize);
                TYPE *base = vector;
                TYPE *dstptr = base + vectorSize;
                TYPE *endptr = base + s;
                for (; dstptr < endptr; dstptr++) {
                    *dstptr = v;
                }
            }
            vectorSize = s;
        }

        size_t size() const { return (size_t) vectorSize; }

        void extend() {
            extend(vectorSize);
        }

        void extend(size_t idx) {
            if (idx >= allocatedSize) {
                uint64_t size = allocatedSize == 0 ? 1 : allocatedSize;
                do {
                    size <<= 1;
                } while (size <= idx);
                if (size > 0xffff) {
                    TLOG_CRITICAL("CompactVector is too big. {}", size);
                    abort();
                }
                reserve(size);
            }
        }

        TYPE *vector;
        uint16_t vectorSize;
        uint16_t allocatedSize;
    };


}  // namespace tann

#endif  // TANN_VECTOR_COMPACT_VECTOR_H_
