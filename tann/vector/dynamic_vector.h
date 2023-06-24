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

#ifndef TANN_VECTOR_DYNAMIC_VECTOR_H_
#define TANN_VECTOR_DYNAMIC_VECTOR_H_

#include "tann/vector/serializer.h"
#include "tann/vector/vector_space.h"
#include "tann/common/shared_memory_allocator.h"

namespace tann {
    template<class TYPE>
    class DynamicVector {
    public:
        typedef TYPE *iterator;

        DynamicVector() : vector(0), vectorSize(0), allocatedSize(0), elementSize(0) {}

        ~DynamicVector() { clear(); }

        void clear() {
            if (vector != 0) {
                delete[] vector;
            }
            vector = 0;
            vectorSize = 0;
            allocatedSize = 0;
        }

        TYPE &front() { return (*this).at(0); }

        TYPE &back() { return (*this).at(vectorSize - 1); }

        bool empty() { return vectorSize == 0; }

        iterator begin() {
            return reinterpret_cast<iterator>(vector);
        }

        iterator end(SharedMemoryAllocator &allocator) {
            return begin() + vectorSize;
        }

        DynamicVector &operator=(DynamicVector<TYPE> &v) {
            std::cerr << "DynamicVector cannot be copied." << std::endl;
            abort();
        }

        TYPE &at(size_t idx) {
            if (idx >= vectorSize) {
                std::stringstream msg;
                msg << "Vector: beyond the range. " << idx << ":" << vectorSize;
                TANN_THROW(msg);
            }
            return *reinterpret_cast<TYPE *>(reinterpret_cast<uint8_t *>(begin()) + idx * elementSize);
        }

        TYPE &operator[](size_t idx) {
            return *reinterpret_cast<TYPE *>(reinterpret_cast<uint8_t *>(begin()) + idx * elementSize);
        }

        void copy(TYPE &dst, const TYPE &src) {
            memcpy(&dst, &src, elementSize);
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
                copy(*i, *(i + 1));
            }
            return back;
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
                copy(*ci, *(ci - 1));
            }
            copy(*i, data);
            vectorSize++;
            return i + 1;
        }

        void push_back(const TYPE &data) {
            extend();
            vectorSize++;
            copy((*this).at(vectorSize - 1), data);
        }

        void reserve(size_t s) {
            if (s <= allocatedSize) {
                return;
            } else {
                uint8_t *newptr = new uint8_t[s * elementSize];
                uint8_t *dstptr = newptr;
                uint8_t *srcptr = vector;
                memcpy(dstptr, srcptr, vectorSize * elementSize);
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
                uint8_t *base = vector;
                for (size_t i = vectorSize; i < s; i++) {
                    copy(*reinterpret_cast<TYPE *>(base + i * elementSize), v);
                }
            }
            vectorSize = s;
        }

        void serializeAsText(std::ostream &os, VectorSpace *objectspace = 0) {
            unsigned int s = size();
            os << s << " ";
            for (unsigned int i = 0; i < s; i++) {
                Serializer::writeAsText(os, (*this)[i]);
                os << " ";
            }
        }


        void deserializeAsText(std::istream &is, VectorSpace *objectspace = 0) {
            clear();
            size_t s;
            Serializer::readAsText(is, s);
            resize(s);
            for (unsigned int i = 0; i < s; i++) {
                Serializer::readAsText(is, (*this)[i]);
            }
        }


        void serialize(std::ofstream &os, tann::VectorSpace *objspace = 0) {
            uint32_t sz = size();
            tann::Serializer::write(os, sz);
            os.write(reinterpret_cast<char *>(vector), size() * elementSize);
        }

        void deserialize(std::ifstream &is, tann::VectorSpace *objectspace = 0) {
            uint32_t sz;
            try {
                tann::Serializer::read(is, sz);
            } catch (tann::Exception &err) {
                std::stringstream msg;
                msg
                        << "DynamicVector::deserialize: It might be caused by inconsistency of the valuable type of the vector. "
                        << err.what();
                TANN_THROW(msg);
            }
            resize(sz);
            is.read(reinterpret_cast<char *>(vector), sz * elementSize);
        }

        size_t size() { return vectorSize; }

    public:
        void extend() {
            extend(vectorSize);
        }

        void extend(size_t idx) {
            if (idx >= allocatedSize) {
                uint64_t size = allocatedSize == 0 ? 1 : allocatedSize;
                do {
                    size <<= 1;
                } while (size <= idx);
                if (size > 0xffffffff) {
                    std::cerr << "Vector is too big. " << size << std::endl;
                    abort();
                }
                reserve(size);
            }
        }

        uint8_t *vector;
        uint32_t vectorSize;
        uint32_t allocatedSize;
        uint32_t elementSize;
    };

}  // namespace

#endif  // TANN_VECTOR_DYNAMIC_VECTOR_H_
