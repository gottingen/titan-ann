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
#ifndef TANN_VECTOR_VECTOR_VIEW_H_
#define TANN_VECTOR_VECTOR_VIEW_H_

#include "tann/common/config.h"
#include "tann/vector/primitive_comparator.h"
#include "tann/vector/vector_space.h"
#include "tann/vector/serializer.h"
#include "turbo/log/logging.h"

namespace tann {

    class BaseObject {
    public:
        virtual uint8_t &operator[](size_t idx) const = 0;

        void serialize(std::ostream &os, VectorSpace *objectspace = 0) {
            assert(objectspace != 0);
            size_t byteSize = objectspace->getByteSizeOfObject();
            tann::Serializer::write(os, (uint8_t * ) & (*this)[0], byteSize);
        }

        void deserialize(std::istream &is, VectorSpace *objectspace = 0) {
            assert(objectspace != 0);
            size_t byteSize = objectspace->getByteSizeOfObject();
            assert(&(*this)[0] != 0);
            tann::Serializer::read(is, (uint8_t * ) & (*this)[0], byteSize);
            if (is.eof()) {
                std::stringstream msg;
                msg
                        << "VectorSpace::BaseObject: Fatal Error! Read beyond the end of the object file. The object file is corrupted?"
                        << byteSize;
                TANN_THROW(msg);
            }
        }

        void serializeAsText(std::ostream &os, VectorSpace *objectspace = 0) {
            assert(objectspace != 0);
            const std::type_info &t = objectspace->getObjectType();
            size_t dimension = objectspace->getDimension();
            void *ref = (void *) &(*this)[0];
            if (t == typeid(uint8_t)) {
                tann::Serializer::writeAsText(os, (uint8_t *) ref, dimension);
            } else if (t == typeid(float)) {
                tann::Serializer::writeAsText(os, (float *) ref, dimension);
#ifdef TANN_ENABLE_HALF_FLOAT
                } else if (t == typeid(float16)) {
                tann::Serializer::writeAsText(os, (float16 *) ref, dimension);
#endif
            } else if (t == typeid(double)) {
                tann::Serializer::writeAsText(os, (double *) ref, dimension);
            } else if (t == typeid(uint16_t)) {
                tann::Serializer::writeAsText(os, (uint16_t *) ref, dimension);
            } else if (t == typeid(uint32_t)) {
                tann::Serializer::writeAsText(os, (uint32_t *) ref, dimension);
            } else {
                TLOG_ERROR("Object::serializeAsText: not supported data type. [{}]",t.name());
                assert(0);
            }
        }

        void deserializeAsText(std::ifstream &is, VectorSpace *objectspace = 0) {
            assert(objectspace != 0);
            const std::type_info &t = objectspace->getObjectType();
            size_t dimension = objectspace->getDimension();
            void *ref = (void *) &(*this)[0];
            assert(ref != 0);
            if (t == typeid(uint8_t)) {
                tann::Serializer::readAsText(is, (uint8_t *) ref, dimension);
            } else if (t == typeid(float)) {
                tann::Serializer::readAsText(is, (float *) ref, dimension);
#ifdef TANN_ENABLE_HALF_FLOAT
                } else if (t == typeid(float16)) {
                tann::Serializer::readAsText(is, (float16 *) ref, dimension);
#endif
            } else if (t == typeid(double)) {
                tann::Serializer::readAsText(is, (double *) ref, dimension);
            } else if (t == typeid(uint16_t)) {
                tann::Serializer::readAsText(is, (uint16_t *) ref, dimension);
            } else if (t == typeid(uint32_t)) {
                tann::Serializer::readAsText(is, (uint32_t *) ref, dimension);
            } else {
                TLOG_ERROR("Object::deserializeAsText: not supported data type. [{}]",t.name());
                assert(0);
            }
        }

    };


    class Object : public BaseObject {
    public:
        Object(tann::VectorSpace *os = 0) : vector(0) {
            if (os == 0) {
                return;
            }
            size_t s = os->getByteSizeOfObject();
            construct(s);
        }

        Object(size_t s) : vector(0) {
            assert(s != 0);
            construct(s);
        }

        void attach(void *ptr) { vector = static_cast<uint8_t *>(ptr); }

        void detach() { vector = 0; }

        void copy(Object &o, size_t s) {
            assert(vector != 0);
            for (size_t i = 0; i < s; i++) {
                vector[i] = o[i];
            }
        }

        virtual ~Object() { clear(); }

        uint8_t &operator[](size_t idx) const { return vector[idx]; }

        void *getPointer(size_t idx = 0) const { return vector + idx; }

        static Object *allocate(VectorSpace &objectspace) { return new Object(&objectspace); }

    private:
        void clear() {
            if (vector != 0) {
                MemoryCache::alignedFree(vector);
            }
            vector = 0;
        }

        void construct(size_t s) {
            assert(vector == 0);
            size_t allocsize = ((s - 1) / 64 + 1) * 64;
            vector = static_cast<uint8_t *>(MemoryCache::alignedAlloc(allocsize));
            memset(vector, 0, allocsize);
        }

        uint8_t *vector;
    };


}  // namespace tann

#endif  // TANN_VECTOR_VECTOR_VIEW_H_
