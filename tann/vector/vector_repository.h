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
#ifndef TANN_VECTOR_VECTOR_REPOSITORY_H_
#define TANN_VECTOR_VECTOR_REPOSITORY_H_

#include "tann/vector/vector_view.h"
#include "tann/vector/repository.h"
#include "turbo/strings/str_split.h"

namespace tann {

    class VectorRepository : public Repository<Object> {
    public:
        typedef Repository<Object> Parent;

        VectorRepository(size_t dim, const std::type_info &ot) : dimension(dim), type(ot), sparse(false) {}

        void initialize() {
            deleteAll();
            Parent::push_back((PersistentObject *) 0);
        }

        void serialize(const std::string &ofile, VectorSpace *ospace) {
            std::ofstream objs(ofile);
            if (!objs.is_open()) {
                TANN_THROW(turbo::Format("tann::VectorSpace: Cannot open the specified file {}.", ofile));
            }
            Parent::serialize(objs, ospace);
        }

        void deserialize(const std::string &ifile, VectorSpace *ospace) {
            assert(ospace != 0);
            std::ifstream objs(ifile);
            if (!objs.is_open()) {
                TANN_THROW(turbo::Format("tann::VectorSpace: Cannot open the specified file {}.", ifile));
            }
            Parent::deserialize(objs, ospace);
        }

        void serializeAsText(const std::string &ofile, VectorSpace *ospace) {
            std::ofstream objs(ofile);
            if (!objs.is_open()) {
                TANN_THROW(turbo::Format("tann::VectorSpace: Cannot open the specified file {}.", ofile));
            }
            Parent::serializeAsText(objs, ospace);
        }

        void deserializeAsText(const std::string &ifile, VectorSpace *ospace) {
            std::ifstream objs(ifile);
            if (!objs.is_open()) {
                TANN_THROW(turbo::Format("tann::VectorSpace: Cannot open the specified file {}.", ifile));
            }
            Parent::deserializeAsText(objs, ospace);
        }

        void readText(std::istream &is, size_t dataSize = 0) {
            initialize();
            appendText(is, dataSize);
        }

        virtual PersistentObject *allocateNormalizedPersistentObject(const std::vector<double> &obj) {
            TLOG_CRITICAL(
                    "VectorRepository::allocateNormalizedPersistentObject(double): Fatal error! Something wrong!");
            abort();
        }

        virtual PersistentObject *allocateNormalizedPersistentObject(const std::vector<float> &obj) {
            TLOG_CRITICAL("VectorRepository::allocateNormalizedPersistentObject(float): Fatal error! Something wrong!");
            abort();
        }

        virtual PersistentObject *allocateNormalizedPersistentObject(const std::vector<uint8_t> &obj) {
            TLOG_CRITICAL(
                    "VectorRepository::allocateNormalizedPersistentObject(uint8_t): Fatal error! Something wrong!");
            abort();
        }

        virtual PersistentObject *allocateNormalizedPersistentObject(const float *obj, size_t size) {
            TLOG_CRITICAL("VectorRepository::allocateNormalizedPersistentObject: Fatal error! Something wrong!");
            abort();
        }

        void appendText(std::istream &is, size_t dataSize = 0) {
            if (dimension == 0) {
                TANN_THROW("VectorSpace::readText: Dimension is not specified.");
            }
            if (size() == 0) {
                // First entry should be always a dummy entry.
                // If it is empty, the dummy entry should be inserted.
                push_back((PersistentObject *) 0);
            }
            size_t prevDataSize = size();
            if (dataSize > 0) {
                reserve(size() + dataSize);
            }
            std::string line;
            size_t lineNo = 0;
            while (getline(is, line)) {
                lineNo++;
                if (dataSize > 0 && (dataSize <= size() - prevDataSize)) {
                    std::cerr
                            << "The size of data reached the specified size. The remaining data in the file are not inserted. "
                            << dataSize << std::endl;
                    break;
                }
                std::vector<double> object;
                try {
                    extractObjectFromText(line, "\t ", object);
                    PersistentObject *obj = 0;
                    try {
                        obj = allocateNormalizedPersistentObject(object);
                    } catch (Exception &err) {
                        std::cerr << err.what() << " continue..." << std::endl;
                        obj = allocatePersistentObject(object);
                    }
                    push_back(obj);
                } catch (Exception &err) {
                    std::cerr << "VectorSpace::readText: Warning! Invalid line. [" << line << "] Skip the line "
                              << lineNo << " and continue." << std::endl;
                }
            }
        }

        template<typename T>
        void append(T *data, size_t objectCount) {
            if (dimension == 0) {
                TANN_THROW("VectorSpace::readText: Dimension is not specified.");
            }
            if (size() == 0) {
                // First entry should be always a dummy entry.
                // If it is empty, the dummy entry should be inserted.
                push_back((PersistentObject *) 0);
            }
            if (objectCount > 0) {
                reserve(size() + objectCount);
            }
            for (size_t idx = 0; idx < objectCount; idx++, data += dimension) {
                std::vector<double> object;
                object.reserve(dimension);
                for (size_t dataidx = 0; dataidx < dimension; dataidx++) {
                    object.push_back(data[dataidx]);
                }
                try {
                    PersistentObject *obj = 0;
                    try {
                        obj = allocateNormalizedPersistentObject(object);
                    } catch (Exception &err) {
                        std::cerr << err.what() << " continue..." << std::endl;
                        obj = allocatePersistentObject(object);
                    }
                    push_back(obj);

                } catch (Exception &err) {
                    TLOG_ERROR("VectorSpace::readText: Warning! Invalid data. Skip the data no. {} and continue", idx);
                }
            }
        }

        Object *allocateObject() {
            return (Object *) new Object(paddedByteSize);
        }

        // This method is called during search to generate query.
        // Therefore the object is not persistent.
        Object *allocateObject(const std::string &textLine, const std::string &sep) {
            std::vector<double> object;
            extractObjectFromText(textLine, sep, object);
            Object *po = (Object *) allocateObject(object);
            return (Object *) po;
        }

        template<typename T>
        void extractObjectFromText(const std::string &textLine, const std::string &sep, std::vector<T> &object) {
            object.resize(dimension);
            std::vector<std::string> tokens = turbo::StrSplit(textLine, sep);
            if (dimension > tokens.size()) {
                TANN_THROW(turbo::Format("VectorSpace::allocate: too few dimension. {}:{}.{}", tokens.size(), dimension,
                                         textLine));
            }
            size_t idx;
            for (idx = 0; idx < dimension; idx++) {
                if (tokens[idx].size() == 0) {
                    TANN_THROW(turbo::Format("VectorSpace::allocate: an empty value string. {}:{}:{}.{}", idx,
                                             tokens.size(), dimension, textLine));
                }
                char *e;
                object[idx] = static_cast<T>(strtod(tokens[idx].c_str(), &e));
                if (*e != 0) {
                    TLOG_ERROR("VectorSpace::readText: Warning! Not numerical value. [{}]", e);
                    break;
                }
            }
        }

        template<typename T>
        Object *allocateObject(T *o, size_t size) {
            size_t osize = paddedByteSize;
            if (sparse) {
                size_t vsize = size * (type == typeid(float) ? 4 : 1);
                osize = osize < vsize ? vsize : osize;
            } else {
                if (dimension != size) {
                    TANN_THROW(turbo::Format(
                            "VectorSpace::allocateObject: Fatal error! The specified dimension is invalid. The indexed objects={} The specified object={}",
                            dimension, size));
                }
            }
            Object *po = new Object(osize);
            void *object = static_cast<void *>(&(*po)[0]);
            if (type == typeid(uint8_t)) {
                uint8_t *obj = static_cast<uint8_t *>(object);
                for (size_t i = 0; i < size; i++) {
                    obj[i] = static_cast<uint8_t>(o[i]);
                }
            } else if (type == typeid(float)) {
                float *obj = static_cast<float *>(object);
                for (size_t i = 0; i < size; i++) {
                    obj[i] = static_cast<float>(o[i]);
                }
#ifdef TANN_ENABLE_HALF_FLOAT
            } else if (type == typeid(float16)) {
                float16 *obj = static_cast<float16 *>(object);
                for (size_t i = 0; i < size; i++) {
                    obj[i] = static_cast<float16>(o[i]);
                }
#endif
            } else {
                TLOG_ERROR("VectorSpace::allocateObject: Fatal error: unsupported type!");
                abort();
            }
            return po;
        }

        template<typename T>
        Object *allocateObject(const std::vector<T> &o) {
            return allocateObject(o.data(), o.size());
        }

        template<typename T>
        PersistentObject *allocatePersistentObject(T *o, size_t size) {
            if (size != 0 && dimension != size) {
                auto msg = turbo::Format(
                        "VectorSpace::allocatePersistentObject: Fatal error! The dimensionality is invalid. The specified dimensionality={}. The specified object={}.",
                        (sparse ? dimension - 1 : dimension), (sparse ? size - 1 : size));
                TANN_THROW(msg);
            }
            return allocateObject(o, size);
        }

        template<typename T>
        PersistentObject *allocatePersistentObject(const std::vector<T> &o) {
            return allocatePersistentObject(o.data(), o.size());
        }

        void deleteObject(Object *po) {
            delete po;
        }

    private:
        void extractObject(void *object, std::vector<double> &d) {
            if (type == typeid(uint8_t)) {
                uint8_t *obj = (uint8_t *) object;
                for (size_t i = 0; i < dimension; i++) {
                    d.push_back(obj[i]);
                }
            } else if (type == typeid(float)) {
                float *obj = (float *) object;
                for (size_t i = 0; i < dimension; i++) {
                    d.push_back(obj[i]);
                }
#ifdef TANN_ENABLE_HALF_FLOAT
            } else if (type == typeid(float16)) {
                float16 *obj = (float16 *) object;
                for (size_t i = 0; i < dimension; i++) {
                    d.push_back(obj[i]);
                }
#endif
            } else {
                std::cerr << "VectorSpace::allocate: Fatal error: unsupported type!" << std::endl;
                abort();
            }
        }

    public:
        void extractObject(Object *o, std::vector<double> &d) {
            void *object = (void *) (&(*o)[0]);
            extractObject(object, d);
        }

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
        void extractObject(PersistentObject *o, std::vector<double> &d) {
          SharedMemoryAllocator &objectAllocator = getAllocator();
          void *object = (void*)(&(*o).at(0, objectAllocator));
          extractObject(object, d);
        }
#endif

        void setLength(size_t l) { byteSize = l; }

        void setPaddedLength(size_t l) { paddedByteSize = l; }

        void setSparse() { sparse = true; }

        size_t getByteSize() { return byteSize; }

        size_t insert(PersistentObject *obj) { return Parent::insert(obj); }

        const size_t dimension;
        const std::type_info &type;
    protected:
        size_t byteSize;        // the length of all of elements.
        size_t paddedByteSize;
        bool sparse;        // sparse data format
    };

} // namespace tann
#endif  // TANN_VECTOR_VECTOR_REPOSITORY_H_
