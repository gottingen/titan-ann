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

#ifndef TANN_VECTOR_REPOSITORY_H_
#define TANN_VECTOR_REPOSITORY_H_

#include "tann/common/config.h"
#include "tann/vector/serializer.h"
#include "tann/vector/vector_space.h"

namespace tann {

    template<class TYPE>
    class Repository : public std::vector<TYPE *> {
    public:

        static TYPE *allocate() { return new TYPE; }

        size_t push(TYPE *n) {
            if (std::vector<TYPE *>::size() == 0) {
                std::vector<TYPE *>::push_back(0);
            }
            std::vector<TYPE *>::push_back(n);
            return std::vector<TYPE *>::size() - 1;
        }

        size_t insert(TYPE *n) {
#ifdef ADVANCED_USE_REMOVED_LIST
            if (!removedList.empty()) {
                size_t idx = removedList.top();
                removedList.pop();
                put(idx, n);
                return idx;
            }
#endif
            return push(n);
        }

        bool isEmpty(size_t idx) {
            if (idx < std::vector<TYPE *>::size()) {
                return (*this)[idx] == 0;
            } else {
                return true;
            }
        }

        void put(size_t idx, TYPE *n) {
            if (std::vector<TYPE *>::size() <= idx) {
                std::vector<TYPE *>::resize(idx + 1, 0);
            }
            if ((*this)[idx] != 0) {
                TANN_THROW("put: Not empty");
            }
            (*this)[idx] = n;
        }

        void erase(size_t idx) {
            if (isEmpty(idx)) {
                TANN_THROW("erase: Not in-memory or invalid id");
            }
            delete (*this)[idx];
            (*this)[idx] = 0;
        }

        void remove(size_t idx) {
            erase(idx);
#ifdef ADVANCED_USE_REMOVED_LIST
            removedList.push(idx);
#endif
        }

        TYPE **getPtr() { return &(*this)[0]; }

        inline TYPE *get(size_t idx) {
            if (isEmpty(idx)) {
                std::stringstream msg;
                msg << "get: Not in-memory or invalid offset of node. idx=" << idx << " size=" << this->size();
                TANN_THROW(msg.str());
            }
            return (*this)[idx];
        }

        inline TYPE *getWithoutCheck(size_t idx) { return (*this)[idx]; }

        void serialize(std::ofstream &os, VectorSpace *objectspace = 0) {
            if (!os.is_open()) {
                TANN_THROW("tann::Common: Not open the specified stream yet.");
            }
            tann::Serializer::write(os, std::vector<TYPE *>::size());
            for (size_t idx = 0; idx < std::vector<TYPE *>::size(); idx++) {
                if ((*this)[idx] == 0) {
                    tann::Serializer::write(os, '-');
                } else {
                    tann::Serializer::write(os, '+');
                    if (objectspace == 0) {
                        (*this)[idx]->serialize(os);
                    } else {
                        (*this)[idx]->serialize(os, objectspace);
                    }
                }
            }
        }

        void deserialize(std::ifstream &is, VectorSpace *objectspace = 0) {
            if (!is.is_open()) {
                TANN_THROW("tann::Common: Not open the specified stream yet.");
            }
            deleteAll();
            size_t s;
            tann::Serializer::read(is, s);
            std::vector<TYPE *>::reserve(s);
            for (size_t i = 0; i < s; i++) {
                char type;
                tann::Serializer::read(is, type);
                switch (type) {
                    case '-': {
                        std::vector<TYPE *>::push_back(0);
#ifdef ADVANCED_USE_REMOVED_LIST
                        if (i != 0) {
                            removedList.push(i);
                        }
#endif
                    }
                        break;
                    case '+': {
                        if (objectspace == 0) {
                            TYPE *v = new TYPE;
                            v->deserialize(is);
                            std::vector<TYPE *>::push_back(v);
                        } else {
                            TYPE *v = new TYPE(objectspace);
                            v->deserialize(is, objectspace);
                            std::vector<TYPE *>::push_back(v);
                        }
                    }
                        break;
                    default: {
                        assert(type == '-' || type == '+');
                        break;
                    }
                }
            }
        }

        void serializeAsText(std::ofstream &os, VectorSpace *objectspace = 0) {
            if (!os.is_open()) {
                TANN_THROW("tann::Common: Not open the specified stream yet.");
            }
            // The format is almost the same as the default and the best in terms of the string length.
            os.setf(std::ios_base::fmtflags(0), std::ios_base::floatfield);
            os << std::setprecision(8);

            os << std::vector<TYPE *>::size() << std::endl;
            for (size_t idx = 0; idx < std::vector<TYPE *>::size(); idx++) {
                if ((*this)[idx] == 0) {
                    os << idx << " - " << std::endl;
                } else {
                    os << idx << " + ";
                    if (objectspace == 0) {
                        (*this)[idx]->serializeAsText(os);
                    } else {
                        (*this)[idx]->serializeAsText(os, objectspace);
                    }
                    os << std::endl;
                }
            }
            os << std::fixed;
        }

        void deserializeAsText(std::ifstream &is, VectorSpace *objectspace = 0) {
            if (!is.is_open()) {
                TANN_THROW("tann::Common: Not open the specified stream yet.");
            }
            deleteAll();
            size_t s;
            tann::Serializer::readAsText(is, s);
            std::vector<TYPE *>::reserve(s);
            for (size_t i = 0; i < s; i++) {
                size_t idx;
                tann::Serializer::readAsText(is, idx);
                if (i != idx) {
                    TLOG_ERROR("Repository: Error. index of a specified import file is invalid. {}:{}", idx, i);
                }
                char type;
                tann::Serializer::readAsText(is, type);
                switch (type) {
                    case '-': {
                        std::vector<TYPE *>::push_back(0);
#ifdef ADVANCED_USE_REMOVED_LIST
                        if (i != 0) {
                            removedList.push(i);
                        }
#endif
                    }
                        break;
                    case '+': {
                        if (objectspace == 0) {
                            TYPE *v = new TYPE;
                            v->deserializeAsText(is);
                            std::vector<TYPE *>::push_back(v);
                        } else {
                            TYPE *v = new TYPE(objectspace);
                            v->deserializeAsText(is, objectspace);
                            std::vector<TYPE *>::push_back(v);
                        }
                    }
                        break;
                    default: {
                        assert(type == '-' || type == '+');
                        break;
                    }
                }
            }
        }

        void deleteAll() {
            for (size_t i = 0; i < this->size(); i++) {
                if ((*this)[i] != 0) {
                    delete (*this)[i];
                    (*this)[i] = 0;
                }
            }
            this->clear();
            this->shrink_to_fit();
#ifdef ADVANCED_USE_REMOVED_LIST
            while (!removedList.empty()) { removedList.pop(); }
#endif
        }

        void set(size_t idx, TYPE *n) {
            (*this)[idx] = n;
        }

#ifdef ADVANCED_USE_REMOVED_LIST

        size_t count() {
            return std::vector<TYPE *>::size() == 0 ? 0 : std::vector<TYPE *>::size() - removedList.size() - 1;
        }

    protected:
        std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t> > removedList;
#endif
    };


}  // namespace tann
#endif  // TANN_VECTOR_REPOSITORY_H_
