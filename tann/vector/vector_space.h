//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include "tann/vector/primitive_comparator.h"
#include "tann/vector/vector_distance.h"
#include "tann/index/option.h"

class VectorSpace;

namespace tann {

    class PersistentVectorDistances;

    class VectorSpace;

    class VectorDistances : public std::vector<VectorDistance> {
    public:
        VectorDistances(tann::VectorSpace *os = 0) {}

        void serialize(std::ofstream &os, VectorSpace *objspace = 0) {
            tann::Serializer::write(os, (std::vector<VectorDistance> &) *this);
        }

        void deserialize(std::ifstream &is, VectorSpace *objspace = 0) {
            tann::Serializer::read(is, (std::vector<VectorDistance> &) *this);
        }

        void serializeAsText(std::ofstream &os, VectorSpace *objspace = 0) {
            tann::Serializer::writeAsText(os, size());
            os << " ";
            for (size_t i = 0; i < size(); i++) {
                (*this)[i].serializeAsText(os);
                os << " ";
            }
        }

        void deserializeAsText(std::ifstream &is, VectorSpace *objspace = 0) {
            size_t s;
            tann::Serializer::readAsText(is, s);
            resize(s);
            for (size_t i = 0; i < size(); i++) {
                (*this)[i].deserializeAsText(is);
            }
        }

        void
        moveFrom(std::priority_queue<VectorDistance, std::vector<VectorDistance>, std::less<VectorDistance> > &pq) {
            this->clear();
            this->resize(pq.size());
            for (int i = pq.size() - 1; i >= 0; i--) {
                (*this)[i] = pq.top();
                pq.pop();
            }
            assert(pq.size() == 0);
        }

        void moveFrom(std::priority_queue<VectorDistance, std::vector<VectorDistance>, std::less<VectorDistance> > &pq,
                      double (&f)(double)) {
            this->clear();
            this->resize(pq.size());
            for (int i = pq.size() - 1; i >= 0; i--) {
                (*this)[i] = pq.top();
                (*this)[i].distance = f((*this)[i].distance);
                pq.pop();
            }
            assert(pq.size() == 0);
        }

        void moveFrom(std::priority_queue<VectorDistance, std::vector<VectorDistance>, std::less<VectorDistance> > &pq,
                      unsigned int id) {
            this->clear();
            if (pq.size() == 0) {
                return;
            }
            this->resize(id == 0 ? pq.size() : pq.size() - 1);
            int i = this->size() - 1;
            while (pq.size() != 0 && i >= 0) {
                if (pq.top().id != id) {
                    (*this)[i] = pq.top();
                    i--;
                }
                pq.pop();
            }
            if (pq.size() != 0 && pq.top().id != id) {
                std::cerr << "moveFrom: Fatal error: somethig wrong! " << pq.size() << ":" << this->size() << ":" << id
                          << ":" << pq.top().id << std::endl;
                assert(pq.size() == 0 || pq.top().id == id);
            }
        }

        VectorDistances &operator=(PersistentVectorDistances &objs);
    };

    typedef VectorDistances GraphNode;

    class Object;

    typedef Object PersistentObject;

    class VectorRepository;

    class VectorSpace {
    public:
        class Comparator {
        public:

            Comparator(size_t d) : dimension(d) {}

            virtual double operator()(Object &objecta, Object &objectb) = 0;

            size_t dimension;

            virtual ~Comparator() {}
        };

        /*
        enum MetricType {
            MetricTypeNone = -1,
            MetricTypeL1 = 0,
            MetricTypeL2 = 1,
            MetricTypeHamming = 2,
            MetricTypeAngle = 3,
            MetricTypeCosine = 4,
            MetricTypeNormalizedAngle = 5,
            MetricTypeNormalizedCosine = 6,
            MetricTypeJaccard = 7,
            MetricTypeSparseJaccard = 8,
            MetricTypeNormalizedL2 = 9,
            MetricTypePoincare = 100,  // added by Nyapicom
            MetricTypeLorentz = 101  // added by Nyapicom
        };

        enum ObjectType {
            ObjectTypeNone = 0,
            Uint8 = 1,
            Float = 2
#ifdef TANN_ENABLE_HALF_FLOAT
            ,
            Float16 = 3
#endif
        };
*/

        typedef std::priority_queue<VectorDistance, std::vector<VectorDistance>, std::less<VectorDistance> > ResultSet;

        VectorSpace(size_t d) : dimension(d), distanceType(MetricType::MetricTypeNone), comparator(0),
                                normalization(false),
                                prefetchOffset(-1), prefetchSize(-1) {}

        virtual ~VectorSpace() { if (comparator != 0) { delete comparator; }}

        virtual size_t insert(Object *obj) = 0;

        Comparator &getComparator() { return *comparator; }

        virtual void serialize(const std::string &of) = 0;

        virtual void deserialize(const std::string &ifile) = 0;

        virtual void serializeAsText(const std::string &of) = 0;

        virtual void deserializeAsText(const std::string &of) = 0;

        virtual void readText(std::istream &is, size_t dataSize) = 0;

        virtual void appendText(std::istream &is, size_t dataSize) = 0;

        virtual void append(const float *data, size_t dataSize) = 0;

        virtual void append(const double *data, size_t dataSize) = 0;

        virtual void copy(Object &objecta, Object &objectb) = 0;

        virtual void linearSearch(Object &query, double radius, size_t size,
                                  VectorSpace::ResultSet &results) = 0;

        virtual const std::type_info &getObjectType() = 0;

        virtual void show(std::ostream &os, Object &object) = 0;

        virtual size_t getSize() = 0;

        virtual size_t getSizeOfElement() = 0;

        virtual size_t getByteSizeOfObject() = 0;

        virtual Object *allocateNormalizedObject(const std::string &textLine, const std::string &sep) = 0;

        virtual Object *allocateNormalizedObject(const std::vector<double> &obj) = 0;

        virtual Object *allocateNormalizedObject(const std::vector<float> &obj) = 0;

        virtual Object *allocateNormalizedObject(const std::vector<uint8_t> &obj) = 0;

        virtual Object *allocateNormalizedObject(const float *obj, size_t size) = 0;

        virtual PersistentObject *allocateNormalizedPersistentObject(const std::vector<double> &obj) = 0;

        virtual PersistentObject *allocateNormalizedPersistentObject(const std::vector<float> &obj) = 0;

        virtual void deleteObject(Object *po) = 0;

        virtual Object *allocateObject() = 0;

        virtual void remove(size_t id) = 0;

        virtual VectorRepository &getRepository() = 0;

        virtual void setDistanceType(MetricType t) = 0;

        virtual void *getObject(size_t idx) = 0;

        virtual void getObject(size_t idx, std::vector<float> &v) = 0;

        virtual std::vector<float> getObject(Object &object) = 0;

        virtual void getObjects(const std::vector<size_t> &idxs, std::vector<std::vector<float>> &vs) = 0;

        MetricType getDistanceType() { return distanceType; }

        size_t getDimension() { return dimension; }

        size_t getPaddedDimension() { return ((dimension - 1) / 16 + 1) * 16; }

        template<typename T>
        void normalize(T *data, size_t dim) {
            float sum = 0.0;
            for (size_t i = 0; i < dim; i++) {
                sum += static_cast<float>(data[i]) * static_cast<float>(data[i]);
            }
            if (sum == 0.0) {
                std::stringstream msg;
                msg
                        << "VectorSpace::normalize: Error! the object is an invalid zero vector for the cosine similarity or normalized distances.";
                TANN_THROW(msg);
            }
            sum = sqrt(sum);
            for (size_t i = 0; i < dim; i++) {
                data[i] = static_cast<float>(data[i]) / sum;
            }
        }

        int32_t getPrefetchOffset() { return prefetchOffset; }

        int32_t setPrefetchOffset(int offset) {
            if (offset > 0) {
                prefetchOffset = offset;
            }
            if (prefetchOffset <= 0) {
                prefetchOffset = floor(300.0 / (static_cast<float>(getPaddedDimension()) + 30.0) + 1.0);
            }
            return prefetchOffset;
        }

        int32_t getPrefetchSize() { return prefetchSize; }

        int32_t setPrefetchSize(int size) {
            if (size > 0) {
                prefetchSize = size;
            }
            if (prefetchSize <= 0) {
                prefetchSize = getByteSizeOfObject();
            }
            return prefetchSize;
        }

    protected:
        const size_t dimension;
        MetricType distanceType;
        Comparator *comparator;
        bool normalization;
        int32_t prefetchOffset;
        int32_t prefetchSize;
    };


#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    class PersistentObject : public BaseObject {
    public:
      PersistentObject(SharedMemoryAllocator &allocator, tann::VectorSpace *os = 0):array(0) {
        assert(os != 0);
        size_t s = os->getByteSizeOfObject();
        construct(s, allocator);
      }
      PersistentObject(SharedMemoryAllocator &allocator, size_t s):array(0) {
        assert(s != 0);
        construct(s, allocator);
      }

      ~PersistentObject() {}

      uint8_t &at(size_t idx, SharedMemoryAllocator &allocator) const {
        uint8_t *a = (uint8_t *)allocator.getAddr(array);
        return a[idx];
      }
      uint8_t &operator[](size_t idx) const {
        std::cerr << "not implemented" << std::endl;
        assert(0);
        uint8_t *a = 0;
        return a[idx];
      }

      void *getPointer(size_t idx, SharedMemoryAllocator &allocator) {
        uint8_t *a = (uint8_t *)allocator.getAddr(array);
        return a + idx;
      }

      // set v in objectspace to this object using allocator.
      void set(PersistentObject &po, VectorSpace &objectspace);

      static off_t allocate(VectorSpace &objectspace);

      void serializeAsText(std::ostream &os, SharedMemoryAllocator &allocator,
               VectorSpace *objectspace = 0) {
        serializeAsText(os, objectspace);
      }

      void serializeAsText(std::ostream &os, VectorSpace *objectspace = 0);

      void deserializeAsText(std::ifstream &is, SharedMemoryAllocator &allocator,
                 VectorSpace *objectspace = 0) {
        deserializeAsText(is, objectspace);
      }

      void deserializeAsText(std::ifstream &is, VectorSpace *objectspace = 0);

      void serialize(std::ostream &os, SharedMemoryAllocator &allocator,
             VectorSpace *objectspace = 0) {
        std::cerr << "serialize is not implemented" << std::endl;
        assert(0);
      }

    private:
      void construct(size_t s, SharedMemoryAllocator &allocator) {
        assert(array == 0);
        assert(s != 0);
        size_t allocsize = ((s - 1) / 64 + 1) * 64;
        array = allocator.getOffset(new(allocator) uint8_t[allocsize]);
        memset(getPointer(0, allocator), 0, allocsize);
      }
      off_t array;
    };
#endif // NGT_SHARED_MEMORY_ALLOCATOR

}

