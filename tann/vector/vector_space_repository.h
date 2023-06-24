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


//#include    "common.h"
#include "tann/vector/vector_space.h"
#include "tann/vector/vector_repository.h"
#include "tann/vector/primitive_comparator.h"
#include "tann/vector/vector_view.h"

class VectorSpace;

namespace tann {

    template<typename OBJECT_TYPE, typename COMPARE_TYPE>
    class VectorSpaceRepository : public VectorSpace, public VectorRepository {
    public:

        class ComparatorL1 : public Comparator {
        public:

            ComparatorL1(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareL1((OBJECT_TYPE *) &objecta[0], (OBJECT_TYPE *) &objectb[0],
                                                      dimension);
            }

        };

        class ComparatorL2 : public Comparator {
        public:
            ComparatorL2(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareL2((OBJECT_TYPE *) &objecta[0], (OBJECT_TYPE *) &objectb[0],
                                                      dimension);
            }
        };

        class ComparatorNormalizedL2 : public Comparator {
        public:

            ComparatorNormalizedL2(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareNormalizedL2((OBJECT_TYPE *) &objecta[0],
                                                                (OBJECT_TYPE *) &objectb[0], dimension);
            }

        };

        class ComparatorHammingDistance : public Comparator {
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorHammingDistance(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
        double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareHammingDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareHammingDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareHammingDistance((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
#else

            ComparatorHammingDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareHammingDistance((OBJECT_TYPE *) &objecta[0],
                                                                   (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorJaccardDistance : public Comparator {
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorJaccardDistance(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
            double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareJaccardDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareJaccardDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareJaccardDistance((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
#else

            ComparatorJaccardDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareJaccardDistance((OBJECT_TYPE *) &objecta[0],
                                                                   (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorSparseJaccardDistance : public Comparator {
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorSparseJaccardDistance(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
            double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareSparseJaccardDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareSparseJaccardDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareSparseJaccardDistance((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
#else

            ComparatorSparseJaccardDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareSparseJaccardDistance((OBJECT_TYPE *) &objecta[0],
                                                                         (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorAngleDistance : public Comparator {
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorAngleDistance(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
        double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareAngleDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareAngleDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareAngleDistance((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
#else

            ComparatorAngleDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareAngleDistance((OBJECT_TYPE *) &objecta[0],
                                                                 (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorNormalizedAngleDistance : public Comparator {
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorNormalizedAngleDistance(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
        double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareNormalizedAngleDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareNormalizedAngleDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareNormalizedAngleDistance((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
#else

            ComparatorNormalizedAngleDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareNormalizedAngleDistance((OBJECT_TYPE *) &objecta[0],
                                                                           (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorCosineSimilarity : public Comparator {
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorCosineSimilarity(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
        double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareCosineSimilarity((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareCosineSimilarity((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareCosineSimilarity((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
#else

            ComparatorCosineSimilarity(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareCosineSimilarity((OBJECT_TYPE *) &objecta[0],
                                                                    (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorNormalizedCosineSimilarity : public Comparator {
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorNormalizedCosineSimilarity(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
        double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareNormalizedCosineSimilarity((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareNormalizedCosineSimilarity((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareNormalizedCosineSimilarity((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
        }
#else

            ComparatorNormalizedCosineSimilarity(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareNormalizedCosineSimilarity((OBJECT_TYPE *) &objecta[0],
                                                                              (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorPoincareDistance : public Comparator {  // added by Nyapicom
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorPoincareDistance(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
            double operator()(Object &objecta, Object &objectb) {
              return PrimitiveComparator::comparePoincareDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
            }
        double operator()(Object &objecta, PersistentObject &objectb) {
              return PrimitiveComparator::comparePoincareDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
            }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
              return PrimitiveComparator::comparePoincareDistance((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
            }
#else

            ComparatorPoincareDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::comparePoincareDistance((OBJECT_TYPE *) &objecta[0],
                                                                    (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        class ComparatorLorentzDistance : public Comparator {  // added by Nyapicom
        public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            ComparatorLorentzDistance(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
            double operator()(Object &objecta, Object &objectb) {
          return PrimitiveComparator::compareLorentzDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
        double operator()(Object &objecta, PersistentObject &objectb) {
          return PrimitiveComparator::compareLorentzDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
            }
        double operator()(PersistentObject &objecta, PersistentObject &objectb) {
              return PrimitiveComparator::compareLorentzDistance((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
            }
#else

            ComparatorLorentzDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareLorentzDistance((OBJECT_TYPE *) &objecta[0],
                                                                   (OBJECT_TYPE *) &objectb[0], dimension);
            }

#endif
        };

        VectorSpaceRepository(size_t d, const std::type_info &ot, DistanceType t) : VectorSpace(d),
                                                                                    VectorRepository(d, ot) {
            size_t objectSize = 0;
            if (ot == typeid(uint8_t)) {
                objectSize = sizeof(uint8_t);
            } else if (ot == typeid(float)) {
                objectSize = sizeof(float);
#ifdef TANN_ENABLE_HALF_FLOAT
            } else if (ot == typeid(float16)) {
                objectSize = sizeof(float16);
#endif
            } else {
                std::stringstream msg;
                msg << "VectorSpace::constructor: Not supported type. " << ot.name();
                TANN_THROW(msg);
            }
            setLength(objectSize * d);
            setPaddedLength(objectSize * VectorSpace::getPaddedDimension());
            setDistanceType(t);
        }

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
        void open(const std::string &f, size_t sharedMemorySize) { VectorRepository::open(f, sharedMemorySize); }
        void copy(PersistentObject &objecta, PersistentObject &objectb) { objecta = objectb; }

        void show(std::ostream &os, PersistentObject &object) {
          const std::type_info &t = getObjectType();
          if (t == typeid(uint8_t)) {
        auto *optr = static_cast<unsigned char*>(&object.at(0,allocator));
        for (size_t i = 0; i < getDimension(); i++) {
          os << (int)optr[i] << " ";
        }
#ifdef TANN_ENABLE_HALF_FLOAT
          } else if (t == typeid(float16)) {
        auto *optr = reinterpret_cast<float16*>(&object.at(0,allocator));
        for (size_t i = 0; i < getDimension(); i++) {
          os << optr[i] << " ";
        }
#endif
          } else if (t == typeid(float)) {
        auto *optr = reinterpret_cast<float*>(&object.at(0,allocator));
        for (size_t i = 0; i < getDimension(); i++) {
          os << optr[i] << " ";
        }
          } else {
        os << " not implement for the type.";
          }
        }

        Object *allocateObject(Object &o) {
          Object *po = new Object(getByteSizeOfObject());
          for (size_t i = 0; i < getByteSizeOfObject(); i++) {
        (*po)[i] = o[i];
          }
          return po;
        }
        Object *allocateObject(PersistentObject &o) {
          PersistentObject &spo = (PersistentObject &)o;
          Object *po = new Object(getByteSizeOfObject());
          for (size_t i = 0; i < getByteSizeOfObject(); i++) {
        (*po)[i] = spo.at(i,VectorRepository::allocator);
          }
          return (Object*)po;
        }
        void deleteObject(PersistentObject *po) {
          delete po;
        }
#endif // NGT_SHARED_MEMORY_ALLOCATOR

        void copy(Object &objecta, Object &objectb) {
            objecta.copy(objectb, getByteSizeOfObject());
        }

        void setDistanceType(DistanceType t) {
            if (comparator != 0) {
                delete comparator;
            }
            assert(VectorSpace::dimension != 0);
            distanceType = t;
            switch (distanceType) {
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
                case DistanceTypeL1:
              comparator = new VectorSpaceRepository::ComparatorL1(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypeL2:
              comparator = new VectorSpaceRepository::ComparatorL2(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypeNormalizedL2:
              comparator = new VectorSpaceRepository::ComparatorNormalizedL2(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              normalization = true;
              break;
                case DistanceTypeHamming:
              comparator = new VectorSpaceRepository::ComparatorHammingDistance(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypeJaccard:
              comparator = new VectorSpaceRepository::ComparatorJaccardDistance(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypeSparseJaccard:
              comparator = new VectorSpaceRepository::ComparatorSparseJaccardDistance(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              setSparse();
              break;
                case DistanceTypeAngle:
              comparator = new VectorSpaceRepository::ComparatorAngleDistance(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypeCosine:
              comparator = new VectorSpaceRepository::ComparatorCosineSimilarity(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypePoincare:  // added by Nyapicom
              comparator = new VectorSpaceRepository::ComparatorPoincareDistance(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypeLorentz:  // added by Nyapicom
              comparator = new VectorSpaceRepository::ComparatorLorentzDistance(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              break;
                case DistanceTypeNormalizedAngle:
              comparator = new VectorSpaceRepository::ComparatorNormalizedAngleDistance(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              normalization = true;
              break;
                case DistanceTypeNormalizedCosine:
              comparator = new VectorSpaceRepository::ComparatorNormalizedCosineSimilarity(VectorSpace::getPaddedDimension(), VectorRepository::allocator);
              normalization = true;
              break;
#else
                case DistanceTypeL1:
                    comparator = new VectorSpaceRepository::ComparatorL1(VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypeL2:
                    comparator = new VectorSpaceRepository::ComparatorL2(VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypeNormalizedL2:
                    comparator = new VectorSpaceRepository::ComparatorNormalizedL2(VectorSpace::getPaddedDimension());
                    normalization = true;
                    break;
                case DistanceTypeHamming:
                    comparator = new VectorSpaceRepository::ComparatorHammingDistance(
                            VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypeJaccard:
                    comparator = new VectorSpaceRepository::ComparatorJaccardDistance(
                            VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypeSparseJaccard:
                    comparator = new VectorSpaceRepository::ComparatorSparseJaccardDistance(
                            VectorSpace::getPaddedDimension());
                    setSparse();
                    break;
                case DistanceTypeAngle:
                    comparator = new VectorSpaceRepository::ComparatorAngleDistance(VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypeCosine:
                    comparator = new VectorSpaceRepository::ComparatorCosineSimilarity(
                            VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypePoincare:  // added by Nyapicom
                    comparator = new VectorSpaceRepository::ComparatorPoincareDistance(
                            VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypeLorentz:  // added by Nyapicom
                    comparator = new VectorSpaceRepository::ComparatorLorentzDistance(
                            VectorSpace::getPaddedDimension());
                    break;
                case DistanceTypeNormalizedAngle:
                    comparator = new VectorSpaceRepository::ComparatorNormalizedAngleDistance(
                            VectorSpace::getPaddedDimension());
                    normalization = true;
                    break;
                case DistanceTypeNormalizedCosine:
                    comparator = new VectorSpaceRepository::ComparatorNormalizedCosineSimilarity(
                            VectorSpace::getPaddedDimension());
                    normalization = true;
                    break;
#endif
                default:
                    std::cerr << "Distance type is not specified" << std::endl;
                    assert(distanceType != DistanceTypeNone);
                    abort();
            }
        }


        void serialize(const std::string &ofile) { VectorRepository::serialize(ofile, this); }

        void deserialize(const std::string &ifile) { VectorRepository::deserialize(ifile, this); }

        void serializeAsText(const std::string &ofile) { VectorRepository::serializeAsText(ofile, this); }

        void deserializeAsText(const std::string &ifile) { VectorRepository::deserializeAsText(ifile, this); }

        void readText(std::istream &is, size_t dataSize) { VectorRepository::readText(is, dataSize); }

        void appendText(std::istream &is, size_t dataSize) { VectorRepository::appendText(is, dataSize); }

        void append(const float *data, size_t dataSize) { VectorRepository::append(data, dataSize); }

        void append(const double *data, size_t dataSize) { VectorRepository::append(data, dataSize); }


#ifdef NGT_SHARED_MEMORY_ALLOCATOR
        PersistentObject *allocatePersistentObject(Object &obj) {
          return VectorRepository::allocatePersistentObject(obj);
        }
        size_t insert(PersistentObject *obj) { return VectorRepository::insert(obj); }
#else

        size_t insert(Object *obj) { return VectorRepository::insert(obj); }

#endif

        void remove(size_t id) { VectorRepository::remove(id); }

        void linearSearch(Object &query, double radius, size_t size, VectorSpace::ResultSet &results) {
            if (!results.empty()) {
                TANN_THROW("lenearSearch: results is not empty");
            }
#ifndef NGT_PREFETCH_DISABLED
            size_t byteSizeOfObject = getByteSizeOfObject();
            const size_t prefetchOffset = getPrefetchOffset();
#endif
            VectorRepository &rep = *this;
            for (size_t idx = 0; idx < rep.size(); idx++) {
#ifndef NGT_PREFETCH_DISABLED
                if (idx + prefetchOffset < rep.size() && rep[idx + prefetchOffset] != 0) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    MemoryCache::prefetch((unsigned char*)&(*static_cast<PersistentObject*>(VectorRepository::get(idx + prefetchOffset))), byteSizeOfObject);
#else
                    MemoryCache::prefetch(
                            (unsigned char *) &(*static_cast<PersistentObject *>(rep[idx + prefetchOffset]))[0],
                            byteSizeOfObject);
#endif
                }
#endif
                if (rep[idx] == 0) {
                    continue;
                }
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
                Distance d = (*comparator)((Object&)query, (PersistentObject&)*rep[idx]);
#else
                Distance d = (*comparator)((Object &) query, (Object &) *rep[idx]);
#endif
                if (radius < 0.0 || d <= radius) {
                    tann::VectorDistance obj(idx, d);
                    results.push(obj);
                    if (results.size() > size) {
                        results.pop();
                    }
                }
            }
            return;
        }

        void *getObject(size_t idx) {
            if (isEmpty(idx)) {
                std::stringstream msg;
                msg
                        << "tann::VectorSpaceRepository: The specified ID is out of the range. The object ID should be greater than zero. "
                        << idx << ":" << VectorRepository::size() << ".";
                TANN_THROW(msg);
            }
            PersistentObject &obj = *(*this)[idx];
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
            return reinterpret_cast<OBJECT_TYPE*>(&obj.at(0, allocator));
#else
            return reinterpret_cast<OBJECT_TYPE *>(&obj[0]);
#endif
        }

        void getObject(size_t idx, std::vector<float> &v) {
            OBJECT_TYPE *obj = static_cast<OBJECT_TYPE *>(getObject(idx));
            size_t dim = getDimension();
            v.resize(dim);
            for (size_t i = 0; i < dim; i++) {
                v[i] = static_cast<float>(obj[i]);
            }
        }

        std::vector<float> getObject(Object &object) {
            std::vector<float> v;
            OBJECT_TYPE *obj = static_cast<OBJECT_TYPE *>(object.getPointer());
            size_t dim = getDimension();
            v.resize(dim);
            for (size_t i = 0; i < dim; i++) {
                v[i] = static_cast<float>(obj[i]);
            }
            return v;
        }

        void getObjects(const std::vector<size_t> &idxs, std::vector<std::vector<float>> &vs) {
            vs.resize(idxs.size());
            auto v = vs.begin();
            for (auto idx = idxs.begin(); idx != idxs.end(); idx++, v++) {
                getObject(*idx, *v);
            }
        }

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
        void normalize(PersistentObject &object) {
          OBJECT_TYPE *obj = (OBJECT_TYPE*)&object.at(0, getRepository().getAllocator());
          VectorSpace::normalize(obj, VectorSpace::dimension);
        }
#endif

        void normalize(Object &object) {
            OBJECT_TYPE *obj = (OBJECT_TYPE *) &object[0];
            VectorSpace::normalize(obj, VectorSpace::dimension);
        }

        Object *allocateObject() { return VectorRepository::allocateObject(); }

        void deleteObject(Object *po) { VectorRepository::deleteObject(po); }

        Object *allocateNormalizedObject(const std::string &textLine, const std::string &sep) {
            Object *allocatedObject = VectorRepository::allocateObject(textLine, sep);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        Object *allocateNormalizedObject(const std::vector<double> &obj) {
            Object *allocatedObject = VectorRepository::allocateObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        Object *allocateNormalizedObject(const std::vector<float> &obj) {
            Object *allocatedObject = VectorRepository::allocateObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        Object *allocateNormalizedObject(const std::vector<uint8_t> &obj) {
            Object *allocatedObject = VectorRepository::allocateObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        Object *allocateNormalizedObject(const float *obj, size_t size) {
            Object *allocatedObject = VectorRepository::allocateObject(obj, size);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        PersistentObject *allocateNormalizedPersistentObject(const std::vector<double> &obj) {
            PersistentObject *allocatedObject = VectorRepository::allocatePersistentObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        PersistentObject *allocateNormalizedPersistentObject(const std::vector<float> &obj) {
            PersistentObject *allocatedObject = VectorRepository::allocatePersistentObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        PersistentObject *allocateNormalizedPersistentObject(const std::vector<uint8_t> &obj) {
            PersistentObject *allocatedObject = VectorRepository::allocatePersistentObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        size_t getSize() { return VectorRepository::size(); }

        size_t getSizeOfElement() { return sizeof(OBJECT_TYPE); }

        const std::type_info &getObjectType() { return typeid(OBJECT_TYPE); };

        size_t getByteSizeOfObject() { return getByteSize(); }

        VectorRepository &getRepository() { return *this; };

        void show(std::ostream &os, Object &object) {
            const std::type_info &t = getObjectType();
            if (t == typeid(uint8_t)) {
                unsigned char *optr = static_cast<unsigned char *>(&object[0]);
                for (size_t i = 0; i < getDimension(); i++) {
                    os << (int) optr[i] << " ";
                }
            } else if (t == typeid(float)) {
                float *optr = reinterpret_cast<float *>(&object[0]);
                for (size_t i = 0; i < getDimension(); i++) {
                    os << optr[i] << " ";
                }
#ifdef TANN_ENABLE_HALF_FLOAT
            } else if (t == typeid(float16)) {
                float16 *optr = reinterpret_cast<float16 *>(&object[0]);
                for (size_t i = 0; i < getDimension(); i++) {
                    os << optr[i] << " ";
                }
#endif
            } else {
                os << " not implement for the type.";
            }
        }

    };

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    // set v in objectspace to this object using allocator.
    inline void PersistentObject::set(PersistentObject &po, VectorSpace &objectspace) {
      SharedMemoryAllocator &allocator = objectspace.getRepository().getAllocator();
      uint8_t *src = (uint8_t *)&po.at(0, allocator);
      uint8_t *dst = (uint8_t *)&(*this).at(0, allocator);
      memcpy(dst, src, objectspace.getByteSizeOfObject());
    }

    inline off_t PersistentObject::allocate(VectorSpace &objectspace) {
      SharedMemoryAllocator &allocator = objectspace.getRepository().getAllocator();
      return allocator.getOffset(new(allocator) PersistentObject(allocator, &objectspace));
    }

    inline void PersistentObject::serializeAsText(std::ostream &os, VectorSpace *objectspace) {
      assert(objectspace != 0);
      SharedMemoryAllocator &allocator = objectspace->getRepository().getAllocator();
      const std::type_info &t = objectspace->getObjectType();
      void *ref = &(*this).at(0, allocator);
      size_t dimension = objectspace->getDimension();
      if (t == typeid(uint8_t)) {
        tann::Serializer::writeAsText(os, (uint8_t*)ref, dimension);
      } else if (t == typeid(float)) {
        tann::Serializer::writeAsText(os, (float*)ref, dimension);
#ifdef TANN_ENABLE_HALF_FLOAT
      } else if (t == typeid(float16)) {
        tann::Serializer::writeAsText(os, (float16*)ref, dimension);
#endif
      } else if (t == typeid(double)) {
        tann::Serializer::writeAsText(os, (double*)ref, dimension);
      } else if (t == typeid(uint16_t)) {
        tann::Serializer::writeAsText(os, (uint16_t*)ref, dimension);
      } else if (t == typeid(uint32_t)) {
        tann::Serializer::writeAsText(os, (uint32_t*)ref, dimension);
      } else {
        std::cerr << "ObjectT::serializeAsText: not supported data type. [" << t.name() << "]" << std::endl;
        assert(0);
      }
    }

    inline void PersistentObject::deserializeAsText(std::ifstream &is, VectorSpace *objectspace) {
      assert(objectspace != 0);
      SharedMemoryAllocator &allocator = objectspace->getRepository().getAllocator();
      const std::type_info &t = objectspace->getObjectType();
      size_t dimension = objectspace->getDimension();
      void *ref = &(*this).at(0, allocator);
      assert(ref != 0);
      if (t == typeid(uint8_t)) {
        tann::Serializer::readAsText(is, (uint8_t*)ref, dimension);
      } else if (t == typeid(float)) {
        tann::Serializer::readAsText(is, (float*)ref, dimension);
#ifdef TANN_ENABLE_HALF_FLOAT
      } else if (t == typeid(float16)) {
        tann::Serializer::readAsText(is, (float16*)ref, dimension);
#endif
      } else if (t == typeid(double)) {
        tann::Serializer::readAsText(is, (double*)ref, dimension);
      } else if (t == typeid(uint16_t)) {
        tann::Serializer::readAsText(is, (uint16_t*)ref, dimension);
      } else if (t == typeid(uint32_t)) {
        tann::Serializer::readAsText(is, (uint32_t*)ref, dimension);
      } else {
        std::cerr << "Object::deserializeAsText: not supported data type. [" << t.name() << "]" << std::endl;
        assert(0);
      }
    }

#endif
} // namespace tann

