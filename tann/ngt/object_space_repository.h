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


#include    "common.h"
#include    "object_space.h"
#include    "object_repository.h"
#include    "primitive_comparator.h"

class ObjectSpace;

namespace tann::ngt {

  template <typename OBJECT_TYPE, typename COMPARE_TYPE> 
    class ObjectSpaceRepository : public ObjectSpace, public ObjectRepository {
  public:

    class ComparatorL1 : public Comparator {
      public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
        ComparatorL1(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
	double operator()(Object &objecta, Object &objectb) {
	  return PrimitiveComparator::compareL1((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
	}
	double operator()(Object &objecta, PersistentObject &objectb) {
	  return PrimitiveComparator::compareL1((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
	}
	double operator()(PersistentObject &objecta, PersistentObject &objectb) {
	  return PrimitiveComparator::compareL1((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
	}
#else
        ComparatorL1(size_t d) : Comparator(d) {}
	double operator()(Object &objecta, Object &objectb) {
	  return PrimitiveComparator::compareL1((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
	}
#endif
    };

    class ComparatorL2 : public Comparator {
      public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
        ComparatorL2(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
	double operator()(Object &objecta, Object &objectb) {
	  return PrimitiveComparator::compareL2((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
	}
	double operator()(Object &objecta, PersistentObject &objectb) {
	  return PrimitiveComparator::compareL2((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
	}
	double operator()(PersistentObject &objecta, PersistentObject &objectb) {
	  return PrimitiveComparator::compareL2((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
	}
#else
        ComparatorL2(size_t d) : Comparator(d) {}
	double operator()(Object &objecta, Object &objectb) {
	  return PrimitiveComparator::compareL2((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
	}
#endif
    };

    class ComparatorNormalizedL2 : public Comparator {
      public:
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
        ComparatorNormalizedL2(size_t d, SharedMemoryAllocator &a) : Comparator(d, a) {}
	double operator()(Object &objecta, Object &objectb) {
	  return PrimitiveComparator::compareNormalizedL2((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
	}
	double operator()(Object &objecta, PersistentObject &objectb) {
	  return PrimitiveComparator::compareNormalizedL2((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
	}
	double operator()(PersistentObject &objecta, PersistentObject &objectb) {
	  return PrimitiveComparator::compareNormalizedL2((OBJECT_TYPE*)&objecta.at(0, allocator), (OBJECT_TYPE*)&objectb.at(0, allocator), dimension);
	}
#else
        ComparatorNormalizedL2(size_t d) : Comparator(d) {}
	double operator()(Object &objecta, Object &objectb) {
	  return PrimitiveComparator::compareNormalizedL2((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
	}
#endif
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
	  return PrimitiveComparator::compareHammingDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
	  return PrimitiveComparator::compareJaccardDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
	  return PrimitiveComparator::compareSparseJaccardDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
	  return PrimitiveComparator::compareAngleDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
	  return PrimitiveComparator::compareNormalizedAngleDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
	  return PrimitiveComparator::compareCosineSimilarity((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
	  return PrimitiveComparator::compareNormalizedCosineSimilarity((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
	  return PrimitiveComparator::comparePoincareDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
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
          return PrimitiveComparator::compareLorentzDistance((OBJECT_TYPE*)&objecta[0], (OBJECT_TYPE*)&objectb[0], dimension);
        }
#endif
    };

    ObjectSpaceRepository(size_t d, const std::type_info &ot, DistanceType t) : ObjectSpace(d), ObjectRepository(d, ot) {
     size_t objectSize = 0;
     if (ot == typeid(uint8_t)) {
       objectSize = sizeof(uint8_t);
     } else if (ot == typeid(float)) {
       objectSize = sizeof(float);
#ifdef NGT_HALF_FLOAT
     } else if (ot == typeid(float16)) {
       objectSize = sizeof(float16);
#endif
     } else {
       std::stringstream msg;
       msg << "ObjectSpace::constructor: Not supported type. " << ot.name();
       NGTThrowException(msg);
     }
     setLength(objectSize * d);
     setPaddedLength(objectSize * ObjectSpace::getPaddedDimension());
     setDistanceType(t);
   }

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    void open(const std::string &f, size_t sharedMemorySize) { ObjectRepository::open(f, sharedMemorySize); }
    void copy(PersistentObject &objecta, PersistentObject &objectb) { objecta = objectb; }

    void show(std::ostream &os, PersistentObject &object) {
      const std::type_info &t = getObjectType();
      if (t == typeid(uint8_t)) {
	auto *optr = static_cast<unsigned char*>(&object.at(0,allocator));
	for (size_t i = 0; i < getDimension(); i++) {
	  os << (int)optr[i] << " ";
	}
#ifdef NGT_HALF_FLOAT
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
	(*po)[i] = spo.at(i,ObjectRepository::allocator);
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
      assert(ObjectSpace::dimension != 0);
      distanceType = t; 
      switch (distanceType) {
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
      case DistanceTypeL1:
	comparator = new ObjectSpaceRepository::ComparatorL1(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypeL2:
	comparator = new ObjectSpaceRepository::ComparatorL2(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypeNormalizedL2:
	comparator = new ObjectSpaceRepository::ComparatorNormalizedL2(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	normalization = true;
	break;
      case DistanceTypeHamming:
	comparator = new ObjectSpaceRepository::ComparatorHammingDistance(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypeJaccard:
	comparator = new ObjectSpaceRepository::ComparatorJaccardDistance(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypeSparseJaccard:
	comparator = new ObjectSpaceRepository::ComparatorSparseJaccardDistance(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	setSparse();
	break;
      case DistanceTypeAngle:
	comparator = new ObjectSpaceRepository::ComparatorAngleDistance(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypeCosine:
	comparator = new ObjectSpaceRepository::ComparatorCosineSimilarity(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypePoincare:  // added by Nyapicom
	comparator = new ObjectSpaceRepository::ComparatorPoincareDistance(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypeLorentz:  // added by Nyapicom
	comparator = new ObjectSpaceRepository::ComparatorLorentzDistance(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	break;
      case DistanceTypeNormalizedAngle:
	comparator = new ObjectSpaceRepository::ComparatorNormalizedAngleDistance(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	normalization = true;
	break;
      case DistanceTypeNormalizedCosine:
	comparator = new ObjectSpaceRepository::ComparatorNormalizedCosineSimilarity(ObjectSpace::getPaddedDimension(), ObjectRepository::allocator);
	normalization = true;
	break;
#else
      case DistanceTypeL1:
	comparator = new ObjectSpaceRepository::ComparatorL1(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypeL2:
	comparator = new ObjectSpaceRepository::ComparatorL2(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypeNormalizedL2:
	comparator = new ObjectSpaceRepository::ComparatorNormalizedL2(ObjectSpace::getPaddedDimension());
	normalization = true;
	break;
      case DistanceTypeHamming:
	comparator = new ObjectSpaceRepository::ComparatorHammingDistance(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypeJaccard:
	comparator = new ObjectSpaceRepository::ComparatorJaccardDistance(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypeSparseJaccard:
	comparator = new ObjectSpaceRepository::ComparatorSparseJaccardDistance(ObjectSpace::getPaddedDimension());
	setSparse();
	break;
      case DistanceTypeAngle:
	comparator = new ObjectSpaceRepository::ComparatorAngleDistance(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypeCosine:
	comparator = new ObjectSpaceRepository::ComparatorCosineSimilarity(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypePoincare:  // added by Nyapicom
	comparator = new ObjectSpaceRepository::ComparatorPoincareDistance(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypeLorentz:  // added by Nyapicom
	comparator = new ObjectSpaceRepository::ComparatorLorentzDistance(ObjectSpace::getPaddedDimension());
	break;
      case DistanceTypeNormalizedAngle:
	comparator = new ObjectSpaceRepository::ComparatorNormalizedAngleDistance(ObjectSpace::getPaddedDimension());
	normalization = true;
	break;
      case DistanceTypeNormalizedCosine:
	comparator = new ObjectSpaceRepository::ComparatorNormalizedCosineSimilarity(ObjectSpace::getPaddedDimension());
	normalization = true;
	break;
#endif
      default:
	std::cerr << "Distance type is not specified" << std::endl;
	assert(distanceType != DistanceTypeNone);
	abort();
      }
    }


    void serialize(const std::string &ofile) { ObjectRepository::serialize(ofile, this); }
    void deserialize(const std::string &ifile) { ObjectRepository::deserialize(ifile, this); }
    void serializeAsText(const std::string &ofile) { ObjectRepository::serializeAsText(ofile, this); }
    void deserializeAsText(const std::string &ifile) { ObjectRepository::deserializeAsText(ifile, this); }
    void readText(std::istream &is, size_t dataSize) { ObjectRepository::readText(is, dataSize); }
    void appendText(std::istream &is, size_t dataSize) { ObjectRepository::appendText(is, dataSize); }

    void append(const float *data, size_t dataSize) { ObjectRepository::append(data, dataSize); }
    void append(const double *data, size_t dataSize) { ObjectRepository::append(data, dataSize); }


#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    PersistentObject *allocatePersistentObject(Object &obj) {
      return ObjectRepository::allocatePersistentObject(obj);
    }
    size_t insert(PersistentObject *obj) { return ObjectRepository::insert(obj); }
#else
    size_t insert(Object *obj) { return ObjectRepository::insert(obj); }
#endif

    void remove(size_t id) { ObjectRepository::remove(id); }

    void linearSearch(Object &query, double radius, size_t size, ObjectSpace::ResultSet &results) {
      if (!results.empty()) {
	NGTThrowException("lenearSearch: results is not empty");
      }
#ifndef NGT_PREFETCH_DISABLED
      size_t byteSizeOfObject = getByteSizeOfObject();
      const size_t prefetchOffset = getPrefetchOffset();
#endif
      ObjectRepository &rep = *this;
      for (size_t idx = 0; idx < rep.size(); idx++) {
#ifndef NGT_PREFETCH_DISABLED
	if (idx + prefetchOffset < rep.size() && rep[idx + prefetchOffset] != 0) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	  MemoryCache::prefetch((unsigned char*)&(*static_cast<PersistentObject*>(ObjectRepository::get(idx + prefetchOffset))), byteSizeOfObject);
#else
	  MemoryCache::prefetch((unsigned char*)&(*static_cast<PersistentObject*>(rep[idx + prefetchOffset]))[0], byteSizeOfObject);
#endif
	}
#endif
	if (rep[idx] == 0) {
	  continue;
	}
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
	Distance d = (*comparator)((Object&)query, (PersistentObject&)*rep[idx]);
#else
	Distance d = (*comparator)((Object&)query, (Object&)*rep[idx]);
#endif
	if (radius < 0.0 || d <= radius) {
	  tann::ngt::ObjectDistance obj(idx, d);
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
	msg << "tann::ngt::ObjectSpaceRepository: The specified ID is out of the range. The object ID should be greater than zero. " << idx << ":" << ObjectRepository::size() << ".";
	NGTThrowException(msg);
      }
      PersistentObject &obj = *(*this)[idx];
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
      return reinterpret_cast<OBJECT_TYPE*>(&obj.at(0, allocator));
#else
      return reinterpret_cast<OBJECT_TYPE*>(&obj[0]);
#endif
    }

    void getObject(size_t idx, std::vector<float> &v) {
      OBJECT_TYPE *obj = static_cast<OBJECT_TYPE*>(getObject(idx));
      size_t dim = getDimension();
      v.resize(dim);
      for (size_t i = 0; i < dim; i++) {
	v[i] = static_cast<float>(obj[i]);
      }
    }

    std::vector<float> getObject(Object &object) {
      std::vector<float> v;
      OBJECT_TYPE *obj = static_cast<OBJECT_TYPE*>(object.getPointer());
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
      ObjectSpace::normalize(obj, ObjectSpace::dimension);
    }
#endif
    void normalize(Object &object) {
      OBJECT_TYPE *obj = (OBJECT_TYPE*)&object[0];
      ObjectSpace::normalize(obj, ObjectSpace::dimension);
    }

    Object *allocateObject() { return ObjectRepository::allocateObject(); }
    void deleteObject(Object *po) { ObjectRepository::deleteObject(po); }

    Object *allocateNormalizedObject(const std::string &textLine, const std::string &sep) {
      Object *allocatedObject = ObjectRepository::allocateObject(textLine, sep);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }
    Object *allocateNormalizedObject(const std::vector<double> &obj) {
      Object *allocatedObject = ObjectRepository::allocateObject(obj);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }
    Object *allocateNormalizedObject(const std::vector<float> &obj) {
      Object *allocatedObject = ObjectRepository::allocateObject(obj);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }
    Object *allocateNormalizedObject(const std::vector<uint8_t> &obj) {
      Object *allocatedObject = ObjectRepository::allocateObject(obj);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }
    Object *allocateNormalizedObject(const float *obj, size_t size) {
      Object *allocatedObject = ObjectRepository::allocateObject(obj, size);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }

    PersistentObject *allocateNormalizedPersistentObject(const std::vector<double> &obj) {
      PersistentObject *allocatedObject = ObjectRepository::allocatePersistentObject(obj);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }

    PersistentObject *allocateNormalizedPersistentObject(const std::vector<float> &obj) {
      PersistentObject *allocatedObject = ObjectRepository::allocatePersistentObject(obj);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }

    PersistentObject *allocateNormalizedPersistentObject(const std::vector<uint8_t> &obj) {
      PersistentObject *allocatedObject = ObjectRepository::allocatePersistentObject(obj);
      if (normalization) {
	normalize(*allocatedObject);
      }
      return allocatedObject;
    }

    size_t getSize() { return ObjectRepository::size(); }
    size_t getSizeOfElement() { return sizeof(OBJECT_TYPE); }
    const std::type_info &getObjectType() { return typeid(OBJECT_TYPE); };
    size_t getByteSizeOfObject() { return getByteSize(); }

    ObjectRepository &getRepository() { return *this; };

    void show(std::ostream &os, Object &object) {
      const std::type_info &t = getObjectType();
      if (t == typeid(uint8_t)) {
	unsigned char *optr = static_cast<unsigned char*>(&object[0]);
	for (size_t i = 0; i < getDimension(); i++) {
	  os << (int)optr[i] << " ";
	}
      } else if (t == typeid(float)) {
	float *optr = reinterpret_cast<float*>(&object[0]);
	for (size_t i = 0; i < getDimension(); i++) {
	  os << optr[i] << " ";
	}
#ifdef NGT_HALF_FLOAT
      } else if (t == typeid(float16)) {
	float16 *optr = reinterpret_cast<float16*>(&object[0]);
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
  inline void PersistentObject::set(PersistentObject &po, ObjectSpace &objectspace) {
    SharedMemoryAllocator &allocator = objectspace.getRepository().getAllocator();
    uint8_t *src = (uint8_t *)&po.at(0, allocator);
    uint8_t *dst = (uint8_t *)&(*this).at(0, allocator);
    memcpy(dst, src, objectspace.getByteSizeOfObject());
  }

  inline off_t PersistentObject::allocate(ObjectSpace &objectspace) {
    SharedMemoryAllocator &allocator = objectspace.getRepository().getAllocator();
    return allocator.getOffset(new(allocator) PersistentObject(allocator, &objectspace));
  }

  inline void PersistentObject::serializeAsText(std::ostream &os, ObjectSpace *objectspace) { 
    assert(objectspace != 0);
    SharedMemoryAllocator &allocator = objectspace->getRepository().getAllocator();
    const std::type_info &t = objectspace->getObjectType();
    void *ref = &(*this).at(0, allocator);
    size_t dimension = objectspace->getDimension();
    if (t == typeid(uint8_t)) {
      tann::ngt::Serializer::writeAsText(os, (uint8_t*)ref, dimension);
    } else if (t == typeid(float)) {
      tann::ngt::Serializer::writeAsText(os, (float*)ref, dimension);
#ifdef NGT_HALF_FLOAT
    } else if (t == typeid(float16)) {
      tann::ngt::Serializer::writeAsText(os, (float16*)ref, dimension);
#endif
    } else if (t == typeid(double)) {
      tann::ngt::Serializer::writeAsText(os, (double*)ref, dimension);
    } else if (t == typeid(uint16_t)) {
      tann::ngt::Serializer::writeAsText(os, (uint16_t*)ref, dimension);
    } else if (t == typeid(uint32_t)) {
      tann::ngt::Serializer::writeAsText(os, (uint32_t*)ref, dimension);
    } else {
      std::cerr << "ObjectT::serializeAsText: not supported data type. [" << t.name() << "]" << std::endl;
      assert(0);
    }
  }

  inline void PersistentObject::deserializeAsText(std::ifstream &is, ObjectSpace *objectspace) {
    assert(objectspace != 0);
    SharedMemoryAllocator &allocator = objectspace->getRepository().getAllocator();
    const std::type_info &t = objectspace->getObjectType();
    size_t dimension = objectspace->getDimension();
    void *ref = &(*this).at(0, allocator);
    assert(ref != 0);
    if (t == typeid(uint8_t)) {
      tann::ngt::Serializer::readAsText(is, (uint8_t*)ref, dimension);
    } else if (t == typeid(float)) {
      tann::ngt::Serializer::readAsText(is, (float*)ref, dimension);
#ifdef NGT_HALF_FLOAT
    } else if (t == typeid(float16)) {
      tann::ngt::Serializer::readAsText(is, (float16*)ref, dimension);
#endif
    } else if (t == typeid(double)) {
      tann::ngt::Serializer::readAsText(is, (double*)ref, dimension);
    } else if (t == typeid(uint16_t)) {
      tann::ngt::Serializer::readAsText(is, (uint16_t*)ref, dimension);
    } else if (t == typeid(uint32_t)) {
      tann::ngt::Serializer::readAsText(is, (uint32_t*)ref, dimension);
    } else {
      std::cerr << "Object::deserializeAsText: not supported data type. [" << t.name() << "]" << std::endl;
      assert(0);
    }
  }

#endif
} // namespace tann::ngt

