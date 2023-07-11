//
// Copyright (C) 2016 Yahoo Japan Corporation
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

#include "tann/ngt/index.h"
#include "tann/ngt/array_file.h"
#include "tann/ngt/clustering.h"
#include <unordered_map>
#include "tann/ngt/ngtq/object_file.h"
#include "tann/vector/dynamic_vector.h"


#if defined(NGT_SHARED_MEMORY_ALLOCATOR) || defined(NGT_QBG_DISABLED)
#undef NGTQ_QBG
#else
#define NGTQ_QBG
#endif

#define NGTQ_VECTOR_OBJECT

#ifdef NGTQ_STATIC_OBJECT_FILE
#define MULTIPLE_OBJECT_LISTS
#endif

#define NGTQBG_MIN

#define MULTIPLE_OBJECT_LISTS
#define NGTQG_ROTATION
#define NGTQ_BLAS_FOR_ROTATION 
#define NGTQG_ROTATED_GLOBAL_CODEBOOKS      

#define NGTQ_UINT8_LUT
#define NGTQ_SIMD_BLOCK_SIZE	16	
#define NGTQ_BATCH_SIZE		2	
#define NGTQ_UINT4_OBJECT  
#define NGTQ_TOTAL_SCALE_OFFSET_COMPRESSION
#define NGTQG_PREFETCH
#if defined(NGT_AVX512)
#define NGTQG_AVX512	
#warning "AVX512 is available for NGTQG"
#elif defined(NGT_AVX2)
#define NGTQG_AVX2
#warning "AVX2 is available for NGTQG"
#else
#undef NGTQG_PREFETCH
#warning "SIMD is not available for NGTQG. NGTQG might not work well."
#endif




#ifdef NGT_SHARED_MEMORY_ALLOCATOR
#define	NGTQ_SHARED_INVERTED_INDEX	
#endif

extern "C" {
  void sgemv_(char *trans, int *m, int *n,
	      float *alpha, float *A, int *ldA, float *x, int *incx,
	      float *beta , float *y, int *incy);
}

namespace NGTQ {

class Rotation : public std::vector<float> {
  typedef std::vector<float> PARENT;
 public:
  Rotation& operator=(const std::vector<float> &r) {
    PARENT::operator=(r);
    dim = sqrt(PARENT::size());
    if ((dim * dim) != PARENT::size()) {
      std::cerr << "Rotation: Fatal inner error! Invalid data. " << dim * dim << ":" << PARENT::size() << std::endl;
      abort();
    }
    return *this;
  }
#ifdef NGTQ_BLAS_FOR_ROTATION
  void mulBlas(float *a) {
    char trans = 'N';
    int m = dim;
    float alpha = 1.0;
    float *matrix = PARENT::data();
    int incx = 1;
    float beta = 0.0;
    float *y = new float[dim];
    int incy = 1;
    sgemv_(&trans, &m, &m, &alpha, matrix, &m, a, &incx, &beta, y, &incy);
    memcpy(a, y, sizeof(float) * dim);
    delete[] y;
  }
  void mulBlas(tann::float16 *a) {
    float *floatv = new float[dim];
    for (size_t d = 0; d < dim; d++) {
      floatv[d] = a[d];
    }
    char trans = 'N';
    int m = dim;
    float alpha = 1.0;
    float *matrix = PARENT::data();
    int incx = 1;
    float beta = 0.0;
    float *y = new float[dim];
    int incy = 1;
    sgemv_(&trans, &m, &m, &alpha, matrix, &m, floatv, &incx, &beta, y, &incy);
    for (size_t d = 0; d < dim; d++) {
      a[d] = y[d];
    }
    delete[] y;
    delete[] floatv;
  }
#endif
  template <typename T>
  void mul(T *a) {
    if (PARENT::size() == 0) {
      return;
    }
#ifdef NGTQ_BLAS_FOR_ROTATION
    mulBlas(a);
    return;
#else
    float *matrix = PARENT::data();
    T vec[dim];
    for (int c = 0; c < dim; c++) {
      double sum = 0;
      for (int p = 0; p < dim; p++) {
	sum += a[p] * matrix[p * dim + c];
      }
      vec[c] = sum;
    }
    memcpy(a, vec, sizeof(T) * dim);
#endif
  }
  void mul(std::vector<float> &a) {
    if (a.size() != dim) {
      std::cerr << "Fatal inner error! " << a.size() << ":" << dim << std::endl;
      abort();
    }
    mul(a.data());
  }
  void serialize(std::ofstream &os) {
    uint32_t v = PARENT::size();
    tann::Serializer::write(os, v);
    os.write(reinterpret_cast<const char*>(PARENT::data()), static_cast<uint64_t>(PARENT::size()) * sizeof(float));
  }

  void deserialize(std::ifstream &is) {
    uint32_t v;
    tann::Serializer::read(is, v);
    PARENT::resize(v);
    dim = sqrt(v);
    if (dim * dim != v) {
      std::cerr << "rotation::deserialize: Fatal inner error. Invalid data. " << dim << ":" << dim * dim << ":" << v << std::endl;
      abort();
    }

    is.read(reinterpret_cast<char*>(PARENT::data()), PARENT::size() * sizeof(float));

  }

  void show() {
    std::cerr << "R=" << std::endl;
    for (size_t i = 0; i < dim; i++) {
      for (size_t j = 0; j < dim; j++) {
	std::cerr << (*this)[i * dim + j] << " ";
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl;
  }

  uint32_t dim;
};

template <typename T>
class QuantizationCodebook : public std::vector<T> {
  typedef std::vector<T> PARENT;
 public:
  QuantizationCodebook(): dimension(0), paddedDimension(0), index(0) {}
  QuantizationCodebook& operator=(const std::vector<std::vector<float>> &qc) {
    if (qc.empty()) {
      TANN_THROW("NGTQ::QuantizationCodebook::operator=: codebook is empty.");
    }
    if (paddedDimension == 0) {
      TANN_THROW("NGTQ::QuantizationCodebook::operator=: paddedDimension is unset.");
    }
    dimension = qc[0].size();
    std::cerr << "dim=" << dimension << ":" << qc.size() << ":" << paddedDimension << std::endl;
    PARENT::resize(paddedDimension * qc.size());
    for (size_t i = 0; i < qc.size(); i++) {
      if (qc[i].size() != dimension) {
	std::stringstream msg;
	msg << "NGTQ::QuantizationCodebook::operator=: paddedDimension is invalid. " << i << ":" << qc[i].size() << ":" << dimension;
	TANN_THROW(msg);
      }
      memcpy(PARENT::data() + i * paddedDimension, qc[i].data(), dimension * sizeof(T));
    }
    buildIndex();
    return *this;
  }
  ~QuantizationCodebook() { delete index;  }
  void setPaddedDimension(size_t pd) { paddedDimension = pd; }
  T at(size_t id, size_t dim) {
    return *(PARENT::data() + id * paddedDimension + dim);
  }
  T *data(size_t id) {
    return PARENT::data() + id * paddedDimension;
  }
  size_t size() { return PARENT::size() == 0 ? 0 : PARENT::size() / paddedDimension; }
  void buildIndex() {
    if (PARENT::size() == 0) {
      return;
    }
    std::cerr << "QuantizationCodebook::buildIndex" << std::endl;
    if (index != 0) {
      std::cerr << "Quantization codebook: something wrong?" << std::endl;
      delete index;
    }
    tann::Property property;
    property.dimension = dimension;
    property.distanceType = tann::MetricType::MetricTypeL2;
#ifdef NGTQ_SHARED_INVERTED_INDEX
    index = new tann::Index("dummy", property);
    std::cerr << "Not implemented" << std::endl;
    abort();
#else
    index = new tann::Index(property);
#endif
    size_t noOfCentroids = PARENT::size() / paddedDimension;
    std::cerr << "QuantizationCodebook::buildIndex # of the centroids=" << noOfCentroids << std::endl;
    for (size_t idx = 0; idx < noOfCentroids; idx++) {
      if ((idx + 1) % 100000 == 0) {
	std::cerr << "QuantizationCodebook::buildIndex processed objects=" << idx << std::endl;
      }
      index->append(data(idx), 1);  
    }
    index->createIndex(50);
  }
  size_t search(tann::Object &object) {
    if (index == 0) {
      std::cerr << "QuantizeationCodebook: Fatal Error! The index is not available." << std::endl;
      abort();
    }
    tann::VectorDistances result;
    tann::SearchContainer sc(object);
    sc.setResults(&result);
    sc.setSize(10);
    sc.radius = FLT_MAX;
    sc.setEpsilon(0.1);
    index->search(sc);
    if (result.size() == 0) {
      std::cerr << "QuantizeationCodebook: Fatal Error! Cannot search." << std::endl;
      abort();
    } else {
      return result[0].id - 1;
    }
  }
  size_t find(tann::Object &object, tann::VectorSpace::Comparator &comparator) {
    size_t noOfCentroids = PARENT::size() / paddedDimension;
    auto mind = std::numeric_limits<float>::max();
    size_t minidx = 0;
    for (size_t idx = 0; idx < noOfCentroids; idx++) {
      auto d = tann::PrimitiveComparator::compareL2(static_cast<float*>(object.getPointer()), data(idx), paddedDimension);
      if (mind > d) {
	mind = d;
	minidx = idx;
      }
    }
    return minidx;
  }

  void rotate(Rotation &r) {
    size_t noOfCentroids = PARENT::size() / paddedDimension;
    for (size_t idx = 0; idx < noOfCentroids; idx++) {
      r.mul(data(idx));
    }
  }
  
  void serialize(std::ofstream &os) {
    uint32_t v = PARENT::size();
    tann::Serializer::write(os, v);
    v = dimension;
    tann::Serializer::write(os, v);
    v = paddedDimension;
    tann::Serializer::write(os, v);
    os.write(reinterpret_cast<const char*>(PARENT::data()), static_cast<uint64_t>(PARENT::size()) * sizeof(T));
  }
  
  void deserialize(std::ifstream &is, bool readOnly) {
    uint32_t v;
    tann::Serializer::read(is, v);
    PARENT::resize(v);
    tann::Serializer::read(is, v);
    dimension = v;
    tann::Serializer::read(is, v);
    paddedDimension = v;
    is.read(reinterpret_cast<char*>(PARENT::data()), PARENT::size() * sizeof(T));
    if (!readOnly) {
      buildIndex();
    }
  }
  uint32_t dimension;
  uint32_t paddedDimension;
  tann::Index *index;
};

class GraphNodeToInvertedIndexEntries : public std::vector<uint32_t> {
  typedef std::vector<uint32_t> PARENT;
 public:
  GraphNodeToInvertedIndexEntries() {}
  ~GraphNodeToInvertedIndexEntries() {}
  void serialize(std::ofstream &os) {
    uint32_t v = PARENT::size();
    tann::Serializer::write(os, v);
    os.write(reinterpret_cast<const char*>(PARENT::data()), static_cast<uint64_t>(v) * sizeof(uint32_t));
  }
  void deserialize(std::ifstream &is) {
    uint32_t v;
    tann::Serializer::read(is, v);
    PARENT::resize(v);
    is.read(reinterpret_cast<char*>(PARENT::data()), static_cast<uint64_t>(v) * sizeof(uint32_t));
  }

};

template <typename T>
class InvertedIndexObject {
public:
  InvertedIndexObject(): id(0) { localID[0] = 0; }
  InvertedIndexObject<T>& operator=(const InvertedIndexObject<T> &iio) {
    std::cerr << "InvertedIndexObject: operator=() should not be used, because it cannot be implemented. Instead of this, use copy()." << std::endl;
    abort();
  }
  InvertedIndexObject<T>& operator=(InvertedIndexObject<T> &iio) {
    std::cerr << "InvertedIndexObject: operator=() should not be used, because it cannot be implemented. Instead of this, use copy()." << std::endl;
    abort();
  }
  void setID(uint32_t i) { id = i; }
  static size_t headerSize() {
    InvertedIndexObject<T> dummy;
    return (size_t)&dummy.localID[0] - (size_t)&dummy;
  }
  void clear(size_t n) {
    id = 0;
    for (size_t i = 0; i < n; i++) {
      localID[i] = 0;
    }
  }
  uint32_t	id;		
  T		localID[1];	
};
 
template <typename T>
class InvertedIndexEntry : public tann::DynamicVector<InvertedIndexObject<T>> {
public:
  typedef tann::DynamicVector<InvertedIndexObject<T>> PARENT;
#ifdef NGTQ_SHARED_INVERTED_INDEX
  InvertedIndexEntry(size_t n, tann::VectorSpace *os = 0):numOfLocalIDs(n)
#ifdef NGTQ_QBG
    , subspaceID(std::numeric_limits<uint32_t>::max())
#endif
  {
    PARENT::elementSize = getSizeOfElement();
  }
  InvertedIndexEntry(size_t n, tann::SharedMemoryAllocator &allocator, tann::VectorSpace *os = 0):numOfLocalIDs(n)
#ifdef NGTQ_QBG
    , subspaceID(std::numeric_limits<uint32_t>::max())
#endif
  {
    PARENT::elementSize = getSizeOfElement();
  }
  void pushBack(tann::SharedMemoryAllocator &allocator) {
    PARENT::push_back(InvertedIndexObject<T>(), allocator);
    PARENT::back(allocator).clear(numOfLocalIDs);    
  }
  void pushBack(size_t id, tann::SharedMemoryAllocator &allocator) {
    pushBack(allocator);
    PARENT::back(allocator).setID(id);
  }
#else // NGTQ_SHARED_INVERTED_INDEX
  InvertedIndexEntry(tann::VectorSpace *os = 0):numOfLocalIDs(0)
#ifdef NGTQ_QBG
    , subspaceID(std::numeric_limits<uint32_t>::max())
#endif
  {}
  InvertedIndexEntry(size_t n, tann::VectorSpace *os = 0):numOfLocalIDs(n)
#ifdef NGTQ_QBG
    , subspaceID(std::numeric_limits<uint32_t>::max())
#endif
  {
    PARENT::elementSize = getSizeOfElement();
  }
  void pushBack() { 
    PARENT::push_back(InvertedIndexObject<T>());
    PARENT::back().clear(numOfLocalIDs);
  }
  void pushBack(size_t id) { 
    pushBack();
    PARENT::back().setID(id);
  }

  void serialize(std::ofstream &os, tann::VectorSpace *objspace = 0) {
    uint32_t sz = PARENT::size();
    tann::Serializer::write(os, sz);
    if (numOfLocalIDs > 0xFFFF) {
      std::cerr << "num of local IDs is too large. " << numOfLocalIDs << std::endl;
      abort();
    }
    uint16_t nids = numOfLocalIDs;
    tann::Serializer::write(os, nids);
#ifdef NGTQ_QBG
    int32_t ssid = subspaceID;
    tann::Serializer::write(os, ssid);
#endif
    os.write(reinterpret_cast<char*>(PARENT::vector), PARENT::size() * PARENT::elementSize);
  }

  void deserialize(std::ifstream &is, tann::VectorSpace *objectspace = 0) {
    uint32_t sz;
    uint16_t nids;
#ifdef NGTQ_QBG
    int32_t ssid;
#endif
    try {
      tann::Serializer::read(is, sz);
      tann::Serializer::read(is, nids);
#ifdef NGTQ_QBG
      tann::Serializer::read(is, ssid);
#endif
    } catch(tann::Exception &err) {
      std::stringstream msg;
      msg << "InvertedIndexEntry::deserialize: It might be caused by inconsistency of the valuable type of the inverted index size. " << err.what();
      TANN_THROW(msg);
    }
    numOfLocalIDs = nids;
#ifdef NGTQ_QBG
    subspaceID = ssid;
#endif
    PARENT::elementSize = getSizeOfElement();
    PARENT::reserve(sz); 
    PARENT::resize(sz);
    is.read(reinterpret_cast<char*>(PARENT::vector), sz * PARENT::elementSize);
  }

#endif 

  size_t getSizeOfElement() {
    InvertedIndexObject<T> dummy;
    size_t dsize = ((numOfLocalIDs * sizeof(T) - 1) / 4 + 1) * 4;
    return InvertedIndexObject<T>::headerSize() + dsize;
  }

  uint32_t numOfLocalIDs;
#ifdef NGTQ_QBG
  uint32_t subspaceID;
#endif
};

class LocalDatam {
public:
  LocalDatam(){};
#ifdef NGTQ_QBG
 LocalDatam(size_t iii, size_t iil, uint32_t ssID = 0) : iiIdx(iii), iiLocalIdx(iil), subspaceID(ssID) {}
#else
 LocalDatam(size_t iii, size_t iil) : iiIdx(iii), iiLocalIdx(iil) {}
#endif
  size_t iiIdx;	
  size_t iiLocalIdx; 
#ifdef NGTQ_QBG
  uint32_t subspaceID;
#endif
};

template <typename TYPE, int SIZE>
class SerializableObject : public tann::Object {
public:
  static size_t getSerializedDataSize() { return SIZE; }
};

 enum DataType {
   DataTypeUint8 = 0,
   DataTypeFloat = 1
#ifdef TANN_ENABLE_HALF_FLOAT
   ,
   DataTypeFloat16 = 2
#endif
 };

 typedef tann::MetricType		MetricType;

 enum CentroidCreationMode {
   CentroidCreationModeDynamic		= 0,
   CentroidCreationModeStatic		= 1,
   CentroidCreationModeDynamicKmeans	= 2,
   CentroidCreationModeStaticLayer	= 3,
   CentroidCreationModeNone		= 9	
 };

 enum AggregationMode {
   AggregationModeApproximateDistance				= 0,
   AggregationModeApproximateDistanceWithLookupTable		= 1,
   AggregationModeApproximateDistanceWithCache			= 2,
   AggregationModeExactDistanceThroughApproximateDistance	= 3,
   AggregationModeExactDistance					= 4
 };

 enum QuantizerType {
   QuantizerTypeNone	= 0,
   QuantizerTypeQG	= 1,
   QuantizerTypeQBG	= 2
 };
 
 class Property {
 public:
  Property() {
    // default values
    threadSize		= 32;
    globalRange		= 200;
    localRange		= 50;
    globalCentroidLimit	= 10000000;
    localCentroidLimit	= 1000000;
    dimension		= 0;
#ifdef NGTQ_QBG
    genuineDimension	= 0;
    genuineDataType	= ObjectFile::DataTypeFloat;
#endif
    dataSize		= 0;
    dataType		= DataTypeFloat;
    distanceType	= MetricType::MetricTypeNone;
    singleLocalCodebook = false;
    localDivisionNo	= 8;
    batchSize		= 1000;
    centroidCreationMode = CentroidCreationModeDynamic;
    localCentroidCreationMode = CentroidCreationModeDynamic;
    localIDByteSize	= 0;		// finally decided by localCentroidLimit
    localCodebookState	= false;	// not completed
    localClusteringSampleCoefficient = 10;	
    quantizerType	= QuantizerTypeNone;
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    invertedIndexSharedMemorySize = 512; // MB
#endif
  }

  void save(const string &path) {
    tann::PropertySet prop;
    prop.set("ThreadSize", 	(long)threadSize);
    prop.set("GlobalRange", 	globalRange);
    prop.set("LocalRange", 	localRange);
    prop.set("GlobalCentroidLimit", (long)globalCentroidLimit);
    prop.set("LocalCentroidLimit", (long)localCentroidLimit);
    prop.set("Dimension", 	(long)dimension);
#ifdef NGTQ_QBG
    prop.set("GenuineDimension", (long)genuineDimension);
    prop.set("GenuineDataType", (long)genuineDataType);
#endif
    prop.set("DataSize", 	(long)dataSize);
    prop.set("DataType", 	(long)dataType);
    prop.set("MetricType",	(long)distanceType);
    prop.set("SingleLocalCodebook", (long)singleLocalCodebook);
    prop.set("LocalDivisionNo", (long)localDivisionNo);
    prop.set("BatchSize", 	(long)batchSize);
    prop.set("CentroidCreationMode", (long)centroidCreationMode);
    prop.set("LocalCentroidCreationMode", (long)localCentroidCreationMode);
    prop.set("LocalIDByteSize",	(long)localIDByteSize);	
    prop.set("LocalCodebookState", (long)localCodebookState);
    prop.set("LocalSampleCoefficient", (long)localClusteringSampleCoefficient);
    prop.set("QuantizerType",	(long)quantizerType);
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    prop.set("InvertedIndexSharedMemorySize", 	(long)invertedIndexSharedMemorySize);
#endif
    prop.save(path + "/prf");
  }

  void setupLocalIDByteSize() {
    if (localCentroidLimit > 0xffff - 1) { 
      if (localIDByteSize == 2) {
	TANN_THROW("NGTQ::Property: The localIDByteSize is illegal for the localCentroidLimit.");
      }
      localIDByteSize = 4;
    } else {
      if (localIDByteSize == INT_MAX) {
	localIDByteSize = 4;
      } else if (localIDByteSize == 0) {
	localIDByteSize = 2;
      } else {
      }
    }
#ifdef NGTQ_QBG
    if (localIDByteSize != 1 && localIDByteSize != 2 && localIDByteSize != 4) {
#else
    if (localIDByteSize != 2 && localIDByteSize != 4) {
#endif
      TANN_THROW("NGTQ::Property: Fatal internal error! localIDByteSize should be 2 or 4.");
    }
  }

  void load(const string &path) {
    tann::PropertySet prop;
    prop.load(path + "/prf");
    threadSize 		= prop.getl("ThreadSize", threadSize);
    globalRange 	= prop.getf("GlobalRange", globalRange);
    localRange		= prop.getf("LocalRange", localRange);
    globalCentroidLimit	= prop.getl("GlobalCentroidLimit", globalCentroidLimit);
    localCentroidLimit	= prop.getl("LocalCentroidLimit", localCentroidLimit);
    dimension		= prop.getl("Dimension", dimension);
#ifdef NGTQ_QBG
    genuineDimension	= prop.getl("GenuineDimension", genuineDimension);
    genuineDataType	= static_cast<ObjectFile::DataType>(prop.getl("GenuineDataType", genuineDataType));
#endif
    dataSize		= prop.getl("DataSize", dataSize);
    dataType		= (DataType)prop.getl("DataType", dataType);
    distanceType	= (MetricType)prop.getl("MetricType", static_cast<long>(distanceType));
    singleLocalCodebook	= prop.getl("SingleLocalCodebook", singleLocalCodebook);
    localDivisionNo	= prop.getl("LocalDivisionNo", localDivisionNo);
    batchSize		= prop.getl("BatchSize", batchSize);
    centroidCreationMode= (CentroidCreationMode)prop.getl("CentroidCreationMode", centroidCreationMode);
    localCentroidCreationMode = (CentroidCreationMode)prop.getl("LocalCentroidCreationMode", localCentroidCreationMode);
    localIDByteSize	= prop.getl("LocalIDByteSize", INT_MAX);
    localCodebookState	= prop.getl("LocalCodebookState", localCodebookState);
    localClusteringSampleCoefficient	= prop.getl("LocalSampleCoefficient", localClusteringSampleCoefficient);
    setupLocalIDByteSize();
    quantizerType	= (QuantizerType)prop.getl("QuantizerType", quantizerType);
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    invertedIndexSharedMemorySize
      = prop.getl("InvertedIndexSharedMemorySize", invertedIndexSharedMemorySize);
#endif
  }

  void setup(const Property &p) {
    *this = p;
#ifdef NGTQ_QBG
    switch (genuineDataType) {
#else
    switch (dataType) {
#endif
    case DataTypeUint8:
#ifdef NGTQ_QBG
      dataSize = sizeof(uint8_t) * genuineDimension;
#else
      dataSize = sizeof(uint8_t) * dimension;
#endif
      break;
    case DataTypeFloat:
#ifdef NGTQ_QBG
      dataSize = sizeof(float) * genuineDimension;
#else
      dataSize = sizeof(float) * dimension;
#endif
      break;
    case DataTypeFloat16:
#ifdef NGTQ_QBG
      dataSize = sizeof(float) * genuineDimension;
#else
      dataSize = sizeof(float) * dimension;
#endif
      break;
    default:
      TANN_THROW("Quantizer constructor: Inner error. Invalid data type.");
      break;
    } 
    setupLocalIDByteSize();
    localDivisionNo = getLocalCodebookNo();
  }

  inline size_t getLocalCodebookNo() { return singleLocalCodebook ? 1 : localDivisionNo; }

  size_t	threadSize;
  float		globalRange;
  float		localRange;
  size_t	globalCentroidLimit;
  size_t	localCentroidLimit;
  size_t	dimension;
#ifdef NGTQ_QBG
  size_t	genuineDimension;
  ObjectFile::DataType	genuineDataType;
#endif
  size_t	dataSize;
  DataType	dataType;
  MetricType	distanceType;
  bool		singleLocalCodebook;
  size_t	localDivisionNo;
  size_t	batchSize;
  CentroidCreationMode centroidCreationMode;
  CentroidCreationMode localCentroidCreationMode;
  size_t	localIDByteSize;
  bool		localCodebookState;
  size_t	localClusteringSampleCoefficient;
  QuantizerType	quantizerType;
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
  size_t	invertedIndexSharedMemorySize;
#endif
};

#ifdef NGTQ_DISTANCE_ANGLE
 class LocalDistanceLookup {
 public:
 LocalDistanceLookup():a(0.0), b(0.0), sum(0.0){};
   void set(double pa, double pb, double psum) {a = pa; b = pb; sum = psum;}
   double a;
   double b;
   double sum;
 };
#endif

class QuantizedObjectDistance {
public:
  class DistanceLookupTable {
  public:
    DistanceLookupTable():localDistanceLookup(0) {}
    ~DistanceLookupTable() {
      if (localDistanceLookup != 0) {
	delete[] localDistanceLookup;
	localDistanceLookup = 0;
      }
    }
    bool isValid(size_t idx) { 
#ifdef NGTQ_QBG
      std::cerr << "isValid() is not implemented" << std::endl;
      abort();
#else
      return flag[idx]; 
#endif
    }
#ifndef NGTQ_DISTANCE_ANGLE
    void set(size_t idx, double d) { 
#ifndef NGTQ_QBG
      flag[idx] = true;
#endif
      localDistanceLookup[idx] = d; 
    }
    double getDistance(size_t idx) { return localDistanceLookup[idx]; }
#endif
    void initialize(size_t s) {
      size = s;
#ifdef NGTQ_DISTANCE_ANGLE
      localDistanceLookup = new LocalDistanceLookup[size];
#else
      localDistanceLookup = new float[size];
#endif
#ifndef NGTQ_QBG
      flag.resize(size, false);
#endif
    }
#ifdef NGTQ_DISTANCE_ANGLE
    LocalDistanceLookup	*localDistanceLookup;
#else
    float		*localDistanceLookup;
#endif
    size_t		size;
#ifndef NGTQ_QBG
    vector<bool>	flag;	
#endif
  };

  class DistanceLookupTableUint8 {
  public:
    DistanceLookupTableUint8():localDistanceLookup(0) {}
    ~DistanceLookupTableUint8() {
      if (localDistanceLookup != 0) {
	delete[] localDistanceLookup;
	localDistanceLookup = 0;
	delete[] scales;
	delete[] offsets;
      }
    }
    void initialize(size_t numOfSubspaces, size_t localCodebookCentroidNo) {
      size_t numOfAlignedSubvectors = ((numOfSubspaces - 1) / NGTQ_BATCH_SIZE + 1) * NGTQ_BATCH_SIZE;
      size = numOfAlignedSubvectors * localCodebookCentroidNo;
      localDistanceLookup = new uint8_t[size];
      scales = new float[numOfAlignedSubvectors];
      offsets = new float[numOfAlignedSubvectors];
      range512 = (numOfSubspaces >> 2) * step512;
      range256 = (((numOfSubspaces - 1) >> 1) + 1) * step256;
    }

    uint8_t		*localDistanceLookup;
    size_t		size;
    size_t		aslignedNumOfSubspaces;
    size_t		localCodebookCentroidNo;
    float		*scales;
    float		*offsets;
    float		totalOffset;
    size_t		range512;
    size_t		range256;
    static constexpr size_t		step512 = 32;
    static constexpr size_t		step256 = 16;
  };

  QuantizedObjectDistance(){}
  virtual ~QuantizedObjectDistance() {
    delete[] localCentroids;
    delete[] localCentroidsForSIMD;
  }

  virtual double operator()(tann::Object &object, size_t objectID, void *localID) = 0;

  virtual double operator()(void *localID, DistanceLookupTable &distanceLUT) = 0;

#ifdef NGTQBG_MIN
  virtual float operator()(void *inv, float *distances, size_t size, DistanceLookupTableUint8 &distanceLUT) = 0;
#else
  virtual void operator()(void *inv, float *distances, size_t size, DistanceLookupTableUint8 &distanceLUT) = 0;
#endif
  virtual double operator()(tann::Object &object, size_t objectID, void *localID, DistanceLookupTable &distanceLUT) = 0;

  template <typename T>
  inline double getAngleDistanceUint8(tann::Object &object, size_t objectID, T localID[]) {
    assert(globalCodebookIndex != 0);
    tann::PersistentObject &gcentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localDataSize = sizeOfObject / localDivisionNo / sizeof(uint8_t);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    unsigned char *gcptr = &gcentroid.at(0, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
    unsigned char *gcptr = &gcentroid[0];
#endif
    unsigned char *optr = &((tann::Object&)object)[0];
    double normA = 0.0F;
    double normB = 0.0F;
    double sum = 0.0F;
    for (size_t li = 0; li < localDivisionNo; li++) {
      size_t idx = localCodebookNo == 1 ? 0 : li;
      tann::PersistentObject &lcentroid = *localCodebookIndexes[idx].getObjectSpace().getRepository().get(localID[li]);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      float *lcptr = (float*)&lcentroid.at(0, localCodebookIndexes[idx].getObjectSpace().getRepository().allocator);
#else
      float *lcptr = (float*)&lcentroid[0];
#endif
      float *lcendptr = lcptr + localDataSize;
      while (lcptr != lcendptr) {
	double a = *optr++;
	double b = *gcptr++ + *lcptr++;
	normA += a * a;
	normB += b * b;
	sum += a * b;
      }
    }
    double cosine = sum / (sqrt(normA) * sqrt(normB));
    if (cosine >= 1.0F) {
      return 0.0F;
    } else if (cosine <= -1.0F) {
      return acos(-1.0F);
    }
    return acos(cosine);
  }

#if defined(NGT_NO_AVX)
  template <typename T>
  inline double getL2DistanceUint8(tann::Object &object, size_t objectID, T localID[]) {
    assert(globalCodebookIndex != 0);
    tann::PersistentObject &gcentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localDataSize = sizeOfObject / localDivisionNo / sizeof(uint8_t);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    unsigned char *gcptr = &gcentroid.at(0, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
    unsigned char *gcptr = &gcentroid[0];
#endif
    unsigned char *optr = &((tann::Object&)object)[0];
    double distance = 0.0;
    for (size_t li = 0; li < localDivisionNo; li++) {
      size_t idx = localCodebookNo == 1 ? 0 : li;
      tann::PersistentObject &lcentroid = *localCodebookIndexes[idx].getObjectSpace().getRepository().get(localID[li]);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      float *lcptr = (float*)&lcentroid.at(0, localCodebookIndexes[idx].getObjectSpace().getRepository().allocator);
#else
      float *lcptr = (float*)&lcentroid[0];
#endif
      double d = 0.0;
      float *lcendptr = lcptr + localDataSize;
      while (lcptr != lcendptr) {
	double sub = ((int)*optr++ - (int)*gcptr++) - *lcptr++;
	d += sub * sub;
      }
      distance += d;
    }
    return sqrt(distance);
  }
#else 
  template <typename T>
  inline double getL2DistanceUint8(tann::Object &object, size_t objectID, T localID[]) {
    assert(globalCodebookIndex != 0);
    tann::PersistentObject &gcentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localDataSize = sizeOfObject / localDivisionNo / sizeof(uint8_t);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    unsigned char *gcptr = &gcentroid.at(0, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
    unsigned char *gcptr = &gcentroid[0];
#endif
    unsigned char *optr = &((tann::Object&)object)[0];
    double distance = 0.0;
    for (size_t li = 0; li < localDivisionNo; li++) {
      size_t idx = localCodebookNo == 1 ? 0 : li;
      tann::PersistentObject &lcentroid = *localCodebookIndexes[idx].getObjectSpace().getRepository().get(localID[li]);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      float *lcptr = (float*)&lcentroid.at(0, localCodebookIndexes[idx].getObjectSpace().getRepository().allocator);
#else
      float *lcptr = (float*)&lcentroid[0];
#endif

      float *lcendptr = lcptr + localDataSize - 3;
      __m128 sum = _mm_setzero_ps();
      while (lcptr < lcendptr) {
	__m128i x1 = _mm_cvtepu8_epi32(_mm_loadu_si128((__m128i const*)optr));
	__m128i x2 = _mm_cvtepu8_epi32(_mm_loadu_si128((__m128i const*)gcptr));
	x1 = _mm_sub_epi32(x1, x2);
	__m128 sub = _mm_sub_ps(_mm_cvtepi32_ps(x1), _mm_loadu_ps(lcptr));
	sum = _mm_add_ps(sum, _mm_mul_ps(sub, sub));
	optr += 4;
	gcptr += 4;
	lcptr += 4;
      }
      __attribute__((aligned(32))) float f[4];
      _mm_store_ps(f, sum);
      double d = f[0] + f[1] + f[2] + f[3];
      while (lcptr < lcendptr) {
	double sub = ((int)*optr++ - (int)*gcptr++) - *lcptr++;
	d += sub * sub;
      }
      distance += d;
    }
    distance = sqrt(distance);
    return distance;
  }
#endif 

  template <typename T>
  inline double getAngleDistanceFloat(tann::Object &object, size_t objectID, T localID[]) {
    assert(globalCodebookIndex != 0);
    tann::PersistentObject &gcentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localDataSize = sizeOfObject / localDivisionNo  / sizeof(float);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    float *gcptr = (float*)&gcentroid.at(0, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
    float *gcptr = (float*)&gcentroid[0];
#endif
    float *optr = (float*)&((tann::Object&)object)[0];
    double normA = 0.0F;
    double normB = 0.0F;
    double sum = 0.0F;
    for (size_t li = 0; li < localDivisionNo; li++) {
      size_t idx = localCodebookNo == 1 ? 0 : li;
      tann::PersistentObject &lcentroid = *localCodebookIndexes[idx].getObjectSpace().getRepository().get(localID[li]);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      float *lcptr = (float*)&lcentroid.at(0, localCodebookIndexes[idx].getObjectSpace().getRepository().allocator);
#else
      float *lcptr = (float*)&lcentroid[0];
#endif
      float *lcendptr = lcptr + localDataSize;
      while (lcptr != lcendptr) {
	double a = *optr++;
	double b = *gcptr++ + *lcptr++;
	normA += a * a;
	normB += b * b;
	sum += a * b;
      }
    }
    double cosine = sum / (sqrt(normA) * sqrt(normB));
    if (cosine >= 1.0F) {
      return 0.0F;
    } else if (cosine <= -1.0F) {
       return acos(-1.0F);
    }
    return acos(cosine);
  }

  template <typename T>
    inline double getL2DistanceFloat(tann::Object &object, size_t objectID, T localID[]) {
    assert(globalCodebookIndex != 0);
    tann::PersistentObject &gcentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localDataSize = sizeOfObject / localDivisionNo  / sizeof(float);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    float *gcptr = (float*)&gcentroid.at(0, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
    float *gcptr = (float*)&gcentroid[0];
#endif
    float *optr = (float*)&((tann::Object&)object)[0];
    double distance = 0.0;
    for (size_t li = 0; li < localDivisionNo; li++) {
      size_t idx = localCodebookNo == 1 ? 0 : li;
      tann::PersistentObject &lcentroid = *localCodebookIndexes[idx].getObjectSpace().getRepository().get(localID[li]);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      float *lcptr = (float*)&lcentroid.at(0, localCodebookIndexes[idx].getObjectSpace().getRepository().allocator);
#else
      float *lcptr = (float*)&lcentroid[0];
#endif
      float *lcendptr = lcptr + localDataSize;
      double d = 0.0;
      while (lcptr != lcendptr) {
	double sub = (*optr++ - *gcptr++) - *lcptr++;
	d += sub * sub;
      }
      distance += d;
    }
    distance = sqrt(distance);
    return distance;
  }

#ifdef NGTQ_DISTANCE_ANGLE
  inline void createDistanceLookup(tann::Object &object, size_t objectID, DistanceLookupTable &distanceLUT) {
    assert(globalCodebookIndex != 0);
    tann::Object &gcentroid = (tann::Object &)*globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localDataSize = sizeOfObject  / localDivisionNo / sizeof(float);
    float *optr = (float*)&((tann::Object&)object)[0];
    float *gcptr = (float*)&gcentroid[0];
    LocalDistanceLookup *dlu = distanceLUT.localDistanceLookup;
    size_t oft = 0;
    for (size_t li = 0; li < localCodebookNo; li++, oft += localDataSize) {
      dlu++;  
      for (size_t k = 1; k < localCodebookCentroidNo; k++) {
	tann::Object &lcentroid = (tann::Object&)*localCodebookIndexes[li].getObjectSpace().getRepository().get(k);
	float *lcptr = (float*)&lcentroid[0];		
	float *lcendptr = lcptr + localDataSize;	
	float *toptr = optr + oft;
	float *tgcptr = gcptr + oft;
	double normA = 0.0F;
	double normB = 0.0F;
	double sum = 0.0F;
	while (lcptr != lcendptr) {
	  double a = *toptr++;
	  double b = *tgcptr++ + *lcptr++;
	  normA += a * a;
	  normB += b * b;
	  sum += a * b;
	}
	dlu->set(normA, normB, sum);
	dlu++;
      }
    }
  }
#else 
  inline void createDistanceLookup(tann::Object &object, size_t objectID, DistanceLookupTable &distanceLUT) {
    assert(globalCodebookIndex != 0);
    tann::Object &gcentroid = (tann::Object &)*globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();

    createFloatL2DistanceLookup(&((tann::Object&)object)[0], sizeOfObject, &gcentroid[0], distanceLUT.localDistanceLookup);

  }
#endif 

  inline void createFloatDotProductLookup(void *object, size_t sizeOfObject, void *globalCentroid, float *lut) {
    size_t localDataSize = sizeOfObject  / localDivisionNo / sizeof(float);

    float *optr = static_cast<float*>(object);
    float *gcptr = static_cast<float*>(globalCentroid);

    size_t oft = 0;
    float *lcptr = static_cast<float*>(&localCentroids[0]);
    for (size_t li = 0; li < localCodebookNo; li++, oft += localDataSize) {
      lut++;  
      lcptr += localDataSize;
      for (size_t k = 1; k < localCodebookCentroidNo; k++) {
	float *lcendptr = lcptr + localDataSize;	
	float *toptr = optr + oft;
	float *tgcptr = gcptr + oft;
	float d = 0.0;
	while (lcptr != lcendptr) {
	  float product = *toptr++ * (*lcptr++ + *tgcptr++);
	  d += product;
	}
	*lut++ = d;
      }
    }

  }



  inline void createFloatL2DistanceLookup(void *object, size_t sizeOfObject, void *globalCentroid, DistanceLookupTableUint8 &distanceLUT) {
    assert(localCodebookCentroidNoSIMD == 16);
    size_t dim = sizeOfObject  / sizeof(float);
    size_t localDim = dim / localDivisionNo;
#if defined(NGT_AVX512) 
    __m512 flut[dim];
    __m512 *flutptr = &flut[0];
#ifdef NGTQ_TOTAL_SCALE_OFFSET_COMPRESSION
    __m512 mmin = _mm512_set1_ps(std::numeric_limits<float>::max());
    __m512 mmax = _mm512_set1_ps(-std::numeric_limits<float>::max());
#else
    std::pair<float, float> range[localDivisionNo];
    auto *rangePtr = &range[0];
#endif
    auto *lcptr = static_cast<float*>(&localCentroidsForSIMD[0]);
    auto *optr = static_cast<float*>(object);
    auto *optrend = optr + dim;
#ifndef NGTQG_ZERO_GLOBAL
    auto *gcptr = static_cast<float*>(globalCentroid);
#endif
    while (optr < optrend) {
      auto *optrllast = optr + localDim;
#ifdef NGTQG_ZERO_GLOBAL
      float rsv = *optr++;
#else
      float rsv = *optr++ - *gcptr++;
#endif
      __m512 vtmp = _mm512_sub_ps(_mm512_set1_ps(rsv), _mm512_loadu_ps(lcptr));
      __m512 v = _mm512_mul_ps(vtmp, vtmp);
      lcptr += 16;
      while (optr < optrllast) {
#ifdef NGTQG_ZERO_GLOBAL
	rsv = *optr++;
#else
	rsv = *optr++ - *gcptr++;
	
#endif
	vtmp = _mm512_sub_ps(_mm512_set1_ps(rsv), _mm512_loadu_ps(lcptr));
	v = _mm512_add_ps(v, _mm512_mul_ps(vtmp, vtmp));
	lcptr += 16;
      }
#ifdef NGTQ_TOTAL_SCALE_OFFSET_COMPRESSION
      mmin = _mm512_min_ps(mmin, v);
      mmax = _mm512_max_ps(mmax, v);
#else
      (*rangePtr).first = _mm512_reduce_min_ps(v);
      (*rangePtr).second = _mm512_reduce_max_ps(v);
      rangePtr++;
#endif
      *flutptr = v;
      flutptr++;
    }
    

#ifdef NGTQ_TOTAL_SCALE_OFFSET_COMPRESSION
    float min = _mm512_reduce_min_ps(mmin);
    float max = _mm512_reduce_max_ps(mmax);
    float offset = min;
    float scale = (max - min) / 255.0;
    distanceLUT.totalOffset = offset * static_cast<float>(localCodebookNo);
    auto *blutptr = distanceLUT.localDistanceLookup;
    flutptr = &flut[0];
    for (size_t li = 0; li < localCodebookNo; li++) {
      __m512 v = _mm512_div_ps(_mm512_sub_ps(*flutptr, _mm512_set1_ps(offset)), _mm512_set1_ps(scale));
      __m512i b4v = _mm512_cvtps_epi32(_mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT));
      __m128i b = _mm512_cvtepi32_epi8(b4v);
      _mm_storeu_si128((__m128i_u*)blutptr, b);
      flutptr++;
      blutptr += 16;
      distanceLUT.offsets[li] = offset; 
      distanceLUT.scales[li] = scale;
    }
#else 
    distanceLUT.totalOffset = 0.0;
    auto *blutptr = distanceLUT.localDistanceLookup;
    flutptr = &flut[0];
    for (size_t li = 0; li < localCodebookNo; li++) {
      float offset = range[li].first; 
      float scale = (range[li].second - range[li].first) / 255.0;
      __m512 v = _mm512_div_ps(_mm512_sub_ps(*flutptr, _mm512_set1_ps(offset)), _mm512_set1_ps(scale));
      __m512i b4v = _mm512_cvtps_epi32(_mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT));
      __m128i b = _mm512_cvtepi32_epi8(b4v);
      _mm_storeu_si128((__m128i_u*)blutptr, b);
      flutptr++;
      blutptr += 16;
      distanceLUT.offsets[li] = offset; 
      distanceLUT.scales[li] = scale;
      distanceLUT.totalOffset += offset; 
    }
#endif

#elif defined(NGT_AVX2)
    __m256 flut[dim * 2];
    __m256 *flutptr = &flut[0];
    __m256 mmin = _mm256_set1_ps(std::numeric_limits<float>::max());
    __m256 mmax = _mm256_set1_ps(-std::numeric_limits<float>::max());
    auto *lcptr = static_cast<float*>(&localCentroidsForSIMD[0]);
    auto *optr = static_cast<float*>(object);
    auto *optrend = optr + dim;
#ifndef NGTQG_ZERO_GLOBAL
    auto *gcptr = static_cast<float*>(globalCentroid);
#endif
    while (optr < optrend) {
      auto *optrllast = optr + localDim;
#ifdef NGTQG_ZERO_GLOBAL
      float rsv = *optr++;
#else
      float rsv = *optr++ - *gcptr++;
#endif
      __m256 vtmp = _mm256_sub_ps(_mm256_set1_ps(rsv), _mm256_loadu_ps(lcptr));
      __m256 vt = _mm256_mul_ps(vtmp, vtmp);
      lcptr += 8;
      vtmp = _mm256_sub_ps(_mm256_set1_ps(rsv), _mm256_loadu_ps(lcptr));
      __m256 vb = _mm256_mul_ps(vtmp, vtmp);
      lcptr += 8;
      while (optr < optrllast) {
#ifdef NGTQG_ZERO_GLOBAL
	rsv = *optr++;
#else
	rsv = *optr++ - *gcptr++;
#endif
	vtmp = _mm256_sub_ps(_mm256_set1_ps(rsv), _mm256_loadu_ps(lcptr));
	vt = _mm256_add_ps(vt, _mm256_mul_ps(vtmp, vtmp));
	lcptr += 8;
	vtmp = _mm256_sub_ps(_mm256_set1_ps(rsv), _mm256_loadu_ps(lcptr));
	vb = _mm256_add_ps(vb, _mm256_mul_ps(vtmp, vtmp));
	lcptr += 8;
      }
      mmin = _mm256_min_ps(mmin, vt);
      mmax = _mm256_max_ps(mmax, vt);
      *flutptr++ = vt;
      mmin = _mm256_min_ps(mmin, vb);
      mmax = _mm256_max_ps(mmax, vb);
      *flutptr++ = vb;
    }
    float f[8];
    _mm256_storeu_ps(f, mmin);
    float min = f[0];
    for (size_t i = 1; i < 8; i++) {
      if (min > f[i]) min = f[i];
    }
    _mm256_storeu_ps(f, mmax);
    float max = f[0];
    for (size_t i = 1; i < 8; i++) {
      if (max < f[i]) max = f[i];
    }
    float offset = min;
    float scale = (max - min) / 255.0;

    distanceLUT.totalOffset = offset * static_cast<float>(localCodebookNo);
    //uint8_t blut[dim / 2 * localCodebookCentroidNoSIMD];
    auto *blutptr = distanceLUT.localDistanceLookup;
    flutptr = &flut[0];
    for (size_t li = 0; li < localCodebookNo; li++) {
      __m256 v = _mm256_div_ps(_mm256_sub_ps(*flutptr++, _mm256_set1_ps(offset)), _mm256_set1_ps(scale));
      __m256i b4v = _mm256_cvtps_epi32(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT));
      *blutptr++ = _mm256_extract_epi8(b4v, 0);
      *blutptr++ = _mm256_extract_epi8(b4v, 4);
      *blutptr++ = _mm256_extract_epi8(b4v, 8);
      *blutptr++ = _mm256_extract_epi8(b4v, 12);
      __m128i b = _mm256_extracti128_si256(b4v, 1);
      *blutptr++ = _mm_extract_epi8(b, 0);
      *blutptr++ = _mm_extract_epi8(b, 4);
      *blutptr++ = _mm_extract_epi8(b, 8);
      *blutptr++ = _mm_extract_epi8(b, 12);
      v = _mm256_div_ps(_mm256_sub_ps(*flutptr++, _mm256_set1_ps(offset)), _mm256_set1_ps(scale));
      b4v = _mm256_cvtps_epi32(_mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT));
      *blutptr++ = _mm256_extract_epi8(b4v, 0);
      *blutptr++ = _mm256_extract_epi8(b4v, 4);
      *blutptr++ = _mm256_extract_epi8(b4v, 8);
      *blutptr++ = _mm256_extract_epi8(b4v, 12);
      b = _mm256_extracti128_si256(b4v, 1);
      *blutptr++ = _mm_extract_epi8(b, 0);
      *blutptr++ = _mm_extract_epi8(b, 4);
      *blutptr++ = _mm_extract_epi8(b, 8);
      *blutptr++ = _mm_extract_epi8(b, 12);
      distanceLUT.offsets[li] = offset;
      distanceLUT.scales[li] = scale;
    }
#else 
    float flut[dim / 2 * localCodebookCentroidNoSIMD];
    auto *flutptr = &flut[0];
    auto min = std::numeric_limits<float>::max();
    auto max = -std::numeric_limits<float>::max();
    auto *lcptr = static_cast<float*>(&localCentroidsForSIMD[0]);
    auto *optr = static_cast<float*>(object);
#ifndef NGTQG_ZERO_GLOBAL
    auto *gcptr = static_cast<float*>(globalCentroid);
#endif
    float sub[localCodebookCentroidNoSIMD];
    for (size_t d = 0; d < dim; d++) {
#ifdef NGTQG_ZERO_GLOBAL
      float rsv = optr[d];
#else
      float rsv = optr[d] - *gcptr++;
#endif
      for (size_t k = 0; k < localCodebookCentroidNoSIMD; k++) {
	auto v = rsv - *lcptr++;
	v *= v;
	if (d % 2 == 0) {
	  sub[k] = v;
	} else {
	  v += sub[k];
	  if (v < min) {
	    min = v;
	  } else if (v > max) {
	    max = v;
	  }
	  *flutptr++ = v;
	}
      }
    }
    float offset = min;
    float scale = (max - min) / 255.0;

    distanceLUT.totalOffset = offset * static_cast<float>(localCodebookNo);
    auto *blutptr = distanceLUT.localDistanceLookup;
    flutptr = &flut[0];
    for (size_t li = 0; li < localCodebookNo; li++) {
      for (size_t k = 0; k < localCodebookCentroidNoSIMD; k++) {
	int32_t tmp = std::round((*flutptr - offset) / scale);
	assert(tmp >= 0 && tmp <= 255);
	*blutptr++ = static_cast<uint8_t>(tmp);
	flutptr++;
      }
      distanceLUT.offsets[li] = offset;
      distanceLUT.scales[li] = scale;
    }

#endif 

    

  }


    
  inline void createFloatL2DistanceLookup(void *object, size_t sizeOfObject, void *globalCentroid, float *lut) {
    size_t localDataSize = sizeOfObject  / localDivisionNo / sizeof(float);
    float *optr = static_cast<float*>(object);
#if !defined(NGTQG_ZERO_GLOBAL)
    float *gcptr = static_cast<float*>(globalCentroid);
#endif
    size_t oft = 0;
    float *lcptr = static_cast<float*>(&localCentroids[0]);
    for (size_t li = 0; li < localCodebookNo; li++, oft += localDataSize) {
      *lut++ = 0;  
      lcptr += localDataSize;
      for (size_t k = 1; k < localCodebookCentroidNo; k++) {
	float *lcendptr = lcptr + localDataSize;	
	float *toptr = optr + oft;
#if !defined(NGTQG_ZERO_GLOBAL)
	float *tgcptr = gcptr + oft;
#endif
	float d = 0.0;
	while (lcptr != lcendptr) {
#if defined(NGTQG_ZERO_GLOBAL)
	  float sub = *toptr++ - *lcptr++;
#else
	  float sub = *toptr++ - *tgcptr++ - *lcptr++;
#endif
	  d += sub * sub;
	}
	*lut++ = d;
      }
    }


  }

  inline void createDistanceLookup(tann::Object &object, size_t objectID, DistanceLookupTableUint8 &distanceLUT) {
    void *objectPtr = &((tann::Object&)object)[0];
    createDistanceLookup(objectPtr, objectID, distanceLUT);
  }
  inline void createDistanceLookup(void *objectPtr, size_t objectID, DistanceLookupTableUint8 &distanceLUT) {
    assert(globalCodebookIndex != 0);
    size_t sizeOfObject = dimension * sizeOfType;
#ifdef NGTQG_DOT_PRODUCT
    std::cerr << "Not implemented." << std::endl;
    abort();
#else
    assert(objectID < quantizationCodebook->size());
    createFloatL2DistanceLookup(objectPtr, sizeOfObject, quantizationCodebook->data(objectID), distanceLUT);
#endif
  }

  void set(tann::Index *gcb, tann::Index lcb[], QuantizationCodebook<float> *qcodebook, size_t dn, size_t lcn,
	   size_t sizeoftype, size_t dim, Rotation *r) {
    globalCodebookIndex = gcb;
    localCodebookIndexes = lcb;
    localDivisionNo = dn;
    dimension = dim;
    assert(dimension % localDivisionNo == 0);
    localDataSize = dimension / localDivisionNo;
    sizeOfType = sizeoftype;
    set(lcb, lcn);
    if (globalCodebookIndex->getObjectSpace().getRepository().size() == 2) {
      tann::ObjectID id = 1;
      try {
	globalCodebookIndex->getObjectSpace().getObject(id, globalCentroid);
      } catch (tann::Exception &err) {
	std::cerr << "Cannot load the global centroid. id=" << id << std::endl;
      }
    }

    quantizationCodebook = qcodebook;

    float *lc = new float[localCodebookNo * localCodebookCentroidNo * localDataSize];
    for (size_t li = 0; li < localCodebookNo; li++) {
      for (size_t k = 1; k < localCodebookCentroidNo; k++) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	tann::PersistentObject &lcentroid = *static_cast<tann::PersistentObject*>(localCodebookIndexes[li].getObjectSpace().getRepository().get(k));
	memcpy(&lc[li * localCodebookCentroidNo * localDataSize + k * localDataSize],
	       &lcentroid.at(0, localCodebookIndexes[li].getObjectSpace().getRepository().allocator), localDataSize * sizeof(float));
#else
	tann::Object &lcentroid = *static_cast<tann::Object*>(localCodebookIndexes[li].getObjectSpace().getRepository().get(k));
	memcpy(&lc[li * localCodebookCentroidNo * localDataSize + k * localDataSize], &lcentroid[0], localDataSize * sizeof(float));
#endif
      }
    }

    localCentroids = lc;

    rotation = r;

    localCodebookCentroidNoSIMD = localCodebookCentroidNo == 0 ? 0 : localCodebookCentroidNo - 1;
    lc = new float[dimension * localCodebookCentroidNoSIMD];
    for (size_t li = 0; li < localCodebookNo; li++) {
      for (size_t k = 1; k < localCodebookCentroidNo; k++) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	tann::PersistentObject &lcentroid = *static_cast<tann::PersistentObject*>(localCodebookIndexes[li].getObjectSpace().getRepository().get(k));
	float *subVector = reinterpret_cast<float*>(&lcentroid.at(0, localCodebookIndexes[li].getObjectSpace().getRepository().allocator));
#else
	tann::Object &lcentroid = *static_cast<tann::Object*>(localCodebookIndexes[li].getObjectSpace().getRepository().get(k));
	float *subVector = reinterpret_cast<float*>(&lcentroid[0]);
#endif
	for (size_t d = 0; d < localDataSize; d++) {
	  lc[(li * localDataSize + d) * localCodebookCentroidNoSIMD + (k - 1)] = subVector[d];
	}
      }
    }
    
    localCentroidsForSIMD = lc;


  }

  void set(tann::Index lcb[], size_t lcn) {
    localCodebookNo = lcn;
    localCodebookCentroidNo = lcb[0].getObjectRepositorySize();
  }

  void initialize(DistanceLookupTable &c) {
    c.initialize(localCodebookNo * localCodebookCentroidNo);
  }

  void initialize(DistanceLookupTableUint8 &c) {
    c.initialize(localCodebookNo, localCodebookCentroidNo);
  }

  tann::Index	*globalCodebookIndex;
  tann::Index	*localCodebookIndexes;
  size_t	localDivisionNo;
  size_t	localCodebookNo;
  size_t	localCodebookCentroidNo;

  size_t	localDataSize;
  size_t	sizeOfType;
  size_t	dimension;
  vector<float>	globalCentroid;
  QuantizationCodebook<float>	*quantizationCodebook;
  
  float		*localCentroids;	
  float		*localCentroidsForSIMD;	

  size_t	localCodebookCentroidNoSIMD;

  Rotation	*rotation;
};  

template <typename T>
class QuantizedObjectDistanceUint8 : public QuantizedObjectDistance {
public:

#ifdef NGTQ_DISTANCE_ANGLE
  inline double operator()(void *l, DistanceLookupTable &distanceLUT) {
    T *localID = static_cast<T*>(l);
    double normA = 0.0F;
    double normB = 0.0F;
    double sum = 0.0F;
    for (size_t li = 0; li < localDivisionNo; li++) {
      LocalDistanceLookup &ldl = *(distanceLUT.localDistanceLookup + li * localCodebookCentroidNo + localID[li]);
      normA += ldl.a;
      normB += ldl.b;
      sum += ldl.sum;
    }
    double cosine = sum / (sqrt(normA) * sqrt(normB));
    if (cosine >= 1.0F) {
      return 0.0F;
    } else if (cosine <= -1.0F) {
      return acos(-1.0F);
    }
    return acos(cosine);
  }
  inline double operator()(tann::Object &object, size_t objectID, void *l) {
    return getAngleDistanceUint8(object, objectID, static_cast<T*>(l));
  }
  inline double operator()(tann::Object &object, size_t objectID, void *l, DistanceLookupTable &distanceLUT) {
    cerr << "operator() is not implemented" << endl;
    abort();
    return 0.0;
  }
#else // NGTQ_DISTANCE_ANGLE
  inline double operator()(void *l, DistanceLookupTable &distanceLUT) {
    T *localID = static_cast<T*>(l);
    double distance = 0.0;
    for (size_t li = 0; li < localDivisionNo; li++) {
      distance += distanceLUT.getDistance(li * localCodebookCentroidNo + localID[li]);
    }
    return sqrt(distance);	
  }

  inline double operator()(tann::Object &object, size_t objectID, void *l) {
    return getL2DistanceUint8(object, objectID, static_cast<T*>(l));
  }
  inline double operator()(tann::Object &object, size_t objectID, void *l, DistanceLookupTable &distanceLUT) {
    T *localID = static_cast<T*>(l);
    tann::PersistentObject &gcentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localDataSize = sizeOfObject / localDivisionNo  / sizeof(uint8_t);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    unsigned char *gcptr = &gcentroid.at(0, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
    unsigned char *gcptr = &gcentroid[0];
#endif
    unsigned char *optr = &((tann::Object&)object)[0];
    double distance = 0.0;
    for (size_t li = 0; li < localDivisionNo; li++) {
      if (distanceLUT.isValid(li * localCodebookCentroidNo + localID[li])) {
	distance += distanceLUT.getDistance(li * localCodebookCentroidNo + localID[li]);
	optr += localDataSize;
	gcptr += localDataSize;
      } else {
	size_t idx = localCodebookNo == 1 ? 0 : li;
	tann::PersistentObject &lcentroid = *localCodebookIndexes[idx].getObjectSpace().getRepository().get(localID[li]);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	float *lcptr = (float*)&lcentroid.at(0, localCodebookIndexes[idx].getObjectSpace().getRepository().allocator);
#else
	float *lcptr = (float*)&lcentroid[0];
#endif
	double d = 0.0;
	float *lcendptr = lcptr + localDataSize;
	while (lcptr != lcendptr) {
	  double sub = ((int)*optr++ - (int)*gcptr++) - *lcptr++;
	  d += sub * sub;
	}
	distance += d;
	distanceLUT.set(li * localCodebookCentroidNo + localID[li], d);
      }
    }
    return sqrt(distance);	
  }
#ifdef NGTQBG_MIN
  inline float operator()(void *inv, float *distances, size_t size, DistanceLookupTableUint8 &distanceLUT) {
#else
  inline void operator()(void *inv, float *distances, size_t size, DistanceLookupTableUint8 &distanceLUT) {
#endif
    cerr << "operator is not implemented" << endl;
    abort();
  }
#endif 

};

template <typename T>
class QuantizedObjectDistanceFloat : public QuantizedObjectDistance {
public:

#ifdef NGTQ_DISTANCE_ANGLE
  inline double operator()(void *l, DistanceLookupTable &distanceLUT) {
    T *localID = static_cast<T*>(l);
    double normA = 0.0F;
    double normB = 0.0F;
    double sum = 0.0F;
    for (size_t li = 0; li < localDivisionNo; li++) {
      LocalDistanceLookup &ldl = *(distanceLUT.localDistanceLookup + li * localCodebookCentroidNo + localID[li]);
      normA += ldl.a;
      normB += ldl.b;
      sum += ldl.sum;
    }
    double cosine = sum / (sqrt(normA) * sqrt(normB));
    if (cosine >= 1.0F) {
      return 0.0F;
    } else if (cosine <= -1.0F) {
      return acos(-1.0F);
    }
    return acos(cosine);
  }
  inline double operator()(tann::Object &object, size_t objectID, void *l) {
    return getAngleDistanceFloat(object, objectID, static_cast<T*>(l));
  }
  inline double operator()(tann::Object &object, size_t objectID, void *l, DistanceLookupTable &distanceLUT) {
    cerr << "operator() is not implemented." << endl;
    abort();
    return 0.0;
  }
#else 
  inline double operator()(void *l, DistanceLookupTable &distanceLUT) {
    T *localID = static_cast<T*>(l);
    float *lut = distanceLUT.localDistanceLookup;
    float *end = lut + localCodebookCentroidNo * localDivisionNo;
    double distance = 0.0;
    while (lut < end) {
      distance += *(lut + *localID);
      localID++;
      lut += localCodebookCentroidNo;
    }
    return sqrt(distance);	
  }



#if defined(NGTQG_AVX2)
  inline float horizontalMin(__m256 &a, __m256 &b, float noOfObjects) {
    __m256 seq = _mm256_set_ps(8, 7, 6, 5, 4, 3, 2, 1);
    __m256 mask = _mm256_sub_ps(_mm256_set1_ps(noOfObjects), seq);
    mask = _mm256_blendv_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(std::numeric_limits<float>::max()), mask);
    __m256 data = _mm256_max_ps(a, mask);
    noOfObjects -= 8;
    if (noOfObjects > 0.0) {
      mask = _mm256_sub_ps(_mm256_set1_ps(noOfObjects), seq);
      mask = _mm256_blendv_ps(_mm256_set1_ps(0.0), _mm256_set1_ps(std::numeric_limits<float>::max()), mask);
      b = _mm256_max_ps(b, mask);
      data = _mm256_min_ps(data, b);
    }

    data = _mm256_min_ps(data, (__m256)_mm256_permute4x64_epi64((__m256i)data, _MM_SHUFFLE(3, 2, 3, 2)));
    data = _mm256_min_ps(data, (__m256)_mm256_srli_si256((__m256i)data, 8));
    data = _mm256_min_ps(data, (__m256)_mm256_srli_si256((__m256i)data, 4));
    //std::cerr << "ret=" << data[0] << std::endl;
    return data[0];
  }
#endif
  
#if defined(NGTQG_AVX512) || defined(NGTQG_AVX2)
#if defined(NGTQ_TOTAL_SCALE_OFFSET_COMPRESSION) 
#ifdef NGTQBG_MIN
  inline float operator()(void *inv, float *distances, size_t noOfObjects, DistanceLookupTableUint8 &distanceLUT) {
#else
  inline void operator()(void *inv, float *distances, size_t noOfObjects, DistanceLookupTableUint8 &distanceLUT) {
#endif


    uint8_t *localID = static_cast<uint8_t*>(inv);
    float *d = distances;
#ifdef NGTQBG_MIN
    float *lastd = distances + noOfObjects;
    float min = std::numeric_limits<float>::max();
#endif
#if defined(NGTQG_AVX512)
    const __m512i mask512x0F = _mm512_set1_epi16(0x000f);
    const __m512i mask512xF0 = _mm512_set1_epi16(0x00f0);
    const size_t range512 = distanceLUT.range512;
    auto step512 = distanceLUT.step512;
#endif 
    const __m256i mask256x0F = _mm256_set1_epi16(0x000f);
    const __m256i mask256xF0 = _mm256_set1_epi16(0x00f0);
    const size_t range256 = distanceLUT.range256;
    auto step256 = distanceLUT.step256;
    auto *last = localID + range256 / NGTQ_SIMD_BLOCK_SIZE * noOfObjects;
    while (localID < last) {
      uint8_t *lut = distanceLUT.localDistanceLookup;
      auto *lastgroup256 = localID + range256;
#if defined(NGTQG_AVX512)
      __m512i depu16 = _mm512_setzero_si512();
      auto *lastgroup512 = localID + range512;
      while (localID < lastgroup512) {
	__m512i lookupTable = _mm512_loadu_si512((__m512i const*)lut);
	_mm_prefetch(&localID[0] + 64 * 8, _MM_HINT_T0); 
	__m512i packedobj = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i const*)&localID[0]));  
	__m512i lo = _mm512_and_si512(packedobj, mask512x0F);  
	__m512i hi = _mm512_slli_epi16(_mm512_and_si512(packedobj, mask512xF0), 4);  
	__m512i obj = _mm512_or_si512(lo, hi);  
	__m512i vtmp = _mm512_shuffle_epi8(lookupTable, obj);  
        depu16 = _mm512_adds_epu16(depu16, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(vtmp, 0)));
	depu16 = _mm512_adds_epu16(depu16, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(vtmp, 1)));
	lut += (localCodebookCentroidNo - 1) * 4;
	localID += step512;
      } 
#else 
      __m256i depu16l = _mm256_setzero_si256();
      __m256i depu16h = _mm256_setzero_si256();
#endif 
      while (localID < lastgroup256) {
	__m256i lookupTable = _mm256_loadu_si256((__m256i const*)lut);
	_mm_prefetch(&localID[0] + 64 * 8, _MM_HINT_T0); 
	__m256i packedobj = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i const*)&localID[0]));
	__m256i lo = _mm256_and_si256(packedobj, mask256x0F);
	__m256i hi = _mm256_slli_epi16(_mm256_and_si256(packedobj, mask256xF0), 4);
	__m256i obj = _mm256_or_si256(lo, hi);
	__m256i vtmp = _mm256_shuffle_epi8(lookupTable, obj);

#if defined(NGTQG_AVX512)
        depu16 = _mm512_adds_epu16(depu16, _mm512_cvtepu8_epi16(vtmp));
#else
	depu16l = _mm256_adds_epu16(depu16l, _mm256_cvtepu8_epi16(_mm256_extractf128_si256(vtmp, 0)));
	depu16h = _mm256_adds_epu16(depu16h, _mm256_cvtepu8_epi16(_mm256_extractf128_si256(vtmp, 1)));
#endif
	lut += (localCodebookCentroidNo - 1) * 2;
	localID += step256;
      } 
#if defined(NGTQG_AVX512)
      __m512i lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(depu16, 0));
      __m512i hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(depu16, 1));

      __m512 distance = _mm512_cvtepi32_ps(_mm512_add_epi32(lo, hi));
      __m512 scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&distanceLUT.scales[0]));
      distance = _mm512_mul_ps(distance, scale);
      distance = _mm512_add_ps(distance, _mm512_set1_ps(distanceLUT.totalOffset));
#if defined(NGTQG_DOT_PRODUCT)
      float one = 1.0;
      float two = 2.0;
      distance = _mm512_mul_ps(_mm512_sub_ps(_mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&one)), distance), _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&two)));
#endif
      distance = _mm512_sqrt_ps(distance);  
      _mm512_storeu_ps(d, distance);
#ifdef NGTQBG_MIN
      {
	float tmpmin;
	int rest = 16 - (lastd - d);
	if (rest > 0) {
	  __mmask16 mask = 0xffff;
	  mask >>= rest;
	  tmpmin = _mm512_mask_reduce_min_ps(mask, distance);
	} else {
	  tmpmin = _mm512_reduce_min_ps(distance);
	}
	//std::cerr << "tmpmin=" << tmpmin << std::endl;
	if (min > tmpmin) min = tmpmin;
      }
#endif
#else 
      __m256i lol = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16l, 0));
      __m256i loh = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16l, 1));
      __m256i hil = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16h, 0));
      __m256i hih = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16h, 1));
      __m256 distancel = _mm256_cvtepi32_ps(_mm256_add_epi32(lol, hil));
      __m256 distanceh = _mm256_cvtepi32_ps(_mm256_add_epi32(loh, hih));
      __m256 scalel = _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&distanceLUT.scales[0]));
      __m256 scaleh = _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&distanceLUT.scales[0]));
      distancel = _mm256_mul_ps(distancel, scalel);
      distancel = _mm256_add_ps(distancel, _mm256_set1_ps(distanceLUT.totalOffset));
      distanceh = _mm256_mul_ps(distanceh, scaleh);
      distanceh = _mm256_add_ps(distanceh, _mm256_set1_ps(distanceLUT.totalOffset));
#if defined(NGTQG_DOT_PRODUCT)
      float one = 1.0;
      float two = 2.0;
      distancel = _mm256_mul_ps(_mm256_sub_ps(_mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&one)), distancel), _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&two)));
      distanceh = _mm256_mul_ps(_mm256_sub_ps(_mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&one)), distanceh), _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&two)));
#endif
      distancel = _mm256_sqrt_ps(distancel);  
      distanceh = _mm256_sqrt_ps(distanceh);  
      _mm256_storeu_ps(d, distancel);
      _mm256_storeu_ps(d + 8, distanceh);
#ifdef NGTQBG_MIN
      {
	float tmpmin = horizontalMin(distancel, distanceh, lastd - d);
	if (min > tmpmin) min = tmpmin;
      }
#endif
#endif 
      d += 16;
    } 
#ifdef NGTQBG_MIN
    return min;
#endif
  }

#else /// NGTQ_TOTAL_SCALE_OFFSET_COMPRESSION  ////////////////////////////////////////
#ifndef NGT_AVX512
#error "AVX512 is *NOT* defined. *INDIVIDUAL* scale offset compression is available only for AVX512!"
#endif
#ifdef NGTQBG_MIN
  inline float operator()(void *inv, float *distances, size_t noOfObjects, DistanceLookupTableUint8 &distanceLUT) {
#else
  inline void operator()(void *inv, float *distances, size_t noOfObjects, DistanceLookupTableUint8 &distanceLUT) {
#endif

    uint8_t *localID = static_cast<uint8_t*>(inv);
    float *d = distances;
#ifdef NGTQBG_MIN
    float *lastd = distances + noOfObjects;
    float min = std::numeric_limits<float>::max();
#endif
#if defined(NGTQG_AVX512)
    __m512i mask512x0F = _mm512_set1_epi16(0x000f);
    __m512i mask512xF0 = _mm512_set1_epi16(0x00f0);
    const size_t range512 = distanceLUT.range512;
    auto step512 = distanceLUT.step512;
#endif 
    const __m256i mask256x0F = _mm256_set1_epi16(0x000f);
    const __m256i mask256xF0 = _mm256_set1_epi16(0x00f0);
    const size_t range256 = distanceLUT.range256;
    auto step256 = distanceLUT.step256;
    auto *last = localID + range256 / NGTQ_SIMD_BLOCK_SIZE * noOfObjects;
    while (localID < last) {
      uint8_t *lut = distanceLUT.localDistanceLookup;
      float *scales = distanceLUT.scales;
      auto *lastgroup256 = localID + range256;
      __m512 distance = _mm512_setzero_ps();
#if defined(NGTQG_AVX512)
      //__m512i depu16 = _mm512_setzero_si512();
      auto *lastgroup512 = localID + range512;
      while (localID < lastgroup512) {
	__m512i lookupTable = _mm512_loadu_si512((__m512i const*)lut);
	_mm_prefetch(&localID[0] + 64 * 8, _MM_HINT_T0); 
	__m512i packedobj = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i const*)&localID[0]));  
	__m512i lo = _mm512_and_si512(packedobj, mask512x0F);  
	__m512i hi = _mm512_slli_epi16(_mm512_and_si512(packedobj, mask512xF0), 4);  
	__m512i obj = _mm512_or_si512(lo, hi);  
	__m512i vtmp = _mm512_shuffle_epi8(lookupTable, obj);  

	__m512 d = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(vtmp, 0))); 
	__m512 scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&scales[0]));
	distance = _mm512_add_ps(distance, _mm512_mul_ps(d, scale));
	d = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(vtmp, 1))); 
	scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&scales[1]));
	distance = _mm512_add_ps(distance, _mm512_mul_ps(d, scale));
	d = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(vtmp, 2))); 
	scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&scales[2]));
	distance = _mm512_add_ps(distance, _mm512_mul_ps(d, scale));
	d = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti64x2_epi64(vtmp, 3))); 
	scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&scales[3]));
	distance = _mm512_add_ps(distance, _mm512_mul_ps(d, scale));

	lut += (localCodebookCentroidNo - 1) * 4;
	scales += 4;
	localID += step512;
      } 
#else 
      __m256i depu16l = _mm256_setzero_si256();
      __m256i depu16h = _mm256_setzero_si256();
#endif 
      while (localID < lastgroup256) {
	__m256i lookupTable = _mm256_loadu_si256((__m256i const*)lut);
	_mm_prefetch(&localID[0] + 64 * 8, _MM_HINT_T0); 
	//std::cerr << "obj=" << (int)(localID[0] & 0x0f) << "," << (int)((localID[0] >> 4) & 0x0f) << std::endl; 
	__m256i packedobj = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i const*)&localID[0]));
	__m256i lo = _mm256_and_si256(packedobj, mask256x0F);
	__m256i hi = _mm256_slli_epi16(_mm256_and_si256(packedobj, mask256xF0), 4);
	__m256i obj = _mm256_or_si256(lo, hi);
	//std::cerr << "LUT=" << (int)*lut << "," << (int)*(lut+1) << std::endl;
	__m256i vtmp = _mm256_shuffle_epi8(lookupTable, obj);

#if defined(NGTQG_AVX512)
	__m512 d = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti32x4_epi32(vtmp, 0)));
	__m512 scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&scales[0]));
	distance = _mm512_add_ps(distance, _mm512_mul_ps(d, scale));
	d = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm256_extracti32x4_epi32(vtmp, 1)));
	scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&scales[1]));
	distance = _mm512_add_ps(distance, _mm512_mul_ps(d, scale));
	////////////////////
#else
	depu16l = _mm256_adds_epu16(depu16l, _mm256_cvtepu8_epi16(_mm256_extractf128_si256(vtmp, 0)));
	depu16h = _mm256_adds_epu16(depu16h, _mm256_cvtepu8_epi16(_mm256_extractf128_si256(vtmp, 1)));
#endif
	lut += (localCodebookCentroidNo - 1) * 2;
	scales += 2;
	localID += step256;
      } 

#if defined(NGTQG_AVX512)
      //__m512i lo = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(depu16, 0));
      //__m512i hi = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(depu16, 1));
      //__m512 scale = _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&distanceLUT.scales[0]));
      //distance = _mm512_mul_ps(distance, scale);
      distance = _mm512_add_ps(distance, _mm512_set1_ps(distanceLUT.totalOffset));
#if defined(NGTQG_DOT_PRODUCT)
      float one = 1.0;
      float two = 2.0;
      distance = _mm512_mul_ps(_mm512_sub_ps(_mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&one)), distance), _mm512_broadcastss_ps(*reinterpret_cast<__m128*>(&two)));
#endif
      distance = _mm512_sqrt_ps(distance);  
      _mm512_storeu_ps(d, distance);
#ifdef NGTQBG_MIN
      {
	float tmpmin;
	int rest = 16 - (lastd - d);
	if (rest > 0) {
	  __mmask16 mask = 0xffff;
	  mask >>= rest;
	  tmpmin = _mm512_mask_reduce_min_ps(mask, distance);
	} else {
	  tmpmin = _mm512_reduce_min_ps(distance);
	}
	//std::cerr << "tmpmin=" << tmpmin << std::endl;
	if (min > tmpmin) min = tmpmin;
      }
#endif
#else 
      __m256i lol = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16l, 0));
      __m256i loh = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16l, 1));
      __m256i hil = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16h, 0));
      __m256i hih = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(depu16h, 1));
      __m256 distancel = _mm256_cvtepi32_ps(_mm256_add_epi32(lol, hil));
      __m256 distanceh = _mm256_cvtepi32_ps(_mm256_add_epi32(loh, hih));
      __attribute__((aligned(32))) float v32[8];
      _mm256_storeu_ps((float*)&v32, distancel);
      _mm256_storeu_ps((float*)&v32, distanceh);
      __m256 scalel = _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&distanceLUT.scales[0]));
      __m256 scaleh = _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&distanceLUT.scales[0]));
      distancel = _mm256_mul_ps(distancel, scalel);
      distancel = _mm256_add_ps(distancel, _mm256_set1_ps(distanceLUT.totalOffset));
      distanceh = _mm256_mul_ps(distanceh, scaleh);
      distanceh = _mm256_add_ps(distanceh, _mm256_set1_ps(distanceLUT.totalOffset));
#if defined(NGTQG_DOT_PRODUCT)
      float one = 1.0;
      float two = 2.0;
      distancel = _mm256_mul_ps(_mm256_sub_ps(_mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&one)), distancel), _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&two)));
      distanceh = _mm256_mul_ps(_mm256_sub_ps(_mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&one)), distanceh), _mm256_broadcastss_ps(*reinterpret_cast<__m128*>(&two)));
#endif
      distancel = _mm256_sqrt_ps(distancel);  
      distanceh = _mm256_sqrt_ps(distanceh);  
      _mm256_storeu_ps(d, distancel);
      _mm256_storeu_ps(d + 8, distanceh);
#endif 
      d += 16;
    } 
#ifdef NGTQBG_MIN
    return min;
#endif
  }
#endif /// NGTQ_TOTAL_SCALE_OFFSET_COMPRESSION  //////////////////////////////////////// 

#else 
#ifdef NGTQBG_MIN
  inline float operator()(void *inv, float *distances, size_t size, DistanceLookupTableUint8 &distanceLUT) {
#else
  inline void operator()(void *inv, float *distances, size_t size, DistanceLookupTableUint8 &distanceLUT) {
#endif
    uint8_t *localID = static_cast<uint8_t*>(inv);
    size_t numOfAlignedSubvectors = ((localDivisionNo - 1) / NGTQ_BATCH_SIZE + 1) * NGTQ_BATCH_SIZE;
    size_t alignedSize = ((size - 1) / 2 + 1) * 2;
    uint32_t d[NGTQ_SIMD_BLOCK_SIZE];
    size_t didx = 0;
    size_t byteSize = numOfAlignedSubvectors * alignedSize / 2;
    auto *last = localID + byteSize;
    while (localID < last) {
      uint8_t *lut = distanceLUT.localDistanceLookup;
      memset(d, 0, sizeof(uint32_t) * NGTQ_SIMD_BLOCK_SIZE);
      for (size_t li = 0; li < numOfAlignedSubvectors; li++) {
	for (size_t i = 0; i < NGTQ_SIMD_BLOCK_SIZE; i++) {
	  uint8_t obj = *localID;
	  if (i % 2 == 0) {
	    obj &= 0x0f;
	  } else {
	    obj >>= 4;
	    localID++;
	  }
	  d[i] += *(lut + obj);
	}
	lut += localCodebookCentroidNo - 1;
      }
      for (size_t i = 0; i < NGTQ_SIMD_BLOCK_SIZE; i++) {
	distances[didx + i] = sqrt(static_cast<float>(d[i]) * distanceLUT.scales[0] + distanceLUT.totalOffset);
      }
      didx += NGTQ_SIMD_BLOCK_SIZE;
    }
  }
#endif 


  inline double operator()(tann::Object &object, size_t objectID, void *l) {
    return getL2DistanceFloat(object, objectID, static_cast<T*>(l));
  }
  inline double operator()(tann::Object &object, size_t objectID, void *l, DistanceLookupTable &distanceLUT) {
    T *localID = static_cast<T*>(l);
    tann::PersistentObject &gcentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(objectID);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    float *gcptr = (float*)&gcentroid.at(0, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
    float *gcptr = (float*)&gcentroid[0];
#endif
    float *optr = (float*)&((tann::Object&)object)[0];
    double distance = 0.0;
    for (size_t li = 0; li < localDivisionNo; li++) {
      size_t distanceLUTidx = li * localCodebookCentroidNo + localID[li];
      if (distanceLUT.isValid(distanceLUTidx)) {
	distance += distanceLUT.getDistance(distanceLUTidx);
	optr += localDataSize;
	gcptr += localDataSize;
      } else {
        size_t idx = li;
	tann::PersistentObject &lcentroid = *localCodebookIndexes[idx].getObjectSpace().getRepository().get(localID[li]);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	float *lcptr = (float*)&lcentroid.at(0, localCodebookIndexes[idx].getObjectSpace().getRepository().allocator);
#else
	float *lcptr = (float*)&lcentroid[0];
#endif
#if defined(NGTQG_AVX512) || defined(NGTQG_AVX2)
	float *lcendptr = lcptr + localDataSize;
	__m256 sum256 = _mm256_setzero_ps();
	__m256 v;
	while (lcptr < lcendptr) {
	  v = _mm256_sub_ps(_mm256_sub_ps(_mm256_loadu_ps(optr), _mm256_loadu_ps(gcptr)), _mm256_loadu_ps(lcptr));
	  sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v, v));
	  optr += 8;
	  gcptr += 8;
	  lcptr += 8;
	}
	__m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
	__attribute__((aligned(32))) float f[4];
	_mm_store_ps(f, sum128);
	double d = f[0] + f[1] + f[2] + f[3];
#else
	float *lcendptr = lcptr + localDataSize;
	double d = 0.0;
	while (lcptr != lcendptr) {
	  double sub = (*optr++ - *gcptr++) - *lcptr++;
	  d += sub * sub;
	}
#endif
	distance += d;
      }
    }
    return sqrt(distance);	
  }
#endif 

};


class Quantizer {
public:
#ifdef NGTQ_STATIC_OBJECT_FILE
  typedef StaticObjectFile<tann::Object>	ObjectList;
#else
  typedef ObjectFile	ObjectList;	
#endif


  virtual ~Quantizer() { }

  virtual void create(const string &index,
		      tann::Property &globalPropertySet,
#ifdef NGTQ_QBG
		      tann::Property &localPropertySet,
		      std::vector<float> *rotation = 0,
		      const string &objectFile = "") = 0;
#else
		      tann::Property &localPropertySet) = 0;
#endif
#ifndef NGTQ_QBG
  virtual void insert(vector<pair<tann::Object*, size_t> > &objects) = 0;
  virtual void insert(vector<float> &object, vector<pair<tann::Object*, size_t> > &objects, size_t id) = 0;
#endif
  virtual void insertIntoObjectRepository(vector<float> &object, size_t id) = 0;
#ifdef NGTQ_QBG
  virtual void createIndex(size_t beginID, size_t endID) = 0;
#endif
  virtual void setupInvertedIndex(std::vector<std::vector<float>> &quantizationCodebook,
					std::vector<uint32_t> &codebookIndex,
					std::vector<uint32_t> &objectIndex) = 0;
#ifndef NGTQ_QBG
  virtual void rebuildIndex() = 0;
#endif
  virtual void save() = 0;
  virtual void saveRotation(const std::vector<float> &rotation) = 0;
  virtual void open(const string &index, tann::Property &globalProperty, bool readOnly) = 0;
  virtual void open(const string &index, bool readOnly) = 0;
  virtual void close() = 0;
  virtual void closeCodebooks() = 0;
#ifdef NGTQ_SHARED_INVERTED_INDEX
  virtual void reconstructInvertedIndex(const string &indexFile) = 0;
#endif

  virtual void validate() = 0;

#ifdef NGTQ_QBG
  virtual void extractInvertedIndexObject(InvertedIndexEntry<uint16_t> &invertedIndexObjects, size_t id) = 0;
  virtual void extractInvertedIndexObject(InvertedIndexEntry<uint16_t> &invertedIndexObjects) = 0;
#endif
  virtual void extractInvertedIndex(std::vector<std::vector<uint32_t>> &invertedIndex) = 0;
  virtual void eraseInvertedIndexObject(size_t id) = 0;
  virtual void eraseInvertedIndexObject() = 0;

  virtual tann::distance_type getApproximateDistance(tann::Object &query, uint32_t globalID, uint16_t *localID, QuantizedObjectDistance::DistanceLookupTable &distanceLUT) {
    std::cerr << "getApproximateDistance() is not implemented." << std::endl;
    abort();
  }

  virtual void search(tann::Object *object, tann::VectorDistances &objs, size_t size,
		      size_t approximateSearchSize,
		      size_t codebookSearchSize, bool resultRefinement, bool lookUpTable,
		      double epsilon) = 0;

  virtual void search(tann::Object *object, tann::VectorDistances &objs, size_t size,
		      size_t approximateSearchSize,
		      size_t codebookSearchSize, AggregationMode aggregationMode,
		      double epsilon) = 0;

  virtual void search(tann::Object *object, tann::VectorDistances &objs, size_t size,
		      float expansion,
		      AggregationMode aggregationMode,
		      double epsilon) = 0;

  virtual void info(ostream &os, char mode) = 0;

  virtual void verify() = 0;

  virtual size_t getInstanceSharedMemorySize(ostream &os, tann::SharedMemoryAllocator::GetMemorySizeType t = tann::SharedMemoryAllocator::GetTotalMemorySize) = 0;

  tann::Object *allocateObject(string &line, const string &sep) {
    return globalCodebookIndex.allocateObject(line, " \t");
  }
  tann::Object *allocateObject(vector<double> &obj) {
    return globalCodebookIndex.allocateObject(obj);
  }
  tann::Object *allocateObject(vector<float> &obj) {
    return globalCodebookIndex.allocateObject(obj);
  }
  void deleteObject(tann::Object *object) { globalCodebookIndex.deleteObject(object); }
  
  void setThreadSize(size_t size) { property.threadSize = size; }
  void setGlobalRange(float r) { property.globalRange = r; }
  void setLocalRange(float r) { property.localRange = r; }
  void setGlobalCentroidLimit(size_t s) { property.globalCentroidLimit = s; }
  void setLocalCentroidLimit(size_t s) { property.localCentroidLimit = s; }
  void setDimension(size_t s) { property.dimension = s; }
  void setDistanceType(MetricType t) { property.distanceType = t; }

  tann::Index &getLocalCodebook(size_t idx) { return localCodebookIndexes[idx]; }
  size_t getLocalCodebookSize(size_t size) { return localCodebookIndexes[size].getObjectRepositorySize(); }

  string getRootDirectory() { return rootDirectory; }

  size_t getSharedMemorySize(ostream &os, tann::SharedMemoryAllocator::GetMemorySizeType t = tann::SharedMemoryAllocator::GetTotalMemorySize) {
    os << "Global centroid:" << endl;
    return globalCodebookIndex.getSharedMemorySize(os, t) + getInstanceSharedMemorySize(os, t);
  }

  virtual QuantizedObjectDistance &getQuantizedObjectDistance() = 0;

  virtual size_t getInvertedIndexSize() = 0;

  ObjectList	objectList;
  string	rootDirectory;

  Property	property;

  tann::Index	globalCodebookIndex;

  size_t	distanceComputationCount;

  size_t	localIDByteSize;
  tann::DataType objectType;
  size_t	divisionNo;

  std::vector<tann::Index>	localCodebookIndexes;

  QuantizationCodebook<float>	quantizationCodebook;
  std::vector<uint32_t>		objectToBlobIndex;
  Rotation			rotation;

};

class QuantizedObjectProcessingStream {
 public:
 QuantizedObjectProcessingStream(Quantizer &quantizer, size_t numOfObjects): stream(0) {
    initialize(quantizer.divisionNo);
    setStreamSize(numOfObjects);
    stream = new uint8_t[streamSize]();
  }

  QuantizedObjectProcessingStream(size_t numOfSubspaces): stream(0) {
    initialize(numOfSubspaces);
  }

  ~QuantizedObjectProcessingStream() {
    delete[] stream;
  }

  void initialize(size_t divisionNo) {
    numOfAlignedSubvectors = ((divisionNo - 1) / NGTQ_BATCH_SIZE + 1) * NGTQ_BATCH_SIZE;
    alignedBlockSize = NGTQ_SIMD_BLOCK_SIZE * numOfAlignedSubvectors;
  }

  static size_t getNumOfAlignedObjects(size_t numOfObjects) {
    return (((numOfObjects - 1) / NGTQ_SIMD_BLOCK_SIZE + 1) * NGTQ_SIMD_BLOCK_SIZE);
  }
  
  void setStreamSize(size_t numOfObjects) {
    numOfAlignedObjects  = getNumOfAlignedObjects(numOfObjects);
    streamSize = numOfAlignedObjects * numOfAlignedSubvectors;
    return;
  }

#ifdef NGTQ_QBG
  void arrangeQuantizedObject(size_t dataNo, size_t subvectorNo, uint8_t quantizedObject) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    abort();
#else
    size_t blkNo = dataNo / NGTQ_SIMD_BLOCK_SIZE;	
    size_t oft = dataNo - blkNo * NGTQ_SIMD_BLOCK_SIZE;	
    stream[blkNo * alignedBlockSize + NGTQ_SIMD_BLOCK_SIZE * subvectorNo + oft] = quantizedObject;
#endif
  }
#endif 

  uint8_t* compressIntoUint4() {
    size_t idx = 0;
    size_t uint4StreamSize = streamSize / 2;
    uint8_t *uint4Objects = new uint8_t[uint4StreamSize]();
    while (idx < streamSize) {
      for (size_t lidx = 0; lidx < numOfAlignedSubvectors; lidx++) {
	for (size_t bidx = 0; bidx < NGTQ_SIMD_BLOCK_SIZE; bidx++) {
	  if (idx / 2 > uint4StreamSize) {
	    std::stringstream msg;
	    msg << "Quantizer::compressIntoUint4: Fatal inner error! " << (idx / 2) << ":" << uint4StreamSize;
	    TANN_THROW(msg);
	  }
	  if (idx % 2 == 0) {
	    uint4Objects[idx / 2] = stream[idx];
	  } else {
	    uint4Objects[idx / 2] |= (stream[idx] << 4);
	  }
	  idx++;
	}
      }
    }
    return uint4Objects;
  }
  
  uint8_t*	getStream() {
    auto s = stream;
    stream = 0;
    return s;
  }

  size_t getUint4StreamSize(size_t numOfObjects) {
    setStreamSize(numOfObjects);
    return streamSize / 2;
  }

  size_t getStreamSize(size_t numOfObjects) {
    setStreamSize(numOfObjects);
    return streamSize;
  }

  uint8_t	*stream;
  size_t	numOfAlignedSubvectors;
  size_t	alignedBlockSize;
  size_t	numOfAlignedObjects ;
  size_t	streamSize;
};
 
class GenerateResidualObject {
public:
  GenerateResidualObject():globalCodebookIndex(0), objectList(0), quantizationCodebook(0) {}
  virtual ~GenerateResidualObject() {}
#ifdef NGTQ_QBG

#ifdef NGTQ_VECTOR_OBJECT
  virtual void operator()(std::vector<float> &object, size_t centroidID, float *subspaceObject) = 0;
#else
  virtual void operator()(tann::Object &object, size_t centroidID, float *subspaceObject) = 0;
#endif
  virtual void operator()(tann::Object &object, size_t centroidID,
			  vector<vector<pair<tann::Object*, size_t> > > &localObjs) = 0;
#else
  virtual void operator()(size_t objectID, size_t centroidID, 
			  vector<vector<pair<tann::Object*, size_t> > > &localObjs) = 0;
#endif
  void set(tann::Index &gc, tann::Index lc[], size_t dn, size_t lcn,
	   Quantizer::ObjectList *ol, QuantizationCodebook<float> *qc) {
    globalCodebookIndex = &(tann::GraphAndTreeIndex&)gc.getIndex();
    divisionNo = dn;
    objectList = ol;
    set(lc, lcn);
    quantizationCodebook = qc;
  }
  void set(tann::Index lc[], size_t lcn) {
    localCodebookIndexes.clear();
    localCodebookNo = lcn;
    for (size_t i = 0; i < localCodebookNo; ++i) {
      localCodebookIndexes.push_back(&(tann::GraphAndTreeIndex&)lc[i].getIndex());
    }
  }

  tann::GraphAndTreeIndex		*globalCodebookIndex;
  vector<tann::GraphAndTreeIndex*>	localCodebookIndexes;
  size_t				divisionNo;
  size_t				localCodebookNo;
  Quantizer::ObjectList			*objectList;
  QuantizationCodebook<float>		*quantizationCodebook;
};

class GenerateResidualObjectUint8 : public GenerateResidualObject {
public:
#ifdef NGTQ_QBG
#ifdef NGTQ_VECTOR_OBJECT
  void operator()(std::vector<float> &object, size_t centroidID, float *subspaceObject) { abort(); }
#else
  void operator()(tann::Object &object, size_t centroidID, float *subspaceObject) { abort(); }
#endif
  void operator()(tann::Object &xobject, size_t centroidID,
		  vector<vector<pair<tann::Object*, size_t> > > &localObjs) { abort(); }
#else
  void operator()(size_t objectID, size_t centroidID, 
		  vector<vector<pair<tann::Object*, size_t> > > &localObjs) {
    tann::PersistentObject &globalCentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(centroidID);
    tann::Object object(&globalCodebookIndex->getObjectSpace());
    objectList->get(objectID, object, &globalCodebookIndex->getObjectSpace());
    size_t sizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t lsize = sizeOfObject / divisionNo;
    for (size_t di = 0; di < divisionNo; di++) {
      vector<double> subObject;
      subObject.resize(lsize);
      for (size_t d = 0; d < lsize; d++) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	subObject[d] = (double)object[di * lsize + d] - 
	  (double)globalCentroid.at(di * lsize + d, globalCodebookIndex->getObjectSpace().getRepository().allocator);
#else
	subObject[d] = (double)object[di * lsize + d] - (double)globalCentroid[di * lsize + d];
#endif
      }
      size_t idx = localCodebookNo == 1 ? 0 : di;
      tann::Object *localObj = localCodebookIndexes[idx]->allocateObject(subObject);
      localObjs[idx].push_back(pair<tann::Object*, size_t>(localObj, 0));
    }
  }
#endif

};

#ifdef NGTQ_QBG
class GenerateResidualObjectFloat : public GenerateResidualObject {
public:
#ifdef NGTQ_VECTOR_OBJECT
  void operator()(std::vector<float> &object, size_t centroidID, float *subspaceObject) {
#else
  void operator()(tann::Object &object, size_t centroidID, float *subspaceObject) {
#endif
    size_t dimension = globalCodebookIndex->getObjectSpace().getPaddedDimension();
#ifdef NGTQ_VECTOR_OBJECT
    auto *vector = object.data();
#else
    auto *vector = static_cast<float*>(object.getPointer());
#endif
    for (size_t d = 0; d < dimension; d++) {
#ifdef NGTQG_ZERO_GLOBAL
      subspaceObject[d] = vector[d];
#else
      if (centroidID == std::numeric_limits<uint32_t>::max()) {
	subspaceObject[d] = vector[d];
      } else {
	subspaceObject[d] = vector[d] - quantizationCodebook->at(centroidID, d);
      }
#endif
    }
  }
  void operator()(tann::Object &object, size_t centroidID,
		  vector<vector<pair<tann::Object*, size_t>>> &localObjs) {
    size_t byteSizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localByteSize = byteSizeOfObject / divisionNo;
    size_t localDimension = localByteSize / sizeof(float);
    for (size_t di = 0; di < divisionNo; di++) {
#ifndef NGTQG_ZERO_GLOBAL
      vector<double> subObject;
      subObject.resize(localDimension);
#endif
      float *subVector = static_cast<float*>(object.getPointer(di * localByteSize));
#ifndef NGTQG_ZERO_GLOBAL
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      float *quantizationCodebookSubvector = quantizationCodebook->data(centroidID) + di * localDimension;
#else
      float *quantizationCodebookSubvector = quantizationCodebook->data(centroidID) + di * localDimension;
#endif
      for (size_t d = 0; d < localDimension; d++) {
	subObject[d] = static_cast<double>(subVector[d]) - static_cast<double>(quantizationCodebookSubvector[d]);
      }
#endif /// /////////////////////
      size_t idx = localCodebookNo == 1 ? 0 : di;
      tann::Object *localObj = localCodebookIndexes[idx]->allocateObject(subVector, localDimension);
      localObjs[idx].push_back(pair<tann::Object*, size_t>(localObj, 0));
    }
  }
};

#else 
class GenerateResidualObjectFloat : public GenerateResidualObject {
public:
  void operator()(size_t objectID, size_t centroidID, 
		  vector<vector<pair<tann::Object*, size_t> > > &localObjs) {
    tann::PersistentObject &globalCentroid = *globalCodebookIndex->getObjectSpace().getRepository().get(centroidID);
    tann::Object object(&globalCodebookIndex->getObjectSpace());
    objectList->get(objectID, object, &globalCodebookIndex->getObjectSpace());
    size_t byteSizeOfObject = globalCodebookIndex->getObjectSpace().getByteSizeOfObject();
    size_t localByteSize = byteSizeOfObject / divisionNo;
    size_t localDimension = localByteSize / sizeof(float);
    for (size_t di = 0; di < divisionNo; di++) {
      vector<double> subObject;
      subObject.resize(localDimension);
      float *subVector = static_cast<float*>(object.getPointer(di * localByteSize));
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      float *globalCentroidSubVector = static_cast<float*>(globalCentroid.getPointer(di * localByteSize, 
										     globalCodebookIndex->getObjectSpace().getRepository().allocator));
#else
      float *globalCentroidSubVector = static_cast<float*>(globalCentroid.getPointer(di * localByteSize));
#endif
      for (size_t d = 0; d < localDimension; d++) {
	subObject[d] = (double)subVector[d] - (double)globalCentroidSubVector[d];
      }
      size_t idx = localCodebookNo == 1 ? 0 : di;
      tann::Object *localObj = localCodebookIndexes[idx]->allocateObject(subObject);
      localObjs[idx].push_back(pair<tann::Object*, size_t>(localObj, 0));
    }
  }
};
#endif 
 
template <typename LOCAL_ID_TYPE>
class QuantizerInstance : public Quantizer {
public:

  typedef void (QuantizerInstance::*AggregateObjectsFunction)(tann::VectorDistance &, tann::Object *, size_t size, tann::VectorSpace::ResultSet &, size_t);
  typedef InvertedIndexEntry<LOCAL_ID_TYPE>	IIEntry;

  QuantizerInstance() {
    quantizedObjectDistance = 0;
    generateResidualObject = 0;
    localCodebooks = 0;
    silence = true;
  }

  virtual ~QuantizerInstance() { close(); }

  void createEmptyIndex(const string &index,
			tann::Property &globalProperty,
#ifdef NGTQ_QBG
			tann::Property &localProperty,
			std::vector<float> *rotation,
			const string &objectFile)
#else
			tann::Property &localProperty)
#endif
  {
    rootDirectory = index;
    tann::Index::mkdir(rootDirectory);
    string global = rootDirectory + "/global";
    tann::Index::mkdir(global);

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    tann::GraphAndTreeIndex globalCodebook(global, globalProperty);
    globalCodebook.saveIndex(global);
    globalCodebook.close();
#else
    tann::GraphAndTreeIndex globalCodebook(globalProperty);
    globalCodebook.saveIndex(global);
    globalCodebook.close();
#endif

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    size_t localCodebookNo = property.getLocalCodebookNo();
    for (size_t i = 0; i < localCodebookNo; ++i) {
      stringstream local;
      local << rootDirectory << "/local-" << i;
      tann::Index::mkdir(local.str());
      tann::GraphAndTreeIndex localCodebook(local.str(), localProperty);
      localCodebook.saveIndex(local.str());
    }
#else
    tann::GraphAndTreeIndex localCodebook(localProperty);
    size_t localCodebookNo = property.getLocalCodebookNo();
    for (size_t i = 0; i < localCodebookNo; ++i) {
      stringstream local;
      local << rootDirectory << "/local-" << i;
      tann::Index::mkdir(local.str());
      localCodebook.saveIndex(local.str());
    }
    localCodebook.close();
#endif
#ifdef NGTQ_SHARED_INVERTED_INDEX
    invertedIndex.open(index + "/ivt", property.invertedIndexSharedMemorySize);
#else
    ofstream of(rootDirectory + "/ivt");
    invertedIndex.serialize(of);
#endif
    string fname = rootDirectory + "/obj";
    if (property.dataSize == 0) {
      std::stringstream msg;
#ifdef NGTQ_QBG
      msg << "Quantizer: data size of the object is zero. " << property.dataSize << ":" << property.dimension 
	  << ":" << property.dataType << ":" << property.genuineDataType;
#else
      msg << "Quantizer: data size of the object is zero. " << property.dataSize << ":" << property.dimension 
	  << ":" << property.dataType;
#endif
      TANN_THROW(msg);
    }
#ifdef NGTQ_STATIC_OBJECT_FILE
    if (!objectList.create(fname, objectFile)) {
      std::stringstream msg;
      msg << "Quantizer::createEmptyIndex: cannot construct the object list. " << fname << ":" << objectFile;
      TANN_THROW(msg);
    }
    objectList.open(fname, property.dimension);
#ifdef MULTIPLE_OBJECT_LISTS
    objectList.openMultipleStreams(omp_get_max_threads());
#endif
#else 
    objectList.create(fname, property.dataSize);
#endif 
#ifdef NGTQ_QBG
    if (rotation != 0) {
      saveRotation(*rotation);
    }
#endif
    property.save(rootDirectory);
  }

  void saveRotation(const std::vector<float> &rotation) {
    Rotation r;
    r = rotation;
    ofstream ofs(rootDirectory + "/qr");
    r.serialize(ofs);
  }

  void open(const string &index, tann::Property &globalProperty, bool readOnly) {
    open(index, readOnly);
    globalCodebookIndex.setProperty(globalProperty);
  }

  void open(const string &index, bool readOnly) {
    tann::StdOstreamRedirector redirector(silence);
    redirector.begin();
    rootDirectory = index;
    property.load(rootDirectory);
    string globalIndex = index + "/global";
    globalCodebookIndex.open(globalIndex, readOnly);
    if ((globalCodebookIndex.getObjectRepositorySize() == 0) && readOnly) {
      std::cerr << "open: Warning. global codebook is empty." << std::endl;
    }
    size_t localCodebookNo = property.getLocalCodebookNo();

    localCodebookIndexes.resize(localCodebookNo);
    for (size_t i = 0; i < localCodebookNo; ++i) {
      stringstream localIndex;
      localIndex << index << "/local-" << i;
      localCodebookIndexes[i].open(localIndex.str());
    }
#ifdef NGTQ_QBG
    if (!readOnly) {
#else
    {
#endif

#ifdef NGTQ_SHARED_INVERTED_INDEX
      invertedIndex.open(index + "/ivt", 0);
#else
      ifstream ifs(index + "/ivt");
      if (!ifs) {
        cerr << "Cannot open " << index + "/ivt" << "." << endl;
        return;
      }
      invertedIndex.deserialize(ifs);
#endif
    }
#ifdef NGTQ_QBG
    if (!objectList.open(index + "/obj", property.genuineDataType, property.distanceType, property.dimension)) {
#else
    if (!objectList.open(index + "/obj", static_cast<ObjectFile::DataType>(property.dataType), property.distanceType, property.dimension)) {
#endif
      stringstream msg;
      msg << "NGTQ::Quantizer::open: cannot open the object file. " << index + "/obj" << std::endl;
      std::cerr << "Ignore. " << msg.str() << std::endl;
    }
#ifdef MULTIPLE_OBJECT_LISTS
    objectList.openMultipleStreams(omp_get_max_threads());
#endif
    tann::Property globalProperty;
    globalCodebookIndex.getProperty(globalProperty);
    size_t sizeoftype = 0;
#ifdef TANN_ENABLE_HALF_FLOAT
    if (globalProperty.objectType == tann::DataType::Float ||
	globalProperty.objectType == tann::DataType::Float16) {
#else
    if (globalProperty.objectType == tann::Property::ObjectType::Float) {
#endif
      if (property.localIDByteSize == 4) {
	quantizedObjectDistance = new QuantizedObjectDistanceFloat<uint32_t>;
      } else if (property.localIDByteSize == 2) {
	quantizedObjectDistance = new QuantizedObjectDistanceFloat<uint16_t>;
#ifdef NGTQ_QBG
      } else if (property.localIDByteSize == 1) {
	quantizedObjectDistance = new QuantizedObjectDistanceFloat<uint8_t>;
#endif
      } else {
	std::cerr << "Invalid localIDByteSize : " << property.localIDByteSize << std::endl;
	abort();
      }
      generateResidualObject = new GenerateResidualObjectFloat;
      sizeoftype = sizeof(float);
    } else if (globalProperty.objectType == tann::DataType::Uint8) {
      if (property.localIDByteSize == 4) {
	quantizedObjectDistance = new QuantizedObjectDistanceUint8<uint32_t>;
      } else if (property.localIDByteSize == 2) {
	quantizedObjectDistance = new QuantizedObjectDistanceUint8<uint16_t>;
      } else {
	abort();
      }
      generateResidualObject = new GenerateResidualObjectUint8;
      sizeoftype = sizeof(uint8_t);
    } else {
      cerr << "NGTQ::open: Fatal Inner Error: invalid object type. " << static_cast<int>(globalProperty.objectType) << endl;
      cerr << "   check NGT version consistency between the caller and the library." << endl;
      abort();
    }
    assert(quantizedObjectDistance != 0);
    quantizedObjectDistance->set(&globalCodebookIndex, localCodebookIndexes.data(), &quantizationCodebook,
				 property.localDivisionNo, property.getLocalCodebookNo(), sizeoftype,
				 property.dimension, &rotation);
    generateResidualObject->set(globalCodebookIndex, localCodebookIndexes.data(), property.localDivisionNo, property.getLocalCodebookNo(), &objectList, &quantizationCodebook);
    localIDByteSize = property.localIDByteSize;
    objectType = globalProperty.objectType;
    divisionNo = property.localDivisionNo;
#ifdef NGTQ_QBG
    {
      std::string streamName(rootDirectory + "/qr");
      ifstream ifs(streamName);
      if (ifs) {
	std::cerr << "loading rotation..." << std::endl;
	rotation.deserialize(ifs);
      }
    }
    {
#ifdef NGTQG_ROTATION
      std::string rqcbName(rootDirectory + "/rqcb");
      ifstream irfs(rqcbName);
      if (!irfs) {
	std::string qcbName(rootDirectory + "/qcb");
	ifstream ifs(qcbName);
	if (!ifs) {
	} else {
	  std::cerr << "loading the global codebooks..." << std::endl;
	  quantizationCodebook.deserialize(ifs, readOnly);
	  std::cerr << "rotating the global codebooks... " << rotation.size() << std::endl;
	  quantizationCodebook.rotate(rotation);
	}
      } else {
	std::cerr << "loading the rotated global codebooks..." << std::endl;
	quantizationCodebook.deserialize(irfs, readOnly);
      }
#else 
      std::string qcbName(rootDirectory + "/qcb");
      ifstream ifs(qcbName);
      quantizationCodebook.deserialize(ifs, readOnly);
#endif 
    }
#endif
    redirector.end();
  }

  void save() {
#ifndef NGT_SHARED_MEMORY_ALLOCATOR
    string global = rootDirectory + "/global";
    globalCodebookIndex.saveIndex(global);
    size_t localCodebookNo = property.getLocalCodebookNo();
    for (size_t i = 0; i < localCodebookNo; ++i) {
      stringstream local;
      local << rootDirectory << "/local-" << i;
      try {
	tann::Index::mkdir(local.str());
      } catch (...) {}
      localCodebookIndexes[i].saveIndex(local.str());
    }
#endif // NGT_SHARED_MEMORY_ALLOCATOR
#ifndef NGTQ_SHARED_INVERTED_INDEX
    {
      ofstream of(rootDirectory + "/ivt");
      invertedIndex.serialize(of);
    }
#endif
#ifdef NGTQ_QBG
    {
#ifdef NGTQG_ROTATED_GLOBAL_CODEBOOKS
      ofstream ofs(rootDirectory + "/rqcb");
#else
      ofstream ofs(rootDirectory + "/qcb");
#endif
      quantizationCodebook.serialize(ofs);
    }
#endif
    property.save(rootDirectory);
  }

  void closeCodebooks() {
    globalCodebookIndex.close();
    for (size_t i = 0; i < localCodebookIndexes.size(); ++i) {
      localCodebookIndexes[i].close();
    }
  }

  void close() {
    objectList.close();
    closeCodebooks();
    if (quantizedObjectDistance != 0) {
      delete quantizedObjectDistance;
      quantizedObjectDistance = 0;
    }
    if (generateResidualObject != 0) {
      delete generateResidualObject;
      generateResidualObject = 0;
    }
#ifndef NGTQ_SHARED_INVERTED_INDEX
    invertedIndex.deleteAll();
#endif
    delete[] localCodebooks;
  }

#ifdef NGTQ_SHARED_INVERTED_INDEX
  void reconstructInvertedIndex(const string &invertedFile) {
    size_t size = invertedIndex.size();
#ifdef NGTQ_RECONSTRUCTION_DISABLE
    cerr << "Reconstruction is disabled!!!!!" << endl;
    return;
#endif
    cerr << "reconstructing to reduce shared memory..." << endl;
    tann::PersistentRepository<IIEntry>	tmpInvertedIndex;
    tmpInvertedIndex.open(invertedFile, 0);
    tmpInvertedIndex.reserve(size);
    for (size_t id = 0; id < size; ++id) {
      if (invertedIndex.isEmpty(id)) {
	continue;
      }
      if (id % 100000 == 0) {
	cerr << "Processed " << id << endl;
      }
      auto codebookSize = property.localCentroidLimit;
      IIEntry *entry = new(tmpInvertedIndex.getAllocator()) InvertedIndexEntry<LOCAL_ID_TYPE>(codebookSize, tmpInvertedIndex.getAllocator());
      size_t esize = (*invertedIndex.at(id)).size();
      (*entry).reserve(esize, tmpInvertedIndex.getAllocator());
      for (size_t i = 0; i < esize; ++i) {
	(*entry).pushBack(tmpInvertedIndex.getAllocator());
	entry->copy((*entry).at(i, tmpInvertedIndex.getAllocator()),
				  (*invertedIndex.at(id)).at(i, invertedIndex.getAllocator()));
      }
      tmpInvertedIndex.put(id, entry);
    }
    cerr << "verifying..." << endl;
    for (size_t id = 0; id < size; ++id) {
      if (invertedIndex.isEmpty(id)) {
	continue;
      }
      if (id % 100000 == 0) {
	cerr << "Processed " << id << endl;
      }
      IIEntry &sentry = *invertedIndex.at(id);
      IIEntry &dentry = *tmpInvertedIndex.at(id);
      size_t esize = sentry.size();
      if (esize != dentry.size()) {
	cerr << id << " : size is inconsistency" << endl;
      }
      for (size_t i = 0; i < esize; ++i) {
	InvertedIndexObject<LOCAL_ID_TYPE> &sobject = (*invertedIndex.at(id)).at(i, invertedIndex.getAllocator());
	InvertedIndexObject<LOCAL_ID_TYPE> &dobject = (*tmpInvertedIndex.at(id)).at(i, tmpInvertedIndex.getAllocator());
	if (sobject.id != dobject.id) {
	  cerr << id << "," << i << " : id is inconsistency" << endl;
	}
	for (size_t d = 0; d < property.localDivisionNo; ++d) {
	  if (sobject.localID[d] != dobject.localID[d]) {
	    cerr << id << "," << i << "," << d << " : local id is inconsistency" << endl;
	  }
	}
      }
    }

    tmpInvertedIndex.close();
  }
#endif

  void createIndex(tann::GraphAndTreeIndex &codebook,
		   size_t centroidLimit,
		   const vector<pair<tann::Object*, size_t> > &objects,
		   vector<tann::Index::InsertionResult> &ids,
		   float &range)
  {
    if (centroidLimit > 0) {
      if (getNumberOfObjects(codebook) >= centroidLimit) {
	range = FLT_MAX;
	codebook.createIndex(objects, ids, range, property.threadSize);
      } else if (getNumberOfObjects(codebook) + objects.size() > centroidLimit) {
	auto start = objects.begin();
	do {
	  size_t s = centroidLimit - getNumberOfObjects(codebook);
	  auto end = start;
	  if (std::distance(objects.begin(), start) + s >= objects.size()) {
	    end = objects.end();
	  } else {
	    end += s;
	  }
	  vector<tann::Index::InsertionResult> idstmp;
	  vector<pair<tann::Object*, size_t> > objtmp;
	  std::copy(start, end, std::back_inserter(objtmp));
	  codebook.createIndex(objtmp, idstmp, range, property.threadSize);
	  assert(idstmp.size() == objtmp.size());
	  std::copy(idstmp.begin(), idstmp.end(), std::back_inserter(ids));
	  start = end;
	} while (start != objects.end() && centroidLimit - getNumberOfObjects(codebook) > 0);
	range = FLT_MAX;
	vector<tann::Index::InsertionResult> idstmp;
	vector<pair<tann::Object*, size_t> > objtmp;
	std::copy(start, objects.end(), std::back_inserter(objtmp));
	codebook.createIndex(objtmp, idstmp, range, property.threadSize);
	std::copy(idstmp.begin(), idstmp.end(), std::back_inserter(ids));
	assert(ids.size() == objects.size());
      } else {
	codebook.createIndex(objects, ids, range, property.threadSize);
      }
    } else {
      codebook.createIndex(objects, ids, range, property.threadSize);
    }
  }

#ifdef NGTQ_VECTOR_OBJECT
  void setGlobalCodeToInvertedEntry(tann::Index::InsertionResult &id, pair<std::vector<float>, size_t> &object, vector<LocalDatam> &localData) {
#else
  void setGlobalCodeToInvertedEntry(tann::Index::InsertionResult &id, pair<tann::Object*, size_t> &object, vector<LocalDatam> &localData) {
#endif
    size_t globalCentroidID = id.id;
    if (invertedIndex.isEmpty(globalCentroidID)) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
      invertedIndex.put(globalCentroidID, new(invertedIndex.allocator) InvertedIndexEntry<LOCAL_ID_TYPE>(localCodebookIndexes.size(), invertedIndex.allocator));
#else
      invertedIndex.put(globalCentroidID, new InvertedIndexEntry<LOCAL_ID_TYPE>(localCodebookIndexes.size()));
#endif
    }
    assert(!invertedIndex.isEmpty(globalCentroidID));
    IIEntry &invertedIndexEntry = *invertedIndex.at(globalCentroidID);
    if (id.identical) {
      if (property.centroidCreationMode == CentroidCreationModeDynamic) {
	assert(invertedIndexEntry.size() != 0);
      }
#ifdef NGTQ_SHARED_INVERTED_INDEX
      invertedIndexEntry.pushBack(object.second, invertedIndex.allocator);
#else
      invertedIndexEntry.pushBack(object.second);
#endif
      if (property.centroidCreationMode == CentroidCreationModeStatic ||
	  property.centroidCreationMode == CentroidCreationModeStaticLayer) {
	localData.push_back(LocalDatam(globalCentroidID, 
				       invertedIndexEntry.size() - 1)); 
      } else {
	if (id.distance != 0.0) {
	  localData.push_back(LocalDatam(globalCentroidID, 
					 invertedIndexEntry.size() - 1)); 
	}
      }
    } else {
      if (property.centroidCreationMode != CentroidCreationModeDynamic) {
	cerr << "Quantizer: Fatal error! Although it is a static global codebook, an object has been added to the global." << endl;
	cerr << "    The actual size of the global codebook=" << globalCodebookIndex.getObjectRepositorySize() - 1
	     << ", The size of the global codebook in the property=" << property.globalCentroidLimit << std::endl;
	cerr << "    The both numbers above should be the same." << std::endl;
	cerr << "    Specify a proper size limitation for the global codebook?" << endl;
	assert(id.identical);
	abort();
      }
      if (invertedIndexEntry.size() == 0) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
	invertedIndexEntry.pushBack(object.second, invertedIndex.allocator);
#else
	invertedIndexEntry.pushBack(object.second);
#endif
      } else {
#ifdef NGTQ_SHARED_INVERTED_INDEX
	invertedIndexEntry.at(0, invertedIndex.allocator).setID(object.second);
#else
	invertedIndexEntry[0].setID(object.second);
#endif
      }
      if (property.quantizerType == QuantizerTypeQBG) {
	localData.push_back(LocalDatam(globalCentroidID, 
				       invertedIndexEntry.size() - 1)); 
      }
    }
  }

  void setSingleLocalCodeToInvertedIndexEntry(vector<tann::GraphAndTreeIndex*> &lcodebook, vector<LocalDatam> &localData, vector<vector<pair<tann::Object*, size_t> > > &localObjs) {
    float lr = property.localRange;
    size_t localCentroidLimit = property.localCentroidLimit;
    if (property.localCodebookState) {
      lr = FLT_MAX;	
      localCentroidLimit = 0;
    }
    vector<tann::Index::InsertionResult> lids;
    createIndex(*lcodebook[0], localCentroidLimit, localObjs[0], lids, lr);
    size_t divisionNo = property.localDivisionNo;
    for (size_t i = 0; i < localData.size(); i++) {
      for (size_t di = 0; di < divisionNo; di++) {
	size_t id = lids[i * divisionNo + di].id;
	assert(!property.localCodebookState || id <= ((1UL << (sizeof(LOCAL_ID_TYPE) * 8)) - 1)); 
#ifdef NGTQ_SHARED_INVERTED_INDEX
	(*invertedIndex.at(localData[i].iiIdx)).at(localData[i].iiLocalIdx, invertedIndex.allocator).localID[di] = id;
#else
	(*invertedIndex.at(localData[i].iiIdx))[localData[i].iiLocalIdx].localID[di] = id;
#endif
      }
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
      localCodebookIndexes[0].deleteObject(localObjs[0][i].first);
#else
      if (lids[i].identical) {
	localCodebookIndexes[0].deleteObject(localObjs[0][i].first);
      }
#endif
    }
  }

#ifndef NGTQ_QBG
  bool setMultipleLocalCodeToInvertedIndexEntry(vector<tann::GraphAndTreeIndex*> &lcodebook, vector<LocalDatam> &localData, vector<vector<pair<tann::Object*, size_t> > > &localObjs) {
    size_t localCodebookNo = property.getLocalCodebookNo();
    bool localCodebookFull = true;  
#pragma omp parallel for
    for (size_t li = 0; li < localCodebookNo; ++li) {
      float lr = property.localRange;
      size_t localCentroidLimit = property.localCentroidLimit;
      if (property.localCentroidCreationMode == CentroidCreationModeDynamicKmeans) {
	localCentroidLimit *= property.localClusteringSampleCoefficient;
      }
      if (property.localCodebookState) {
	lr = FLT_MAX;	
	localCentroidLimit = 0;
      } else {
	if (property.localCentroidCreationMode == CentroidCreationModeDynamicKmeans) {
	  lr = -1.0;
	}
      }
      vector<tann::Index::InsertionResult> lids;
      createIndex(*lcodebook[li], localCentroidLimit, localObjs[li], lids, lr);
      if (lr != FLT_MAX) { 
	localCodebookFull = false;
      }
      assert(localData.size() == lids.size());
      for (size_t i = 0; i < localData.size(); i++) {
	size_t id = lids[i].id;
	assert(!property.localCodebookState || id <= ((1UL << (sizeof(LOCAL_ID_TYPE) * 8)) - 1)); 
#ifdef NGTQ_SHARED_INVERTED_INDEX
	(*invertedIndex.at(localData[i].iiIdx)).at(localData[i].iiLocalIdx, invertedIndex.allocator).localID[li] = id;
#else
	(*invertedIndex.at(localData[i].iiIdx))[localData[i].iiLocalIdx].localID[li] = id;
#endif
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
	localCodebookIndexes[li].deleteObject(localObjs[li][i].first);
#else
	if (lids[i].identical) {
	  localCodebookIndexes[li].deleteObject(localObjs[li][i].first);
	}
#endif
      } 
    } 
    return localCodebookFull;
  }
#endif 

#ifdef NGTQ_QBG
  bool setMultipleLocalCodeToInvertedIndexEntry(vector<tann::GraphAndTreeIndex*> &lcodebook,
						   vector<LocalDatam> &localData,
						   float *subspaceObjects) {
    size_t paddedDimension = globalCodebookIndex.getObjectSpace().getPaddedDimension();
    size_t localCodebookNo = property.getLocalCodebookNo();
    bool localCodebookFull = true;  
    for (size_t li = 0; li < localCodebookNo; ++li) {
      float lr = property.localRange;
      size_t localCentroidLimit = property.localCentroidLimit;
      if (property.localCentroidCreationMode == CentroidCreationModeDynamicKmeans) {
	localCentroidLimit *= property.localClusteringSampleCoefficient;
      }
      if (property.localCodebookState) {
	lr = FLT_MAX;	
	localCentroidLimit = 0;
      } else {
	if (property.localCentroidCreationMode == CentroidCreationModeDynamicKmeans) {
	  lr = -1.0;
	}
      }
      vector<tann::Index::InsertionResult> lids;
      size_t localDimension = lcodebook[li]->getObjectSpace().getDimension();
      vector<pair<tann::Object*, size_t>> localObjects(localData.size());
      for (size_t i = 0; i < localData.size(); i++) {
	localObjects[i].first = lcodebook[li]->allocateObject(&subspaceObjects[i * paddedDimension + (li * localDimension)], localDimension);
	localObjects[i].second = 0;
      }
      createIndex(*lcodebook[li], localCentroidLimit, localObjects, lids, lr);
      if (lr != FLT_MAX) { 
	localCodebookFull = false;
      }
      assert(localData.size() == lids.size());
      for (size_t i = 0; i < localData.size(); i++) {
	size_t id = lids[i].id;
	assert(!property.localCodebookState || id <= ((1UL << (sizeof(LOCAL_ID_TYPE) * 8)) - 1)); 
#ifdef NGTQ_SHARED_INVERTED_INDEX
	(*invertedIndex.at(localData[i].iiIdx)).at(localData[i].iiLocalIdx, invertedIndex.allocator).localID[li] = id;
#else
	(*invertedIndex.at(localData[i].iiIdx))[localData[i].iiLocalIdx].localID[li] = id;
#endif
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
	lcodebook[li]->deleteObject(localObjects[i].first);
#else
	if (lids[i].identical) {
	  lcodebook[li]->deleteObject(localObjects[i].first);
	}
#endif
      } 
    } 
    return localCodebookFull;
  }
#endif 

  void constructLocalCodebooks() {
    delete localCodebooks;
    size_t localCodebookNo = property.getLocalCodebookNo();
    size_t codebookSize = localCodebookIndexes[0].getObjectSpace().getSize() - 1;
    size_t paddedDimension = globalCodebookIndex.getObjectSpace().getPaddedDimension();
    size_t localDimension = localCodebookIndexes[0].getObjectSpace().getDimension();
    localCodebooks = new float[codebookSize * paddedDimension];
    size_t oft = 0;
    for (size_t li = 0; li < localCodebookNo; li++) {
      if (localCodebookIndexes[li].getObjectSpace().getSize() - 1 != codebookSize) {
	std::cerr << "Fatal Error!" << std::endl;
	abort();
      }
      for (size_t cid = 1; cid <= codebookSize; cid++) {
	std::vector<float> v;
	localCodebookIndexes[li].getObjectSpace().getObject(cid, v);
	for (size_t ld = 0; ld < v.size(); ld++) {
	  localCodebooks[(cid - 1) * paddedDimension + oft + ld] = v[ld];
	}
      }
      oft += localDimension;
    }
    if (oft != globalCodebookIndex.getObjectSpace().getDimension()) {
      std::cerr << "somethig wrong " << oft << ":" << globalCodebookIndex.getObjectSpace().getDimension() << std::endl;
      abort();
    }
  }

#ifdef NGTQ_QBG
  void setMultipleLocalCodeToInvertedIndexEntryFixed(vector<LocalDatam> &localData,
							float *subspaceObjects) {
    if (localData.empty()) {
      return;
    }
    if (localCodebooks == 0) {
      constructLocalCodebooks();
    }
    size_t paddedDimension = globalCodebookIndex.getObjectSpace().getPaddedDimension();
    size_t localCodebookNo = property.getLocalCodebookNo();
    size_t codebookSize = property.localCentroidLimit;
    if (property.dimension % property.localDivisionNo != 0) {
      std::cerr << "setMultipleLocalCodeToInvertedIndexEntry!!!" << std::endl;
      abort();
    }
    size_t localDimension = property.dimension / property.localDivisionNo;
    std::unique_ptr<float[]> distance(new float[localData.size() * codebookSize * localCodebookNo]());
    std::vector<std::pair<float, uint32_t>> min(localData.size() * localCodebookNo, std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<uint32_t>::max()));
    if (localCodebooks == 0) {
      std::cerr << "Quantizer::setMultipleLocalCodeToInvertedEntry: FatalError!" << std::endl;
      abort();
    }
    {
#pragma omp parallel for
      for (size_t idx = 0; idx < localData.size(); idx++) {
	for (size_t cid = 0; cid < codebookSize; cid++) {
	  for (size_t svi = 0; svi < localCodebookNo; svi++) {
	    float localDistance = 0.0;
	    for (size_t ld = 0; ld < localDimension; ld++) {
	      size_t d = svi * localDimension + ld;
	      auto dist = subspaceObjects[idx * paddedDimension + d] - localCodebooks[cid * paddedDimension + d];
	      dist *= dist;
	      distance[idx * codebookSize * localCodebookNo + cid * localCodebookNo + svi] += dist; 
	      localDistance = distance[idx * codebookSize * localCodebookNo + cid * localCodebookNo + svi];
	    }
	    if (localDistance < min[idx * localCodebookNo + svi].first) {
	      min[idx * localCodebookNo + svi].first = localDistance;
	      min[idx * localCodebookNo + svi].second = cid;
	    }
	  }
	}
      }

    }
#pragma omp parallel for
    for (size_t li = 0; li < localCodebookNo; ++li) {
      for (size_t i = 0; i < localData.size(); i++) {
	size_t id = min[i * localCodebookNo + li].second + 1;
	assert(!property.localCodebookState || id <= ((1UL << (sizeof(LOCAL_ID_TYPE) * 8)) - 1)); 
#ifdef NGTQ_SHARED_INVERTED_INDEX
	(*invertedIndex.at(localData[i].iiIdx)).at(localData[i].iiLocalIdx, invertedIndex.allocator).localID[li] = id;
#else
	(*invertedIndex.at(localData[i].iiIdx))[localData[i].iiLocalIdx].localID[li] = id;
#endif
      }
    }
    return;
  }
#endif 

  void buildMultipleLocalCodebooks(tann::Index *localCodebook, size_t localCodebookNo, size_t numberOfCentroids) {
    tann::Clustering clustering;
    clustering.epsilonFrom = 0.10;
    clustering.epsilonTo = 0.50;
    clustering.epsilonStep = 0.05;
    clustering.maximumIteration = 20;
    clustering.clusterSizeConstraint = false; 
    for (size_t li = 0; li < localCodebookNo; ++li) {
      double diff = clustering.kmeansWithNGT(localCodebook[li], numberOfCentroids);
      if (diff > 0.0) {
	cerr << "Not converge. " << diff << endl;
      }
      cerr << "Clustering of the subvector is complete. " << localCodebook[li].getPath() << ":" << localCodebook[li].getObjectRepositorySize() << endl;
    }
  }


#ifdef NGTQ_QBG
  void replaceInvertedIndexEntry(size_t localCodebookNo) {
    vector<LocalDatam> localData;
    for (size_t gidx = 1; gidx < invertedIndex.size(); gidx++) {
      if (invertedIndex.at(gidx) == 0) {
	std::cerr << "replaceInvertedIndexEntry: Warning. empty inverted index entry. " << gidx << ":" << invertedIndex.size() << std::endl;
	continue;
      }
      IIEntry &invertedIndexEntry = *invertedIndex.at(gidx);
      for (size_t oi = ((property.centroidCreationMode == CentroidCreationModeStatic ||
			 property.centroidCreationMode == CentroidCreationModeStaticLayer) ||
			property.quantizerType == QuantizerTypeQBG) ? 0 : 1;
	   oi < invertedIndexEntry.size(); oi++) {
#ifdef NGTQ_QBG
	localData.push_back(LocalDatam(gidx, oi, invertedIndexEntry.subspaceID)); 
#else
	localData.push_back(LocalDatam(gidx, oi)); 
#endif
      }
    }
    float subspaceObjects[localData.size()][globalCodebookIndex.getObjectSpace().getPaddedDimension()];
    for (size_t i = 0; i < localData.size(); i++) {
      IIEntry &invertedIndexEntry = *invertedIndex.at(localData[i].iiIdx);
#ifdef NGTQ_SHARED_INVERTED_INDEX
#ifdef NGTQ_QBG
      std::cerr << "not implemented" << std::endl;
      abort();
#else
      tann::Object object(&globalCodebookIndex.getObjectSpace());
      objectList.get(invertedIndexEntry[localData[i].iiLocalIdx].id, object, &globalCodebookIndex.getObjectSpace());
      (*generateResidualObject)(object, // object
				invertedIndexEntry.subspaceID,
				subspaceObjects[i]); // subspace objects
#endif 
#else
#ifdef NGTQ_VECTOR_OBJECT
      std::vector<float> object;
      objectList.get(invertedIndexEntry[localData[i].iiLocalIdx].id, object, &globalCodebookIndex.getObjectSpace());
#else
      tann::Object object(&globalCodebookIndex.getObjectSpace());
#endif
      (*generateResidualObject)(object, // object
				invertedIndexEntry.subspaceID,
				subspaceObjects[i]); // subspace objects
#endif
    }
    setMultipleLocalCodeToInvertedIndexEntryFixed(localData, &subspaceObjects[0][0]);    
  }
#endif 

#ifndef NGTQ_QBG
  void insert(vector<pair<tann::Object*, size_t> > &objects) {
    std::cerr << "insert() is not implemented." << std::endl;
    abort();
  }
#endif 

  void searchIndex(tann::GraphAndTreeIndex &codebook,
		   size_t centroidLimit,
#ifdef NGTQ_VECTOR_OBJECT
		   const vector<pair<std::vector<float>, size_t>> &objects, 
#else
		   const vector<pair<tann::Object*, size_t>> &objects,
#endif
		   vector<tann::Index::InsertionResult> &ids,
		   float &range, tann::Index *gqindex)
  {
    if (quantizationCodebook.size() == 0) {
      std::cerr << "Fatal error. quantizationCodebook is empty" << std::endl;
      abort();
    }
    ids.clear();
    ids.resize(objects.size());
    size_t foundCount = 0;
    double foundRank = 0.0;
#pragma omp parallel for 
    for (size_t idx = 0; idx < objects.size(); idx++) {
#ifdef NGTQ_VECTOR_OBJECT
      auto *object = globalCodebookIndex.allocateObject(objects[idx].first);
      auto qid = quantizationCodebook.search(*object);
      globalCodebookIndex.deleteObject(object);
#else
      auto qid = quantizationCodebook.search(*objects[idx].first);
#endif
      tann::VectorDistances result;
      if (gqindex != 0) {
	std::vector<float> object(globalCodebookIndex.getObjectSpace().getDimension());
#ifdef NGTQ_VECTOR_OBJECT
	memcpy(object.data(), objects[idx].first.data(), sizeof(float) * object.size());
#else
	memcpy(object.data(), objects[idx].first->getPointer(), sizeof(float) * object.size());
#endif
#define QID_WEIGHT	100	
	object.push_back(qid * QID_WEIGHT);
	tann::SearchQuery sc(object);;
	sc.setResults(&result);
	sc.setSize(50);
	sc.radius = FLT_MAX;
	sc.setEpsilon(0.1);
	gqindex->search(sc);
      } else {
#ifdef NGTQ_VECTOR_OBJECT
	auto *object = globalCodebookIndex.allocateObject(objects[idx].first);
	tann::SearchContainer sc(*object);
#else
	tann::SearchContainer sc(*objects[idx].first);
#endif
	sc.setResults(&result);
	sc.setSize(50);
	sc.radius = FLT_MAX;
	sc.setEpsilon(0.1);
	globalCodebookIndex.search(sc);
#ifdef NGTQ_VECTOR_OBJECT
	globalCodebookIndex.deleteObject(object);
#endif
      }
      int32_t eqi = -1;
      for (size_t i = 0; i < result.size(); i++) {
	auto &invertedIndexEntry = *invertedIndex.at(result[i].id);
	if (invertedIndexEntry.subspaceID == qid) {
#pragma omp critical
	  {
	    foundCount++;
	    foundRank += i;
	  }
	  eqi = i;
	  break;
	}
      }
      if (eqi < 0) {
	eqi = 0;
      }
      ids[idx].id = result[eqi].id;
      ids[idx].distance = result[eqi].distance;
      ids[idx].identical = true;
    }
    return;

  }

#ifdef NGTQ_VECTOR_OBJECT
  void getBlobIDFromObjectToBlobIndex(const vector<pair<std::vector<float>, size_t>> &objects, 
				      vector<tann::Index::InsertionResult> &ids)
#else
  void getBlobIDFromObjectToBlobIndex(const vector<pair<tann::Object*, size_t>> &objects,
				      vector<tann::Index::InsertionResult> &ids)
#endif
  {
    ids.clear();
    ids.resize(objects.size());
#ifdef GET_BLOG_EVAL
    size_t identicalObjectCount = 0;
#endif
    for (size_t idx = 0; idx < objects.size(); idx++) {
      if (objects[idx].second - 1 >= objectToBlobIndex.size()) {
	std::cerr << "Quantizer::insert: Fatal Error! Object ID is invalid. " 
		  << idx << ":" << objects[idx].second - 1 << ":" << objectToBlobIndex.size()
		  << ":" << objects.size()<< std::endl;
        abort();
      }
      ids[idx].id = objectToBlobIndex[objects[idx].second - 1] + 1;
      ids[idx].distance = 0.0;
      ids[idx].identical = true;
#ifdef GET_BLOG_EVAL
      {
	tann::VectorDistances result;
	tann::SearchContainer sc(*objects[idx].first);
	sc.setResults(&result);
	sc.setSize(50);
	sc.radius = FLT_MAX;
	sc.setEpsilon(0.1);
	globalCodebookIndex.search(sc);
	//std::cerr << "insert:Eval: ";
	if (result[0].id == ids[idx].id) {
	  identicalObjectCount++;
	} else {
	}
      }
#endif
    }
#ifdef GET_BLOG_EVAL
    std::cerr << identicalObjectCount << "/" << objects.size() << std::endl;
#endif
    return;
  }

#ifdef NGTQ_QBG
  tann::Index *buildGlobalCodebookWithQIDIndex() {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    std::cerr << "buildGlobalCodebookWithQIDIndex: Not implemented." << std::endl;
    abort();
#else
    tann::Property property;
    
    property.dimension = globalCodebookIndex.getObjectSpace().getDimension() + 1;
    property.distanceType = tann::MetricType::MetricTypeL2;
#ifdef NGTQ_SHARED_INVERTED_INDEX
    tann::Index *index = new tann::Index("dummy", property);
    std::cerr << "Not implemented" << std::endl;
    abort();
#else
    tann::Index *index = new tann::Index(property);
#endif
    for (size_t id = 1; id < globalCodebookIndex.getObjectRepositorySize(); id++) {
      std::vector<float> object;
      if (id % 10000 == 0) {
	std::cerr << "# of processed objects=" << id << std::endl;
      }
      globalCodebookIndex.getObjectSpace().getObject(id, object);
      object.push_back(invertedIndex.at(id)->subspaceID * QID_WEIGHT);
      index->append(object);
    }
    index->createIndex(50);
    return index;
#endif 
  }
#endif 

#ifdef NGTQ_QBG
#ifdef NGTQ_VECTOR_OBJECT
  void insert(vector<pair<std::vector<float>, size_t> > &objects, tann::Index *gqindex) {
#else
  void insert(vector<pair<tann::Object*, size_t> > &objects, tann::Index *gqindex) {
#endif
#ifdef NGTQ_SHARED_INVERTED_INDEX
    std::cerr << "insert: Not implemented." << std::endl;
    abort();
#else
    tann::GraphAndTreeIndex &gcodebook = (tann::GraphAndTreeIndex &)globalCodebookIndex.getIndex();
    size_t localCodebookNo = property.getLocalCodebookNo();
    vector<tann::GraphAndTreeIndex*> lcodebook;
    lcodebook.reserve(localCodebookNo);
    for (size_t i = 0; i < localCodebookNo; i++) {
      lcodebook.push_back(&static_cast<tann::GraphAndTreeIndex &>(localCodebookIndexes[i].getIndex()));
    }
    float gr = property.globalRange;
    vector<tann::Index::InsertionResult> ids;
    if (property.centroidCreationMode == CentroidCreationModeStaticLayer ||
	property.centroidCreationMode == CentroidCreationModeStatic) {
      if (objectToBlobIndex.empty()) {
	searchIndex(gcodebook, property.globalCentroidLimit, objects, ids, gr, gqindex);
      } else {
	getBlobIDFromObjectToBlobIndex(objects, ids);
      }
    } else {
      std::cerr << "Quantizer::InsertQBG: Warning! invalid centroidCreationMode. " << property.centroidCreationMode << std::endl;
      abort();
    }
#ifdef NGTQ_SHARED_INVERTED_INDEX
    if (invertedIndex.getAllocatedSize() <= invertedIndex.size() + objects.size()) {
      invertedIndex.reserve(invertedIndex.getAllocatedSize() * 2);
    }
#else
    invertedIndex.reserve(invertedIndex.size() + objects.size());
#endif
    vector<LocalDatam> localData;
    for (size_t i = 0; i < ids.size(); i++) {
      setGlobalCodeToInvertedEntry(ids[i], objects[i], localData);
    } 
    float subspaceObjects[localData.size()][globalCodebookIndex.getObjectSpace().getPaddedDimension()];
#pragma omp parallel for  
    for (size_t i = 0; i < localData.size(); i++) {
      IIEntry &invertedIndexEntry = *invertedIndex.at(localData[i].iiIdx);
#ifdef NGTQ_SHARED_INVERTED_INDEX
#ifdef NGTQ_QBG
      std::cerr << "Not implemented" << std::endl;
      abort();
#else
      (*generateResidualObject)(invertedIndexEntry.at(localData[i].iiLocalIdx, invertedIndex.allocator).id,
				localData[i].iiIdx, // centroid:ID of global codebook
				localObjs);
#endif
#else
#ifdef NGTQ_QBG

#ifdef NGTQG_ROTATED_GLOBAL_CODEBOOKS
      assert(!rotation.empty());
      if (!rotation.empty()) {
#ifdef NGTQ_VECTOR_OBJECT
	rotation.mul(objects[i].first.data());
#else
	rotation.mul(static_cast<float*>(objects[i].first->getPointer()));
#endif
      }
#endif
#ifdef NGTQ_VECTOR_OBJECT
      (*generateResidualObject)(objects[i].first, // object
				invertedIndexEntry.subspaceID,
				subspaceObjects[i]); // subspace objects
#else
      (*generateResidualObject)(*objects[i].first, // object
				invertedIndexEntry.subspaceID,
				subspaceObjects[i]); // subspace objects
#endif
#ifndef NGTQG_ROTATED_GLOBAL_CODEBOOKS
      rotation.mul(subspaceObjects[i]);
#endif
#else
      (*generateResidualObject)(invertedIndexEntry[localData[i].iiLocalIdx].id,
				localData[i].iiIdx, // centroid:ID of global codebook
				localObjs);
#endif
#endif

    } 
    
    if (property.singleLocalCodebook) {
      // single local codebook
      std::cerr << "insert: Fatal Error. single local codebook isn't available." << std::endl;
      abort();
    } else { 
      // multiple local codebooks
      bool localCodebookFull = true;
      if (property.localCodebookState) {
	setMultipleLocalCodeToInvertedIndexEntryFixed(localData, &subspaceObjects[0][0]);
      } else {
	localCodebookFull = setMultipleLocalCodeToInvertedIndexEntry(lcodebook, localData, &subspaceObjects[0][0]);
      }
      if ((!property.localCodebookState) && localCodebookFull) {
	if (property.localCentroidCreationMode == CentroidCreationModeDynamicKmeans) {
	  buildMultipleLocalCodebooks(localCodebookIndexes.data(), localCodebookNo, property.localCentroidLimit);
	  (*generateResidualObject).set(localCodebookIndexes.data(), localCodebookNo);
	  property.localCodebookState = true;	
	  localCodebookFull = false;		
	  replaceInvertedIndexEntry(localCodebookNo);
	} else {
	  property.localCodebookState = true;	
	  localCodebookFull = false;
	}
      }
    } 
#pragma omp parallel for
    for (size_t i = 0; i < objects.size(); i++) {
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
      globalCodebookIndex.deleteObject(objects[i].first);
#else
#ifndef NGTQ_VECTOR_OBJECT
      if (ids[i].identical == true) {
	globalCodebookIndex.deleteObject(objects[i].first);
      }
#endif
#endif
    }
    objects.clear();
#endif 
  }
#endif 


#ifndef NGTQ_QBG
  void insert(vector<float> &objvector, vector<pair<tann::Object*, size_t> > &objects, size_t count) {
    size_t id = count;
    if (count == 0) {
      id = objectList.size();
      id = id == 0 ? 1 : id;
    }

    tann::Object *object = globalCodebookIndex.allocateObject(objvector);
    objectList.put(id, *object, &globalCodebookIndex.getObjectSpace());

    objects.push_back(pair<tann::Object*, size_t>(object, id));

    if (objects.size() >= property.batchSize) {
      insert(objects);   // batch insert
    }
  }
#endif 

  void insertIntoObjectRepository(vector<float> &objvector, size_t count) {
    size_t id = count;
    if (count == 0) {
      id = objectList.size();
      id = id == 0 ? 1 : id;
    }
    tann::Object *object = globalCodebookIndex.allocateObject(objvector);
    std::vector<float> vs = globalCodebookIndex.getObjectSpace().getObject(*object);
    objectList.put(id, *object, &globalCodebookIndex.getObjectSpace());
    globalCodebookIndex.deleteObject(object);
  }

#ifdef NGTQ_QBG
  void createIndex(size_t beginID = 1, size_t endID = 0) {
    tann::Index *gqindex = 0;
    if (property.centroidCreationMode == CentroidCreationModeStaticLayer) {
      gqindex = buildGlobalCodebookWithQIDIndex();
    }
#ifdef NGTQ_VECTOR_OBJECT
    vector<pair<std::vector<float>, size_t>> objects;
#else
    vector<pair<tann::Object*, size_t>> objects;
#endif
    objects.reserve(property.batchSize);
    if (endID == 0) {
      endID = objectList.size() - 1;
    }
    tann::Timer timer;
    timer.start();
    for (size_t id = beginID; id <= endID; id++) {
      if (id % 1000000 == 0) {
	timer.stop();
	std::cerr << "# of processed objects=" << id << " Time=" << timer << ", vm size="
		  << tann::Common::getProcessVmSizeStr() << std::endl;
	timer.restart();
      }
#ifdef NGTQ_VECTOR_OBJECT
      std::vector<float> object;
      if (!objectList.get(id, object, &globalCodebookIndex.getObjectSpace())) {
	std::cerr << "Cannot get object. ID=" << id << std::endl;
	continue;
      }
#else
      tann::Object *object = globalCodebookIndex.getObjectSpace().allocateObject();
      objectList.get(id, *object, &globalCodebookIndex.getObjectSpace());
#endif
#ifdef NGTQ_VECTOR_OBJECT
      objects.push_back(pair<std::vector<float>, size_t>(object, id));
#else
      objects.push_back(pair<tann::Object*, size_t>(object, id));
#endif
      if (objects.size() >= property.batchSize) {
	insert(objects, gqindex);   // batch insert
      }
    }
    if (objects.size() > 0) {
      insert(objects, gqindex);   // batch insert
    }
    delete gqindex;
  }
#endif 

  void setupInvertedIndex(std::vector<std::vector<float>> &qCodebook,
				std::vector<uint32_t> &codebookIndex,
				std::vector<uint32_t> &objectIndex) {
#if !defined(NGTQ_QBG)
    std::cerr << "setupInvertedIndex: Not implemented." << std::endl;
    abort();
#else
    if (globalCodebookIndex.getObjectRepositorySize() != codebookIndex.size() + 1) {
      std::cerr << "Error? " << globalCodebookIndex.getObjectRepositorySize() << ":" <<  codebookIndex.size() + 1 << std::endl;
    }
    if (!invertedIndex.empty()) {
      stringstream msg;
      msg << "Fatal Error! inverted index is not empty. " << invertedIndex.size();
      TANN_THROW(msg);
    }
    invertedIndex.reserve(codebookIndex.size() + 1);
    std::cerr << "codebook Index size=" << codebookIndex.size()<< std::endl;
    for (size_t idx = 0; idx < codebookIndex.size(); idx++) {
      auto gid = idx + 1;
#ifdef NGTQ_SHARED_INVERTED_INDEX
      invertedIndex.put(gid, new(invertedIndex.allocator) InvertedIndexEntry<LOCAL_ID_TYPE>(localCodebookIndexes.size(), invertedIndex.allocator));
#else
      invertedIndex.put(gid, new InvertedIndexEntry<LOCAL_ID_TYPE>(localCodebookIndexes.size()));
#endif
      invertedIndex.at(gid)->subspaceID = codebookIndex[idx];
    }


    
    quantizationCodebook.setPaddedDimension(globalCodebookIndex.getObjectSpace().getPaddedDimension());
    quantizationCodebook = qCodebook;
#ifdef NGTQG_ROTATED_GLOBAL_CODEBOOKS
    quantizationCodebook.rotate(rotation);
#endif

    objectToBlobIndex = std::move(objectIndex);
    objectIndex.clear();
    std::vector<uint32_t> invertedIndexCount(codebookIndex.size());
    for (size_t idx = 0; idx < objectToBlobIndex.size(); idx++) {
      invertedIndexCount[objectToBlobIndex[idx]]++;
    }
    for (size_t idx = 0; idx < codebookIndex.size(); idx++) {
      auto gid = idx + 1;
#ifdef NGTQ_SHARED_INVERTED_INDEX
      IIEntry &invertedIndexEntry = *invertedIndex.at(gid, invertedIndex.allocator);
#else
      IIEntry &invertedIndexEntry = *invertedIndex.at(gid);
#endif
      invertedIndexEntry.reserve(invertedIndexCount[idx]);
    }
#endif 
  }


#ifndef NGTQ_QBG
  void rebuildIndex() {
    abort();
  }
#endif 

  void create(const string &index,
	      tann::Property &globalProperty,
#ifdef NGTQ_QBG
	      tann::Property &localProperty,
	      std::vector<float> *rotation = 0,
	      const string &objectFile = ""
#else
	      tann::Property &localProperty
#endif
	      ) {
    if (property.localCentroidLimit > ((1UL << (sizeof(LOCAL_ID_TYPE) * 8)) - 1)) {
      stringstream msg;
      msg << "Quantizer::Error. Local centroid limit " << property.localCentroidLimit << " is too large. It must be less than " << (1UL << (sizeof(LOCAL_ID_TYPE) * 8));
      TANN_THROW(msg);
    }

    tann::Property gp;
    tann::Property lp;

    gp.setDefault();
    lp.setDefault();

    gp.batchSizeForCreation = 500;
    gp.edgeSizeLimitForCreation = 0;
    gp.edgeSizeForCreation = 10;
    gp.graphType = tann::Index::Property::GraphType::GraphTypeANNG;
    gp.insertionRadiusCoefficient = 1.1;
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    gp.graphSharedMemorySize	= 512; // MB
    gp.treeSharedMemorySize	= 512; // MB
    gp.objectSharedMemorySize	= 512; // MB  512 is for up to 20M objects.
#endif

    lp.batchSizeForCreation = 500;
    lp.edgeSizeLimitForCreation = 0;
    lp.edgeSizeForCreation = 10;
    lp.graphType = tann::Index::Property::GraphType::GraphTypeANNG;
    lp.insertionRadiusCoefficient = 1.1;
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    lp.graphSharedMemorySize	= 128; // MB
    lp.treeSharedMemorySize	= 128; // MB
    lp.objectSharedMemorySize	= 128; // MB  128 is for up to 5M objects?
#endif
    gp.set(globalProperty);
    lp.set(localProperty);

    gp.edgeSizeForSearch = 40;	
    lp.edgeSizeForSearch = 40;	

    lp.objectType = tann::DataType::Float;
    gp.dimension = property.dimension;
    if (gp.dimension == 0) {
      stringstream msg;
      msg << "NGTQ::Quantizer::create: specified dimension is zero!";
      TANN_THROW(msg);
    }
    if (property.localDivisionNo == 0) {
      TANN_THROW("NGTQ::Quantizer::create: # of subvectors is zero");
    }
    if (property.localDivisionNo != 1 && property.dimension % property.localDivisionNo != 0) {
      stringstream msg;
      msg << "NGTQ::Quantizer::create: dimension and localDivisionNo are not proper. "
	  << property.dimension << ":" << property.localDivisionNo;
      TANN_THROW(msg);
    }
    lp.dimension = property.dimension / property.localDivisionNo;

    switch (property.dataType) {
    case DataTypeFloat:
      gp.objectType = tann::DataType::Float;
      break;
    case DataTypeFloat16:
      gp.objectType = tann::DataType::Float16;
      break;
    case DataTypeUint8:
      gp.objectType = tann::DataType::Uint8;
      break;
    default:
      {
	stringstream msg;
	msg << "NGTQ::Quantizer::create: Inner error! Invalid data type.";
	TANN_THROW(msg);
      }
    }

    switch (property.distanceType) {
    case MetricType::MetricTypeL1:
      gp.distanceType = tann::MetricType::MetricTypeL1;
      lp.distanceType = tann::MetricType::MetricTypeL1;
      break;
    case MetricType::MetricTypeL2:
#ifdef NGTQ_DISTANCE_ANGLE
      {
	stringstream msg;
	msg << "NGTQ::Quantizer::create: L2 is unavailable!!! you have to rebuild.";
	TANN_THROW(msg);
      }
#endif
      gp.distanceType = tann::MetricType::MetricTypeL2;
      lp.distanceType = tann::MetricType::MetricTypeL2;
      break;
    case MetricType::MetricTypeHamming:
      gp.distanceType = tann::MetricType::MetricTypeHamming;
      lp.distanceType = tann::MetricType::MetricTypeHamming;
      break;
    case MetricType::MetricTypeAngle:
#ifndef NGTQ_DISTANCE_ANGLE
      {
	stringstream msg;
	msg << "NGTQ::Quantizer::create: Angle is unavailable!!! you have to rebuild.";
	TANN_THROW(msg);
      }
#endif
      gp.distanceType = tann::MetricType::MetricTypeAngle;
      lp.distanceType = tann::MetricType::MetricTypeAngle;
      break;
    case MetricType::MetricTypeNormalizedCosine:
      gp.distanceType = tann::MetricType::MetricTypeNormalizedCosine;
      lp.distanceType = tann::MetricType::MetricTypeL2;
      break;
    case MetricType::MetricTypeCosine:
      gp.distanceType = tann::MetricType::MetricTypeCosine;
      lp.distanceType = tann::MetricType::MetricTypeL2;
      break;
    case MetricType::MetricTypeNormalizedL2:
      gp.distanceType = tann::MetricType::MetricTypeNormalizedL2;
      lp.distanceType = tann::MetricType::MetricTypeL2;
      break;
    default:
      {
	stringstream msg;
	msg << "NGTQ::Quantizer::create Inner error! Invalid distance type.";
	TANN_THROW(msg);
      }
    }

#ifdef NGTQ_QBG
    createEmptyIndex(index, gp, lp, rotation, objectFile);
#else
    createEmptyIndex(index, gp, lp);
#endif
  }

  void validate() {
#ifndef NGTQ_QBG
    size_t gcbSize = globalCodebookIndex.getObjectRepositorySize();
    for (size_t gidx = 1; gidx < 4 && gidx < gcbSize; gidx++) {
      if (invertedIndex[gidx] == 0) {
	cerr << "something wrong" << endl;
	exit(1);
      }
      if ((*invertedIndex[gidx]).size() == 0) {
	cerr << "something wrong" << endl;
	continue;
      }

      tann::PersistentObject &gcentroid = *globalCodebookIndex.getObjectSpace().getRepository().get(gidx);
      vector<double> gco;
      globalCodebookIndex.getObjectSpace().getRepository().extractObject(&gcentroid, gco);
      {
#ifdef NGTQ_SHARED_INVERTED_INDEX
	InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[gidx]).at(0, invertedIndex.allocator);
#else
	InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[gidx])[0];
#endif
        if (invertedIndexEntry.id != gidx) {
	  cerr << "a global centroid id is wrong in the inverted index." << gidx << ":" << invertedIndexEntry.id << endl;
	  exit(1);
        }
      }
      tann::Object *gcentroidFromList = globalCodebookIndex.getObjectSpace().getRepository().allocateObject();
      vector<double> gcolist;
      globalCodebookIndex.getObjectSpace().getRepository().extractObject(gcentroidFromList, gcolist);
      if (gco != gcolist) {
	cerr << "Fatal error! centroid in NGT is different from object list in NGTQ" << endl;
	exit(1);
      }
      vector<size_t> elements;
      for (size_t iidx = 0; iidx < (*invertedIndex[gidx]).size(); iidx++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
	InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[gidx]).at(iidx, invertedIndex.allocator);
#else
	InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[gidx])[iidx];
#endif
        elements.push_back(invertedIndexEntry.id);
	cerr << "  object ID=" << invertedIndexEntry.id;
	{
	  tann::Object *o = globalCodebookIndex.getObjectSpace().getRepository().allocateObject();
	  objectList.get(invertedIndexEntry.id, *o, &globalCodebookIndex.getObjectSpace());
	  tann::distance_type distance = globalCodebookIndex.getObjectSpace().getComparator()(*gcentroidFromList, *o);
	  cerr << ":distance=" << distance;
	}
	cerr << ":local codebook IDs=";
	for (size_t li = 0; li < property.localDivisionNo; li++) {
	  cerr << invertedIndexEntry.localID[li] << " ";
	}
	cerr << endl;
	for (size_t li = 0; li < property.localDivisionNo; li++) {
	  if (invertedIndexEntry.localID[li] == 0) {
	    if (property.centroidCreationMode != CentroidCreationModeStatic &&
		property.centroidCreationMode != CentroidCreationModeStaticLayer) {
	      if (iidx == 0) {
		break;
	      }
	    }
	    cerr << "local ID is unexpected zero." << endl;
	  }
	}
      }
      vector<size_t> ngid;
      {
	size_t resultSize = 30;
	size_t approximateSearchSize = 1000;
	size_t codebookSearchSize = 50;
	bool refine = true;
	bool lookuptable = false;
	double epsilon = 0.1;
	tann::VectorDistances objects;
	search(gcentroidFromList, objects, resultSize, approximateSearchSize, codebookSearchSize, 
	       refine, lookuptable, epsilon);
	for (size_t resulti = 0; resulti < objects.size(); resulti++) {
	  if (std::find(elements.begin(), elements.end(), objects[resulti].id) != elements.end()) {
	    cerr << "  ";
	  } else {
	    cerr << "x ";
	    ngid.push_back(objects[resulti].id);
	    tann::VectorDistances result;
	    tann::Object *o = globalCodebookIndex.getObjectSpace().getRepository().allocateObject();
	    objectList.get(ngid.back(), *o, &globalCodebookIndex.getObjectSpace());
	    tann::GraphAndTreeIndex &graphIndex = (tann::GraphAndTreeIndex &)globalCodebookIndex.getIndex();
	    graphIndex.searchForNNGInsertion(*o, result);
	    if (result[0].distance > objects[resulti].distance) {
	      cerr << " Strange! ";
	      cerr << result[0].distance << ":" << objects[resulti].distance << " ";
	    }
	    globalCodebookIndex.getObjectSpace().getRepository().deleteObject(o);
	  }
	  cerr << "  search object " << resulti << " ID=" << objects[resulti].id << " distance=" << objects[resulti].distance << endl;
	}
      }
      globalCodebookIndex.getObjectSpace().getRepository().deleteObject(gcentroidFromList);
    }
#endif  // NGTQ_QBG
  }

  void searchGlobalCodebook(tann::Object *query, size_t size, tann::VectorDistances &objects,
			    size_t &approximateSearchSize,
			    size_t codebookSearchSize, 
			    double epsilon) {
#ifdef	NGTQ_TRACE
    std::cerr << "searchGlobalCodebook codebookSearchSize=" << codebookSearchSize << std::endl;
#endif
    tann::SearchContainer sc(*query);
    sc.setResults(&objects);
    sc.size = codebookSearchSize;
    sc.radius = FLT_MAX;
    sc.explorationCoefficient = epsilon + 1.0;
    if (epsilon >= FLT_MAX) {
      globalCodebookIndex.linearSearch(sc);
    } else {
      globalCodebookIndex.search(sc);
    }

  }

  inline void aggregateObjectsWithExactDistance(tann::VectorDistance &globalCentroid, tann::Object *query, size_t size, tann::VectorSpace::ResultSet &results, size_t approximateSearchSize) {
    abort();
  }

   inline void aggregateObjectsWithLookupTable(tann::VectorDistance &globalCentroid, tann::Object *query, size_t size, tann::VectorSpace::ResultSet &results, size_t approximateSearchSize) {
     QuantizedObjectDistance::DistanceLookupTable distanceLUT;
     (*quantizedObjectDistance).initialize(distanceLUT);
     (*quantizedObjectDistance).createDistanceLookup(*query, globalCentroid.id, distanceLUT);
     for (size_t j = 0; j < invertedIndex[globalCentroid.id]->size() && results.size() < approximateSearchSize; j++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
       InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[globalCentroid.id]).at(j, invertedIndex.allocator);
#else
       InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[globalCentroid.id])[j];
#endif
       double distance;
       if (invertedIndexEntry.localID[0] == 0) {
	 distance = globalCentroid.distance;
       } else { 
	 distance = (*quantizedObjectDistance)(invertedIndexEntry.localID, distanceLUT);
       }  


       tann::VectorDistance obj;
       obj.id = invertedIndexEntry.id;
       obj.distance = distance;
       assert(obj.id > 0);
       results.push(obj);

     } 
  }

   void eraseInvertedIndexObject(size_t id) {
     invertedIndex.erase(id);
   }
   void eraseInvertedIndexObject() {
     for (size_t id = 0; id < invertedIndex.size(); id++) {
       try {
	 invertedIndex.erase(id);
       } catch(...) {}
     }
   }
#ifdef NGTQ_QBG
   void extractInvertedIndexObject(InvertedIndexEntry<uint16_t> &invertedIndexObjects, size_t gid) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
     std::cerr << "InvertedIndex: Not implemented." << std::endl;
     abort();
#else
     if (gid >= invertedIndex.size()) {
       std::stringstream msg;
       msg << "Quantizer::extractInvertedIndexObject: Fatal error! Invalid gid. " << invertedIndex.size() << ":" << gid;
       TANN_THROW(msg);
     }
     if (invertedIndex[gid] == 0) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
       std::cerr << "Not implemented" << std::endl;
#else
       invertedIndexObjects.clear();
#endif
       return;
     }
     invertedIndexObjects.subspaceID = invertedIndex[gid]->subspaceID;
     invertedIndexObjects.resize(invertedIndex[gid]->size());
     for (size_t idx = 0; idx < invertedIndex[gid]->size(); idx++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
       NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid]).at(idx, invertedIndex.allocator);
       invertedIndexObjects.at(idx, invertedIndex.allocator).id = entry.id;
       if (sizeof(entry.localID[0]) > sizeof(invertedIndexObjects.at(idx, invertedIndex.allocator).localID[0])) {
	 std::cerr << "you should change the object ID type." << std::endl;
	 abort();
       }
#else
       NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid])[idx];
       invertedIndexObjects[idx].id = entry.id;
       if (sizeof(entry.localID[0]) > sizeof(invertedIndexObjects[idx].localID[0])) {
	 std::cerr << "you should change the object ID type." << std::endl;
	 abort();
       }
#endif
       for (size_t i = 0; i < localCodebookIndexes.size(); i++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
	 invertedIndexObjects.at(idx,invertedIndex.allocator).localID[i] = entry.localID[i];
#else
	 invertedIndexObjects[idx].localID[i] = entry.localID[i];
#endif
       }
     }
#endif 
   }
#endif 
   
#ifdef NGTQ_QBG
   void extractInvertedIndexObject(InvertedIndexEntry<uint16_t> &invertedIndexObjects) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
     std::cerr << "not implemented" << std::endl;
     abort();
#else
    size_t lastID = 0;
    for (size_t gid = 0; gid < invertedIndex.size(); gid++) {
      if (invertedIndex[gid] == 0) {
	continue;
      }
      for (size_t idx = 0; idx < invertedIndex[gid]->size(); idx++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
        NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid]).at(idx, invertedIndex.allocator);
#else
        NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid])[idx];
#endif
	if (entry.id > lastID) {
	  lastID = entry.id;
	}
      }
    }
    invertedIndexObjects.resize(lastID + 1);
    for (size_t gid = 1; gid < invertedIndex.size(); gid++) {
      for (size_t idx = 0; idx < invertedIndex[gid]->size(); idx++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
        NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid]).at(idx, invertedIndex.allocator);
	invertedIndexObjects.at(entry.id, invertedIndex.allocator).id = entry.id;
#else
        NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid])[idx];
	invertedIndexObjects[entry.id].id = entry.id;
#endif
	if (sizeof(entry.localID[0]) > sizeof(invertedIndexObjects[entry.id].localID[0])) {
	  std::cerr << "you should change the object ID type." << std::endl;
	  abort();
	}
	for (size_t i = 0; i < invertedIndexObjects.numOfLocalIDs; i++) {
	  invertedIndexObjects[entry.id].localID[i] = entry.localID[i];
	}
	assert(invertedIndexObjects[entry.id].localID[0] == entry.localID[0]);
      }
    } 
#endif 
  }
#endif 

  void extractInvertedIndex(std::vector<std::vector<uint32_t>> &ii) { 
    ii.resize(invertedIndex.size());
    for (size_t gid = 1; gid < invertedIndex.size(); gid++) {
      if (invertedIndex[gid] == 0 || invertedIndex[gid]->size() == 0) {
	continue;
      }
      ii[gid].reserve(invertedIndex[gid]->size());
      for (size_t idx = 0; idx < invertedIndex[gid]->size(); idx++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
	NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid]).at(idx, invertedIndex.allocator);
#else
        NGTQ::InvertedIndexObject<LOCAL_ID_TYPE> &entry = (*invertedIndex[gid])[idx];
#endif
	ii[gid].push_back(entry.id);
      }
    }
  }

  inline tann::distance_type getApproximateDistance(tann::Object &query, uint32_t globalID, LOCAL_ID_TYPE *localID, QuantizedObjectDistance::DistanceLookupTable &distanceLUT) {
    double distance;
      distance = (*quantizedObjectDistance)(query, globalID, localID, distanceLUT);

    return distance;
  }


   inline void aggregateObjectsWithCache(tann::VectorDistance &globalCentroid, tann::Object *query, size_t size, tann::VectorSpace::ResultSet &results, size_t approximateSearchSize) {

     QuantizedObjectDistance::DistanceLookupTable distanceLUT;
     (*quantizedObjectDistance).initialize(distanceLUT);

     for (size_t j = 0; j < invertedIndex[globalCentroid.id]->size() && results.size() < approximateSearchSize; j++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
       InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[globalCentroid.id]).at(j, invertedIndex.allocator);
#else
       InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[globalCentroid.id])[j];
#endif
       double distance;
       if (invertedIndexEntry.localID[0] == 0) {
	 distance = globalCentroid.distance;
       } else { 
	 distance = getApproximateDistance(*query, globalCentroid.id, invertedIndexEntry.localID, distanceLUT);
       }  

       tann::VectorDistance obj;
       obj.id = invertedIndexEntry.id;
       obj.distance = distance;
       assert(obj.id > 0);
       results.push(obj);

     } 
  }


  inline void aggregateObjects(tann::VectorDistance &globalCentroid, tann::Object *query, size_t size, tann::VectorSpace::ResultSet &results, size_t approximateSearchSize) {
    for (size_t j = 0; j < invertedIndex[globalCentroid.id]->size() && results.size() < approximateSearchSize; j++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
      InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[globalCentroid.id]).at(j, invertedIndex.allocator);
#else
      InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[globalCentroid.id])[j];
#endif
      double distance;
      if (invertedIndexEntry.localID[0] == 0) {
	distance = globalCentroid.distance;
      } else { 
	distance = (*quantizedObjectDistance)(*query, globalCentroid.id, invertedIndexEntry.localID);
      }  


      tann::VectorDistance obj;
      obj.id = invertedIndexEntry.id;
      obj.distance = distance;
      assert(obj.id > 0);
      results.push(obj);
      if (results.size() >= approximateSearchSize) {
	return;
      }

    } 
  }


  inline void aggregateObjects(tann::Object *query, size_t size, tann::VectorDistances &objects, tann::VectorSpace::ResultSet &results, size_t approximateSearchSize, AggregateObjectsFunction aggregateObjectsFunction) {
    for (size_t i = 0; i < objects.size(); i++) {
      if (invertedIndex[objects[i].id] == 0) {
	if (property.centroidCreationMode == CentroidCreationModeDynamic) {
	  cerr << "Inverted index is empty. " << objects[i].id << endl;
	}
	continue;
      }
      ((*this).*aggregateObjectsFunction)(objects[i], query, size, results, results.size() == 0 ? INT_MAX : approximateSearchSize);
      if (results.size() >= approximateSearchSize) {
	return;
      }
    } 
  }

  void refineDistance(tann::Object *query, tann::VectorDistances &results) {
#ifndef NGTQ_QBG
     tann::VectorSpace &objectSpace = globalCodebookIndex.getObjectSpace();
     for (auto i = results.begin(); i != results.end(); ++i) {
       tann::VectorDistance &result = *i;
       tann::Object o(&objectSpace);
       objectList.get(result.id, (tann::Object&)o, &objectSpace);
       double distance = objectSpace.getComparator()(*query, (tann::Object&)o);
       result.distance = distance;
     }
     std::sort(results.begin(), results.end());
#endif 
  }

  void search(tann::Object *query, tann::VectorDistances &objs,
	      size_t size, 
      	      float expansion,
	      AggregationMode aggregationMode,
	      double epsilon = FLT_MAX) {
    size_t approximateSearchSize = size * expansion;
    size_t codebookSearchSize = approximateSearchSize / (objectList.size() / globalCodebookIndex.getObjectRepositorySize()) + 1;
    search(query, objs, size, approximateSearchSize, codebookSearchSize, aggregationMode, epsilon);
  }

  void search(tann::Object *query, tann::VectorDistances &objs,
	      size_t size, size_t approximateSearchSize,
	      size_t codebookSearchSize, bool resultRefinement,
	      bool lookUpTable = false,
	      double epsilon = FLT_MAX) {
    AggregationMode aggregationMode;
    if (resultRefinement) {
      aggregationMode = AggregationModeExactDistance;
    } else {
      if (lookUpTable) {
	aggregationMode = AggregationModeApproximateDistanceWithLookupTable;
      } else {
	aggregationMode = AggregationModeApproximateDistanceWithCache;
      }
    }
    search(query, objs, size, approximateSearchSize, codebookSearchSize, aggregationMode, epsilon);
  }

  void search(tann::Object *query, tann::VectorDistances &objs,
	      size_t size, size_t approximateSearchSize,
	      size_t codebookSearchSize, 
	      AggregationMode aggregationMode,
	      double epsilon = FLT_MAX) {
    if (aggregationMode == AggregationModeApproximateDistanceWithLookupTable) {
      if (property.dataType != DataTypeFloat) {
	TANN_THROW("NGTQ: Fatal inner error. the lookup table is only for dataType float!");
      }
    }
    tann::VectorDistances objects;
    searchGlobalCodebook(query, size, objects, approximateSearchSize, codebookSearchSize, epsilon);

    objs.clear();
    tann::VectorSpace::ResultSet results;
    distanceComputationCount = 0;

    AggregateObjectsFunction aggregateObjectsFunction = &QuantizerInstance::aggregateObjectsWithCache;
    switch(aggregationMode) {
    case AggregationModeExactDistance :
      aggregateObjectsFunction = &QuantizerInstance::aggregateObjectsWithExactDistance;
      break;
    case AggregationModeApproximateDistanceWithLookupTable :
      aggregateObjectsFunction = &QuantizerInstance::aggregateObjectsWithLookupTable;
      break;
    case AggregationModeExactDistanceThroughApproximateDistance :
    case AggregationModeApproximateDistanceWithCache :
      aggregateObjectsFunction = &QuantizerInstance::aggregateObjectsWithCache;
      break;
    case AggregationModeApproximateDistance :
      aggregateObjectsFunction = &QuantizerInstance::aggregateObjects;
      break;
    default:
      cerr << "NGTQ::Fatal Error. invalid aggregation mode. " << aggregationMode << endl;
      abort();
    }

    aggregateObjects(query, size, objects, results, approximateSearchSize, aggregateObjectsFunction);

    objs.resize(results.size());
    while (!results.empty()) {
      objs[results.size() - 1] = results.top();
      results.pop();
    }
    if (objs.size() > size) {
      objs.resize(size);
    }
    if (aggregationMode == AggregationModeExactDistanceThroughApproximateDistance) {
      refineDistance(query, objs);
    }
  }


  double calculateQuantizationError() {
#ifdef NGTQ_QBG
    std::cerr << "calculateQuantizationError: Not implemented." << std::endl;
    return 0.0;
#else
    tann::VectorSpace &objectSpace = globalCodebookIndex.getObjectSpace();
    double distance = 0.0;
    double globalDistance = 0.0;
    size_t count = 0;
    for (size_t gi = 0; gi < invertedIndex.size(); gi++) {
      if (invertedIndex[gi] != 0) {
	tann::PersistentObject &gcentroid = *globalCodebookIndex.getObjectSpace().getRepository().get(gi);
	for (size_t li = 0; li < invertedIndex[gi]->size(); li++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
          size_t id = invertedIndex[gi]->at(li, invertedIndex.allocator).id;
#else
	  size_t id = invertedIndex[gi]->at(li).id;
#endif
	  tann::Object object(&objectSpace);
	  objectList.get(id, (tann::Object&)object, &objectSpace);
#ifdef NGTQ_SHARED_INVERTED_INDEX
	  double d = (*quantizedObjectDistance)(object, gi, invertedIndex[gi]->at(li, invertedIndex.allocator).localID);
#else
	  double d = (*quantizedObjectDistance)(object, gi, invertedIndex[gi]->at(li).localID);
#endif
	  distance += d;
	  count++;
	  tann::distance_type gd = globalCodebookIndex.getObjectSpace().getComparator()(object, gcentroid);
	  globalDistance += gd;
	}
      }
    }
    distance /= count;
    globalDistance /= count;
    std::cerr << distance << ":" << globalDistance << std::endl;
    return distance;
#endif 
  }

  void info(ostream &os, char mode) {
    cerr << "info" << endl;
    os << "Inverted index size=" << invertedIndex.size() << endl;
    for (size_t i = 0; i < invertedIndex.size(); i++) {
      if (invertedIndex[i] != 0) {
	os << i << " " << invertedIndex[i]->size();
	if (mode == 'a' || mode == 'l') {
	  os << ": ";
	  for (size_t li = 0; li < invertedIndex[i]->size(); li++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
            os << invertedIndex[i]->at(li, invertedIndex.allocator).id << " ";
#else
	    os << invertedIndex[i]->at(li).id << " ";
#endif
	  }
	}
	os << endl;
      }
    }
    if (mode == 'a' || mode == 'e') {
      os << "Quantization Error=" << calculateQuantizationError() << endl;
    }
  }

  void verify() {
    cerr << "sizeof(LOCAL_ID_TYPE)=" << sizeof(LOCAL_ID_TYPE) << endl;
    size_t objcount = objectList.size();
    cerr << "Object count=" << objcount << endl;
    size_t gcount = globalCodebookIndex.getObjectRepositorySize();
    cerr << "Global codebook size=" << gcount << endl;
    size_t lcount = localCodebookIndexes[0].getObjectRepositorySize();
    cerr << "Local codebook size=" << lcount << endl;
    lcount *= 1.1;
    cerr << "Inverted index size=" << invertedIndex.size() << endl;

    cerr << "Started verifying global codebook..." << endl;
    vector<uint8_t> status;
    globalCodebookIndex.verify(status);

    cerr << "Started verifing the inverted index." << endl;
    size_t errorCount = 0;
    for (size_t i = 1; i < invertedIndex.size(); i++) {
      if (i % 1000000 == 0) {
	cerr << "  verified " << i << " entries" << endl;
      }
      if (errorCount > 100) {
	cerr << "Too many errors. Stop..." << endl;
	return;
      }
      if (invertedIndex[i] == 0) {
	cerr << "Warning inverted index is zero. " << i << endl;
	continue;
      }
      for (size_t j = 1; j < invertedIndex[i]->size(); j++) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
	InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[i]).at(j, invertedIndex.allocator);
#else
	InvertedIndexObject<LOCAL_ID_TYPE> &invertedIndexEntry = (*invertedIndex[i])[j];
#endif
	if (invertedIndexEntry.id >= objcount) {
	  cerr << "The object ID of the inverted index entry is too big! " << invertedIndexEntry.id << ":" << objcount << endl;
	  cerr << "  No. of the wrong entry in the inverted index is " << i << endl;
	  errorCount++;
	}
	if (invertedIndexEntry.id == 0) {
	  cerr << "The object ID of the inverted index entry is zero! " << invertedIndexEntry.id << ":" << objcount << endl;
	  cerr << "  No. of the wrong entry in the inverted index is " << i << endl;
	  errorCount++;
	}
	for (size_t li = 0; li < property.localDivisionNo; li++) {
	  if (lcount != 0 && invertedIndexEntry.localID[li] >= lcount) {
	    cerr << "The local centroid ID of the inverted index entry is wrong. " << invertedIndexEntry.localID[li] << ":" << lcount << endl;
	    cerr << "  No. of the wrong entry in the inverted index is " << i << ". No. of local centroid db is " << li << endl;
	    errorCount++;
	  }
	  if (invertedIndexEntry.localID[li] == 0) {
	  }
	}
      }
    }
  }


  size_t getInstanceSharedMemorySize(ostream &os, tann::SharedMemoryAllocator::GetMemorySizeType t = tann::SharedMemoryAllocator::GetTotalMemorySize) {
#ifdef NGTQ_SHARED_INVERTED_INDEX
    size_t size = invertedIndex.getAllocator().getMemorySize(t);
#else
    size_t size = 0;
#endif
    os << "inverted=" << size << endl;
    os << "Local centroid:" << endl;
    for (size_t di = 0; di < localCodebookIndexes.size(); di++) {
      size += localCodebookIndexes[di].getSharedMemorySize(os, t);
    }
    return size;
  }

  size_t getNumberOfObjects(tann::GraphAndTreeIndex &index) {
    return index.getObjectRepositorySize() == 0 ? 0 : static_cast<int>(index.getObjectRepositorySize()) - 1;
  }

  QuantizedObjectDistance &getQuantizedObjectDistance() { return *quantizedObjectDistance; }

  size_t getInvertedIndexSize() { return invertedIndex.size(); }

#ifdef NGTQ_SHARED_INVERTED_INDEX
  tann::PersistentRepository<IIEntry>	invertedIndex;
#else
  tann::Repository<IIEntry>	invertedIndex;
#endif
  QuantizedObjectDistance	*quantizedObjectDistance;
  GenerateResidualObject	*generateResidualObject;
  float				*localCodebooks;
  bool				silence;
};
  
class Quantization {
public:
  static Quantizer *generate(Property &property) {
    size_t localIDByteSize  = property.localIDByteSize;
    Quantizer *quantizer    = 0;
    if (property.centroidCreationMode == CentroidCreationModeNone) {
      TANN_THROW("Centroid creation mode is not specified");
    } else {
      if (localIDByteSize == 4) {
	quantizer = new QuantizerInstance<uint32_t>;
      } else if (localIDByteSize == 2) {
        quantizer = new QuantizerInstance<uint16_t>;
#ifdef NGTQ_QBG
      } else if (localIDByteSize == 1) {
	quantizer = new QuantizerInstance<uint8_t>;
#endif
      } else {
	std::stringstream msg;
	msg << "Not support the specified size of local ID. " << localIDByteSize;
	TANN_THROW(msg);
      }

    }

    return quantizer;
  }
};

 class Index {
 public:
   Index():quantizer(0) {}
   Index(const string& index, bool rdOnly = false):quantizer(0) {
     open(index, rdOnly);
   }
   ~Index() { close(); }



   static void create(const string &index, Property &property, 
		      tann::Property &globalProperty,
#ifdef NGTQ_QBG
		      tann::Property &localProperty,
		      std::vector<float> *rotation = 0,
		      const std::string &objectFile = "") {
#else
		      tann::Property &localProperty) {
#endif
     if (property.dimension == 0) {
       TANN_THROW("NGTQ::create: Error. The dimension is zero.");
     }
     property.setup(property);
     NGTQ::Quantizer *quantizer = NGTQ::Quantization::generate(property);
     try {
#ifdef NGTQ_QBG
       if (property.dimension == 0) {
	 property.dimension = property.genuineDimension;
       }
       if (property.dimension % 4 != 0) {
	 property.dimension = ((property.dimension - 1) / 4 + 1) * 4;
       }
       quantizer->property = property;
       quantizer->create(index, globalProperty, localProperty, rotation, objectFile);
#else
       quantizer->property = property;
       quantizer->create(index, globalProperty, localProperty);
#endif
       if (property.dimension == 0) {
	 TANN_THROW("Quantizer: Dimension is zero.");
       }
     } catch(tann::Exception &err) {
       delete quantizer;
       throw err;
     }
     delete quantizer;
   }
#ifdef NGTQ_SHARED_INVERTED_INDEX
   static void compress(const string &indexFile) {
     Index index;
     index.open(indexFile);
     string tmpivt = indexFile + "/ivt-tmp";
     index.getQuantizer().reconstructInvertedIndex(tmpivt);
     index.close();
     string ivt = indexFile + "/ivt";
     unlink(ivt.c_str());
     rename(tmpivt.c_str(), ivt.c_str());
     string ivtc = ivt + "c";
     unlink(ivtc.c_str());
     string tmpivtc = tmpivt + "c";
     rename(tmpivtc.c_str(), ivtc.c_str());
   }
#endif

#ifndef NGTQ_QBG
  static void rebuild(const string &indexName,		
		      const string &rebuiltIndexName	
		     ) {

    const string srcObjectList = indexName + "/obj";
    const string dstObjectList = rebuiltIndexName + "/obj";

    if (std::rename(srcObjectList.c_str(), dstObjectList.c_str()) != 0) {
      stringstream msg;
      msg << "Quantizer::rebuild: Cannot rename an object file. " << srcObjectList << "=>" << dstObjectList ;
      TANN_THROW(msg);
    }

    try {
      NGTQ::Index index(rebuiltIndexName);
    
      index.rebuildIndex();

      index.save();
      index.close();
    } catch(tann::Exception &err) {
      std::rename(dstObjectList.c_str(), srcObjectList.c_str());
      throw err;
    }

  }
#endif

  void open(const string &index, bool readOnly = false) {
     close();
     tann::Property globalProperty;
     globalProperty.clear();
     globalProperty.edgeSizeForSearch = 40;
     quantizer = getQuantizer(index, globalProperty, readOnly);
     if ((quantizer->property.quantizerType == NGTQ::QuantizerTypeQG) && readOnly) {
       quantizer->closeCodebooks();
     }

  }

   void save() {
     getQuantizer().save();
   }

   void close() {
     if (quantizer != 0) {
       delete quantizer;
       quantizer = 0;
     }
   }

#ifndef NGTQ_QBG
   void insert(vector<pair<tann::Object*, size_t> > &objects) {
     std::cerr << "Not implemented." << std::endl;
     abort();
   }
#endif

   void insertIntoObjectRepository(std::vector<float> object) {
     getQuantizer().insertIntoObjectRepository(object, 0);
   }
#ifdef NGTQ_QBG
   void createIndex(size_t beginID = 1, size_t endID = 0) {
     getQuantizer().createIndex(beginID, endID);
   }

   void createIndex(std::vector<std::vector<float>> &quantizationCodebook,
		    std::vector<uint32_t> &codebookIndex,
		    std::vector<uint32_t> &objectIndex,
		    size_t beginID = 1, size_t endID = 0) {
     setupInvertedIndex(quantizationCodebook, codebookIndex, objectIndex);
     createIndex(beginID, endID);
   }
#endif 

   void setupInvertedIndex(std::vector<std::vector<float>> &quantizationCodebook,
				 std::vector<uint32_t> &codebookIndex,
				 std::vector<uint32_t> &objectIndex) {
     getQuantizer().setupInvertedIndex(quantizationCodebook, codebookIndex, objectIndex);
   }


#ifndef NGTQ_QBG
   void rebuildIndex() {
     getQuantizer().rebuildIndex();
   }
#endif

   tann::Object *allocateObject(string &line, const string &sep, size_t dimension) {
     return getQuantizer().allocateObject(line, sep);
   }

   tann::Object *allocateObject(vector<double> &obj) {
     return getQuantizer().allocateObject(obj);
   }

   tann::Object *allocateObject(vector<float> &obj) {
     return getQuantizer().allocateObject(obj);
   }

   void deleteObject(tann::Object *object) { getQuantizer().deleteObject(object); }

   void search(tann::Object *object, tann::VectorDistances &objs,
	       size_t size, size_t approximateSearchSize,
	       size_t codebookSearchSize, bool resultRefinement, 
	       bool lookUpTable, double epsilon) {
     getQuantizer().search(object, objs, size, approximateSearchSize, codebookSearchSize, 
			   resultRefinement, lookUpTable, epsilon);
   }

   void search(tann::Object *object, tann::VectorDistances &objs,
	       size_t size, float expansion,
	       AggregationMode aggregationMode,
	       double epsilon) {
     getQuantizer().search(object, objs, size, expansion, 
			   aggregationMode, epsilon);
   }

   void info(ostream &os, char mode) { getQuantizer().info(os, mode); }

   void verify() { getQuantizer().verify(); }

   NGTQ::Quantizer &getQuantizer() { 
     if (quantizer == 0) {
       TANN_THROW("NGTQ::Index: Not open.");
     }
     return *quantizer; 
   }

   size_t getGlobalCodebookSize() { return quantizer->globalCodebookIndex.getObjectRepositorySize(); }
   size_t getLocalCodebookSize(size_t idx) { return quantizer->getLocalCodebookSize(idx); }

   size_t getInvertedIndexSize() { return quantizer->getInvertedIndexSize(); }

   size_t getSharedMemorySize(ostream &os, tann::SharedMemoryAllocator::GetMemorySizeType t = tann::SharedMemoryAllocator::GetTotalMemorySize) {
     return quantizer->getSharedMemorySize(os, t);
   }

   std::vector<float> getObject(size_t id) {
     std::vector<float> object;
     auto &quantizer = getQuantizer();
     if (!quantizer.objectList.get(id, object, &quantizer.globalCodebookIndex.getObjectSpace())) {
       std::stringstream msg;
       msg << "cannot get the specified object. " << id;
       TANN_THROW(msg);
     }
     return object;
   }

 protected:
   static NGTQ::Quantizer *getQuantizer(const string &index, tann::Property &globalProperty, bool readOnly) {
     NGTQ::Property property;
     try {
       property.load(index);
     } catch (tann::Exception &err) {
       stringstream msg;
       msg << "Quantizer::getQuantizer: Cannot load the property. " << index << " : " << err.what();
       TANN_THROW(msg);
     }
     NGTQ::Quantizer *quantizer = NGTQ::Quantization::generate(property);
     if (quantizer == 0) {
       TANN_THROW("NGTQ::Index: Cannot get quantizer.");
     }
     try {
       quantizer->open(index, globalProperty, property.quantizerType == NGTQ::QuantizerTypeQBG ? readOnly : false);
     } catch(tann::Exception &err) {
       delete quantizer;
       throw err;
     }
     return quantizer;
   }

   NGTQ::Quantizer *quantizer;

   bool silence;
 };

} // namespace NGTQ
