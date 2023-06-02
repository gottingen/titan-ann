//
// Copyright (C) 2021 Yahoo Japan Corporation
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

#include <string>
#include <iostream>
#include <sstream>

#include "tann/ngt/capi.h"
#include "tann/ngt/ngtq/quantizer.h"
#include "tann/ngt/ngtq/capi.h"
#include "tann/ngt/ngtq/quantized_graph.h"
#include "tann/ngt/ngtq/quantized_blob_graph.h"
#include "tann/ngt/ngtq/optimizer.h"
#include "tann/ngt/ngtq/hierarchical_kmeans.h"

#ifdef NGTQ_QBG

static bool operate_error_string_(const std::stringstream &ss, NGTError error){
  if(error != NULL){  
    try{
      std::string *error_str = static_cast<std::string*>(error);
      *error_str = ss.str();
    }catch(std::exception &err){
      std::cerr << ss.str() << " > " << err.what() << std::endl;
      return false;
    }
  }else{
    std::cerr << ss.str() << std::endl;
  }
  return true;
}

void ngtqg_initialize_query(NGTQGQuery *query) {
  query->query = 0;
  query->size = 20;
  query->epsilon = 0.03;
  query->result_expansion = 3.0;
  query->radius = FLT_MAX;
}

NGTQGIndex ngtqg_open_index(const char *index_path, NGTError error) {
  try{
    std::string index_path_str(index_path);
    auto *index = new NGTQG::Index(index_path_str);
    index->disableLog();
    return static_cast<NGTQGIndex>(index);
  }catch(std::exception &err){
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return NULL;
  }
}

void ngtqg_close_index(NGTQGIndex index) {
    if(index == NULL) return;
    (static_cast<NGTQG::Index*>(index))->close();
    delete static_cast<NGTQG::Index*>(index);
}

static bool ngtqg_search_index_(NGTQG::Index* pindex, std::vector<float> &query, NGTQGQuery &param, NGTObjectDistances results) {
  // set search parameters.
  NGTQG::SearchQuery sq(query);  // Query.

  sq.setResults(static_cast<tann::ngt::ObjectDistances*>(results));          // set the result set.
  sq.setSize(param.size);                        // the number of resultant objects.
  sq.setRadius(param.radius);                    // search radius.
  sq.setEpsilon(param.epsilon);                  // exploration coefficient.
  sq.setResultExpansion(param.result_expansion); // result expansion.

  pindex->search(sq);

  return true;
}

bool ngtqg_search_index(NGTQGIndex index, NGTQGQuery query, NGTObjectDistances results, NGTError error) {
  if(index == NULL || query.query == NULL || results == NULL){
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : parametor error: index = " << index << " query = " << query.query << " results = " << results;
    operate_error_string_(ss, error);
    return false;
  }
  
  NGTQG::Index* pindex = static_cast<NGTQG::Index*>(index);   
  int32_t dim = pindex->getObjectSpace().getDimension();

  tann::ngt::Object *ngtquery = NULL;

  if(query.radius < 0.0){
    query.radius = FLT_MAX;
  }

  try{
    std::vector<float> vquery(&query.query[0], &query.query[dim]);
    ngtqg_search_index_(pindex, vquery, query, results);
  }catch(std::exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    if(ngtquery != NULL){
      pindex->deleteObject(ngtquery);
    }
    return false;
  }
  return true;
}

void ngtqg_initialize_quantization_parameters(NGTQGQuantizationParameters *parameters) {
  parameters->dimension_of_subvector = 0;
  parameters->max_number_of_edges = 128;
}

bool ngtqg_quantize(const char *indexPath, NGTQGQuantizationParameters parameters, NGTError error) {
  try{
    NGTQG::Index::quantize(indexPath, parameters.dimension_of_subvector, parameters.max_number_of_edges, true);
    return true;
  }catch(std::exception &err){
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);    
    return false;
  }
}



uint32_t qbg_get_result_size(QBGObjectDistances results, NGTError error) {
  return ngt_get_result_size(results, error);
}

NGTObjectDistance qbg_get_result(const QBGObjectDistances results, const uint32_t idx, NGTError error) {
  return ngt_get_result(results, idx, error);
}

void qbg_destroy_results(QBGObjectDistances results) {
  ngt_destroy_results(results);
}


void qbg_initialize_construction_parameters(QBGConstructionParameters *parameters)
{
  parameters->extended_dimension = 0;
  parameters->dimension = 0;
  parameters->number_of_subvectors = 1;
  parameters->number_of_blobs = 0;
  parameters->internal_data_type = NGTQ::DataTypeFloat;
  parameters->data_type = NGTQ::DataTypeFloat;
  parameters->distance_type = NGTQ::DistanceType::DistanceTypeL2;
}

bool qbg_create(const char *indexPath, QBGConstructionParameters *parameters, NGTQGError error)
{

  try {
    cerr << "qbgcapi: Create" << endl;
    std::vector<float> r;
    NGTQ::Property property;
    tann::ngt::Property globalProperty;
    tann::ngt::Property localProperty;
    property.dimension = parameters->extended_dimension; 
    if (property.dimension == 0) {
      property.dimension = parameters->dimension;
    }
    property.genuineDimension = parameters->dimension; 
    property.globalRange = 0; 
    property.localRange = 0; 
    property.globalCentroidLimit = parameters->number_of_blobs;
    property.localCentroidLimit = 16;
    property.localDivisionNo = parameters->number_of_subvectors;
    property.singleLocalCodebook = false;
    property.centroidCreationMode = NGTQ::CentroidCreationModeStaticLayer;
    property.localCentroidCreationMode = NGTQ::CentroidCreationModeStatic;
    property.localIDByteSize = 1;
    property.dataType = static_cast<NGTQ::DataType>(parameters->internal_data_type); 
    property.genuineDataType = static_cast<ObjectFile::DataType>(parameters->data_type); 
    property.distanceType = static_cast<NGTQ::DistanceType>(parameters->distance_type);

    globalProperty.edgeSizeForCreation = 10;
    globalProperty.edgeSizeForSearch = 40;
    globalProperty.indexType = tann::ngt::Property::GraphAndTree;
    globalProperty.insertionRadiusCoefficient = 1.1;

    localProperty.indexType = globalProperty.indexType;
    localProperty.insertionRadiusCoefficient = globalProperty.insertionRadiusCoefficient;

    std::vector<float> *rotation = 0;
    const std::string objectPath;
    QBG::Index::create(indexPath, property, globalProperty, localProperty, rotation, objectPath);
  } catch(tann::ngt::Exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);    
    return false;
  }

  return true;
}

QBGIndex qbg_open_index(const char *index_path, bool read_only, QBGError error) {
  try {
    std::string index_path_str(index_path);
    auto *index = new QBG::Index(index_path_str, read_only);
    return static_cast<QBGIndex>(index);
  } catch(std::exception &err){
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return NULL;
  }
}

void qbg_close_index(QBGIndex index) {
  if (index == NULL) return;
  (static_cast<QBG::Index*>(index))->close();
  delete static_cast<QBG::Index*>(index);
  index = 0;
}

bool qbg_save_index(QBGIndex index, QBGError error) {
  if (index == NULL) return false;
  try {
    (static_cast<QBG::Index*>(index))->save();
  } catch(std::exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return false;
  }
  return true;
}

ObjectID qbg_append_object(QBGIndex index, float *obj, uint32_t obj_dim, QBGError error) {
  if (index == NULL || obj == NULL || obj_dim == 0){
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : parametor error: index = " << index << " obj = " << obj << " obj_dim = " << obj_dim;
    operate_error_string_(ss, error);
    return 0;
  }

  try {
    auto *pindex = static_cast<QBG::Index*>(index);
    std::vector<float> vobj(&obj[0], &obj[obj_dim]);
    return pindex->append(vobj);
  } catch(std::exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return 0;
  }
}

void qbg_initialize_build_parameters(QBGBuildParameters *parameters) {
  parameters->hierarchical_clustering_init_mode = static_cast<int>(tann::ngt::Clustering::InitializationModeKmeansPlusPlus);
  parameters->number_of_first_objects = 0;
  parameters->number_of_first_clusters = 0;
  parameters->number_of_second_objects = 0;
  parameters->number_of_second_clusters = 0;
  parameters->number_of_third_clusters = 0;

  parameters->number_of_objects = 1000;
  parameters->number_of_subvectors = 1;
  parameters->optimization_clustering_init_mode = static_cast<int>(tann::ngt::Clustering::InitializationModeKmeansPlusPlus);
  parameters->rotation_iteration = 2000;
  parameters->subvector_iteration = 400;
  parameters->number_of_matrices = 3;
  parameters->rotation = true;
  parameters->repositioning = false;
}

bool qbg_build_index(const char *index_path, QBGBuildParameters *parameters, QBGError error) {

  QBG::HierarchicalKmeans hierarchicalKmeans;

  hierarchicalKmeans.maxSize = 1000; 
  hierarchicalKmeans.numOfClusters = 2; 
  hierarchicalKmeans.numOfTotalClusters = 0;
  hierarchicalKmeans.numOfTotalBlobs = 0; 
  hierarchicalKmeans.clusterID = -1;
  hierarchicalKmeans.initMode = static_cast<tann::ngt::Clustering::InitializationMode>(parameters->hierarchical_clustering_init_mode);
  hierarchicalKmeans.numOfRandomObjects = 0;
  hierarchicalKmeans.extractCentroid = false;
  hierarchicalKmeans.numOfFirstObjects = parameters->number_of_first_objects;
  hierarchicalKmeans.numOfFirstClusters = parameters->number_of_first_clusters;
  hierarchicalKmeans.numOfSecondObjects = parameters->number_of_second_objects;
  hierarchicalKmeans.numOfSecondClusters = parameters->number_of_second_clusters;
  hierarchicalKmeans.numOfThirdClusters = parameters->number_of_third_clusters;
  hierarchicalKmeans.numOfObjects = 0; 
  hierarchicalKmeans.threeLayerClustering = true;
  hierarchicalKmeans.silence = true;

  try {
    hierarchicalKmeans.clustering(index_path);
  } catch (tann::ngt::Exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return false;
  }

  QBG::Optimizer optimizer;

  optimizer.numberOfObjects = parameters->number_of_objects;
  optimizer.numberOfClusters = 16;	
  optimizer.numberOfSubvectors = 0;	
  optimizer.clusteringType = tann::ngt::Clustering::ClusteringTypeKmeansWithNGT;
  optimizer.initMode = static_cast<tann::ngt::Clustering::InitializationMode>(parameters->optimization_clustering_init_mode);
  optimizer.convergenceLimitTimes = 5;
  optimizer.iteration = parameters->rotation_iteration;
  optimizer.clusterIteration = parameters->subvector_iteration;
  optimizer.clusterSizeConstraint = false;
  optimizer.nOfMatrices = parameters->number_of_matrices;
  optimizer.seedStartObjectSizeRate = 0.1;
  optimizer.seedStep = 2;
  optimizer.reject = 0.9;
  optimizer.timelimit = 24 * 2; 
  optimizer.timelimit *= 60.0 * 60.0; 
  optimizer.rotation = parameters->rotation;
  optimizer.repositioning = parameters->repositioning;
  optimizer.globalType = QBG::Optimizer::GlobalTypeNone;
  optimizer.silence = true;

  try {
    auto nthreads = omp_get_max_threads();
    optimizer.optimize(index_path, nthreads);
  } catch (tann::ngt::Exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return false;
  }

  try {
    auto silence = true;
    QBG::Index::build(index_path, silence);
  } catch (tann::ngt::Exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return false;
  }
  return true;
}

void qbg_initialize_query(QBGQuery *parameters) {
  parameters->query = 0;
  parameters->number_of_results = 20;
  parameters->epsilon = 0.1;
  parameters->blob_epsilon = 0.0;
  parameters->result_expansion = 3.0;
  parameters->number_of_explored_blobs = 256;
  parameters->number_of_edges = 0;
  parameters->radius = 0;
}

static bool qbg_search_index_(QBG::Index* pindex, std::vector<float> &query, QBGQuery &param, NGTObjectDistances results) {
  // set search parameters.
  if (param.radius < 0.0){
    param.radius = FLT_MAX;
  }

  QBG::SearchContainer sc;
  sc.setObjectVector(query);
  sc.setResults(static_cast<tann::ngt::ObjectDistances*>(results));
  if (param.result_expansion >= 1.0) {
    sc.setSize(static_cast<float>(param.number_of_results) * param.result_expansion);
    sc.setExactResultSize(param.number_of_results);
  } else {
    sc.setSize(param.number_of_results);
    sc.setExactResultSize(0);
  }
  sc.setEpsilon(param.epsilon);
  sc.setBlobEpsilon(param.blob_epsilon);
  sc.setEdgeSize(param.number_of_edges);
  sc.setGraphExplorationSize(param.number_of_explored_blobs);

  pindex->searchBlobGraph(sc);

  return true;
}

bool qbg_search_index(QBGIndex index, QBGQuery query, NGTObjectDistances results, QBGError error) {
  if (index == NULL || query.query == NULL || results == NULL) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : parametor error: index = " << index << " query = " << query.query << " results = " << results;
    operate_error_string_(ss, error);
    return false;
  }

  auto *pindex = static_cast<QBG::Index*>(index);
  int32_t dim = pindex->getQuantizer().property.genuineDimension;

  try {
    std::vector<float> vquery(&query.query[0], &query.query[dim]);
    qbg_search_index_(pindex, vquery, query, results);
  } catch(std::exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return false;
  }

  return true;
}

float* qbg_get_object(QBGIndex index, ObjectID id, QBGError error) {
  if (index == NULL) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : parametor error: index = " << index;
    operate_error_string_(ss, error);
    return 0;
  }

  auto *pindex = static_cast<QBG::Index*>(index);

  try {
    std::vector<float> object = pindex->getObject(id);
    auto obj = malloc(sizeof(float) * object.size());
    if (obj == 0) {
      std::stringstream ss;
      ss << "Capi : " << __FUNCTION__ << "() : Error: Cannot allocate memory.";
      operate_error_string_(ss, error);
      return 0;
    }
    memcpy(obj, object.data(), sizeof(float) * object.size());
    return static_cast<float*>(obj);
  } catch(std::exception &err) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : Error: " << err.what();
    operate_error_string_(ss, error);
    return 0;
  }
}

size_t qbg_get_dimension(QBGIndex index, QBGError error) {
  if (index == NULL) {
    std::stringstream ss;
    ss << "Capi : " << __FUNCTION__ << "() : parametor error: index = " << index;
    operate_error_string_(ss, error);
    return 0;
  }
  auto *pindex = static_cast<QBG::Index*>(index);
  return pindex->getQuantizer().property.genuineDimension;
}

#endif 
