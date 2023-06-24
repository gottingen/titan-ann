//
// Copyright (C) 2020 Yahoo Japan Corporation
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

#include "quantizer.h"

#if defined(NGTQ_QBG) && !defined(NGTQ_SHARED_INVERTED_INDEX)

#include "tann/ngt/ngtq/qbg_cli.h"
#include "tann/ngt/ngtq/optimizer.h"
#include "tann/ngt/ngtq/hierarchical_kmeans.h"

typedef NGTQ::Quantizer::ObjectList QBGObjectList;


class QbgCliBuildParameters : public QBG::BuildParameters {
public:
  QbgCliBuildParameters(tann::Args &a):args(a){
    args.parse("Zv");
  }

  void getBuildParameters() {
    getHierarchicalClustringParameters();
    getOptimizationParameters();
  }

  void getCreationParameters() {
    char objectType = args.getChar("o", 'f');
    char distanceType = args.getChar("D", '2');
    creation.numOfObjects = args.getl("n", 0);

    creation.threadSize = args.getl("p", 24);
    creation.dimension = args.getl("d", 0);
#ifdef NGTQ_QBG
    creation.localCentroidLimit = args.getl("c", 16);
#else
    creation.localCentroidLimit = args.getl("c", 65000);
#endif
    creation.localDivisionNo = args.getl("N", 0);
    creation.batchSize = args.getl("b", 1000);
    creation.localClusteringSampleCoefficient = args.getl("s", 10);
    {
      char localCentroidType = args.getChar("T", 'f');
      creation.singleLocalCodebook = localCentroidType == 't' ? true : false;
    }
    {
      char centroidCreationMode = args.getChar("M", 'l');
      switch(centroidCreationMode) {
      case 'd': creation.centroidCreationMode = NGTQ::CentroidCreationModeDynamic; break;
      case 's': creation.centroidCreationMode = NGTQ::CentroidCreationModeStatic; break;
      case 'l': creation.centroidCreationMode = NGTQ::CentroidCreationModeStaticLayer; break;
      default:
	std::stringstream msg;
	msg << "Command::CreateParameters: Error: Invalid centroid creation mode. " << centroidCreationMode;
	TANN_THROW(msg);
      }
    }
    {
      char localCentroidCreationMode = args.getChar("L", 's');
      switch(localCentroidCreationMode) {
      case 'd': creation.localCentroidCreationMode = NGTQ::CentroidCreationModeDynamic; break;
      case 's': creation.localCentroidCreationMode = NGTQ::CentroidCreationModeStatic; break;
      case 'k': creation.localCentroidCreationMode = NGTQ::CentroidCreationModeDynamicKmeans; break;
      default:
	std::stringstream msg;
	msg << "Command::CreateParameters: Error: Invalid centroid creation mode. " << localCentroidCreationMode;
	TANN_THROW(msg);
      }
    }
#ifdef NGTQ_QBG
    creation.localIDByteSize = args.getl("B", 1);
#endif
      
    creation.globalEdgeSizeForCreation = args.getl("E", 10);
    creation.globalEdgeSizeForSearch = args.getl("S", 40);
    {
      char indexType = args.getChar("i", 't');
      creation.globalIndexType = indexType == 't' ? tann::Property::GraphAndTree : tann::Property::Graph;
      creation.localIndexType = creation.globalIndexType;
    }
    creation.globalInsertionRadiusCoefficient = args.getf("e", 0.1) + 1.0;
    creation.localInsertionRadiusCoefficient = creation.globalInsertionRadiusCoefficient;


    switch (objectType) {
    case 'f': creation.dataType = NGTQ::DataTypeFloat; break;
#ifdef TANN_ENABLE_HALF_FLOAT
    case 'h': creation.dataType = NGTQ::DataTypeFloat16; break;
#endif
    case 'c': creation.dataType = NGTQ::DataTypeUint8; break;
    default:
      std::stringstream msg;
      msg << "Command::CreateParameters: Error: Invalid object type. " << objectType;
      TANN_THROW(msg);
    }

    switch (distanceType) {
    case '2': creation.distanceType = NGTQ::MetricType::MetricTypeL2; break;
    case '1': creation.distanceType = NGTQ::MetricType::MetricTypeL1; break;
    case 'a': creation.distanceType = NGTQ::MetricType::MetricTypeAngle; break;
    case 'C': creation.distanceType = NGTQ::MetricType::MetricTypeNormalizedCosine; break;
    case 'E': creation.distanceType = NGTQ::MetricType::MetricTypeL2; break;
    default:
      std::stringstream msg;
      msg << "Command::CreateParameters: Error: Invalid distance type. " << distanceType;
      TANN_THROW(msg);
    }
#ifdef NGTQ_QBG
    creation.genuineDimension = creation.dimension;
    creation.dimension = args.getl("P", creation.genuineDimension);
    creation.dimensionOfSubvector = args.getl("Q", 0);
    {
      char objectType = args.getChar("O", 'f');
      switch (objectType) {
      case 'f': creation.genuineDataType = ObjectFile::DataTypeFloat; break;
#ifdef TANN_ENABLE_HALF_FLOAT
      case 'h': creation.genuineDataType = ObjectFile::DataTypeFloat16; break;
#endif
      case 'c': creation.genuineDataType = ObjectFile::DataTypeUint8; break;
      default:
	std::stringstream msg;
	msg << "Command::CreateParameters: Error: Invalid genuine object type. " << objectType;
	TANN_THROW(msg);
      }
    }
#endif


  }

  void getHierarchicalClustringParameters() {
    hierarchicalClustering.maxSize = args.getl("r", 1000); 
    hierarchicalClustering.numOfObjects = args.getl("O", 0); 
    hierarchicalClustering.numOfClusters = args.getl("E", 2); 
    try {
      hierarchicalClustering.numOfTotalClusters = args.getl("C", 0);
    } catch (...) {
      hierarchicalClustering.numOfTotalClusters = 0;
    }
    hierarchicalClustering.numOfTotalBlobs = args.getl("b", 0);
    hierarchicalClustering.clusterID = args.getl("c", -1);
    silence = !args.getBool("v");
  
    char iMode = args.getChar("i", '-');
    hierarchicalClustering.initMode = tann::Clustering::InitializationModeKmeansPlusPlus;
    switch (iMode) {
    case 'l':
    case 'h': hierarchicalClustering.initMode = tann::Clustering::InitializationModeHead; break;
    case 'r': hierarchicalClustering.initMode = tann::Clustering::InitializationModeRandom; break;
    case 'R': hierarchicalClustering.initMode = tann::Clustering::InitializationModeRandomFixedSeed; break;
    case 'P': hierarchicalClustering.initMode = tann::Clustering::InitializationModeKmeansPlusPlusFixedSeed; break;
    default:
    case '-':
    case 'p': hierarchicalClustering.initMode = tann::Clustering::InitializationModeKmeansPlusPlus; break;
    }

    hierarchicalClustering.numOfRandomObjects = args.getl("r", 0);
    char rmode = args.getChar("R", '-');
    if (rmode == 'c') {
      hierarchicalClustering.extractCentroid = true;
    } else {
      hierarchicalClustering.extractCentroid = false;
    }

    hierarchicalClustering.numOfFirstObjects = 0;
    hierarchicalClustering.numOfFirstClusters = 0;
    hierarchicalClustering.numOfSecondObjects = 0;
    hierarchicalClustering.numOfSecondClusters = 0;
    hierarchicalClustering.numOfThirdClusters = 0;

    hierarchicalClustering.threeLayerClustering = true;

    std::string blob = args.getString("B", "");
    if (blob == "-") {
      hierarchicalClustering.threeLayerClustering = false;
    } else {
      hierarchicalClustering.threeLayerClustering = true;
    }
    if (hierarchicalClustering.threeLayerClustering) {
      std::vector<std::string> tokens;
      tann::Common::tokenize(blob, tokens, ",");
      if (tokens.size() > 0) {
	std::vector<std::string> ftokens;
	tann::Common::tokenize(tokens[0], ftokens, ":");
	if (ftokens.size() >= 1) {
	  hierarchicalClustering.numOfFirstObjects = tann::Common::strtof(ftokens[0]);
	}
	if (ftokens.size() >= 2) {
	  hierarchicalClustering.numOfFirstClusters = tann::Common::strtof(ftokens[1]);
	}
      }
      if (tokens.size() > 1) {
	std::vector<std::string> ftokens;
	tann::Common::tokenize(tokens[1], ftokens, ":");
	if (ftokens.size() >= 1) {
	  hierarchicalClustering.numOfSecondObjects = tann::Common::strtof(ftokens[0]);
	}
	if (ftokens.size() >= 2) {
	  hierarchicalClustering.numOfSecondClusters = tann::Common::strtof(ftokens[1]);
	}
      }
      if (tokens.size() > 2) {
	std::vector<std::string> ftokens;
	tann::Common::tokenize(tokens[2], ftokens, ":");
	if (ftokens.size() >= 1) {
	  if (ftokens[0] == "" || ftokens[0] == "-") {
	    hierarchicalClustering.numOfObjects = 0;
	  } else {
	    hierarchicalClustering.numOfObjects = tann::Common::strtof(ftokens[0]);
	  }
	}
	if (ftokens.size() >= 2) {
	  hierarchicalClustering.numOfThirdClusters = tann::Common::strtof(ftokens[1]);
	}
      }
    }
  }

  void getOptimizationParameters() {
    optimization.numberOfObjects = args.getl("o", 1000);
    optimization.numberOfClusters = args.getl("n", 0);
    optimization.numberOfSubvectors = args.getl("m", 0);

    optimization.randomizedObjectExtraction = true;

#ifdef NGT_CLUSTERING
    string cType;
    try {
      cType = args.getString("C", "k");
    } catch(...) {}

    optimization.clusteringType = tann::Clustering::ClusteringTypeKmeansWithNGT;
    if (cType == "k") {
      optimization.clusteringType = tann::Clustering::ClusteringTypeKmeansWithoutNGT;
    } else if (cType == "KS") {
      optimization.clusteringType = tann::Clustering::ClusteringTypeKmeansWithNGT;
    } else if (cType == "i") {
      optimization.clusteringType = tann::Clustering::ClusteringTypeKmeansWithIteration;
    } else {
      std::stringstream msg;
      msg << "invalid clustering type. " << cType;
      TANN_THROW(msg);
    }
#else
    char clusteringType;
    try {
      clusteringType = args.getChar("C", 'k');
    } catch(...) {}
#endif
  
#ifdef NGT_CLUSTERING
    char iMode = args.getChar("i", '-');
    optimization.initMode = tann::Clustering::InitializationModeKmeansPlusPlus;
    switch (iMode) {
    case 'h': optimization.initMode = tann::Clustering::InitializationModeHead; break;
    case 'r': optimization.initMode = tann::Clustering::InitializationModeRandom; break;
    case 'p': optimization.initMode = tann::Clustering::InitializationModeKmeansPlusPlus; break;
    case 'R': optimization.initMode = tann::Clustering::InitializationModeRandomFixedSeed; break;
    case 'P': optimization.initMode = tann::Clustering::InitializationModeKmeansPlusPlusFixedSeed; break;
    default:
    case '-':
    case 'b': optimization.initMode = tann::Clustering::InitializationModeBest; break;
    }
#else
    optimization.initMode = args.getChar("i", '-');
#endif
  
    optimization.convergenceLimitTimes = args.getl("c", 5);
    optimization.iteration = args.getl("t", 100);
    optimization.clusterIteration = args.getl("I", 100);

    optimization.clusterSizeConstraint = false;
    if (args.getChar("s", 'f') == 't') {
      optimization.clusterSizeConstraintCoefficient = 5.0;
      optimization.clusterSizeConstraint = true;
    } else if (args.getChar("s", 'f') == 'f') {
      optimization.clusterSizeConstraint = false;
    } else {
      optimization.clusterSizeConstraint = true;
      optimization.clusterSizeConstraintCoefficient = args.getf("s", 5.0);
    }
  
    optimization.nOfMatrices = args.getl("M", 2);
    optimization.seedStartObjectSizeRate = args.getf("S", 0.1);
    optimization.seedStep = args.getl("X", 2);
    optimization.reject = args.getf("R", 0.9);
    optimization.timelimit = args.getf("L", 24 * 1); 
    optimization.timelimit *= 60.0 * 60.0; 
    optimization.showClusterInfo = args.getBool("Z");
    silence = !args.getBool("v");

#ifdef NGTQG_NO_ROTATION
    char positionMode = args.getChar("P", 'n');
#else
    char positionMode = args.getChar("P", 'r');
#endif
    switch (positionMode) {
    case 'r':
      optimization.rotation = true;
      optimization.repositioning = false;
      break;
    case 'R':
      optimization.rotation = true;
      optimization.repositioning = true;
      break;
    case 'p':
      optimization.rotation = false;
      optimization.repositioning = true;
      break;
    case 'n':
    default:
      optimization.rotation = false;
      optimization.repositioning = false;
    }
    char globalType = args.getChar("G", '-');
    switch (globalType) {
    case 'z':
      optimization.globalType = QBG::Optimizer::GlobalTypeZero; break;
    case 'm':
      optimization.globalType = QBG::Optimizer::GlobalTypeMean; break;
    default:
    case 'n':
      optimization.globalType = QBG::Optimizer::GlobalTypeNone; break;
      break;
    }
  }
protected:
  tann::Args &args;
};


class SearchParameters : public tann::Command::SearchParameters {
public:
  SearchParameters(tann::Args &args): tann::Command::SearchParameters(args, "0.02") {
    stepOfResultExpansion = 2;
    std::string resultExpansion = args.getString("p", "3.0");
    std::vector<std::string> tokens;
    tann::Common::tokenize(resultExpansion, tokens, ":");
    if (tokens.size() >= 1) { beginOfResultExpansion = endOfResultExpansion = tann::Common::strtod(tokens[0]); }
    if (tokens.size() >= 2) { endOfResultExpansion = tann::Common::strtod(tokens[1]); }
    if (tokens.size() >= 3) { stepOfResultExpansion = tann::Common::strtod(tokens[2]); }
  }
  float	beginOfResultExpansion;
  float	endOfResultExpansion;
  float	stepOfResultExpansion;
};


void 
QBG::CLI::buildQG(tann::Args &args)
{
  const std::string usage = "Usage: qbg build-qg [-Q dimension-of-subvector] [-E max-number-of-edges] index";

  QbgCliBuildParameters buildParameters(args);
  buildParameters.getBuildParameters();

  string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "An index is not specified." << endl;
    cerr << usage << endl;
    return;
  }

  size_t phase = args.getl("p", 0);
  size_t maxNumOfEdges = args.getl("E", 128);
  
  const std::string qgPath = indexPath + "/qg";

  if (phase == 0 || phase == 1) {
    QBG::Optimizer optimizer(buildParameters);
    optimizer.globalType = QBG::Optimizer::GlobalTypeZero;

#ifdef NGTQG_NO_ROTATION
    if (optimizer.rotation || optimizer.repositioning) {
      std::cerr << "build-qg: Warning! Although rotation or repositioning is specified, turn off rotation and repositioning because of unavailable options." << std::endl;
      optimizer.rotation = false;
      optimizer.repositioning = false;
    }
#endif

    std::cerr << "optimizing..." << std::endl;
    optimizer.optimize(qgPath);
  }
  if (phase == 0 || phase == 2) {
    std::cerr << "building the inverted index..." << std::endl;
    bool silence = true;
    QBG::Index::buildNGTQ(qgPath, silence);
  }
  if (phase == 0 || phase == 3) {
    std::cerr << "building the quantized graph... " << std::endl;
    bool silence = true;
    NGTQG::Index::realign(indexPath, maxNumOfEdges, silence);
  }
}

void
searchQG(NGTQG::Index &index, SearchParameters &searchParameters, ostream &stream)
{

  std::ifstream		is(searchParameters.query);
  if (!is) {
    std::cerr << "Cannot open the specified file. " << searchParameters.query << std::endl;
    return;
  }

  if (searchParameters.outputMode[0] == 'e') { 
    stream << "# Beginning of Evaluation" << endl; 
  }

  string line;
  double totalTime	= 0;
  size_t queryCount	= 0;

  while(getline(is, line)) {
    if (searchParameters.querySize > 0 && queryCount >= searchParameters.querySize) {
      break;
    }
    vector<float>	query;
    stringstream	linestream(line);
    while (!linestream.eof()) {
      float value;
      linestream >> value;
      if (linestream.fail()) {
	TANN_THROW("NGTQG: invalid stream.");
      }
      query.push_back(value);
    }
    queryCount++;
    float beginOfParam = 0;
    float endOfParam = 0;
    float stepOfParam = FLT_MAX;
    bool resultExpansionEnabled = true;
    if (searchParameters.beginOfResultExpansion != searchParameters.endOfResultExpansion) {
      beginOfParam = searchParameters.beginOfResultExpansion;
      endOfParam   = searchParameters.endOfResultExpansion;
      stepOfParam  = searchParameters.stepOfResultExpansion;
      resultExpansionEnabled = true;
    } else if (searchParameters.beginOfEpsilon != searchParameters.endOfEpsilon) {
      beginOfParam = searchParameters.beginOfEpsilon;
      endOfParam   = searchParameters.endOfEpsilon;
      stepOfParam  = searchParameters.stepOfEpsilon;
      resultExpansionEnabled = false;
    }
    for (float param = beginOfParam; param <= endOfParam; param += stepOfParam) {
      NGTQG::SearchQuery	searchQuery(query);
      tann::VectorDistances	objects;
      searchQuery.setResults(&objects);
      searchQuery.setSize(searchParameters.size);
      searchQuery.setRadius(searchParameters.radius);
      float epsilon, resultExpansion;
      if (stepOfParam == FLT_MAX) {
	resultExpansion = searchParameters.beginOfResultExpansion;
	epsilon = searchParameters.beginOfEpsilon;
      } else if (resultExpansionEnabled) {
	resultExpansion = param;
	epsilon = searchParameters.beginOfEpsilon;
      } else {
	resultExpansion = searchParameters.beginOfResultExpansion;
	epsilon = param;
      }
      searchQuery.setResultExpansion(resultExpansion);
      searchQuery.setEpsilon(epsilon);
      searchQuery.setEdgeSize(searchParameters.edgeSize);
      tann::Timer timer;
      switch (searchParameters.indexType) {
      case 't': timer.start(); index.NGTQG::Index::search(searchQuery); timer.stop(); break;
      case 's': timer.start(); index.linearSearch(searchQuery); timer.stop(); break;
      }
      totalTime += timer.time;
      if (searchParameters.outputMode[0] == 'e') {
	stream << "# Query No.=" << queryCount << endl;
	stream << "# Query=" << line.substr(0, 20) + " ..." << endl;
	stream << "# Index Type=" << searchParameters.indexType << endl;
	stream << "# Size=" << searchParameters.size << endl;
	stream << "# Radius=" << searchParameters.radius << endl;
	stream << "# Epsilon*=" << epsilon << endl;
	stream << "# Result Expansion*=" << resultExpansion << endl;
	stream << "# Factor=" << param << endl;
	stream << "# Query Time (msec)=" << timer.time * 1000.0 << endl;
	stream << "# Distance Computation=" << searchQuery.distanceComputationCount << endl;
	stream << "# Visit Count=" << searchQuery.visitCount << endl;
      } else {
	stream << "Query No." << queryCount << endl;
	stream << "Rank\tID\tDistance" << endl;
      }
      for (size_t i = 0; i < objects.size(); i++) {
	stream << i + 1 << "\t" << objects[i].id << "\t";
	stream << objects[i].distance << endl;
      }
      if (searchParameters.outputMode[0] == 'e') {
	stream << "# End of Search" << endl;
      } else {
	stream << "Query Time= " << timer.time << " (sec), " << timer.time * 1000.0 << " (msec)" << endl;
      }
    } 
    if (searchParameters.outputMode[0] == 'e') {
      stream << "# End of Query" << endl;
    }
  }
  if (searchParameters.outputMode[0] == 'e') {
    stream << "# Average Query Time (msec)=" << totalTime * 1000.0 / (double)queryCount << endl;
    stream << "# Number of queries=" << queryCount << endl;
    stream << "# End of Evaluation" << endl;
  } else {
    stream << "Average Query Time= " << totalTime / (double)queryCount  << " (sec), " 
	   << totalTime * 1000.0 / (double)queryCount << " (msec), (" 
	   << totalTime << "/" << queryCount << ")" << endl;
  }
}

void
QBG::CLI::searchQG(tann::Args &args) {
  const string usage = "Usage: ngtqg search-qg [-i index-type(g|t|s)] [-n result-size] [-e epsilon] [-E edge-size] "
    "[-o output-mode] [-p result-expansion] index(input) query.tsv(input)";

  args.parse("v");

  string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "An index is not specified." << endl;
    cerr << usage << endl;
    return;
  }

  SearchParameters searchParameters(args);

  bool readOnly = true; 
  NGTQG::Index index(indexPath, 128, readOnly);

  if (debugLevel >= 1) {
    cerr << "indexType=" << searchParameters.indexType << endl;
    cerr << "size=" << searchParameters.size << endl;
    cerr << "edgeSize=" << searchParameters.edgeSize << endl;
    cerr << "epsilon=" << searchParameters.beginOfEpsilon << "<->" << searchParameters.endOfEpsilon << "," 
	 << searchParameters.stepOfEpsilon << endl;
  }

  try {
    ::searchQG(index, searchParameters, std::cout);
  } catch (tann::Exception &err) {
    cerr << "qbg: Error " << err.what() << endl;
    cerr << usage << endl;
  } catch (std::exception &err) {
    cerr << "qbg: Error " << err.what() << endl;
    cerr << usage << endl;
  } catch (...) {
    cerr << "qbg: Error" << endl;
    cerr << usage << endl;
  }

}


void 
QBG::CLI::createQG(tann::Args &args)
{
  const std::string usage = "Usage: qbg create-qg [-Q dimension-of-subvector] index";

  QbgCliBuildParameters buildParameters(args);
  buildParameters.getCreationParameters();
  
  string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "An index is not specified." << endl;
    cerr << usage << endl;
    return;
  }
  std::cerr << "creating..."  << std::endl;
  NGTQG::Index::create(indexPath, buildParameters);
  NGTQG::Index::append(indexPath, buildParameters);
}

void 
QBG::CLI::appendQG(tann::Args &args)
{
  const std::string usage = "Usage: qbg append-qbg ngt-index";
  string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "An index is not specified." << endl;
    cerr << usage << endl;
    return;
  }
  QBG::Index::appendFromObjectRepository(indexPath, indexPath + "/qg", false);
}


void 
QBG::CLI::create(tann::Args &args)
{
  const string usage = "Usage: qbg create "
    " -d dimension [-o object-type (f:float|c:unsigned char)] [-D distance-function] [-n data-size] "
    "[-p #-of-thread] [-R global-codebook-range] [-r local-codebook-range] "
    "[-C global-codebook-size-limit] [-c local-codebook-size-limit] [-N local-division-no] "
    "[-T single-local-centroid (t|f)] [-e epsilon] [-i index-type (t:Tree|g:Graph)] "
    "[-M global-centroid-creation-mode (d|s)] [-L local-centroid-creation-mode (d|k|s)] "
    "[-s local-sample-coefficient] "
    "index(OUT) data.tsv(IN) rotation(IN)";

  try {
    cerr << "qbg: Create" << endl;
    QbgCliBuildParameters buildParameters(args);
    buildParameters.getCreationParameters();

    std::vector<float> r;
    auto *rotation = &r;
    {
      try {
	std::string rotationPath = args.get("#3");
	cerr << "rotation is " << rotationPath << "." << endl;
	std::ifstream stream(rotationPath);
	if (!stream) {
	  std::cerr << "Cannot open the rotation. " << rotationPath << std::endl;
	  cerr << usage << endl;
	  return;
	}
	std::string line;
	while (getline(stream, line)) {
	  std::vector<std::string> tokens;
	  tann::Common::tokenize(line, tokens, " \t");
	  for (auto &token : tokens) {
	    r.push_back(tann::Common::strtof(token));
	  }
	}
      } catch (...) {
	rotation = 0;
      }
      std::cerr << "rotation matrix size=" << r.size() << std::endl;
    }
    std::string indexPath = args.get("#1");
    std::string objectPath;
    try {
      objectPath = args.get("#2");
    } catch(...) {}

    QBG::Index::create(indexPath, buildParameters, rotation, objectPath);
  } catch(tann::Exception &err) {
    std::cerr << err.what() << std::endl;
    cerr << usage << endl;
  }
}


void
QBG::CLI::load(tann::Args &args)
{
  const string usage = "Usage: qbg load ";

  int threadSize = args.getl("p", 50);

  std::string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    std::cerr << "Not specified the index." << std::endl;
    std::cerr << usage << std::endl;
    return;
  }

  std::cerr << "qbg: loading the specified blobs..." << std::endl;
  std::string blobs;
  try {
    blobs = args.get("#2");
  } catch(...) {}

  std::string localCodebooks;
  try {
    localCodebooks = args.get("#3");
  } catch (...) {}

  std::string rotationPath;
  try {
    rotationPath = args.get("#4");
  } catch (...) {}
  cerr << "rotation is " << rotationPath << "." << endl;

  QBG::Index::load(indexPath, blobs, localCodebooks, rotationPath, threadSize);
}

void
QBG::CLI::search(tann::Args &args)
{
  
  const string usage = "Usage: qbg search [-i g|t|s] [-n result-size] [-e epsilon] [-m mode(r|l|c|a)] "
    "[-E edge-size] [-o output-mode] [-b result expansion(begin:end:[x]step)] "
    "index(input) query.tsv(input)";
  string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "DB is not specified" << endl;
    cerr << usage << endl;
    return;
  }

  string query;
  try {
    query = args.get("#2");
  } catch (...) {
    cerr << "Query is not specified" << endl;
    cerr << usage << endl;
    return;
  }

  size_t size		= args.getl("n", 20);
  char outputMode	= args.getChar("o", '-');
  float epsilon	= 0.1;

  char searchMode	= args.getChar("M", 'g');
  if (args.getString("e", "none") == "-") {
    // linear search
    epsilon = FLT_MAX;
  } else {
    epsilon = args.getf("e", 0.1);
  }
  float blobEpsilon = args.getf("B", 0.0);
  size_t edgeSize = args.getl("E", 0);
  float cutback = args.getf("C", 0.0);
  size_t explorationSize = args.getf("N", 256);
  size_t nOfProbes = args.getl("P", 10);

  float beginOfResultExpansion, endOfResultExpansion, stepOfResultExpansion;
  bool mulStep = false;
  {
    beginOfResultExpansion = 0.0;
    endOfResultExpansion = 0.0;
    stepOfResultExpansion = 1;
    string str = args.getString("p", "0.0");
    vector<string> tokens;
    tann::Common::tokenize(str, tokens, ":");
    if (tokens.size() >= 1) { 
      beginOfResultExpansion = tann::Common::strtod(tokens[0]);
      endOfResultExpansion = beginOfResultExpansion;
    }
    if (tokens.size() >= 2) { endOfResultExpansion = tann::Common::strtod(tokens[1]); }
    if (tokens.size() >= 3) { 
      if (tokens[2][0] == 'x') {
	mulStep = true;
	stepOfResultExpansion = tann::Common::strtod(tokens[2].substr(1));
      } else {
	stepOfResultExpansion = tann::Common::strtod(tokens[2]);
      }
    }
  }
  if (debugLevel >= 1) {
    cerr << "size=" << size << endl;
    cerr << "result expansion=" << beginOfResultExpansion << "->" << endOfResultExpansion << "," << stepOfResultExpansion << endl;
  }

  QBG::Index index(indexPath, true);
  std::cerr << "qbg::The index is open." << std::endl;
  std::cerr << "  vmsize==" << tann::Common::getProcessVmSizeStr() << std::endl;
  std::cerr << "  peak vmsize==" << tann::Common::getProcessVmPeakStr() << std::endl;
  auto dimension = index.getQuantizer().globalCodebookIndex.getObjectSpace().getDimension();
  try {
    ifstream		is(query);
    if (!is) {
      cerr << "Cannot open the specified file. " << query << endl;
      return;
    }
    if (outputMode == 's') { cout << "# Beginning of Evaluation" << endl; }
    string line;
    double totalTime = 0;
    int queryCount = 0;
    while(getline(is, line)) {
      vector<float>	queryVector;
      stringstream	linestream(line);
      while (!linestream.eof()) {
	float value;
	linestream >> value;
	queryVector.push_back(value);
      }
      queryVector.resize(dimension);
      queryCount++;
      for (auto resultExpansion = beginOfResultExpansion; 
	   resultExpansion <= endOfResultExpansion; 
	   resultExpansion = mulStep ? resultExpansion * stepOfResultExpansion : 
			     resultExpansion + stepOfResultExpansion) {
	tann::VectorDistances objects;
	tann::Timer timer;
	timer.start();
	QBG::SearchContainer searchContainer;
	auto query = queryVector;
	searchContainer.setObjectVector(query);
	searchContainer.setResults(&objects);
	if (resultExpansion >= 1.0) {
	  searchContainer.setSize(static_cast<float>(size) * resultExpansion);
	  searchContainer.setExactResultSize(size);
	} else {
	  searchContainer.setSize(size);
	  searchContainer.setExactResultSize(0);
	}
	searchContainer.setEpsilon(epsilon);
	searchContainer.setBlobEpsilon(blobEpsilon);
	searchContainer.setEdgeSize(edgeSize);
	searchContainer.setCutback(cutback);
	searchContainer.setGraphExplorationSize(explorationSize);
	searchContainer.setNumOfProbes(nOfProbes);
	switch (searchMode) {
	case 'b':
	  index.searchBlobGraphNaively(searchContainer);
	  break;
	case 'n':
	  index.searchBlobNaively(searchContainer);
	  break;
	case 'g':
	default:
	  index.searchBlobGraph(searchContainer);
	  break;
	}
	if (objects.size() > size) {
	  objects.resize(size);
	}
	timer.stop();
	totalTime += timer.time;
	if (outputMode == 'e') {
	  cout << "# Query No.=" << queryCount << endl;
	  cout << "# Query=" << line.substr(0, 20) + " ..." << endl;
	  cout << "# Index Type=" << "----" << endl;
	  cout << "# Size=" << size << endl;
	  cout << "# Epsilon=" << epsilon << endl;
	  cout << "# Result expansion=" << resultExpansion << endl;
	  cout << "# Distance Computation=" << index.getQuantizer().distanceComputationCount << endl;
	  cout << "# Query Time (msec)=" << timer.time * 1000.0 << endl;
	} else {
	  cout << "Query No." << queryCount << endl;
	  cout << "Rank\tIN-ID\tID\tDistance" << endl;
	}

	for (size_t i = 0; i < objects.size(); i++) {
	  cout << i + 1 << "\t" << objects[i].id << "\t";
	  cout << objects[i].distance << endl;
	}

	if (outputMode == 'e') {
	  cout << "# End of Search" << endl;
	} else {
	  cout << "Query Time= " << timer.time << " (sec), " << timer.time * 1000.0 << " (msec)" << endl;
	}
      } 
      if (outputMode == 'e') {
	cout << "# End of Query" << endl;
      }
    } 
    if (outputMode == 'e') {
      cout << "# Average Query Time (msec)=" << totalTime * 1000.0 / (double)queryCount << endl;
      cout << "# Number of queries=" << queryCount << endl;
      cout << "# End of Evaluation" << endl;
    } else {
      cout << "Average Query Time= " << totalTime / (double)queryCount  << " (sec), " 
	   << totalTime * 1000.0 / (double)queryCount << " (msec), (" 
	   << totalTime << "/" << queryCount << ")" << endl;
    }
  } catch (tann::Exception &err) {
    cerr << "Error " << err.what() << endl;
    cerr << usage << endl;
  } catch (...) {
    cerr << "Error" << endl;
    cerr << usage << endl;
  }
  std::cerr << "qbg::The end of search" << std::endl;
  std::cerr << "  vmsize==" << tann::Common::getProcessVmSizeStr() << std::endl;
  std::cerr << "  peak vmsize==" << tann::Common::getProcessVmPeakStr() << std::endl;
  index.close();
}


void 
QBG::CLI::append(tann::Args &args)
{
  const string usage = "Usage: qbg append [-n data-size] index(output) data.tsv(input)";
  string index;
  try {
    index = args.get("#1");
  } catch (...) {
    cerr << "DB is not specified." << endl;
    cerr << usage << endl;
    return;
  }
  string data;
  try {
    data = args.get("#2");
  } catch (...) {
    cerr << "Data is not specified." << endl;
  }

  size_t dataSize = args.getl("n", 0);
  char mode = args.getChar("m", '-');

  if (mode == 'b') {
    QBG::Index::appendBinary(index, data, dataSize);
  } else {
    QBG::Index::append(index, data, dataSize);
  }
}


void 
QBG::CLI::buildIndex(tann::Args &args)
{
  const std::string usage = "Usage: qbg build-index  [-Q dimension-of-subvector] [-E max-number-of-edges] index";
  string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "An index is not specified." << endl;
    cerr << usage << endl;
    return;
  }
  char mode = args.getChar("m", '-');

  size_t beginID = args.getl("s", 1);
  size_t size = args.getl("n", 0);
  size_t endID = beginID + size - 1;

  std::vector<std::vector<float>> quantizerCodebook;
  std::vector<uint32_t> codebookIndex;
  std::vector<uint32_t> objectIndex;

  if (mode == 'q' || mode == '-') {
    {
      try {
	std::string codebookPath;
	try {
	  codebookPath = args.get("#2");
	} catch(...) {
	  codebookPath = indexPath + "/ws/kmeans-cluster_qcentroid.tsv";
	}
	std::ifstream stream(codebookPath);
	if (!stream) {
	  std::cerr << "Cannot open the codebook. " << codebookPath << std::endl;
	  cerr << usage << endl;
	  return;
	}
	std::string line;
	while (getline(stream, line)) {
	  std::vector<std::string> tokens;
	  tann::Common::tokenize(line, tokens, " \t");
	  std::vector<float> object;
	  for (auto &token : tokens) {
	    object.push_back(tann::Common::strtof(token));
	  }
	  if (!quantizerCodebook.empty() && quantizerCodebook[0].size() != object.size()) {
	    cerr << "The specified quantizer codebook is invalid. " << quantizerCodebook[0].size()
		 << ":" << object.size() << ":" << quantizerCodebook.size() << ":" << line << endl;
	    cerr << usage << endl;
	    return;
	  }
	  if (!object.empty()) {
	    quantizerCodebook.push_back(object);
	  }
	}
	std::cerr << "The size of quantizerCodebook is " << quantizerCodebook.size() << std::endl;
      } catch (...) {}
    }

    {
      try {
	std::string codebookIndexPath;
	try {
	  codebookIndexPath = args.get("#3");
	} catch (...) {
	  codebookIndexPath = indexPath + "/ws/kmeans-cluster_bqindex.tsv";
	}
	cerr << "codebook index is " << codebookIndexPath << "." << endl;
	std::ifstream stream(codebookIndexPath);
	if (!stream) {
	  std::cerr << "Cannot open the codebook index. " << codebookIndexPath << std::endl;
	  cerr << usage << endl;
	  return;
	}
	std::string line;
	while (getline(stream, line)) {
	  std::vector<std::string> tokens;
	  tann::Common::tokenize(line, tokens, " \t");
	  std::vector<float> object;
	  if (tokens.size() != 1) {
	    cerr << "The specified codebook index is invalid. " << line << std::endl;
	    cerr << usage << endl;
	    return;
	  }
	  codebookIndex.push_back(tann::Common::strtol(tokens[0]));
	}

      } catch (...) {}
    }

    {
      try {
	std::string objectIndexPath;
	try {
	  objectIndexPath = args.get("#4");
	} catch (...) {
	  objectIndexPath = indexPath + "/ws/kmeans-cluster_index.tsv";
	}
	std::ifstream stream(objectIndexPath);
	if (!stream) {
	  std::cerr << "Cannot open the codebook index. " << objectIndexPath << std::endl;
	  cerr << usage << endl;
	  return;
	}
	std::string line;
	while (getline(stream, line)) {
	  std::vector<std::string> tokens;
	  tann::Common::tokenize(line, tokens, " \t");
	  std::vector<float> object;
	  if (tokens.size() != 1) {
	    cerr << "The specified codebook index is invalid. " << line << std::endl;
	    cerr << usage << endl;
	    return;
	  }
	  objectIndex.push_back(tann::Common::strtol(tokens[0]));
	}

      } catch (...) {}
    }

    std::cerr << "quantizer codebook size=" << quantizerCodebook.size() << std::endl;
    std::cerr << "codebook index size=" << codebookIndex.size() << std::endl;
    std::cerr << "object index size=" << objectIndex.size() << std::endl;

    if (mode == 'q') {
      QBG::Index::buildNGTQ(indexPath, quantizerCodebook, codebookIndex, objectIndex, beginID, endID);
      return;
    }
  }

  if (mode == 'g') {
    QBG::Index::buildQBG(indexPath);
    return;
  }

  QBG::Index::build(indexPath, quantizerCodebook, codebookIndex, objectIndex, beginID, endID);

}

void 
QBG::CLI::build(tann::Args &args)
{
  const std::string usage = "Usage: qbg build [-Q dimension-of-subvector] [-E max-number-of-edges] index";

  QbgCliBuildParameters buildParameters(args);
  buildParameters.getBuildParameters();

  string indexPath;
  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "An index is not specified." << endl;
    cerr << usage << endl;
    return;
  }

  size_t phase = args.getl("p", 0);

  HierarchicalKmeans hierarchicalKmeans(buildParameters);

  if (phase == 0 || phase == 1) {
    hierarchicalKmeans.clustering(indexPath);
  }

  QBG::Optimizer optimizer(buildParameters);

  if (phase == 0 || phase == 2) {
    std::cerr << "optimizing..." << std::endl;
    optimizer.optimize(indexPath);
  }

  if (phase == 0 || phase == 3) {
    std::cerr << "building..." << std::endl;
    QBG::Index::build(indexPath, optimizer.silence);
  }
}



void
QBG::CLI::hierarchicalKmeans(tann::Args &args)
{
  const std::string usage = "qbg kmeans -O #-of-objects -B x1:y1,x2,y2,x3 index [prefix] [object-ID-file]";
  std::string indexPath;

  QbgCliBuildParameters buildParameters(args);

  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "Index is not specified" << endl;
    cerr << usage << endl;
    return;
  }

  std::string prefix;
  try {
    prefix = args.get("#2");
    std::cerr << "prefix=" << prefix << std::endl;
  } catch (...) {
    std::cerr << "Prefix is not specified." << std::endl;
  }

  std::string objectIDsFile;
  try {
    objectIDsFile = args.get("#3");
    std::cerr << "object IDs=" << objectIDsFile << std::endl;
  } catch (...) {
    cerr << "Object ID file is not specified" << endl;
  }

  HierarchicalKmeans hierarchicalKmeans(buildParameters);
  

  hierarchicalKmeans.clustering(indexPath, prefix, objectIDsFile);
}

void
QBG::CLI::assign(tann::Args &args)
{
  const std::string usage = "qbg assign";
  std::string indexPath;

  try {
    indexPath = args.get("#1");
  } catch (...) {
    cerr << "Any index is not specified" << endl;
    cerr << usage << endl;
    return;
  }

  std::string queryPath;

  try {
    queryPath = args.get("#2");
  } catch (...) {
    cerr << "Any query is not specified" << endl;
    cerr << usage << endl;
    return;
  }

  auto epsilon = args.getf("e", 0.1);
  auto numOfObjects = args.getl("n", 10);
  auto mode = args.getChar("m", '-');

  try {
    tann::Index		index(indexPath);
    tann::Property	property;
    index.getProperty(property);
    ifstream		is(queryPath);
    if (!is) {
      std::cerr << "Cannot open the query file. " << queryPath << std::endl;
      return;
    }
    string		line;
    while (getline(is, line)) {
      vector<float>	query;
      {
	stringstream	linestream(line);
	while (!linestream.eof()) {
	  float value;
	  linestream >> value;
	  query.push_back(value);
	}
	if (static_cast<size_t>(property.dimension) != query.size()) {
	  std::cerr << "Dimension is invallid. " << property.dimension << ":" << query.size() << std::endl;
	  return;
	}
      }
      tann::SearchQuery		sc(query);
      tann::VectorDistances	objects;
      sc.setResults(&objects);
      sc.setSize(numOfObjects);
      sc.setEpsilon(epsilon);

      index.search(sc);
      if (objects.size() == 0) {
	std::cerr << "The result is empty. Something wrong." << std::endl;
	return;
      }
      if (mode == '-') {
	std::cout << objects[0].id - 1 << std::endl;
      } else {
	std::cout << objects[0].id << std::endl;
      }
    }
  } catch (tann::Exception &err) {
    cerr << "Error " << err.what() << endl;
    return;
  } catch (...) {
    cerr << "Error" << endl;
    return;
  }
}

void
QBG::CLI::extract(tann::Args &args)
{ 

  const string usage = "Usage: qbg extract binary-file|index [output-file]";

  std::string	objectPath;
  try {
    objectPath = args.get("#1");
  } catch (...) {
    std::cerr << "Object file is not specified." << std::endl;
    std::cerr << usage << std::endl;
    return;
  }

  std::ostream *os;
  std::ofstream ofs;
  std::string outputFile;

  size_t n = args.getl("n", 100);
  size_t dim = args.getl("d", 0);
  std::string type = args.getString("t", "float32");
  char mode = args.getChar("m", 'r');

  try {
    QBG::Index index(objectPath, true);

    try {
      outputFile = args.get("#2");
      if (outputFile == "-") {
	os = &std::cout;
      } else {
	ofs.open(outputFile);
	os = &ofs;
      }
    } catch (...) {
      ofs.open(objectPath + "/ws/base.10000.u8.tsv");
      os = &ofs;
    }
    index.extract(*os, n, mode == 'r');
  } catch (tann::Exception &err) {
    try {
      outputFile = args.get("#2");
      ofs.open(outputFile);
      os = &ofs;
    } catch (...) {
      os = &std::cout;
    }
    try {
      StaticObjectFileLoader loader(objectPath, type);
      if (mode == 'r') {
	struct timeval randTime;
	gettimeofday(&randTime, 0);
	srand(randTime.tv_usec);
	for (size_t cnt = 0; cnt < n; cnt++) {
	  double random = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
	  size_t id = 0;
	  do {
	    id = floor(loader.noOfObjects * random);
	  } while (id >= loader.noOfObjects);
	  loader.seek(id);
	  auto object = loader.getObject();
	  if (cnt + 1 % 100000 == 0) {
	    std::cerr << "loaded " << static_cast<float>(cnt + 1) / 1000000.0 << "M objects." << std::endl;
	  }
	  if (dim != 0) {
	    object.resize(dim, 0.0);
	  }
	  for (auto v = object.begin(); v != object.end(); ++v) {
	    if (v + 1 != object.end()) {
	      *os << *v << "\t";
	    } else {
	      *os << *v << std::endl;;
	    }
	  }
	}
      } else {
	size_t cnt = 0;
	while (!loader.isEmpty()) {
	  auto object = loader.getObject();
	  cnt++;
	  if (cnt % 100000 == 0) {
	    std::cerr << "loaded " << static_cast<float>(cnt) / 1000000.0 << "M objects." << std::endl;
	  }
	  if (dim != 0) {
	    object.resize(dim, 0.0);
	  }
	  for (auto v = object.begin(); v != object.end(); ++v) {
	    if (v + 1 != object.end()) {
	      *os << *v << "\t";
	    } else {
	      *os << *v << std::endl;;
	    }
	  }
	  if (n > 0 && cnt >= n) {
	    break;
	  }
	}
      }
    } catch (...) {
      cerr << "Error" << endl;
      return;
    }
  }
  return;
}

void 
QBG::CLI::gt(tann::Args &args)
{ 
  string	path;

  try {
    path = args.get("#1");
  } catch (...) {
    std::cerr << "An object file is not specified." << std::endl;
    return;
  }

  std::ifstream stream;
  stream.open(path, std::ios::in | std::ios::binary);
  if (!stream) {
    std::cerr << "Error!" << std::endl;
    return;
  }
  uint32_t numQueries;
  uint32_t k;
  
  stream.read(reinterpret_cast<char*>(&numQueries), sizeof(numQueries));
  stream.read(reinterpret_cast<char*>(&k), sizeof(k));
  std::cerr << "# of queries=" << numQueries << std::endl;
  std::cerr << "k=" << k << std::endl;

  {
    std::ofstream idf;
    idf.open(path + "_gt.tsv");
    for (uint32_t qidx = 0; qidx < numQueries; qidx++) {
      for (uint32_t rank = 0; rank < k; rank++) {
	uint32_t id;
	stream.read(reinterpret_cast<char*>(&id), sizeof(id));
	idf << id;
	if (rank + 1 == k) {
	  idf << std::endl;
	} else {
	  idf << "\t";
	}
      }
    }
  }
  
  {
    std::ofstream df;
    df.open(path + "_gtdist.tsv");
    for (uint32_t qidx = 0; qidx < numQueries; qidx++) {
      for (uint32_t rank = 0; rank < k; rank++) {
	float distance;
	stream.read(reinterpret_cast<char*>(&distance), sizeof(distance));
	df << distance;
	if (rank + 1 == k) {
	  df << std::endl;
	} else {
	  df << "\t";
	}
      }
    }
  }

}

void 
QBG::CLI::gtRange(tann::Args &args)
{ 
  string	path;

  try {
    path = args.get("#1");
  } catch (...) {
    std::cerr << "An object file is not specified." << std::endl;
    return;
  }

  std::ifstream stream;
  stream.open(path, std::ios::in | std::ios::binary);
  if (!stream) {
    std::cerr << "Error!" << std::endl;
    return;
  }
  uint32_t numQueries;
  uint32_t totalRes;
  
  stream.read(reinterpret_cast<char*>(&numQueries), sizeof(numQueries));
  stream.read(reinterpret_cast<char*>(&totalRes), sizeof(totalRes));
  std::cerr << "# of queries=" << numQueries << std::endl;
  std::cerr << "totalRes=" << totalRes << std::endl;

  std::vector<uint32_t> numResultsPerQuery(numQueries);
  for (size_t qidx = 0; qidx < numQueries; qidx++) {
    uint32_t v;
    stream.read(reinterpret_cast<char*>(&v), sizeof(v));
    //std::cerr << qidx << ":" << v << std::endl;
    numResultsPerQuery[qidx] = v;
  }
  {
    std::ofstream idf;
    idf.open(path + "_gt.tsv");
    std::ofstream df;
    df.open(path + "_gtdist.tsv");
    size_t count = 0;
    for (size_t qidx = 0; qidx < numQueries; qidx++) {
      if (numResultsPerQuery[qidx] == 0) {
	idf << std::endl;
	df << std::endl;
	continue;
      }
      for (size_t rank = 0; rank < numResultsPerQuery[qidx]; rank++) {
	uint32_t v;
	stream.read(reinterpret_cast<char*>(&v), sizeof(v));
	idf << v;
	df << 0.0;
	count++;
	if (rank + 1 == numResultsPerQuery[qidx]) {
	  idf << std::endl;
	  df << std::endl;
	} else {
	  idf << "\t";
	  df << "\t";
	}
      }
    }
    if (count != totalRes) {
      std::cerr << "Fatal error. " << count << ":" << totalRes << std::endl;
    }
  }
}


void
QBG::CLI::optimize(tann::Args &args)
{

  string usage = "Usage: qbg optimize -n number-of-clusters -m number-of subspaces [-O t|f] [-s t|f] [-I cluster-iteration] [-t R-max-iteration] [-c convergence-limit-times] vector-file [output-file-prefix]\n"
    "       qbg optimize -e E -n number-of-clusters -m number-of index [subspaces] [vector-file] [local-centroid-file] [global-centroid-file]";

  QbgCliBuildParameters buildParameters(args);

  std::string indexPath;
  try {
    indexPath = args.get("#1");
  } catch(...) {
    cerr << "qbg: index is not specified." << endl;
    cerr << usage << endl;
    return;
  }

  string invector;
  try {
    invector = args.get("#2");
  } catch(...) {}

  string ofile;
  try {
    ofile = args.get("#3");
  } catch(...) {}

  string global;
  try {
    global = args.get("#4");
  } catch(...) {}

  QBG::Optimizer optimizer(buildParameters);


  if (invector.empty() || ofile.empty() || global.empty()) {
    optimizer.optimize(indexPath);
  } else {
    optimizer.optimize(invector, ofile, global);
  }

  return;

}

#endif 
