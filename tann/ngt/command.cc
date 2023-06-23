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

#include "tann/ngt/command.h"
#include "tann/ngt/graph_reconstructor.h"
#include "tann/ngt/optimizer.h"
#include "tann/ngt/graph_optimizer.h"


using namespace std;


  tann::Command::CreateParameters::CreateParameters(Args &args) {
    try {
      index = args.get("#1");
    } catch (...) {
      std::stringstream msg;
      msg << "Command::CreateParameter: Error: An index is not specified.";
      TANN_THROW(msg);
    }

    try {
      objectPath = args.get("#2");
    } catch (...) {}

    property.edgeSizeForCreation = args.getl("E", 10);
    property.edgeSizeForSearch = args.getl("S", 40);
    property.batchSizeForCreation = args.getl("b", 200);
    property.insertionRadiusCoefficient = args.getf("e", 0.1) + 1.0;
    property.truncationThreshold = args.getl("t", 0);
    property.dimension = args.getl("d", 0);
    property.threadPoolSize = args.getl("p", 24);
    property.pathAdjustmentInterval = args.getl("P", 0);
    property.dynamicEdgeSizeBase = args.getl("B", 30);
    property.buildTimeLimit = args.getf("T", 0.0);

    if (property.dimension <= 0) {
      std::stringstream msg;
      msg << "Command::CreateParameter: Error: Specify greater than 0 for # of your data dimension by a parameter -d.";
      TANN_THROW(msg);
    }

    property.objectAlignment = args.getChar("A", 'f') == 't' ? tann::Property::ObjectAlignmentTrue : tann::Property::ObjectAlignmentFalse;

    char graphType = args.getChar("g", 'a');
    switch(graphType) {
    case 'a': property.graphType = tann::Property::GraphType::GraphTypeANNG; break;
    case 'k': property.graphType = tann::Property::GraphType::GraphTypeKNNG; break;
    case 'b': property.graphType = tann::Property::GraphType::GraphTypeBKNNG; break;
    case 'd': property.graphType = tann::Property::GraphType::GraphTypeDNNG; break;
    case 'o': property.graphType = tann::Property::GraphType::GraphTypeONNG; break;
    case 'i': property.graphType = tann::Property::GraphType::GraphTypeIANNG; break;
    default:
      std::stringstream msg;
      msg << "Command::CreateParameter: Error: Invalid graph type. " << graphType;
      TANN_THROW(msg);
    }    

    if (property.graphType == tann::Property::GraphType::GraphTypeONNG) {
      property.outgoingEdge = 10;
      property.incomingEdge = 80;
      string str = args.getString("O", "-");
      if (str != "-") {
	vector<string> tokens;
	tann::Common::tokenize(str, tokens, "x");
	if (str != "-" && tokens.size() != 2) {
	  std::stringstream msg;
	  msg << "Command::CreateParameter: Error: outgoing/incoming edge size specification is invalid. (out)x(in) " << str;
	  TANN_THROW(msg);
	}
	property.outgoingEdge = tann::Common::strtod(tokens[0]);
	property.incomingEdge = tann::Common::strtod(tokens[1]);
      }
    }

    char seedType = args.getChar("s", '-');
    switch(seedType) {
    case 'f': property.seedType = tann::Property::SeedType::SeedTypeFixedNodes; break;
    case '1': property.seedType = tann::Property::SeedType::SeedTypeFirstNode; break;
    case 'r': property.seedType = tann::Property::SeedType::SeedTypeRandomNodes; break;
    case 'l': property.seedType = tann::Property::SeedType::SeedTypeAllLeafNodes; break;
    default:
    case '-': property.seedType = tann::Property::SeedType::SeedTypeNone; break;
    }

    char objectType = args.getChar("o", 'f');
    char distanceType = args.getChar("D", '2');

    numOfObjects = args.getl("n", 0);
    indexType = args.getChar("i", 't');

    switch (objectType) {
    case 'f': 
      property.objectType = tann::Index::Property::ObjectType::Float;
      break;
    case 'c':
      property.objectType = tann::Index::Property::ObjectType::Uint8;
      break;
#ifdef TANN_ENABLE_HALF_FLOAT
    case 'h':
      property.objectType = tann::Index::Property::ObjectType::Float16;
      break;
#endif
    default:
      std::stringstream msg;
      msg << "Command::CreateParameter: Error: Invalid object type. " << objectType;
      TANN_THROW(msg);
    }

    switch (distanceType) {
    case '1': 
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeL1;
      break;
    case '2':
    case 'e':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeL2;
      break;
    case 'a':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeAngle;
      break;
    case 'A':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeNormalizedAngle;
      break;
    case 'h':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeHamming;
      break;
    case 'j':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeJaccard;
      break;
    case 'J':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeSparseJaccard;
      break;
    case 'c':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeCosine;
      break;
    case 'C':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeNormalizedCosine;
      break;
    case 'E':
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeNormalizedL2;
      break;
    case 'p':  // added by Nyapicom
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypePoincare;
      break;
    case 'l':  // added by Nyapicom
      property.distanceType = tann::Index::Property::DistanceType::DistanceTypeLorentz;
      break;
    default:
      std::stringstream msg;
      msg << "Command::CreateParameter: Error: Invalid distance type. " << distanceType << endl;
      TANN_THROW(msg);
    }

#ifdef NGT_SHARED_MEMORY_ALLOCATOR
    size_t maxNoOfObjects = args.getl("N", 0);
    if (maxNoOfObjects > 0) {
      property.graphSharedMemorySize 
	= property.treeSharedMemorySize
	= property.objectSharedMemorySize = 512 * ceil(maxNoOfObjects / 50000000);
    }
#endif
  }

  void 
  tann::Command::create(Args &args)
  {
    const string usage = "Usage: ngt create "
      "-d dimension [-p #-of-thread] [-i index-type(t|g)] [-g graph-type(a|k|b|o|i)] "
      "[-t truncation-edge-limit] [-E edge-size] [-S edge-size-for-search] [-L edge-size-limit] "
      "[-e epsilon] "
#ifdef TANN_ENABLE_HALF_FLOAT
      "[-o object-type(f|h|c)] "
#else
      "[-o object-type(f|c)] "
#endif
      "[-D distance-function(1|2|a|A|h|j|c|C|E|p|l)] [-n #-of-inserted-objects] "  // added by Nyapicom
      "[-P path-adjustment-interval] [-B dynamic-edge-size-base] [-A object-alignment(t|f)] "
      "[-T build-time-limit] [-O outgoing x incoming] "
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
      "[-N maximum-#-of-inserted-objects] "
#endif
      "index(output) [data.tsv(input)]";

    try {
      CreateParameters createParameters(args);

      if (debugLevel >= 1) {
	cerr << "edgeSizeForCreation=" << createParameters.property.edgeSizeForCreation << endl;
	cerr << "edgeSizeForSearch=" << createParameters.property.edgeSizeForSearch << endl;
	cerr << "edgeSizeLimit=" << createParameters.property.edgeSizeLimitForCreation << endl;
	cerr << "batch size=" << createParameters.property.batchSizeForCreation << endl;
	cerr << "graphType=" << createParameters.property.graphType << endl;
	cerr << "epsilon=" << createParameters.property.insertionRadiusCoefficient - 1.0 << endl;
	cerr << "thread size=" << createParameters.property.threadPoolSize << endl;
	cerr << "dimension=" << createParameters.property.dimension << endl;
	cerr << "indexType=" << createParameters.indexType << endl;
      }

      switch (createParameters.indexType) {
      case 't':
	tann::Index::createGraphAndTree(createParameters.index, createParameters.property, createParameters.objectPath, createParameters.numOfObjects);
	break;
      case 'g':
	tann::Index::createGraph(createParameters.index, createParameters.property, createParameters.objectPath, createParameters.numOfObjects);
	break;
      }
    } catch(tann::Exception &err) {
      std::cerr << err.what() << std::endl;
      cerr << usage << endl;
    }
  }

  void 
  tann::Command::append(Args &args)
  {
    const string usage = "Usage: ngt append [-p #-of-thread] [-d dimension] [-n data-size] "
      "index(output) [data.tsv(input)]";
    string database;
    try {
      database = args.get("#1");
    } catch (...) {
      cerr << "ngt: Error: DB is not specified." << endl;
      cerr << usage << endl;
      return;
    }
    string data;
    try {
      data = args.get("#2");
    } catch (...) {
      cerr << "ngt: Warning: No specified object file. Just build an index for the existing objects." << endl;
    }

    int threadSize = args.getl("p", 50);
    size_t dimension = args.getl("d", 0);
    size_t dataSize = args.getl("n", 0);

    if (debugLevel >= 1) {
      cerr << "thread size=" << threadSize << endl;
      cerr << "dimension=" << dimension << endl;
    }


    try {
      tann::Index::append(database, data, threadSize, dataSize);
    } catch (tann::Exception &err) {
      cerr << "ngt: Error " << err.what() << endl;
      cerr << usage << endl;
    } catch (...) {
      cerr << "ngt: Error" << endl;
      cerr << usage << endl;
    }
  }


  void
  tann::Command::search(tann::Index &index, tann::Command::SearchParameters &searchParameters, istream &is, ostream &stream)
  {

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
      tann::Object *object = index.allocateObject(line, " \t");
      queryCount++;
      size_t step = searchParameters.step == 0 ? UINT_MAX : searchParameters.step;
      for (size_t n = 0; n <= step; n++) {
	tann::SearchContainer sc(*object);
	double epsilon;
	if (searchParameters.step != 0) {
	  epsilon = searchParameters.beginOfEpsilon + (searchParameters.endOfEpsilon - searchParameters.beginOfEpsilon) * n / step; 
	} else {
	  epsilon = searchParameters.beginOfEpsilon + searchParameters.stepOfEpsilon * n;
	  if (epsilon > searchParameters.endOfEpsilon) {
	    break;
	  }
	}
	tann::ObjectDistances objects;
	sc.setResults(&objects);
	sc.setSize(searchParameters.size);
	sc.setRadius(searchParameters.radius);
	if (searchParameters.accuracy > 0.0) {
	  sc.setExpectedAccuracy(searchParameters.accuracy);
	} else {
	  sc.setEpsilon(epsilon);
	}
 	sc.setEdgeSize(searchParameters.edgeSize);
	tann::Timer timer;
	try {
	  if (searchParameters.outputMode[0] == 'e') {
	    double time = 0.0;
	    uint64_t ntime = 0;
	    double minTime = DBL_MAX;
	    size_t trial = searchParameters.trial <= 0 ? 1 : searchParameters.trial;
	    for (size_t t = 0; t < trial; t++) {
	      switch (searchParameters.indexType) {
	      case 't': timer.start(); index.search(sc); timer.stop(); break;
	      case 'g': timer.start(); index.searchUsingOnlyGraph(sc); timer.stop(); break;
	      case 's': timer.start(); index.linearSearch(sc); timer.stop(); break;
	      }
	      if (minTime > timer.time) {
		minTime = timer.time;
	      }
	      time += timer.time;
	      ntime += timer.ntime;
	    }
	    time /= (double)trial;
	    ntime /= trial;
	    timer.time = minTime;
	    timer.ntime = ntime;
	  } else {
	    switch (searchParameters.indexType) {
	    case 't': timer.start(); index.search(sc); timer.stop(); break;
	    case 'g': timer.start(); index.searchUsingOnlyGraph(sc); timer.stop(); break;
	    case 's': timer.start(); index.linearSearch(sc); timer.stop(); break;
	    }
	  }
	} catch (tann::Exception &err) {
	  if (searchParameters.outputMode != "ei") {
	    // not ignore exceptions
	    throw err;
	  }
	}
	totalTime += timer.time;
	if (searchParameters.outputMode[0] == 'e') {
	  stream << "# Query No.=" << queryCount << endl;
	  stream << "# Query=" << line.substr(0, 20) + " ..." << endl;
	  stream << "# Index Type=" << searchParameters.indexType << endl;
	  stream << "# Size=" << searchParameters.size << endl;
	  stream << "# Radius=" << searchParameters.radius << endl;
	  stream << "# Epsilon=" << epsilon << endl;
	  stream << "# Query Time (msec)=" << timer.time * 1000.0 << endl;
	  stream << "# Distance Computation=" << sc.distanceComputationCount << endl;
	  stream << "# Visit Count=" << sc.visitCount << endl;
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
      } // for
      index.deleteObject(object);
      if (searchParameters.outputMode[0] == 'e') {
	stream << "# End of Query" << endl;
      }
    } // while
    if (searchParameters.outputMode[0] == 'e') {
      stream << "# Average Query Time (msec)=" << totalTime * 1000.0 / (double)queryCount << endl;
      stream << "# Number of queries=" << queryCount << endl;
      stream << "# End of Evaluation" << endl;

      if (searchParameters.outputMode == "e+") {
	// show graph information
	size_t esize = searchParameters.edgeSize;
	long double distance = 0.0;
	size_t numberOfNodes = 0;
	size_t numberOfEdges = 0;

	tann::GraphIndex	&graph = (tann::GraphIndex&)index.getIndex();
	for (size_t id = 1; id < graph.repository.size(); id++) {
	  tann::GraphNode *node = 0;
	  try {
	    node = graph.getNode(id);
	  } catch(tann::Exception &err) {
	    cerr << "Graph::search: Warning. Cannot get the node. ID=" << id << ":" << err.what() << " If the node was removed, no problem." << endl;
	    continue;
	  }
	  numberOfNodes++;
	  if (numberOfNodes % 1000000 == 0) {
	    cerr << "Processed " << numberOfNodes << endl;
	  }
	  for (size_t i = 0; i < node->size(); i++) {
	    if (esize != 0 && i >= esize) {
	      break;
	    }
	    numberOfEdges++;
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	    distance += (*node).at(i, graph.repository.allocator).distance;
#else
	    distance += (*node)[i].distance;
#endif
	  }
	}

	stream << "# # of nodes=" << numberOfNodes << endl;
	stream << "# # of edges=" << numberOfEdges << endl;
	stream << "# Average number of edges=" << (double)numberOfEdges / (double)numberOfNodes << endl;
	stream << "# Average distance of edges=" << setprecision(10) << distance / (double)numberOfEdges << endl;
      }
    } else {
      stream << "Average Query Time= " << totalTime / (double)queryCount  << " (sec), " 
	   << totalTime * 1000.0 / (double)queryCount << " (msec), (" 
	   << totalTime << "/" << queryCount << ")" << endl;
    }
  }


  void
  tann::Command::search(Args &args) {
    const string usage = "Usage: ngt search [-i index-type(g|t|s)] [-n result-size] [-e epsilon] [-E edge-size] "
      "[-m open-mode(r|w)] [-o output-mode] index(input) query.tsv(input)";

    string database;
    try {
      database = args.get("#1");
    } catch (...) {
      cerr << "ngt: Error: DB is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    SearchParameters searchParameters(args);

    if (debugLevel >= 1) {
      cerr << "indexType=" << searchParameters.indexType << endl;
      cerr << "size=" << searchParameters.size << endl;
      cerr << "edgeSize=" << searchParameters.edgeSize << endl;
      cerr << "epsilon=" << searchParameters.beginOfEpsilon << "<->" << searchParameters.endOfEpsilon << "," 
	   << searchParameters.stepOfEpsilon << endl;
    }

    try {
      tann::Index	index(database, searchParameters.openMode == 'r');
      search(index, searchParameters, cout);
    } catch (tann::Exception &err) {
      cerr << "ngt: Error " << err.what() << endl;
      cerr << usage << endl;
    } catch (...) {
      cerr << "ngt: Error" << endl;
      cerr << usage << endl;
    }

  }


  void
  tann::Command::remove(Args &args)
  {
    const string usage = "Usage: ngt remove [-d object-ID-type(f|d)] [-m f] index(input) object-ID(input)";
    string database;
    try {
      database = args.get("#1");
    } catch (...) {
      cerr << "ngt: Error: DB is not specified" << endl;
      cerr << usage << endl;
      return;
    }
    try {
      args.get("#2");
    } catch (...) {
      cerr << "ngt: Error: ID is not specified" << endl;
      cerr << usage << endl;
      return;
    }
    char dataType = args.getChar("d", 'f');
    char mode = args.getChar("m", '-');
    bool force = false;
    if (mode == 'f') {
      force = true;
    }
    if (debugLevel >= 1) {
      cerr << "dataType=" << dataType << endl;
    }

    try {
      vector<tann::ObjectID> objects;
      if (dataType == 'f') {
	string ids;
	try {
	  ids = args.get("#2");
	} catch (...) {
	  cerr << "ngt: Error: Data file is not specified" << endl;
	  cerr << usage << endl;
	  return;
	}
	ifstream is(ids);
	if (!is) {
	  cerr << "ngt: Error: Cannot open the specified file. " << ids << endl;
	  cerr << usage << endl;
	  return;
	}
	string line;
	int count = 0;
	while(getline(is, line)) {
	  count++;
	  vector<string> tokens;
	  tann::Common::tokenize(line, tokens, "\t ");
	  if (tokens.size() == 0 || tokens[0].size() == 0) {
	    continue;
	  }
	  char *e;
	  size_t id;
	  try {
	    id = strtol(tokens[0].c_str(), &e, 10);
	    objects.push_back(id);
	  } catch (...) {
	    cerr << "Illegal data. " << tokens[0] << endl;
	  }
	  if (*e != 0) {
	    cerr << "Illegal data. " << e << endl;
	  }
	  cerr << "removed ID=" << id << endl;	
	}
      } else {
	size_t id = args.getl("#2", 0);
	cerr << "removed ID=" << id << endl;
	objects.push_back(id);
      }
      tann::Index::remove(database, objects, force);
    } catch (tann::Exception &err) {
      cerr << "ngt: Error " << err.what() << endl;
      cerr << usage << endl;
    } catch (...) {
      cerr << "ngt: Error" << endl;
      cerr << usage << endl;
    }
  }

  void
  tann::Command::exportIndex(Args &args)
  {
    const string usage = "Usage: ngt export index(input) export-file(output)";
    string database;
    try {
      database = args.get("#1");
    } catch (...) {
      cerr << "ngt: Error: DB is not specified" << endl;
      cerr << usage << endl;
      return;
    }
    string exportFile;
    try {
      exportFile = args.get("#2");
    } catch (...) {
      cerr << "ngt: Error: ID is not specified" << endl;
      cerr << usage << endl;
      return;
    }
    try {
      tann::Index::exportIndex(database, exportFile);
    } catch (tann::Exception &err) {
      cerr << "ngt: Error " << err.what() << endl;
      cerr << usage << endl;
    } catch (...) {
      cerr << "ngt: Error" << endl;
      cerr << usage << endl;
    }
  }

  void
  tann::Command::importIndex(Args &args)
  {
    const string usage = "Usage: ngt import index(output) import-file(input)";
    string database;
    try {
      database = args.get("#1");
    } catch (...) {
      cerr << "ngt: Error: DB is not specified" << endl;
      cerr << usage << endl;
      return;
    }
    string importFile;
    try {
      importFile = args.get("#2");
    } catch (...) {
      cerr << "ngt: Error: ID is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    try {
      tann::Index::importIndex(database, importFile);
    } catch (tann::Exception &err) {
      cerr << "ngt: Error " << err.what() << endl;
      cerr << usage << endl;
    } catch (...) {
      cerr << "ngt: Error" << endl;
      cerr << usage << endl;
    }

  }

  void
  tann::Command::prune(Args &args)
  {
    const string usage = "Usage: ngt prune -e #-of-forcedly-pruned-edges -s #-of-selecively-pruned-edge index(in/out)";
    string indexName;
    try {
      indexName = args.get("#1");
    } catch (...) {
      cerr << "Index is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    // the number of forcedly pruned edges
    size_t forcedlyPrunedEdgeSize	= args.getl("e", 0);
    // the number of selectively pruned edges
    size_t selectivelyPrunedEdgeSize	= args.getl("s", 0);

    cerr << "forcedly pruned edge size=" << forcedlyPrunedEdgeSize << endl;
    cerr << "selectively pruned edge size=" << selectivelyPrunedEdgeSize << endl;

    if (selectivelyPrunedEdgeSize == 0 && forcedlyPrunedEdgeSize == 0) {
      cerr << "prune: Error! Either of selective edge size or remaining edge size should be specified." << endl;
      cerr << usage << endl;
      return;
    }

    if (forcedlyPrunedEdgeSize != 0 && selectivelyPrunedEdgeSize != 0 && selectivelyPrunedEdgeSize >= forcedlyPrunedEdgeSize) {
      cerr << "prune: Error! selective edge size is less than remaining edge size." << endl;
      cerr << usage << endl;
      return;
    }

    tann::Index	index(indexName);
    cerr << "loaded the input index." << endl;

    tann::GraphIndex	&graph = (tann::GraphIndex&)index.getIndex();

    for (size_t id = 1; id < graph.repository.size(); id++) {
      try {
	tann::GraphNode &node = *graph.getNode(id);
	if (id % 1000000 == 0) {
	  cerr << "Processed " << id << endl;
	}
	if (forcedlyPrunedEdgeSize > 0 && node.size() >= forcedlyPrunedEdgeSize) {
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
	  node.resize(forcedlyPrunedEdgeSize, graph.repository.allocator);
#else
	  node.resize(forcedlyPrunedEdgeSize);
#endif
	}
	if (selectivelyPrunedEdgeSize > 0 && node.size() >= selectivelyPrunedEdgeSize) {
#ifdef NGT_SHARED_MEMORY_ALLOCATOR
	  cerr << "not implemented" << endl;
	  abort();
#else
	  size_t rank = 0;
	  for (tann::GraphNode::iterator i = node.begin(); i != node.end(); ++rank) {
	    if (rank >= selectivelyPrunedEdgeSize) {
	      bool found = false;
	      for (size_t t1 = 0; t1 < node.size() && found == false; ++t1) {
		if (t1 >= selectivelyPrunedEdgeSize) {
		  break;
		}
		if (rank == t1) {
		  continue;
		}
		tann::GraphNode &node2 = *graph.getNode(node[t1].id);
		for (size_t t2 = 0; t2 < node2.size(); ++t2) {		
		  if (t2 >= selectivelyPrunedEdgeSize) {
		    break;
		  }
		  if (node2[t2].id == (*i).id) {
		    found = true;
		    break;
		  }
		} // for
	      } // for
	      if (found) {
		//remove
		i = node.erase(i);
		continue;
	      }
	    }
	    i++;
	  } // for
#endif
	}
	  
      } catch(tann::Exception &err) {
	cerr << "Graph::search: Warning. Cannot get the node. ID=" << id << ":" << err.what() << endl;
	continue;
      }
    }

    graph.saveIndex(indexName);

  }

  void
  tann::Command::reconstructGraph(Args &args)
  {
    const string usage = "Usage: ngt reconstruct-graph [-m mode] [-P path-adjustment-mode] -o #-of-outgoing-edges -i #-of-incoming(reversed)-edges [-q #-of-queries] [-n #-of-results] [-E minimum-#-of-edges] index(input) index(output)\n"
      "\t-m mode\n"
      "\t\ts: Edge adjustment.\n"
      "\t\tS: Edge adjustment and path adjustment. (default)\n"
      "\t\tc: Edge adjustment with the constraint.\n"
      "\t\tC: Edge adjustment with the constraint and path adjustment.\n"
      "\t\tP: Path adjustment.\n"
      "\t-P path-adjustment-mode\n"
      "\t\ta: Advanced method. High-speed. Not guarantee the paper's method. (default)\n"
      "\t\tothers: Slow and less memory usage, but guarantee the paper's method.\n";

    string inIndexPath;
    try {
      inIndexPath = args.get("#1");
    } catch (...) {
      cerr << "ngt::reconstructGraph: Input index is not specified." << endl;
      cerr << usage << endl;
      return;
    }
    string outIndexPath;
    try {
      outIndexPath = args.get("#2");
    } catch (...) {
      cerr << "ngt::reconstructGraph: Output index is not specified." << endl;
      cerr << usage << endl;
      return;
    }

    char mode = args.getChar("m", 'S');
    size_t nOfQueries = args.getl("q", 100);		// # of query objects
    size_t nOfResults = args.getl("n", 20);		// # of resultant objects
    double gtEpsilon = args.getf("e", 0.1);
    double margin = args.getf("M", 0.2);
    char smode = args.getChar("s", '-');

    // the number (rank) of original edges
    int numOfOutgoingEdges	= args.getl("o", -1);
    // the number (rank) of reverse edges
    int numOfIncomingEdges		= args.getl("i", -1);

    tann::GraphOptimizer graphOptimizer(false);

    if (mode == 'P') {
      numOfOutgoingEdges = 0;
      numOfIncomingEdges = 0;
      std::cerr << "ngt::reconstructGraph: Warning. \'-m P\' and not zero for # of in/out edges are specified at the same time." << std::endl;
    }
    graphOptimizer.shortcutReduction = (mode == 'S' || mode == 'C' || mode == 'P') ? true : false;
    graphOptimizer.searchParameterOptimization = (smode == '-' || smode == 's') ? true : false;
    graphOptimizer.prefetchParameterOptimization = (smode == '-' || smode == 'p') ? true : false;
    graphOptimizer.accuracyTableGeneration = (smode == '-' || smode == 'a') ? true : false;
    graphOptimizer.margin = margin;
    graphOptimizer.gtEpsilon = gtEpsilon;
    graphOptimizer.minNumOfEdges = args.getl("E", 0);
    
    graphOptimizer.set(numOfOutgoingEdges, numOfIncomingEdges, nOfQueries, nOfResults);
    graphOptimizer.execute(inIndexPath, outIndexPath);

    std::cout << "Successfully completed." << std::endl;
  }

  void
  tann::Command::optimizeSearchParameters(Args &args)
  {
    const string usage = "Usage: ngt optimize-search-parameters [-m optimization-target(s|p|a)] [-q #-of-queries] [-n #-of-results] index\n"
      "\t-m mode\n"
      "\t\ts: optimize search parameters (the number of explored edges).\n"
      "\t\tp: optimize prefetch parameters.\n"
      "\t\ta: generate an accuracy table to specify an expected accuracy instead of an epsilon for search.\n";
    
    string indexPath;
    try {
      indexPath = args.get("#1");
    } catch (...) {
      cerr << "Index is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    char mode = args.getChar("m", '-');

    size_t nOfQueries = args.getl("q", 100);		// # of query objects
    size_t nOfResults = args.getl("n", 20);		// # of resultant objects


    try {
      tann::GraphOptimizer graphOptimizer(false);

      graphOptimizer.searchParameterOptimization = (mode == '-' || mode == 's') ? true : false;
      graphOptimizer.prefetchParameterOptimization = (mode == '-' || mode == 'p') ? true : false;
      graphOptimizer.accuracyTableGeneration = (mode == '-' || mode == 'a') ? true : false;
      graphOptimizer.numOfQueries = nOfQueries;
      graphOptimizer.numOfResults = nOfResults;

      graphOptimizer.set(0, 0, nOfQueries, nOfResults);
      graphOptimizer.optimizeSearchParameters(indexPath);

      std::cout << "Successfully completed." << std::endl;
    } catch (tann::Exception &err) {
      cerr << "ngt: Error " << err.what() << endl;
      cerr << usage << endl;
    }

  }

  void
  tann::Command::refineANNG(Args &args)
  {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    std::cerr << "refineANNG. Not implemented." << std::endl;
    abort();
#else
    const string usage = "Usage: ngt refine-anng [-e epsilon] [-a expected-accuracy] anng-index refined-anng-index";

    string inIndexPath;
    try {
      inIndexPath = args.get("#1");
    } catch (...) {
      cerr << "Input index is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    string outIndexPath;
    try {
      outIndexPath = args.get("#2");
    } catch (...) {
      cerr << "Output index is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    tann::Index	index(inIndexPath);

    float  epsilon		= args.getf("e", 0.1);
    float  expectedAccuracy	= args.getf("a", 0.0);
    int    noOfEdges		= args.getl("k", 0);	// to reconstruct kNNG
    int    exploreEdgeSize	= args.getf("E", INT_MIN);
    size_t batchSize		= args.getl("b", 10000);

    try {
      GraphReconstructor::refineANNG(index, epsilon, expectedAccuracy, noOfEdges, exploreEdgeSize, batchSize);
    } catch (tann::Exception &err) {
      std::cerr << "Error!! Cannot refine the index. " << err.what() << std::endl;
      return;
    }
    index.saveIndex(outIndexPath);
#endif
  }

  void
  tann::Command::repair(Args &args)
  {
    const string usage = "Usage: ngt [-m c|r|R] repair index \n"
      "\t-m mode\n"
      "\t\tc: Check. (default)\n"
      "\t\tr: Repair and save it as [index].repair.\n"
      "\t\tR: Repair and overwrite into the specified index.\n";

    string indexPath;
    try {
      indexPath = args.get("#1");
    } catch (...) {
      cerr << "Index is not specified" << endl;
      cerr << usage << endl;
      return;
    }
    
    char mode = args.getChar("m", 'c');

    bool repair = false;
    if (mode == 'r' || mode == 'R') {
      repair = true;
    }
    string path = indexPath;
    if (mode == 'r') {
      path = indexPath + ".repair";
      const string com = "cp -r " + indexPath + " " + path;
      int stat = system(com.c_str());
      if (stat != 0) {
	std::cerr << "ngt::repair: Cannot create the specified index. " << path << std::endl;
	cerr << usage << endl;
	return;
      }
    }

    tann::Index	index(path);

    tann::ObjectRepository &objectRepository = index.getObjectSpace().getRepository();
    tann::GraphIndex &graphIndex = static_cast<GraphIndex&>(index.getIndex());
    tann::GraphAndTreeIndex &graphAndTreeIndex = static_cast<GraphAndTreeIndex&>(index.getIndex());
    size_t objSize = objectRepository.size();
    std::cerr << "aggregate removed objects from the repository." << std::endl;
    std::set<ObjectID> removedIDs;
    for (ObjectID id = 1; id < objSize; id++) {
      if (objectRepository.isEmpty(id)) {
	removedIDs.insert(id);
      }
    }

    std::cerr << "aggregate objects from the tree." << std::endl;
    std::set<ObjectID> ids;
    graphAndTreeIndex.DVPTree::getAllObjectIDs(ids);
    size_t idsSize = ids.size() == 0 ? 0 : (*ids.rbegin()) + 1;
    if (objSize < idsSize) {
      std::cerr << "The sizes of the repository and tree are inconsistent. " << objSize << ":" << idsSize << std::endl;
    }
    size_t invalidTreeObjectCount = 0;
    size_t uninsertedTreeObjectCount = 0;
    std::cerr << "remove invalid objects from the tree." << std::endl;
    size_t size = objSize > idsSize ? objSize : idsSize;
    for (size_t id = 1; id < size; id++) {    
      if (ids.find(id) != ids.end()) {
	if (removedIDs.find(id) != removedIDs.end() || id >= objSize) {
	  if (repair) {
	    graphAndTreeIndex.DVPTree::removeNaively(id);
	    std::cerr << "Found the removed object in the tree. Removed it from the tree. " << id << std::endl;
	  } else {
	    std::cerr << "Found the removed object in the tree. " << id << std::endl;
	  }
	  invalidTreeObjectCount++;
	}
      } else {
	if (removedIDs.find(id) == removedIDs.end() && id < objSize) {
          std::cerr << "Not found an object in the tree. However, it might be a duplicated object. " << id << std::endl;
	  uninsertedTreeObjectCount++;
	  if (repair) {
	    try {
	      graphIndex.repository.remove(id);
	    } catch(...) {}
	  }
	}
      }
    }

    if (objSize != graphIndex.repository.size()) {
      std::cerr << "The sizes of the repository and graph are inconsistent. " << objSize << ":" << graphIndex.repository.size() << std::endl;
    }
    size_t invalidGraphObjectCount = 0;
    size_t uninsertedGraphObjectCount = 0;
    size = objSize > graphIndex.repository.size() ? objSize : graphIndex.repository.size();
    std::cerr << "remove invalid objects from the graph." << std::endl;
    for (size_t id = 1; id < size; id++) {
      try {
	graphIndex.getNode(id);
	if (removedIDs.find(id) != removedIDs.end() || id >= objSize) {
	  if (repair) {
	    graphAndTreeIndex.DVPTree::removeNaively(id);
	    try {
	      graphIndex.repository.remove(id);
	    } catch(...) {}
	    std::cerr << "Found the removed object in the graph. Removed it from the graph. " << id << std::endl;
	  } else {
	    std::cerr << "Found the removed object in the graph. " << id << std::endl;
	  }
	  invalidGraphObjectCount++;
	}
      } catch(tann::Exception &err) {
        if (removedIDs.find(id) == removedIDs.end() && id < objSize) {
          std::cerr << "Not found an object in the graph. It should be inserted into the graph. " << err.what() << " ID=" << id << std::endl;
	  uninsertedGraphObjectCount++;
	  if (repair) {
	    try {
	      graphAndTreeIndex.DVPTree::removeNaively(id);
	    } catch(...) {}
	  }
	}
      } catch(...) {
	std::cerr << "Unexpected error!" << std::endl;
      }
    }

    size_t invalidEdgeCount = 0;
//#pragma omp parallel for
    for (size_t id = 1; id < graphIndex.repository.size(); id++) {
      try {
        tann::GraphNode &node = *graphIndex.getNode(id);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	for (auto n = node.begin(graphIndex.repository.allocator); n != node.end(graphIndex.repository.allocator);) {
#else
	for (auto n = node.begin(); n != node.end();) {
#endif
	  if (removedIDs.find((*n).id) != removedIDs.end() || (*n).id >= objSize) {

	    std::cerr << "Not found the destination object of the edge. " << id << ":" << (*n).id << std::endl;
	    invalidEdgeCount++;
            if (repair) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
	      n = node.erase(n, graphIndex.repository.allocator);
#else
	      n = node.erase(n);
#endif
	      continue;
	    }
	  }
	  ++n;
	}
      } catch(...) {}
    }

    if (repair) {
      if (objSize < graphIndex.repository.size()) {
	graphIndex.repository.resize(objSize);
      }
    }

    std::cerr << "The number of invalid tree objects=" << invalidTreeObjectCount << std::endl;
    std::cerr << "The number of invalid graph objects=" << invalidGraphObjectCount << std::endl;
    std::cerr << "The number of uninserted tree objects (Can be ignored)=" << uninsertedTreeObjectCount << std::endl;
    std::cerr << "The number of uninserted graph objects=" << uninsertedGraphObjectCount << std::endl;
    std::cerr << "The number of invalid edges=" << invalidEdgeCount << std::endl;

    if (repair) {
      try {
	if (uninsertedGraphObjectCount > 0) {
	  std::cerr << "Building index." << std::endl;
	  index.createIndex(16);
	}
	std::cerr << "Saving index." << std::endl;
	index.saveIndex(path);
      } catch (tann::Exception &err) {
	cerr << "ngt: Error " << err.what() << endl;
	cerr << usage << endl;
	return;
      }
    }
  }


  void
  tann::Command::optimizeNumberOfEdgesForANNG(Args &args)
  {
    const string usage = "Usage: ngt optimize-#-of-edges [-q #-of-queries] [-k #-of-retrieved-objects] "
      "[-p #-of-threads] [-a target-accuracy] [-o target-#-of-objects] [-s #-of-sampe-objects] "
      "[-e maximum-#-of-edges] anng-index";

    string indexPath;
    try {
      indexPath = args.get("#1");
    } catch (...) {
      cerr << "Index is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    GraphOptimizer::ANNGEdgeOptimizationParameter parameter;

    parameter.noOfQueries	= args.getl("q", 200);
    parameter.noOfResults	= args.getl("k", 50);
    parameter.noOfThreads	= args.getl("p", 16);
    parameter.targetAccuracy	= args.getf("a", 0.9);
    parameter.targetNoOfObjects	= args.getl("o", 0);	// zero will replaced # of the repository size.
    parameter.noOfSampleObjects	= args.getl("s", 100000);
    parameter.maxNoOfEdges	= args.getl("e", 100);

    tann::GraphOptimizer graphOptimizer(false); // false=log
    auto optimizedEdge = graphOptimizer.optimizeNumberOfEdgesForANNG(indexPath, parameter);
    std::cout << "The optimized # of edges=" << optimizedEdge.first << "(" << optimizedEdge.second << ")" << std::endl;
    std::cout << "Successfully completed." << std::endl;
  }



  void
  tann::Command::info(Args &args)
  {
    const string usage = "Usage: ngt info [-E #-of-edges] [-m h|e] index";

    std::cout << "NGT version: " << tann::Index::getVersion() << std::endl;
    std::cout << "CPU SIMD types: ";
    CpuInfo::showSimdTypes();

    string database;
    try {
      database = args.get("#1");
    } catch (...) {
      cerr << "ngt: Error: DB is not specified" << endl;
      cerr << usage << endl;
      return;
    }

    size_t edgeSize = args.getl("E", UINT_MAX);
    char mode = args.getChar("m", '-');

    try {
      tann::Index	index(database);
      tann::GraphIndex::showStatisticsOfGraph(static_cast<tann::GraphIndex&>(index.getIndex()), mode, edgeSize);
      if (mode == 'v') {
	vector<uint8_t> status;
	index.verify(status);
      }
    } catch (tann::Exception &err) {
      cerr << "ngt: Error " << err.what() << endl;
      cerr << usage << endl;
    } catch (...) {
      cerr << "ngt: Error" << endl;
      cerr << usage << endl;
    }
  }


  void tann::Command::exportGraph(Args &args) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    std::cerr << "ngt: Error: exportGraph is not implemented." << std::endl;
    abort();
#else
    std::string usage = "ngt export-graph [-k #-of-edges] index";
    string indexPath;
    try {
      indexPath = args.get("#1");
    } catch (...) {
      cerr << "ngt::exportGraph: Index is not specified." << endl;
      cerr << usage << endl;
      return;
    }

    int k = args.getl("k", 0);

    tann::Index		index(indexPath);
    tann::GraphIndex	&graph = static_cast<tann::GraphIndex&>(index.getIndex());

    size_t size = index.getObjectRepositorySize();

    for (size_t id = 1; id < size; ++id) {
      tann::GraphNode *node = 0;
      try {
	node = graph.getNode(id);
      } catch(...) {
	continue;
      }
      std::cout << id << "\t";
      for (auto ei = (*node).begin(); ei != (*node).end(); ++ei) {
	if (k != 0 && k <= distance((*node).begin(), ei)) {
	  break;
	}
	std::cout << (*ei).id << "\t" << (*ei).distance;
	if (ei + 1 != (*node).end()) {
	  std::cout << "\t";
	}
      }
      std::cout << std::endl;
    }
#endif
  }

  void tann::Command::exportObjects(Args &args) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
    std::cerr << "ngt: Error: exportObjects is not implemented." << std::endl;
    abort();
#else
    std::string usage = "ngt export-objects index";
    string indexPath;
    try {
      indexPath = args.get("#1");
    } catch (...) {
      cerr << "ngt::exportGraph: Index is not specified." << endl;
      cerr << usage << endl;
      return;
    }

    tann::Index		index(indexPath);
    auto &objectSpace = index.getObjectSpace();
    size_t size = objectSpace.getRepository().size();

    for (size_t id = 1; id < size; ++id) {
      std::vector<float> object;
      objectSpace.getObject(id, object);
      for (auto v = object.begin(); v != object.end(); ++v) {
	std::cout << *v;
	if (v + 1 != object.end()) {
	  std::cout << "\t";
	}
      }
      std::cout << std::endl;
    }
#endif
  }

