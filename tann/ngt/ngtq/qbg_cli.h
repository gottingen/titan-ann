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

#pragma once

#include "tann/ngt/ngtq/quantized_blob_graph.h"
#include "tann/ngt/command.h"

namespace QBG {
  
  class CLI {
  public:

    int debugLevel;

#if !defined(NGTQ_QBG) || defined(NGTQ_SHARED_INVERTED_INDEX)
    void create(tann::ngt::Args &args) {};
    void load(tann::ngt::Args &args) {};
    void append(tann::ngt::Args &args) {};
    void buildIndex(tann::ngt::Args &args) {};
    void hierarchicalKmeans(tann::ngt::Args &args) {};
    void search(tann::ngt::Args &args) {};
    void assign(tann::ngt::Args &args) {};
    void extract(tann::ngt::Args &args) {};
    void gt(tann::ngt::Args &args) {};
    void gtRange(tann::ngt::Args &args) {};
    void optimize(tann::ngt::Args &args) {};
    void build(tann::ngt::Args &args) {};
    void createQG(tann::ngt::Args &args) {};
    void buildQG(tann::ngt::Args &args) {};
    void appendQG(tann::ngt::Args &args) {};
    void searchQG(tann::ngt::Args &args) {};
    void info(tann::ngt::Args &args) {};
#else
    void create(tann::ngt::Args &args);
    void load(tann::ngt::Args &args);
    void append(tann::ngt::Args &args);
    void buildIndex(tann::ngt::Args &args);
    void hierarchicalKmeans(tann::ngt::Args &args);
    void search(tann::ngt::Args &args);
    void assign(tann::ngt::Args &args);
    void extract(tann::ngt::Args &args);
    void gt(tann::ngt::Args &args);
    void gtRange(tann::ngt::Args &args);
    void optimize(tann::ngt::Args &args);
    void build(tann::ngt::Args &args);
    void createQG(tann::ngt::Args &args);
    void buildQG(tann::ngt::Args &args);
    void appendQG(tann::ngt::Args &args);
    void searchQG(tann::ngt::Args &args);
    void info(tann::ngt::Args &args);
#endif
    
    void setDebugLevel(int level) { debugLevel = level; }
    int getDebugLevel() { return debugLevel; }

    void help() {
      cerr << "Usage : qbg command database [data]" << endl;
      cerr << "           command : create build quantize search" << endl;
    }

    void execute(tann::ngt::Args args) {
      string command;
      try {
	command = args.get("#0");
      } catch(...) {
	help();
	return;
      }

      debugLevel = args.getl("X", 0);

      try {
	if (debugLevel >= 1) {
	  cerr << "ngt::command=" << command << endl;
	}
	if (command == "search") {
	  search(args);
	} else if (command == "create") {
	  create(args);
	} else if (command == "load") {
	  load(args);
	} else if (command == "append") {
	  append(args);
	} else if (command == "build-index") {
	  buildIndex(args);
	} else if (command == "kmeans") {
	  hierarchicalKmeans(args);
	} else if (command == "assign") {
	  assign(args);
	} else if (command == "extract") {
	  extract(args);
	} else if (command == "gt") {
	  gt(args);
	} else if (command == "gt-range") {
	  gtRange(args);
	} else if (command == "optimize") {
	  optimize(args);
	} else if (command == "build") {
	  build(args);
	} else if (command == "create-qg") {
	  createQG(args);
	} else if (command == "build-qg") {
	  buildQG(args);
	} else if (command == "append-qg") {
	  appendQG(args);
	} else if (command == "search-qg") {
	  searchQG(args);
	} else {
	  cerr << "Illegal command. " << command << endl;
	}
      } catch(tann::ngt::Exception &err) {
	cerr << "qbg: Fatal error: " << err.what() << endl;
      }
    }

  };

}; // NGTQBG
