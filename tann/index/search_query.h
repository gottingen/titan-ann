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
#ifndef TANN_INDEX_SEARCH_QUERY_H_
#define TANN_INDEX_SEARCH_QUERY_H_

#include <queue>
#include "tann/index/container.h"
#include "tann/index/query_container.h"
#include "tann/vector/vector_distance.h"
#include "tann/vector/vector_view.h"

namespace tann {

    typedef std::priority_queue<VectorDistance, std::vector<VectorDistance>, std::less<VectorDistance> > ResultPriorityQueue;

    class SearchContainer : public tann::Container {
    public:
        SearchContainer(Object &f, ObjectID i) : Container(f, i) { initialize(); }

        SearchContainer(Object &f) : Container(f, 0) { initialize(); }

        SearchContainer(SearchContainer &sc) : Container(sc) { *this = sc; }

        SearchContainer(SearchContainer &sc, Object &f) : Container(f, sc.id) { *this = sc; }

        SearchContainer() : Container(*reinterpret_cast<Object *>(0), 0) { initialize(); }

        SearchContainer &operator=(SearchContainer &sc) {
            size = sc.size;
            radius = sc.radius;
            explorationCoefficient = sc.explorationCoefficient;
            result = sc.result;
            distanceComputationCount = sc.distanceComputationCount;
            edgeSize = sc.edgeSize;
            workingResult = sc.workingResult;
            useAllNodesInLeaf = sc.useAllNodesInLeaf;
            expectedAccuracy = sc.expectedAccuracy;
            visitCount = sc.visitCount;
            return *this;
        }

        virtual ~SearchContainer() {}

        virtual void initialize() {
            size = 10;
            radius = FLT_MAX;
            explorationCoefficient = 1.1;
            result = 0;
            edgeSize = -1;    // dynamically prune the edges during search. -1 means following the index property. 0 means using all edges.
            useAllNodesInLeaf = false;
            expectedAccuracy = -1.0;
        }

        void setSize(size_t s) { size = s; };

        void setResults(VectorDistances *r) { result = r; }

        void setRadius(distance_type r) { radius = r; }

        void setEpsilon(float e) { explorationCoefficient = e + 1.0; }

        void setEdgeSize(int e) { edgeSize = e; }

        void setExpectedAccuracy(float a) { expectedAccuracy = a; }

        inline bool resultIsAvailable() { return result != 0; }

        VectorDistances &getResult() {
            if (result == 0) {
                TANN_THROW("Inner error: results is not set");
            }
            return *result;
        }

        ResultPriorityQueue &getWorkingResult() { return workingResult; }


        size_t size;
        distance_type radius;
        float explorationCoefficient;
        int edgeSize;
        size_t distanceComputationCount;
        ResultPriorityQueue workingResult;
        bool useAllNodesInLeaf;
        size_t visitCount;
        float expectedAccuracy;

    private:
        VectorDistances *result;
    };


    class SearchQuery : public tann::QueryContainer, public tann::SearchContainer {
    public:
        template<typename QTYPE>
        SearchQuery(const std::vector<QTYPE> &q):tann::QueryContainer(q) {}
    };
}  // namespace tann
#endif //TANN_SEARCH_QUERY_H
