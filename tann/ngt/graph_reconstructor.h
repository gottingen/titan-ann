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

#include <unordered_map>
#include <unordered_set>
#include <list>
#include "tann/common/std_ostream_redirector.h"
#include "tann/common/timer.h"

#ifdef _OPENMP

#include <omp.h>

#else
#warning "*** OMP is *NOT* available! ***"
#endif

namespace tann {

    class GraphReconstructor {
    public:
        static void extractGraph(std::vector<tann::VectorDistances> &graph, tann::GraphIndex &graphIndex) {
            graph.reserve(graphIndex.repository.size());
            for (size_t id = 1; id < graphIndex.repository.size(); id++) {
                if (id % 1000000 == 0) {
                    std::cerr << "GraphReconstructor::extractGraph: Processed " << id << " objects." << std::endl;
                }
                try {
                    tann::GraphNode &node = *graphIndex.getNode(id);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    tann::VectorDistances nd;
                    nd.reserve(node.size());
                    for (auto n = node.begin(graphIndex.repository.allocator); n != node.end(graphIndex.repository.allocator); ++n) {
                      nd.push_back(VectorDistance((*n).id, (*n).distance));
                        }
                    graph.push_back(nd);
#else
                    graph.push_back(node);
#endif
                    if (graph.back().size() != graph.back().capacity()) {
                        std::cerr
                                << "GraphReconstructor::extractGraph: Warning! The graph size must be the same as the capacity. "
                                << id << std::endl;
                    }
                } catch (tann::Exception &err) {
                    graph.push_back(tann::VectorDistances());
                    continue;
                }
            }

        }


        static void
        adjustPaths(tann::Index &outIndex) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
            std::cerr << "construct index is not implemented." << std::endl;
            exit(1);
#else
            tann::GraphIndex &outGraph = dynamic_cast<tann::GraphIndex &>(outIndex.getIndex());
            size_t rStartRank = 0;
            std::list<std::pair<size_t, tann::GraphNode> > tmpGraph;
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                tann::GraphNode &node = *outGraph.getNode(id);
                tmpGraph.push_back(std::pair<size_t, tann::GraphNode>(id, node));
                if (node.size() > rStartRank) {
                    node.resize(rStartRank);
                }
            }
            size_t removeCount = 0;
            for (size_t rank = rStartRank;; rank++) {
                bool edge = false;
                Timer timer;
                for (auto it = tmpGraph.begin(); it != tmpGraph.end();) {
                    size_t id = (*it).first;
                    try {
                        tann::GraphNode &node = (*it).second;
                        if (rank >= node.size()) {
                            it = tmpGraph.erase(it);
                            continue;
                        }
                        edge = true;
                        if (rank >= 1 && node[rank - 1].distance > node[rank].distance) {
                            std::cerr << "distance order is wrong!" << std::endl;
                            std::cerr << id << ":" << rank << ":" << node[rank - 1].id << ":" << node[rank].id
                                      << std::endl;
                        }
                        tann::GraphNode &tn = *outGraph.getNode(id);
                        volatile bool found = false;
                        if (rank < 1000) {
                            for (size_t tni = 0; tni < tn.size() && !found; tni++) {
                                if (tn[tni].id == node[rank].id) {
                                    continue;
                                }
                                tann::GraphNode &dstNode = *outGraph.getNode(tn[tni].id);
                                for (size_t dni = 0; dni < dstNode.size(); dni++) {
                                    if ((dstNode[dni].id == node[rank].id) &&
                                        (dstNode[dni].distance < node[rank].distance)) {
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(10)
#endif
                            for (size_t tni = 0; tni < tn.size(); tni++) {
                                if (found) {
                                    continue;
                                }
                                if (tn[tni].id == node[rank].id) {
                                    continue;
                                }
                                tann::GraphNode &dstNode = *outGraph.getNode(tn[tni].id);
                                for (size_t dni = 0; dni < dstNode.size(); dni++) {
                                    if ((dstNode[dni].id == node[rank].id) &&
                                        (dstNode[dni].distance < node[rank].distance)) {
                                        found = true;
                                    }
                                }
                            }
                        }
                        if (!found) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                            outGraph.addEdge(id, node.at(i, outGraph.repository.allocator).id,
                                     node.at(i, outGraph.repository.allocator).distance, true);
#else
                            tn.push_back(tann::VectorDistance(node[rank].id, node[rank].distance));
#endif
                        } else {
                            removeCount++;
                        }
                    } catch (tann::Exception &err) {
                        std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                                  << std::endl;
                        it++;
                        continue;
                    }
                    it++;
                }
                if (edge == false) {
                    break;
                }
            }
#endif // NGT_SHARED_MEMORY_ALLOCATOR
        }

        static void
        adjustPathsEffectively(tann::Index &outIndex, size_t minNoOfEdges = 0) {
            tann::GraphIndex &outGraph = dynamic_cast<tann::GraphIndex &>(outIndex.getIndex());
            adjustPathsEffectively(outGraph, minNoOfEdges);
        }

        static bool edgeComp(tann::VectorDistance a, tann::VectorDistance b) {
            return a.id < b.id;
        }

#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
        static void insert(tann::GraphNode &node, size_t edgeID, tann::distance_type edgeDistance, tann::GraphIndex &graph) {
          tann::VectorDistance edge(edgeID, edgeDistance);
          GraphNode::iterator ni = std::lower_bound(node.begin(graph.repository.allocator), node.end(graph.repository.allocator), edge, edgeComp);
          node.insert(ni, edge, graph.repository.allocator);
        }

        static bool hasEdge(tann::GraphIndex &graph, size_t srcNodeID, size_t dstNodeID)
        {
           tann::GraphNode &srcNode = *graph.getNode(srcNodeID);
           GraphNode::iterator ni = std::lower_bound(srcNode.begin(graph.repository.allocator), srcNode.end(graph.repository.allocator), VectorDistance(dstNodeID, 0.0), edgeComp);
           return (ni != srcNode.end(graph.repository.allocator)) && ((*ni).id == dstNodeID);
        }
#else

        static void insert(tann::GraphNode &node, size_t edgeID, tann::distance_type edgeDistance) {
            tann::VectorDistance edge(edgeID, edgeDistance);
            GraphNode::iterator ni = std::lower_bound(node.begin(), node.end(), edge, edgeComp);
            node.insert(ni, edge);
        }

        static bool hasEdge(tann::GraphIndex &graph, size_t srcNodeID, size_t dstNodeID) {
            tann::GraphNode &srcNode = *graph.getNode(srcNodeID);
            GraphNode::iterator ni = std::lower_bound(srcNode.begin(), srcNode.end(), VectorDistance(dstNodeID, 0.0),
                                                      edgeComp);
            return (ni != srcNode.end()) && ((*ni).id == dstNodeID);
        }

#endif


        static void
        adjustPathsEffectively(tann::GraphIndex &outGraph,
                               size_t minNoOfEdges) {
            Timer timer;
            timer.start();
            std::vector<tann::GraphNode> tmpGraph;
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                try {
                    tann::GraphNode &node = *outGraph.getNode(id);
                    tmpGraph.push_back(node);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    node.clear(outGraph.repository.allocator);
#else
                    node.clear();
#endif
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    tmpGraph.push_back(tann::GraphNode(outGraph.repository.allocator));
#else
                    tmpGraph.push_back(tann::GraphNode());
#endif
                }
            }
            if (outGraph.repository.size() != tmpGraph.size() + 1) {
                std::stringstream msg;
                msg << "GraphReconstructor: Fatal inner error. " << outGraph.repository.size() << ":"
                    << tmpGraph.size();
                TANN_THROW(msg);
            }
            timer.stop();
            std::cerr << "GraphReconstructor::adjustPaths: graph preparing time=" << timer << std::endl;
            timer.reset();
            timer.start();

            std::vector<std::vector<std::pair<uint32_t, uint32_t> > > removeCandidates(tmpGraph.size());
            int removeCandidateCount = 0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (size_t idx = 0; idx < tmpGraph.size(); ++idx) {
                auto it = tmpGraph.begin() + idx;
                size_t id = idx + 1;
                try {
                    tann::GraphNode &srcNode = *it;
                    std::unordered_map<uint32_t, std::pair<size_t, double> > neighbors;
                    for (size_t sni = 0; sni < srcNode.size(); ++sni) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                        neighbors[srcNode.at(sni, outGraph.repository.allocator).id] = std::pair<size_t, double>(sni, srcNode.at(sni, outGraph.repository.allocator).distance);
#else
                        neighbors[srcNode[sni].id] = std::pair<size_t, double>(sni, srcNode[sni].distance);
#endif
                    }

                    std::vector<std::pair<int, std::pair<uint32_t, uint32_t> > > candidates;
                    for (size_t sni = 0; sni < srcNode.size(); sni++) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                        tann::GraphNode &pathNode = tmpGraph[srcNode.at(sni, outGraph.repository.allocator).id - 1];
#else
                        tann::GraphNode &pathNode = tmpGraph[srcNode[sni].id - 1];
#endif
                        for (size_t pni = 0; pni < pathNode.size(); pni++) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                            auto dstNodeID = pathNode.at(pni, outGraph.repository.allocator).id;
#else
                            auto dstNodeID = pathNode[pni].id;
#endif
                            auto dstNode = neighbors.find(dstNodeID);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                            if (dstNode != neighbors.end()
                            && srcNode.at(sni, outGraph.repository.allocator).distance < (*dstNode).second.second
                            && pathNode.at(pni, outGraph.repository.allocator).distance < (*dstNode).second.second
                            ) {
#else
                            if (dstNode != neighbors.end()
                                && srcNode[sni].distance < (*dstNode).second.second
                                && pathNode[pni].distance < (*dstNode).second.second
                                    ) {
#endif
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                                candidates.push_back(std::pair<int, std::pair<uint32_t, uint32_t> >((*dstNode).second.first, std::pair<uint32_t, uint32_t>(srcNode.at(sni, outGraph.repository.allocator).id, dstNodeID)));
#else
                                candidates.push_back(
                                        std::pair<int, std::pair<uint32_t, uint32_t> >((*dstNode).second.first,
                                                                                       std::pair<uint32_t, uint32_t>(
                                                                                               srcNode[sni].id,
                                                                                               dstNodeID)));
#endif
                                removeCandidateCount++;
                            }
                        }
                    }
                    sort(candidates.begin(), candidates.end(),
                         std::greater<std::pair<int, std::pair<uint32_t, uint32_t>>>());
                    removeCandidates[id - 1].reserve(candidates.size());
                    for (size_t i = 0; i < candidates.size(); i++) {
                        removeCandidates[id - 1].push_back(candidates[i].second);
                    }
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
                    continue;
                }
            }
            timer.stop();
            std::cerr << "GraphReconstructor::adjustPaths: extracting removed edge candidates time=" << timer
                      << std::endl;
            timer.reset();
            timer.start();

            std::list<size_t> ids;
            for (size_t idx = 0; idx < tmpGraph.size(); ++idx) {
                ids.push_back(idx + 1);
            }

            int removeCount = 0;
            removeCandidateCount = 0;
            for (size_t rank = 0; ids.size() != 0; rank++) {
                for (auto it = ids.begin(); it != ids.end();) {
                    size_t id = *it;
                    size_t idx = id - 1;
                    try {
                        tann::GraphNode &srcNode = tmpGraph[idx];
                        if (rank >= srcNode.size()) {
                            if (!removeCandidates[idx].empty() && minNoOfEdges == 0) {
                                std::cerr << "Something wrong! ID=" << id << " # of remaining candidates="
                                          << removeCandidates[idx].size() << std::endl;
                                abort();
                            }
#if !defined(NGT_SHARED_MEMORY_ALLOCATOR)
                            tann::GraphNode empty;
                            tmpGraph[idx] = empty;
#endif
                            it = ids.erase(it);
                            continue;
                        }
                        if (removeCandidates[idx].size() > 0 &&
                            ((*outGraph.getNode(id)).size() + srcNode.size() - rank) > minNoOfEdges) {
                            removeCandidateCount++;
                            bool pathExist = false;
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                            while (!removeCandidates[idx].empty() && (removeCandidates[idx].back().second == srcNode.at(rank, outGraph.repository.allocator).id)) {
#else
                            while (!removeCandidates[idx].empty() &&
                                   (removeCandidates[idx].back().second == srcNode[rank].id)) {
#endif
                                size_t path = removeCandidates[idx].back().first;
                                size_t dst = removeCandidates[idx].back().second;
                                removeCandidates[idx].pop_back();
                                if (removeCandidates[idx].empty()) {
                                    std::vector<std::pair<uint32_t, uint32_t>> empty;
                                    removeCandidates[idx] = empty;
                                }
                                if ((hasEdge(outGraph, id, path)) && (hasEdge(outGraph, path, dst))) {
                                    pathExist = true;
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                                    while (!removeCandidates[idx].empty() && (removeCandidates[idx].back().second == srcNode.at(rank, outGraph.repository.allocator).id)) {
#else
                                    while (!removeCandidates[idx].empty() &&
                                           (removeCandidates[idx].back().second == srcNode[rank].id)) {
#endif
                                        removeCandidates[idx].pop_back();
                                        if (removeCandidates[idx].empty()) {
                                            std::vector<std::pair<uint32_t, uint32_t>> empty;
                                            removeCandidates[idx] = empty;
                                        }
                                    }
                                    break;
                                }
                            }
                            if (pathExist) {
                                removeCount++;
                                it++;
                                continue;
                            }
                        }
                        tann::GraphNode &outSrcNode = *outGraph.getNode(id);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                        insert(outSrcNode, srcNode.at(rank, outGraph.repository.allocator).id, srcNode.at(rank, outGraph.repository.allocator).distance, outGraph);
#else
                        insert(outSrcNode, srcNode[rank].id, srcNode[rank].distance);
#endif
                    } catch (tann::Exception &err) {
                        std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                                  << std::endl;
                        it++;
                        continue;
                    }
                    it++;
                }
            }
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                try {
                    tann::GraphNode &node = *outGraph.getNode(id);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    std::sort(node.begin(outGraph.repository.allocator), node.end(outGraph.repository.allocator));
#else
                    std::sort(node.begin(), node.end());
#endif
                } catch (...) {}
            }
        }


        static
        void convertToANNG(std::vector<tann::VectorDistances> &graph) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
            std::cerr << "convertToANNG is not implemented for shared memory." << std::endl;
            return;
#else
            std::cerr << "convertToANNG begin" << std::endl;
            for (size_t idx = 0; idx < graph.size(); idx++) {
                tann::GraphNode &node = graph[idx];
                for (auto ni = node.begin(); ni != node.end(); ++ni) {
                    graph[(*ni).id - 1].push_back(tann::VectorDistance(idx + 1, (*ni).distance));
                }
            }
            for (size_t idx = 0; idx < graph.size(); idx++) {
                tann::GraphNode &node = graph[idx];
                if (node.size() == 0) {
                    continue;
                }
                std::sort(node.begin(), node.end());
                tann::ObjectID prev = 0;
                for (auto it = node.begin(); it != node.end();) {
                    if (prev == (*it).id) {
                        it = node.erase(it);
                        continue;
                    }
                    prev = (*it).id;
                    it++;
                }
                tann::GraphNode tmp = node;
                node.swap(tmp);
            }
            std::cerr << "convertToANNG end" << std::endl;
#endif
        }

        static
        void
        reconstructGraph(std::vector<tann::VectorDistances> &graph, tann::GraphIndex &outGraph, size_t originalEdgeSize,
                         size_t reverseEdgeSize) {
            if (reverseEdgeSize > 10000) {
                std::cerr << "something wrong. Edge size=" << reverseEdgeSize << std::endl;
                exit(1);
            }

            tann::Timer originalEdgeTimer, reverseEdgeTimer, normalizeEdgeTimer;
            originalEdgeTimer.start();

            size_t warningCount = 0;
            const size_t warningLimit = 10;
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                try {
                    tann::GraphNode &node = *outGraph.getNode(id);
                    if (originalEdgeSize == 0) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                        node.clear(outGraph.repository.allocator);
#else
                        tann::GraphNode empty;
                        node.swap(empty);
#endif
                    } else {
                        tann::VectorDistances n = graph[id - 1];
                        if (n.size() < originalEdgeSize) {
                            warningCount++;
                            if (warningCount <= warningLimit) {
                                std::cerr << "GraphReconstructor: Warning. The edges are too few. " << n.size() << ":"
                                          << originalEdgeSize << " for " << id << std::endl;
                            }
                            if (warningCount == warningLimit) {
                                std::cerr << "GraphReconstructor: Info. Too many warnings. Warning is disabled."
                                          << std::endl;
                            }
                            continue;
                        }
                        n.resize(originalEdgeSize);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                        node.copy(n, outGraph.repository.allocator);
#else
                        node.swap(n);
#endif
                    }
                } catch (tann::Exception &err) {
                    warningCount++;
                    if (warningCount <= warningLimit) {
                        std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                                  << std::endl;
                    }
                    if (warningCount == warningLimit) {
                        std::cerr << "GraphReconstructor: Info. Too many warnings. Warning is disabled." << std::endl;
                    }
                    continue;
                }
            }
            if (warningCount > warningLimit) {
                std::cerr << "GraphReconstructor: The total " << warningCount << " Warnings." << std::endl;
            }
            originalEdgeTimer.stop();

            reverseEdgeTimer.start();
            int insufficientNodeCount = 0;
            for (size_t id = 1; id <= graph.size(); ++id) {
                try {
                    tann::VectorDistances &node = graph[id - 1];
                    size_t rsize = reverseEdgeSize;
                    if (rsize > node.size()) {
                        insufficientNodeCount++;
                        rsize = node.size();
                    }
                    for (size_t i = 0; i < rsize; ++i) {
                        tann::distance_type distance = node[i].distance;
                        size_t nodeID = node[i].id;
                        try {
                            tann::GraphNode &n = *outGraph.getNode(nodeID);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                            n.push_back(tann::VectorDistance(id, distance), outGraph.repository.allocator);
#else
                            n.push_back(tann::VectorDistance(id, distance));
#endif
                        } catch (...) {}
                    }
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
                    continue;
                }
            }
            reverseEdgeTimer.stop();
            if (insufficientNodeCount != 0) {
                std::cerr << "# of the nodes edges of which are in short = " << insufficientNodeCount << std::endl;
            }

            normalizeEdgeTimer.start();
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                try {
                    tann::GraphNode &n = *outGraph.getNode(id);
                    if (id % 100000 == 0) {
                        std::cerr << "Processed " << id << " nodes" << std::endl;
                    }
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    std::sort(n.begin(outGraph.repository.allocator), n.end(outGraph.repository.allocator));
#else
                    std::sort(n.begin(), n.end());
#endif
                    tann::ObjectID prev = 0;
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    for (auto it = n.begin(outGraph.repository.allocator); it != n.end(outGraph.repository.allocator);) {
#else
                    for (auto it = n.begin(); it != n.end();) {
#endif
                        if (prev == (*it).id) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                            it = n.erase(it, outGraph.repository.allocator);
#else
                            it = n.erase(it);
#endif
                            continue;
                        }
                        prev = (*it).id;
                        it++;
                    }
#if !defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    tann::GraphNode tmp = n;
                    n.swap(tmp);
#endif
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
                    continue;
                }
            }
            normalizeEdgeTimer.stop();
            std::cerr << "Reconstruction time=" << originalEdgeTimer.time << ":" << reverseEdgeTimer.time
                      << ":" << normalizeEdgeTimer.time << std::endl;

            tann::Property prop;
            outGraph.getProperty().get(prop);
            prop.graphType = tann::NeighborhoodGraph::GraphTypeONNG;
            outGraph.getProperty().set(prop);
        }


        static
        void reconstructGraphWithConstraint(std::vector<tann::VectorDistances> &graph, tann::GraphIndex &outGraph,
                                            size_t originalEdgeSize, size_t reverseEdgeSize,
                                            char mode = 'a') {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
            std::cerr << "reconstructGraphWithConstraint is not implemented." << std::endl;
            abort();
#else

            tann::Timer originalEdgeTimer, reverseEdgeTimer, normalizeEdgeTimer;

            if (reverseEdgeSize > 10000) {
                std::cerr << "something wrong. Edge size=" << reverseEdgeSize << std::endl;
                exit(1);
            }

            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                if (id % 1000000 == 0) {
                    std::cerr << "Processed " << id << std::endl;
                }
                try {
                    tann::GraphNode &node = *outGraph.getNode(id);
                    if (node.size() == 0) {
                        continue;
                    }
                    node.clear();
                    tann::GraphNode empty;
                    node.swap(empty);
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
                    continue;
                }
            }
            tann::GraphIndex::showStatisticsOfGraph(outGraph);

            std::vector<VectorDistances> reverse(graph.size() + 1);
            for (size_t id = 1; id <= graph.size(); ++id) {
                try {
                    tann::GraphNode &node = graph[id - 1];
                    if (id % 100000 == 0) {
                        std::cerr << "Processed (summing up) " << id << std::endl;
                    }
                    for (size_t rank = 0; rank < node.size(); rank++) {
                        reverse[node[rank].id].push_back(VectorDistance(id, node[rank].distance));
                    }
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
                    continue;
                }
            }

            std::vector<std::pair<size_t, size_t> > reverseSize(graph.size() + 1);
            reverseSize[0] = std::pair<size_t, size_t>(0, 0);
            for (size_t rid = 1; rid <= graph.size(); ++rid) {
                reverseSize[rid] = std::pair<size_t, size_t>(reverse[rid].size(), rid);
            }
            std::sort(reverseSize.begin(), reverseSize.end());


            std::vector<uint32_t> indegreeCount(graph.size(), 0);
            size_t zeroCount = 0;
            for (size_t sizerank = 0; sizerank <= reverseSize.size(); sizerank++) {

                if (reverseSize[sizerank].first == 0) {
                    zeroCount++;
                    continue;
                }
                size_t rid = reverseSize[sizerank].second;
                VectorDistances &rnode = reverse[rid];
                for (auto rni = rnode.begin(); rni != rnode.end(); ++rni) {
                    if (indegreeCount[(*rni).id] >= reverseEdgeSize) {
                        continue;
                    }
                    tann::GraphNode &node = *outGraph.getNode(rid);
                    if (indegreeCount[(*rni).id] > 0 && node.size() >= originalEdgeSize) {
                        continue;
                    }

                    node.push_back(tann::VectorDistance((*rni).id, (*rni).distance));
                    indegreeCount[(*rni).id]++;
                }
            }
            reverseEdgeTimer.stop();
            std::cerr << "The number of nodes with zero outdegree by reverse edges=" << zeroCount << std::endl;
            tann::GraphIndex::showStatisticsOfGraph(outGraph);

            normalizeEdgeTimer.start();
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                try {
                    tann::GraphNode &n = *outGraph.getNode(id);
                    if (id % 100000 == 0) {
                        std::cerr << "Processed " << id << std::endl;
                    }
                    std::sort(n.begin(), n.end());
                    tann::ObjectID prev = 0;
                    for (auto it = n.begin(); it != n.end();) {
                        if (prev == (*it).id) {
                            it = n.erase(it);
                            continue;
                        }
                        prev = (*it).id;
                        it++;
                    }
                    tann::GraphNode tmp = n;
                    n.swap(tmp);
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
                    continue;
                }
            }
            normalizeEdgeTimer.stop();
            tann::GraphIndex::showStatisticsOfGraph(outGraph);

            originalEdgeTimer.start();
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                if (id % 1000000 == 0) {
                    std::cerr << "Processed " << id << std::endl;
                }
                tann::GraphNode &node = graph[id - 1];
                try {
                    tann::GraphNode &onode = *outGraph.getNode(id);
                    bool stop = false;
                    for (size_t rank = 0; (rank < node.size() && rank < originalEdgeSize) && stop == false; rank++) {
                        switch (mode) {
                            case 'a':
                                if (onode.size() >= originalEdgeSize) {
                                    stop = true;
                                    continue;
                                }
                                break;
                            case 'c':
                                break;
                        }
                        tann::distance_type distance = node[rank].distance;
                        size_t nodeID = node[rank].id;
                        outGraph.addEdge(id, nodeID, distance, false);
                    }
                } catch (tann::Exception &err) {
                    std::cerr << "GraphReconstructor: Warning. Cannot get the node. ID=" << id << ":" << err.what()
                              << std::endl;
                    continue;
                }
            }
            originalEdgeTimer.stop();
            tann::GraphIndex::showStatisticsOfGraph(outGraph);

            std::cerr << "Reconstruction time=" << originalEdgeTimer.time << ":" << reverseEdgeTimer.time
                      << ":" << normalizeEdgeTimer.time << std::endl;

#endif
        }

        // reconstruct a pseudo ANNG with a fewer edges from an actual ANNG with more edges.
        // graph is a source ANNG
        // index is an index with a reconstructed ANNG
        static
        void reconstructANNGFromANNG(std::vector<tann::VectorDistances> &graph, tann::Index &index, size_t edgeSize) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
            std::cerr << "reconstructANNGFromANNG is not implemented." << std::endl;
            abort();
#else

            tann::GraphIndex &outGraph = dynamic_cast<tann::GraphIndex &>(index.getIndex());

            // remove all edges in the index.
            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                if (id % 1000000 == 0) {
                    std::cerr << "Processed " << id << " nodes." << std::endl;
                }
                try {
                    tann::GraphNode &node = *outGraph.getNode(id);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                    node.clear(outGraph.repository.allocator);
#else
                    tann::GraphNode empty;
                    node.swap(empty);
#endif
                } catch (tann::Exception &err) {
                }
            }

            for (size_t id = 1; id <= graph.size(); ++id) {
                size_t edgeCount = 0;
                try {
                    tann::VectorDistances &node = graph[id - 1];
                    tann::GraphNode &n = *outGraph.getNode(id);
                    tann::distance_type prevDistance = 0.0;
                    assert(n.size() == 0);
                    for (size_t i = 0; i < node.size(); ++i) {
                        tann::distance_type distance = node[i].distance;
                        if (prevDistance > distance) {
                            TANN_THROW("Edge distance order is invalid");
                        }
                        prevDistance = distance;
                        size_t nodeID = node[i].id;
                        if (node[i].id < id) {
                            try {
                                tann::GraphNode &dn = *outGraph.getNode(nodeID);
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
                                n.push_back(tann::VectorDistance(nodeID, distance), outGraph.repository.allocator);
                                dn.push_back(tann::VectorDistance(id, distance), outGraph.repository.allocator);
#else
                                n.push_back(tann::VectorDistance(nodeID, distance));
                                dn.push_back(tann::VectorDistance(id, distance));
#endif
                            } catch (...) {}
                            edgeCount++;
                        }
                        if (edgeCount >= edgeSize) {
                            break;
                        }
                    }
                } catch (tann::Exception &err) {
                }
            }

            for (size_t id = 1; id < outGraph.repository.size(); id++) {
                try {
                    tann::GraphNode &n = *outGraph.getNode(id);
                    std::sort(n.begin(), n.end());
                    tann::ObjectID prev = 0;
                    for (auto it = n.begin(); it != n.end();) {
                        if (prev == (*it).id) {
                            it = n.erase(it);
                            continue;
                        }
                        prev = (*it).id;
                        it++;
                    }
                    tann::GraphNode tmp = n;
                    n.swap(tmp);
                } catch (...) {
                }
            }
#endif
        }

        static void
        refineANNG(tann::Index &index, bool unlog, float epsilon = 0.1, float accuracy = 0.0, int noOfEdges = 0,
                   int exploreEdgeSize = INT_MIN, size_t batchSize = 10000) {
            tann::StdOstreamRedirector redirector(unlog);
            redirector.begin();
            try {
                refineANNG(index, epsilon, accuracy, noOfEdges, exploreEdgeSize, batchSize);
            } catch (tann::Exception &err) {
                redirector.end();
                throw (err);
            }
        }

        static void refineANNG(tann::Index &index, float epsilon = 0.1, float accuracy = 0.0, int noOfEdges = 0,
                               int exploreEdgeSize = INT_MIN, size_t batchSize = 10000) {
#if defined(NGT_SHARED_MEMORY_ALLOCATOR)
            TANN_THROW("GraphReconstructor::refineANNG: Not implemented for the shared memory option.");
#else
            auto prop = static_cast<GraphIndex &>(index.getIndex()).getGraphProperty();
            tann::VectorRepository &objectRepository = index.getObjectSpace().getRepository();
            tann::GraphIndex &graphIndex = static_cast<GraphIndex &>(index.getIndex());
            size_t nOfObjects = objectRepository.size();
            bool error = false;
            std::string errorMessage;

            size_t noOfSearchedEdges = noOfEdges < 0 ? -noOfEdges : (noOfEdges > prop.edgeSizeForCreation ? noOfEdges
                                                                                                          : prop.edgeSizeForCreation);
            noOfSearchedEdges++;
            for (size_t bid = 1; bid < nOfObjects; bid += batchSize) {
                tann::VectorDistances results[batchSize];
                // search
#pragma omp parallel for
                for (size_t idx = 0; idx < batchSize; idx++) {
                    size_t id = bid + idx;
                    if (id % 100000 == 0) {
                        std::cerr << "# of processed objects=" << id << std::endl;
                    }
                    if (objectRepository.isEmpty(id)) {
                        continue;
                    }
                    tann::SearchContainer searchContainer(*objectRepository.get(id));
                    searchContainer.setResults(&results[idx]);
                    assert(prop.edgeSizeForCreation > 0);
                    searchContainer.setSize(noOfSearchedEdges);
                    if (accuracy > 0.0) {
                        searchContainer.setExpectedAccuracy(accuracy);
                    } else {
                        searchContainer.setEpsilon(epsilon);
                    }
                    if (exploreEdgeSize != INT_MIN) {
                        searchContainer.setEdgeSize(exploreEdgeSize);
                    }
                    if (!error) {
                        try {
                            index.search(searchContainer);
                        } catch (tann::Exception &err) {
#pragma omp critical
                            {
                                error = true;
                                errorMessage = err.what();
                            }
                        }
                    }
                }
                if (error) {
                    std::stringstream msg;
                    msg << "GraphReconstructor::refineANNG: " << errorMessage;
                    TANN_THROW(msg);
                }
                // outgoing edges
#pragma omp parallel for
                for (size_t idx = 0; idx < batchSize; idx++) {
                    size_t id = bid + idx;
                    if (objectRepository.isEmpty(id)) {
                        continue;
                    }
                    tann::GraphNode &node = *graphIndex.getNode(id);
                    for (auto i = results[idx].begin(); i != results[idx].end(); ++i) {
                        if ((*i).id != id) {
                            node.push_back(*i);
                        }
                    }
                    std::sort(node.begin(), node.end());
                    // dedupe
                    ObjectID prev = 0;
                    for (GraphNode::iterator ni = node.begin(); ni != node.end();) {
                        if (prev == (*ni).id) {
                            ni = node.erase(ni);
                            continue;
                        }
                        prev = (*ni).id;
                        ni++;
                    }
                }
                // incomming edges
                if (noOfEdges != 0) {
                    continue;
                }
                for (size_t idx = 0; idx < batchSize; idx++) {
                    size_t id = bid + idx;
                    if (id % 10000 == 0) {
                        std::cerr << "# of processed objects=" << id << std::endl;
                    }
                    for (auto i = results[idx].begin(); i != results[idx].end(); ++i) {
                        if ((*i).id != id) {
                            tann::GraphNode &node = *graphIndex.getNode((*i).id);
                            graphIndex.addEdge(node, id, (*i).distance, false);
                        }
                    }
                }
            }
            if (noOfEdges > 0) {
                // prune to build knng
                size_t nedges = noOfEdges < 0 ? -noOfEdges : noOfEdges;
#pragma omp parallel for
                for (ObjectID id = 1; id < nOfObjects; ++id) {
                    if (objectRepository.isEmpty(id)) {
                        continue;
                    }
                    tann::GraphNode &node = *graphIndex.getNode(id);
                    if (node.size() > nedges) {
                        node.resize(nedges);
                    }
                }
            }
#endif // defined(NGT_SHARED_MEMORY_ALLOCATOR)
        }
    };

}; // NGT
