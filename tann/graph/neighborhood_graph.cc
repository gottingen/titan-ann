// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tann/graph/neighborhood_graph.h"
#include "tann/graph/k_nearest_neighborhood_graph.h"
#include "tann/graph/relative_neighborhood_graph.h"

using namespace tann::COMMON;

std::shared_ptr<NeighborhoodGraph> NeighborhoodGraph::CreateInstance(std::string type) {
    std::shared_ptr<NeighborhoodGraph> res;
    if (type == "RNG") {
        res.reset(new RelativeNeighborhoodGraph);
    } else if (type == "NNG") {
        res.reset(new KNearestNeighborhoodGraph);
    }
    return res;
}