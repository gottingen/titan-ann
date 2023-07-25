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
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "tann/hnsw/leveled_graph.h"

TEST_CASE("leveled graph") {
    tann::LeveledGraph graph;
    graph.initialize(10000, 16);
    // level 0 idx 0
    CHECK_EQ(graph.setup_location(0,0).ok(), true);
    CHECK_EQ(graph.level(0), 0);
    CHECK_EQ(graph.mutable_node(0,0).size(), 0);
    CHECK_EQ(graph.mutable_node(0,0).capacity(), 32);
    // level 1 idx 2
    CHECK_EQ(graph.setup_location(2,1).ok(), true);
    CHECK_EQ(graph.level(2), 1);
    CHECK_EQ(graph.mutable_node(2,0).size(), 0);
    CHECK_EQ(graph.mutable_node(2,0).capacity(), 32);
    CHECK_EQ(graph.mutable_node(2,1).capacity(), 16);
}