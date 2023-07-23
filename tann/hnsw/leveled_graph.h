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


#ifndef TANN_HNSW_LEVELED_GRAPH_H_
#define TANN_HNSW_LEVELED_GRAPH_H_

#include <vector>
#include "turbo/base/status.h"
#include "tann/core/types.h"
#include "turbo/meta/span.h"

namespace tann {

    class LeveledGraph {
    private:
        class LeveledNode {
        public:
            int level{-1};
            std::vector<location_t> links;
        };

    public:
        class Node {
        public:
            Node() = default;

            Node(int level, turbo::Span<location_t> span) : _level(level),
                      _data(span.data(), span.size()) {

            }

            Node(const Node&) = default;

            Node &operator=(const Node&) = default;

            [[nodiscard]] int level() const {
                return _level;
            }

            [[nodiscard]] uint32_t size() const {
                return _data[0];
            }

            [[nodiscard]] uint32_t capacity() const {
                return _data.size() - 1;
            }

            location_t &operator[](std::size_t i) {
                return _data[i + 1];
            }

            location_t &at(location_t i) {
                return _data[i + 1];
            }

            void set_size(location_t n) {
                _data[0] = n;
            }

            turbo::Span<location_t> links() {
                return turbo::Span<location_t>{_data.data() + 1, _data.size() - 1};
            }

        private:
            friend class LeveledGraph;

            int _level{-1};
            turbo::Span<location_t> _data;
        };

    public:
        LeveledGraph() = default;

        void initialize(location_t max_elements, location_t max_nbor) {
                          _max_nbor = max_nbor;
                          _nodes.resize(max_elements);

        }

        LeveledNode &at(location_t n) {
            return _nodes[n];
        }

        turbo::Status setup_location(location_t lid, int level) {
            TLOG_CHECK(lid < _nodes.size());
            _nodes[lid].level = level;
            if (level == 0) {
                _nodes[lid].links.resize(_max_nbor * 2 + 1);
            } else {
                _nodes[lid].links.resize((_max_nbor + 1) * (level + 1));
            }
            return turbo::OkStatus();
        }

        [[nodiscard]] int level(location_t lid) const {
            return _nodes[lid].level;
        }

        Node mutable_node(location_t lid, int level) {
            auto &n = _nodes[lid];
            TLOG_CHECK(level <= n.level);
            auto sp = turbo::Span<location_t>(const_cast<location_t*>(n.links.data() + (_max_nbor + 1) * level), _max_nbor + 1);
            Node node(n.level, sp);
            return node;
        }

        [[nodiscard]] Node const_node(location_t lid, int level) const {
            auto &n = _nodes[lid];
            TLOG_CHECK(level <= n.level);
            auto sp = turbo::Span<location_t>(const_cast<location_t*>(n.links.data() + (_max_nbor + 1) * level), _max_nbor + 1);
            Node node(n.level, static_cast<turbo::Span<location_t>>(sp));
            return node;
        }

    private:
        location_t _max_nbor{0};
        std::vector<LeveledNode> _nodes;
    };
}  // namespace tann
#endif  // TANN_HNSW_LEVELED_GRAPH_H_
