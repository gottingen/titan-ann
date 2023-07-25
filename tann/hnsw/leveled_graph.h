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
#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"

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

            Node(location_t lid, int level, int cur_level, turbo::Span<location_t> span) : _lid(lid), _level(level),
                                                                                           _cur_level(cur_level),
                                                                                           _data(span.data(),
                                                                                                 span.size()) {

            }

            Node(const Node &) = default;

            Node &operator=(const Node &) = default;

            [[nodiscard]] int level() const {
                return _level;
            }

            [[nodiscard]] uint32_t size() const {
                return _data[0];
            }

            [[nodiscard]] uint32_t capacity() const {
                TLOG_TRACE("capacity lid: {} level: {} capacity: {}", _lid, _level, _data.size() - 1);
                return _data.size() - 1;
            }

            const location_t &operator[](std::size_t i) const {
                TLOG_CHECK(i + 1 <= _data.size(), "at lid: {} cur level {} loc {}", _lid, _cur_level, i);
                return _data[i + 1];
            }

            location_t &at(location_t i) const {
                TLOG_TRACE("at lid: {} cur level {} loc {}", _lid, _cur_level, i);
                TLOG_CHECK(i + 1 < _data.size(), "at lid: {} cur level {} loc {}", _lid, _cur_level, i);
                return _data[i + 1];
            }

            void set_link(location_t i, location_t link) {
                TLOG_CHECK(i + 1 < _data.size(), "set_size lid: {} cur level {} loc {} link {}", _lid, _cur_level, i,
                           link);
                _data[i + 1] = link;
            }

            void set_size(location_t n) {
                TLOG_CHECK(n < _data.size(), "set_size lid: {} cur level {} loc {} capacity {}", _lid, _cur_level, n,
                           _data.size() - 1);
                TLOG_TRACE("set_size lid: {} cur level {} size {} capacity {}", _lid, _cur_level, n, _data.size() - 1);
                _data[0] = n;
            }

            turbo::Span<location_t> links() {
                return turbo::Span<location_t>{_data.data() + 1, _data.size() - 1};
            }

        private:
            friend class LeveledGraph;

            location_t _lid{0};
            int _level{-1};
            int _cur_level{-1};
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
            TLOG_TRACE("set up location: {}, level: {}", lid, level);
            _nodes[lid].level = level;
            if (level == 0) {
                _nodes[lid].links.resize(_max_nbor * 2 + 1);
            } else {
                _nodes[lid].links.resize((_max_nbor + 1) * (level + 2));
            }
            return turbo::OkStatus();
        }

        [[nodiscard]] int level(location_t lid) const {
            return _nodes[lid].level;
        }

        Node mutable_node(location_t lid, int level) {
            auto &n = _nodes[lid];
            TLOG_CHECK(level <= n.level);
            if (level == 0) {
                auto sp = turbo::Span<location_t>(const_cast<location_t *>(n.links.data()),
                                                  _max_nbor * 2 + 1);
                Node node(lid, n.level, level, sp);
                return node;
            } else {
                auto sp = turbo::Span<location_t>(
                        const_cast<location_t *>(n.links.data() + (_max_nbor + 1) * (level + 1)),
                        _max_nbor + 1);
                Node node(lid, n.level, level, sp);
                return node;
            }
        }

        [[nodiscard]] Node const_node(location_t lid, int level) const {
            auto &n = _nodes[lid];
            TLOG_CHECK(level <= n.level);
            if (level == 0) {
                auto sp = turbo::Span<location_t>(const_cast<location_t *>(n.links.data()),
                                                  _max_nbor * 2 + 1);
                Node node(lid, n.level, level, sp);
                return node;
            } else {
                auto sp = turbo::Span<location_t>(
                        const_cast<location_t *>(n.links.data() + (_max_nbor + 1) * (level + 1)),
                        _max_nbor + 1);
                Node node(lid, n.level, level, sp);
                return node;
            }
        }

        [[nodiscard]] turbo::Status save(turbo::SequentialWriteFile &file);

        [[nodiscard]] turbo::Status load(turbo::SequentialReadFile &file);

    private:
        location_t _max_nbor{0};
        std::vector<LeveledNode> _nodes;
    };
}  // namespace tann
#endif  // TANN_HNSW_LEVELED_GRAPH_H_
