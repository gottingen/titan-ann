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

#include "tann/hnsw/leveled_graph.h"
#include "tann/io/utility.h"

namespace tann {

    [[nodiscard]] turbo::Status LeveledGraph::save(turbo::SequentialWriteFile &file) {
        auto r = write_binary_pod(file, _max_nbor);
        if(!r.ok()) {
            return r;
        }
        size_t nsize = _nodes.size();
        r = write_binary_pod(file, nsize);
        if(!r.ok()) {
            return r;
        }

        for (size_t i = 0; i < nsize; ++i) {
            auto &ref  = _nodes[i];
            r = write_binary_pod(file, ref.level);
            if(!r.ok()) {
                return r;
            }
            r = write_binary_vector<location_t>(file, ref.links);
            if(!r.ok()) {
                return r;
            }
        }
        return turbo::OkStatus();
    }

    [[nodiscard]] turbo::Status LeveledGraph::load(turbo::SequentialReadFile &file) {
        auto r = read_binary_pod(file, _max_nbor);
        if(!r.ok()) {
            return r;
        }
        size_t nsize;
        r = read_binary_pod(file, nsize);
        if(!r.ok()) {
            return r;
        }
        _nodes.resize(nsize);

        for (size_t i = 0; i < _nodes.size(); ++i) {
            auto &ref  = _nodes[i];
            r = read_binary_pod(file, ref.level);
            if(!r.ok()) {
                return r;
            }
            if(ref.level == -1) {
                continue;
            }
            r = read_binary_vector(file, ref.links);
            if(!r.ok()) {
                return r;
            }
        }
        return turbo::OkStatus();
    }
}  // namespace tann
