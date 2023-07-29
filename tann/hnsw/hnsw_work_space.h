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

#ifndef TANN_HNSW_HNSW_WORK_SPACE_H_
#define TANN_HNSW_HNSW_WORK_SPACE_H_

#include "tann/core/worker_space.h"

namespace tann {

    struct HnswWorkSpace : public WorkSpace {
        // frequently using memory
        // make it pooling avoid alloc and free
        NeighborQueue top_candidates;
        NeighborQueue candidate_set;
        std::vector<std::pair<distance_type, location_t>> return_list;
        uint32_t search_l{0};

        void clear_sub() override {
            top_candidates.clear();
            candidate_set.clear();
            return_list.clear();
        }
    };
}  // namespace tann

#endif  // TANN_HNSW_HNSW_WORK_SPACE_H_
