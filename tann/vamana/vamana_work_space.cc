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
#include "tann/vamana/vamana_work_space.h"
#include "tann/vamana/constants.h"
namespace tann {

    VamanaWorkSpace::VamanaWorkSpace(uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc,
                                            size_t dim,
                                            size_t aligned_dim, size_t alignment_factor, bool init_pq_scratch)
            : _L(0), _R(r), _maxc(maxc) {
        if (init_pq_scratch)
            _pq_scratch = new QuantizeScratch(constants::kMaxGraphDegree, aligned_dim);
        else
            _pq_scratch = nullptr;

        _occlude_factor.reserve(maxc);
        _id_scratch.reserve((size_t) std::ceil(1.5 * constants::kGraphSlackFactor * _R));
        _dist_scratch.reserve((size_t) std::ceil(1.5 * constants::kGraphSlackFactor * _R));

        resize_for_new_L(std::max(search_l, indexing_l));
    }

    void VamanaWorkSpace::clear_sub() {
        _pool.clear();
        _best_l_nodes.clear();
        _occlude_factor.clear();

        _inserted_into_pool_rs.clear();
        _inserted_into_pool_bs.reset();

        _id_scratch.clear();
        _dist_scratch.clear();

        _expanded_nodes_set.clear();
        _expanded_nghrs_vec.clear();
        _occlude_list_output.clear();
    }

    void VamanaWorkSpace::resize_for_new_L(uint32_t new_l) {
        if (new_l > _L) {
            _L = new_l;
            _pool.reserve(3 * _L + _R);
            _best_l_nodes.reserve(_L);

            _inserted_into_pool_rs.reserve(20 * _L);
        }
    }
}  // namespace tann
