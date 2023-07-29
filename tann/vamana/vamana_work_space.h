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
#ifndef TANN_VAMANA_VAMANA_WORK_SPACE_H_
#define TANN_VAMANA_VAMANA_WORK_SPACE_H_

#include "tann/core/worker_space.h"
#include "turbo/container/flat_hash_set.h"
#include "turbo/container/flat_hash_map.h"
#include "turbo/container/dynamic_bitset.h"
#include "tann/vamana/pq.h"
namespace tann {

    class VamanaWorkSpace : public WorkSpace {
    public:
        ~VamanaWorkSpace() override;
        VamanaWorkSpace() = default;

        VamanaWorkSpace(uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc, size_t dim,
                          size_t aligned_dim,
                          size_t alignment_factor, bool init_pq_scratch = false);

        void resize_for_new_L(uint32_t new_search_l);

        void clear_sub() override;

        inline uint32_t get_L() {
            return _L;
        }

        inline uint32_t get_R() {
            return _R;
        }

        inline uint32_t get_maxc() {
            return _maxc;
        }

        inline QuantizeScratch *pq_scratch() {
            return _pq_scratch;
        }

        inline std::vector<NeighborEntity> &pool() {
            return _pool;
        }

        inline NeighborQueue &best_l_nodes() {
            return _best_l_nodes;
        }

        inline std::vector<double> &occlude_factor() {
            return _occlude_factor;
        }

        inline turbo::flat_hash_set<uint32_t> &inserted_into_pool_rs() {
            return _inserted_into_pool_rs;
        }

        inline turbo::dynamic_bitset<> &inserted_into_pool_bs() {
            return _inserted_into_pool_bs;
        }

        inline std::vector<uint32_t> &id_scratch() {
            return _id_scratch;
        }

        inline std::vector<float> &dist_scratch() {
            return _dist_scratch;
        }

        inline turbo::flat_hash_set<uint32_t> &expanded_nodes_set() {
            return _expanded_nodes_set;
        }

        inline std::vector<NeighborEntity> &expanded_nodes_vec() {
            return _expanded_nghrs_vec;
        }

        inline std::vector<uint32_t> &occlude_list_output() {
            return _occlude_list_output;
        }

    private:
        uint32_t _L;
        uint32_t _R;
        uint32_t _maxc;

        QuantizeScratch *_pq_scratch = nullptr;

        // _pool stores all neighbors explored from best_L_nodes.
        // Usually around L+R, but could be higher.
        // Initialized to 3L+R for some slack, expands as needed.
        std::vector<NeighborEntity> _pool;

        // _best_l_nodes is reserved for storing best L entries
        // Underlying storage is L+1 to support inserts
        NeighborQueue _best_l_nodes;

        // _occlude_factor.size() >= pool.size() in occlude_list function
        // _pool is clipped to maxc in occlude_list before affecting _occlude_factor
        // _occlude_factor is initialized to maxc size
        std::vector<double> _occlude_factor;

        // Capacity initialized to 20L
        turbo::flat_hash_set<uint32_t> _inserted_into_pool_rs;

        turbo::dynamic_bitset<> _inserted_into_pool_bs;

        // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
        std::vector<uint32_t> _id_scratch;

        // _dist_scratch must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
        // _dist_scratch should be at least the size of id_scratch
        std::vector<float> _dist_scratch;

        //  Buffers used in process delete, capacity increases as needed
        turbo::flat_hash_set<uint32_t> _expanded_nodes_set;
        std::vector<NeighborEntity> _expanded_nghrs_vec;
        std::vector<uint32_t> _occlude_list_output;
    };

}
#endif  // TANN_VAMANA_VAMANA_WORK_SPACE_H_
