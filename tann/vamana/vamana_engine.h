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
#ifndef TANN_VAMANA_VAMANA_ENGINE_H_
#define TANN_VAMANA_VAMANA_ENGINE_H_

#include "tann/core/engine.h"
#include "tann/vamana/vamana_work_space.h"

namespace tann {

    class VamaneEngine : public Engine {

    public:
        ~VamaneEngine() override = default;

        turbo::Status
        initialize(const IndexOption &base_option, const std::any &option, MemVectorStore *store) override;

        turbo::Status add_vector(WorkSpace *ws, location_t lid) override;

        WorkSpace *make_workspace() override;

        void setup_workspace(WorkSpace *ws) override {

        }

        turbo::Status remove_vector(location_t lid) override;

        turbo::Status search_vector(WorkSpace *ws) override;

        turbo::Status save(turbo::SequentialWriteFile *file) override;

        turbo::Status load(turbo::SequentialReadFile *file) override;

        bool support_dynamic() const override {
            return true;
        }

        bool need_model() const override {
            return false;
        }

    private:
        turbo::Status search_for_point_and_prune(int location, uint32_t Lindex, std::vector<uint32_t> &pruned_list,
                                        VamanaWorkSpace *scratch);

        turbo::ResultStatus<std::pair<uint32_t, uint32_t>>
        iterate_to_fixed_point(const uint32_t Lsize, const std::vector<uint32_t> &init_ids, VamanaWorkSpace *vws,
                               bool search_invocation);

        std::vector<uint32_t> get_init_ids();

        void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list,
                          VamanaWorkSpace *vws);

        void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                          VamanaWorkSpace *scratch);

        void prune_neighbors(const uint32_t location, std::vector<NeighborEntity> &pool,
                             std::vector<uint32_t> &pruned_list, VamanaWorkSpace *scratch);

        void prune_neighbors(const uint32_t location, std::vector<NeighborEntity> &pool, const uint32_t range,
                             const uint32_t max_candidate_size, const float alpha,
                             std::vector<uint32_t> &pruned_list, VamanaWorkSpace *scratch);

        void occlude_list(const uint32_t location, std::vector<NeighborEntity> &pool, const double alpha,
                          const uint32_t degree, const uint32_t maxc, std::vector<uint32_t> &result,
                          VamanaWorkSpace *scratch,
                          const turbo::flat_hash_set<uint32_t> *const delete_set_ptr);

    private:
        IndexOption _base_option;
        MemVectorStore *_data_store;
        bool _has_built{false};
        bool _dynamic_index{true};

        bool _pq_dist = false;
        bool _use_opq = false;
        size_t _num_pq_chunks = 0;
        uint8_t *_pq_data = nullptr;
        FixedChunkQuantizeTable _pq_table;

        uint32_t _indexingQueueSize;
        uint32_t _indexingRange;
        uint32_t _indexingMaxC;
        float _indexingAlpha;
        bool _data_compacted{true};
        bool _saturate_graph = false;

        size_t _num_frozen_pts{0};

        uint32_t _max_observed_degree = 0;

        location_t _start{0};
        uint32_t _filterIndexingQueueSize;
        bool _conc_consolidate = false; // use _lock while searching

        // Graph related data structures
        std::vector<std::vector<uint32_t>> _final_graph;
        std::vector<std::mutex> _locks;
    };
}

#endif  // TANN_VAMANA_VAMANA_ENGINE_H_
