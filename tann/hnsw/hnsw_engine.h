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


#ifndef TANN_HNSW_HNSW_ENGINE_H_
#define TANN_HNSW_HNSW_ENGINE_H_

#include <random>
#include "tann/core/engine.h"
#include "tann/hnsw/leveled_graph.h"
#include "tann/hnsw/visited_list_pool.h"

namespace tann {

    class HnswEngine : public Engine {

    public:
        ~HnswEngine() override = default;

        turbo::Status initialize(const std::any &option, VectorSpace *vs, MemVectorStore *store) override;

        turbo::Status add_vector(const WriteOption &options, location_t lid, bool ever_added) override;

        turbo::Status remove_vector(location_t lid);

        turbo::Status search_vector(QueryContext *qctx) override;

        virtual turbo::Status save(turbo::SequentialWriteFile *file) = 0;

        virtual turbo::Status load(turbo::SequentialReadFile *file) = 0;

    private:
        turbo::Status add_vector_internal(const WriteOption &options, location_t lid);

        turbo::Status update_vector_internal(const WriteOption &options, location_t lid);

        std::vector<location_t> get_connections_with_lock(location_t lid, int level);

        void get_neighbors_by_heuristic2(links_priority_queue &top_candidates, const size_t M);

        turbo::Status repair_connections_for_update(location_t entryPointInternalId, location_t dataPointInternalId,
                                                    int dataPointLevel, int maxLevel);

        links_priority_queue search_base_layer(location_t ep_id, location_t loc, int layer);

        turbo::ResultStatus<location_t>
        mutually_connect_new_element(location_t cur_c, links_priority_queue &top_candidates, int level,
                                     bool isUpdate);

        int get_random_level(double reverse_size);

        template<bool has_deletions, bool collect_metrics>
        links_priority_queue search_base_layer_st(location_t ep_id, QueryContext *qctx, size_t ef) const;
    private:
        HnswIndexOption _option;
        MemVectorStore *_data_store;
        double _mult{0.0};
        int _max_level{0};
        size_t _maxM{0};
        location_t _enterpoint_node{0};
        LeveledGraph _final_graph;

        std::mutex _global_lock;
        std::vector<std::mutex> _link_list_locks;
        std::unique_ptr<VisitedListPool> _visited_list_pool;
        std::default_random_engine _level_generator;

        mutable std::atomic<size_t> metric_distance_computations{0};
        mutable std::atomic<size_t> metric_hops{0};
    };
}  // namespace tann

#endif // TANN_HNSW_HNSW_ENGINE_H_
