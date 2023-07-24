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

#ifndef TANN_HNSW_HNSW_INDEX_H_
#define TANN_HNSW_HNSW_INDEX_H_

#include <memory>
#include <shared_mutex>
#include <vector>
#include <queue>
#include <random>
#include "tann/core/types.h"
#include "turbo/container/flat_hash_map.h"
#include "tann/distance/distance_factory.h"
#include "tann/store/vector_set.h"
#include "tann/hnsw/leveled_graph.h"
#include "turbo/container/flat_hash_map.h"
#include "tann/common/natural_number_set.h"
#include "tann/core/query_context.h"
#include "tann/core/index_option.h"
#include "tann/core/index_interface.h"
#include "tann/hnsw/visited_list_pool.h"
#include "turbo/log/logging.h"

namespace tann {

    class HnswIndex : public IndexInterface {
    public:
        HnswIndex() = default;

        ~HnswIndex() override = default;

        //////////////////////////////////////////
        // Let the engine use lazy initialization and aggregate the initialization
        // parameters into the index configuration, which will be more convenient when
        // using the factory class method
        [[nodiscard]] turbo::Status initialize(std::unique_ptr<IndexOption> option) override;

        [[nodiscard]] turbo::Status add_vector(const WriteOption &option, turbo::Span<uint8_t> data_point, const label_type &label) override;

        [[nodiscard]] turbo::Status remove_vector(const label_type &label) override;

        [[nodiscard]] turbo::Status search_vector(QueryContext *qctx) override;

        [[nodiscard]] turbo::Status save_index(const std::string &path, const SerializeOption &option) override;

        [[nodiscard]] turbo::Status load_index(const std::string &path, const SerializeOption &option) override;

        [[nodiscard]] std::size_t size() const override {
            return _cur_element_count;
        }

        [[nodiscard]] std::size_t remove_size() const {
            return _data.deleted_size();
        }



        [[nodiscard]] std::size_t dimension() const override {
            return _vs.dimension;
        }

        [[nodiscard]] bool dynamic() const override {
            return true;
        }

        [[nodiscard]] bool need_model() const override {
            return false;
        }

    private:
        static constexpr location_t MAX_LABEL_OPERATION_LOCKS = 65536;

        std::mutex& get_label_op_mutex(label_type label) const;

        label_type get_external_label(location_t internal_id) const;

        void set_external_label(location_t internal_id, label_type label);

        [[nodiscard]] turbo::Status unmark_deleted_internal(location_t internal_id);

        turbo::ResultStatus<location_t> add_vector(const WriteOption &option, turbo::Span<uint8_t> data_point, label_type label, int level);

        int get_random_level(double reverse_size);

        links_priority_queue search_base_layer(location_t ep_id, location_t loc, int layer);

        turbo::ResultStatus<location_t> mutually_connect_new_element(
                location_t cur_c,
                links_priority_queue &top_candidates,
                int level,
                bool isUpdate);
        void get_neighbors_by_heuristic2(links_priority_queue &top_candidates, size_t M);

        turbo::Status update_point(const WriteOption &option, turbo::Span<uint8_t> data_point, location_t internalId, float updateNeighborProbability);

        std::vector<location_t> get_connections_with_lock(location_t internalId, int level);

        [[nodiscard]] turbo::Status repair_connections_for_update(
                location_t entryPointInternalId,
                location_t dataPointInternalId,
                int dataPointLevel,
                int maxLevel);

        template <bool has_deletions, bool collect_metrics = false>
        links_priority_queue
        search_base_layer_st(location_t ep_id, QueryContext *qctx, size_t ef) const;

        turbo::Status mark_deleted_internal(location_t internalId);
        turbo::Status check_option();
    private:
        std::unique_ptr<HnswIndexOption> _option;
        VectorSpace _vs;
        VectorSet _data;
        std::vector<label_type> _index_scratch;
        mutable std::mutex _label_lookup_lock;  // lock for label_lookup_
        std::unordered_map<label_type , location_t> _label_lookup;

        size_t _max_elements{0};
        size_t _ef_construction{0};
        size_t _ef{ 0 };

        size_t _M{0};
        size_t _maxM{0};

        double _mult{0.0};
        int _max_level{0};

        std::default_random_engine _level_generator;
        std::default_random_engine _update_probability_generator;

        LeveledGraph _final_graph;
        std::mutex _global_lock;
        std::vector<std::mutex> _link_list_locks;

        location_t _enterpoint_node{0};

        mutable std::atomic<size_t> _cur_element_count{0};
        std::unique_ptr<VisitedListPool> _visited_list_pool;
        // Locks operations with element by label value
        mutable std::vector<std::mutex> _label_op_locks;
        std::mutex _deleted_elements_lock;
        std::unordered_set<location_t> _deleted_elements;
        bool _allow_replace_deleted{false};

        mutable std::atomic<size_t> metric_distance_computations{0};
        mutable std::atomic<size_t> metric_hops{0};
    };

    inline std::mutex& HnswIndex::get_label_op_mutex(label_type label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return _label_op_locks[lock_id];
    }

    inline label_type HnswIndex::get_external_label(location_t internal_id) const {
        TLOG_CHECK(internal_id < _index_scratch.size());
        return _index_scratch[internal_id];
    }

    inline void HnswIndex::set_external_label(location_t internal_id, label_type label) {
        TLOG_CHECK(internal_id < _index_scratch.size());
        _index_scratch[internal_id] = label;
    }

}  // namespace tann

#endif  // TANN_HNSW_HNSW_INDEX_H_
