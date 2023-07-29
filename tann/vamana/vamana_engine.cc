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

#include "tann/vamana/vamana_engine.h"
#include "tann/vamana/constants.h"

namespace tann {

    turbo::Status VamaneEngine::add_vector(WorkSpace *ws, location_t location) {
        assert(_has_built);

        // Find and add appropriate graph edges
        auto *vws = reinterpret_cast<VamanaWorkSpace *>(ws);
        std::vector<uint32_t> pruned_list;
        auto r = search_for_point_and_prune(location, _indexingQueueSize, pruned_list, vws);
        if(!r.ok()) {
            return r;
        }
        {

            std::unique_lock guard(_locks[location]);
            _final_graph[location].clear();
            _final_graph[location].reserve((size_t) (_indexingRange * constants::kGraphSlackFactor * 1.05));

            for (auto link: pruned_list) {
                if (_conc_consolidate)
                    if (_data_store->is_deleted(link))
                        continue;
                _final_graph[location].emplace_back(link);
            }
            assert(_final_graph[location].size() <= _indexingRange);
        }

        inter_insert(location, pruned_list, vws);

        return turbo::OkStatus();
    }

    WorkSpace *VamaneEngine::make_workspace() {

    }

    turbo::Status VamaneEngine::remove_vector(location_t lid) {
        TURBO_UNUSED(lid);
        _data_compacted = false;
        return turbo::OkStatus();
    }
    turbo::Status VamaneEngine::search_vector(WorkSpace *ws) {
        auto *vws = reinterpret_cast<VamanaWorkSpace*>(ws);
        if (ws->search_context->k > (uint64_t) ws->search_context->search_list) {
            return turbo::InvalidArgumentError("Set L to a value of at least K");
        }

        if (ws->search_context->search_list > vws->get_L()) {
            vws->resize_for_new_L(ws->search_context->search_list);
        }
        const std::vector<uint32_t> init_ids = get_init_ids();
        auto rs =
                iterate_to_fixed_point(vws->get_L(), init_ids, vws, false);
        if(!rs.ok()) {
            return rs.status();
        }
        return turbo::OkStatus();
    }

    void VamaneEngine::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                      VamanaWorkSpace *scratch) {
        const auto &src_pool = pruned_list;

        assert(!src_pool.empty());

        for (auto des: src_pool) {
            // des.loc is the loc of the neighbors of n
            assert(des < _base_option.max_elements + _num_frozen_pts);
            // des_pool contains the neighbors of the neighbors of n
            std::vector<uint32_t> copy_of_neighbors;
            bool prune_needed = false;
            {
                std::unique_lock guard(_locks[des]);
                auto &des_pool = _final_graph[des];
                if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
                    if (des_pool.size() < (uint64_t) (constants::kGraphSlackFactor * range)) {
                        des_pool.emplace_back(n);
                        prune_needed = false;
                    } else {
                        copy_of_neighbors.reserve(des_pool.size() + 1);
                        copy_of_neighbors = des_pool;
                        copy_of_neighbors.push_back(n);
                        prune_needed = true;
                    }
                }
            } // des lock is released by this point

            if (prune_needed) {
                turbo::flat_hash_set<uint32_t> dummy_visited(0);
                std::vector<NeighborEntity> dummy_pool(0);

                size_t reserveSize = (size_t) (std::ceil(1.05 * constants::kGraphSlackFactor * range));
                dummy_visited.reserve(reserveSize);
                dummy_pool.reserve(reserveSize);

                for (auto cur_nbr: copy_of_neighbors) {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des) {
                        float dist = _data_store->get_distance(des, cur_nbr);
                        dummy_pool.emplace_back(NeighborEntity(cur_nbr, dist));
                        dummy_visited.insert(cur_nbr);
                    }
                }
                std::vector<uint32_t> new_out_neighbors;
                prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);
                {
                    std::unique_lock guard(_locks[des]);

                    _final_graph[des] = new_out_neighbors;
                }
            }
        }
    }

    void VamaneEngine::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list,
                      VamanaWorkSpace *vws) {
        inter_insert(n, pruned_list, _indexingRange, vws);
    }

    void VamaneEngine::prune_neighbors(const uint32_t location, std::vector<NeighborEntity> &pool,
                         std::vector<uint32_t> &pruned_list, VamanaWorkSpace *scratch) {
        prune_neighbors(location, pool, _indexingRange, _indexingMaxC, _indexingAlpha, pruned_list, scratch);
    }

    void VamaneEngine::prune_neighbors(const uint32_t location, std::vector<NeighborEntity> &pool, const uint32_t range,
                         const uint32_t max_candidate_size, const float alpha,
                         std::vector<uint32_t> &pruned_list, VamanaWorkSpace *vws) {
        if (pool.size() == 0) {
            // if the pool is empty, behave like a noop
            pruned_list.clear();
            return;
        }

        _max_observed_degree = (std::max)(_max_observed_degree, range);

        // If using _pq_build, over-write the PQ distances with actual distances
        if (_pq_dist) {
            for (auto &ngh: pool)
                ngh.distance = _data_store->get_distance(ngh.lid, location);
        }

        // sort the pool based on distance to query and prune it with occlude_list
        std::sort(pool.begin(), pool.end());
        pruned_list.clear();
        pruned_list.reserve(range);

        occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list, vws, nullptr);
        assert(pruned_list.size() <= range);

        if (_saturate_graph && alpha > 1) {
            for (const auto &node: pool) {
                if (pruned_list.size() >= range)
                    break;
                if ((std::find(pruned_list.begin(), pruned_list.end(), node.lid) == pruned_list.end()) &&
                    node.lid != location)
                    pruned_list.push_back(node.lid);
            }
        }
    }

    void VamaneEngine::occlude_list(const uint32_t location, std::vector<NeighborEntity> &pool, const double alpha,
                 const uint32_t degree, const uint32_t maxc, std::vector<uint32_t> &result,
                 VamanaWorkSpace *scratch,
                 const turbo::flat_hash_set<uint32_t> *const delete_set_ptr) {
        if (pool.size() == 0)
            return;

        // Truncate pool at maxc and initialize scratch spaces
        assert(std::is_sorted(pool.begin(), pool.end()));
        assert(result.size() == 0);
        if (pool.size() > maxc)
            pool.resize(maxc);
        std::vector<double> &occlude_factor = scratch->occlude_factor();
        // occlude_list can be called with the same scratch more than once by
        // search_for_point_and_add_link through inter_insert.
        occlude_factor.clear();
        // Initialize occlude_factor to pool.size() many 0.0f values for correctness
        occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

        float cur_alpha = 1;
        while (cur_alpha <= alpha && result.size() < degree) {
            // used for MIPS, where we store a value of eps in cur_alpha to
            // denote pruned out entries which we can skip in later rounds.
            double eps = cur_alpha + 0.01f;

            for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
                if (occlude_factor[iter - pool.begin()] > cur_alpha) {
                    continue;
                }
                // Set the entry to float::max so that is not considered again
                occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
                // Add the entry to the result if its not been deleted, and doesn't
                // add a self loop
                if (delete_set_ptr == nullptr || delete_set_ptr->find(iter->lid) == delete_set_ptr->end()) {
                    if (iter->lid != location) {
                        result.push_back(iter->lid);
                    }
                }

                // Update occlude factor for points from iter+1 to pool.end()
                for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
                    auto t = iter2 - pool.begin();
                    if (occlude_factor[t] > alpha)
                        continue;

                    bool prune_allowed = true;
                    if (!prune_allowed)
                        continue;

                    float djk = _data_store->get_distance(iter2->lid, iter->lid);
                    if (_base_option.metric == tann::MetricType::METRIC_L2 || _base_option.metric == tann::MetricType::METRIC_COSINE) {
                        occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
                                                       : std::max(occlude_factor[t], iter2->distance / djk);
                    } else if (_base_option.metric == tann::MetricType::METRIC_IP) {
                        // Improvization for flipping max and min dist for MIPS
                        double x = -iter2->distance;
                        double y = -djk;
                        if (y > cur_alpha * x) {
                            occlude_factor[t] = std::max(occlude_factor[t], eps);
                        }
                    }
                }
            }
            cur_alpha *= 1.2f;
        }
    }

    turbo::ResultStatus<std::pair<uint32_t, uint32_t>> VamaneEngine::iterate_to_fixed_point(const uint32_t Lsize, const std::vector<uint32_t> &init_ids, VamanaWorkSpace *vws,
            bool search_invocation) {
        std::vector<NeighborEntity> &expanded_nodes = vws->pool();
        auto &best_L_nodes = vws->best_l_nodes();
        best_L_nodes.reserve(Lsize);
        turbo::flat_hash_set<uint32_t> &inserted_into_pool_rs = vws->inserted_into_pool_rs();
        turbo::dynamic_bitset<> &inserted_into_pool_bs = vws->inserted_into_pool_bs();
        std::vector<uint32_t> &id_scratch = vws->id_scratch();
        std::vector<float> &dist_scratch = vws->dist_scratch();
        assert(id_scratch.size() == 0);

        auto aligned_query = to_span<float>(vws->query_view);

        float *query_float = nullptr;
        float *query_rotated = nullptr;
        float *pq_dists = nullptr;
        uint8_t *pq_coord_scratch = nullptr;
        // Intialize PQ related scratch to use PQ based distances
        if (_pq_dist) {
            // Get scratch spaces
            auto *pq_query_scratch = vws->pq_scratch();
            query_float = pq_query_scratch->aligned_query_float;
            query_rotated = pq_query_scratch->rotated_query;
            pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;

            // Copy query vector to float and then to "rotated" query
            for (size_t d = 0; d < _base_option.dimension; d++) {
                query_float[d] = (float) aligned_query[d];
            }
            pq_query_scratch->set(_base_option.dimension, vws->query_view);

            // center the query and rotate if we have a rotation matrix
            _pq_table.preprocess_query(query_rotated);
            _pq_table.populate_chunk_distances(query_rotated, pq_dists);

            pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;
        }

        if (!expanded_nodes.empty() || !id_scratch.empty()) {
            return turbo::InternalError("ERROR: [{}:{}]Clear scratch space before passing.", __FILE__, __LINE__);
        }

        // Decide whether to use bitset or robin set to mark visited nodes
        auto total_num_points = _base_option.max_elements + _num_frozen_pts;
        bool fast_iterate = total_num_points <= constants::kMaxPointsForUsingBitset;

        if (fast_iterate) {
            if (inserted_into_pool_bs.size() < total_num_points) {
                // hopefully using 2X will reduce the number of allocations.
                auto resize_size =
                        2 * total_num_points > constants::kMaxPointsForUsingBitset ? constants::kMaxPointsForUsingBitset : 2 *
                                                                                                           total_num_points;
                inserted_into_pool_bs.resize(resize_size);
            }
        }

        // Lambda to determine if a node has been visited
        auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
            return fast_iterate ? inserted_into_pool_bs[id] == 0
                                : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
        };

        // Lambda to batch compute query<-> node distances in PQ space
        auto compute_dists = [this, pq_coord_scratch, pq_dists](const std::vector<uint32_t> &ids,
                                                                std::vector<float> &dists_out) {
            tann::Quantize::aggregate_coords(ids, this->_pq_data, this->_num_pq_chunks, pq_coord_scratch);
            tann::Quantize::pq_dist_lookup(pq_coord_scratch, ids.size(), this->_num_pq_chunks, pq_dists, dists_out);
        };

        // Initialize the candidate pool with starting points
        for (auto id: init_ids) {
            if (id >= _base_option.max_elements + _num_frozen_pts) {
                TLOG_ERROR("Out of range loc found as an edge : {}",id);
                return turbo::InternalError(std::string("Wrong loc") + std::to_string(id));
            }

            if (is_not_visited(id)) {
                if (fast_iterate) {
                    inserted_into_pool_bs[id] = 1;
                } else {
                    inserted_into_pool_rs.insert(id);
                }

                float distance;
                if (_pq_dist) {
                    Quantize::pq_dist_lookup(pq_coord_scratch, 1, this->_num_pq_chunks, pq_dists, &distance);
                } else {
                    distance = _data_store->get_distance(to_span<uint8_t>(aligned_query), id);
                }
                NeighborEntity nn = NeighborEntity(distance, id);
                best_L_nodes.insert(nn);
            }
        }

        uint32_t hops = 0;
        uint32_t cmps = 0;

        while (best_L_nodes.has_unexpanded_node()) {
            auto nbr = best_L_nodes.closest_unexpanded();
            auto n = nbr.lid;

            // Add node to expanded nodes to create pool for prune later
            if (!search_invocation) {
                expanded_nodes.emplace_back(nbr);
            }

            // Find which of the nodes in des have not been visited before
            id_scratch.clear();
            dist_scratch.clear();
            {
                if (_dynamic_index)
                    _locks[n].lock();
                for (auto id: _final_graph[n]) {
                    assert(id < _base_option.max_elements + _num_frozen_pts);

                    if (is_not_visited(id)) {
                        id_scratch.push_back(id);
                    }
                }

                if (_dynamic_index)
                    _locks[n].unlock();
            }

            // Mark nodes visited
            for (auto id: id_scratch) {
                if (fast_iterate) {
                    inserted_into_pool_bs[id] = 1;
                } else {
                    inserted_into_pool_rs.insert(id);
                }
            }

            // Compute distances to unvisited nodes in the expansion
            if (_pq_dist) {
                assert(dist_scratch.capacity() >= id_scratch.size());
                compute_dists(id_scratch, dist_scratch);
            } else {
                assert(dist_scratch.size() == 0);
                for (size_t m = 0; m < id_scratch.size(); ++m) {
                    uint32_t id = id_scratch[m];

                    if (m + 1 < id_scratch.size()) {
                        auto nextn = id_scratch[m + 1];
                        //_data_store->prefetch_vector(nextn);
                    }

                    dist_scratch.push_back(_data_store->get_distance(to_span<uint8_t>(aligned_query), id));
                }
            }
            cmps += (uint32_t) id_scratch.size();

            // Insert <id, dist> pairs into the pool of candidates
            for (size_t m = 0; m < id_scratch.size(); ++m) {
                best_L_nodes.insert(NeighborEntity(dist_scratch[m], id_scratch[m]));
            }
        }
        return std::make_pair(hops, cmps);
    }

    turbo::Status VamaneEngine::search_for_point_and_prune(int location, uint32_t Lindex,
                                                  std::vector<uint32_t> &pruned_list,
                                                  VamanaWorkSpace *vws) {
        const std::vector<uint32_t> init_ids = get_init_ids();
        iterate_to_fixed_point(Lindex, init_ids, vws, false);

        auto &pool = vws->pool();

        for (uint32_t i = 0; i < pool.size(); i++) {
            if (pool[i].lid == (uint32_t) location) {
                pool.erase(pool.begin() + i);
                i--;
            }
        }

        if (pruned_list.size() > 0) {
            return turbo::InternalError("ERROR: non-empty pruned_list passed");
        }

        prune_neighbors(location, pool, pruned_list, vws);

        assert(!pruned_list.empty());
        assert(_final_graph.size() == _base_option.max_elements + _num_frozen_pts);
    }

    std::vector<uint32_t> VamaneEngine::get_init_ids() {
        std::vector<uint32_t> init_ids;
        init_ids.reserve(1 + _num_frozen_pts);

        init_ids.emplace_back(_start);

        for (uint32_t frozen = (uint32_t) _base_option.max_elements;
             frozen < _base_option.max_elements + _num_frozen_pts; frozen++) {
            if (frozen != _start) {
                init_ids.emplace_back(frozen);
            }
        }

        return init_ids;
    }

}