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

#include "tann/hnsw/hnsw_engine.h"
#include "tann/common/utility.h"

namespace tann {
    turbo::Status HnswEngine::initialize(const IndexOption& base_option, const std::any &option, MemVectorStore *store) {
        _data_store = store;
        _base_option = base_option;
        _option = std::any_cast<HnswIndexOption>(option);
        _level_generator.seed(_option.random_seed);
        _maxM = _option.m;

        _final_graph.initialize(_base_option.max_elements, _maxM);
        _visited_list_pool = std::make_unique<VisitedListPool>(1, _base_option.max_elements);
        std::vector<std::mutex> temp(_base_option.max_elements);
        _link_list_locks = std::move(temp);
        _mult = 1 / log(1.0 * static_cast<double >(_option.m));
        return turbo::OkStatus();
    }
    turbo::Status HnswEngine::add_vector(WorkSpace*ws, location_t lid) {
        auto hws = reinterpret_cast<HnswWorkSpace*>(ws);

        if(ws->is_update) {
            return update_vector_internal(hws, lid);
        } else {
            return add_vector_internal(hws, lid);
        }
    }

    WorkSpace* HnswEngine::make_workspace() {
        return new HnswWorkSpace();
    }

    void HnswEngine::setup_workspace(WorkSpace*ws)  {
        auto *hnsw_ws = reinterpret_cast<HnswWorkSpace *>(ws);
        hnsw_ws->search_l = std::max(_option.ef, hnsw_ws->search_context->k);
    }

    turbo::Status HnswEngine::remove_vector(location_t lid) {
        return turbo::OkStatus();
    }

    turbo::Status HnswEngine::search_vector(WorkSpace *base_ws) {
        if (_data_store->size() == 0) {
            return turbo::OkStatus();
        }
        auto *hnsw_ws = reinterpret_cast<HnswWorkSpace *>(base_ws);
        location_t currObj = _enterpoint_node;
        auto query_data = to_span<uint8_t>(hnsw_ws->query_view);
        distance_type curdist = _data_store->get_distance(query_data, _enterpoint_node);
        // travel all level > 0 (only 1 level) and find nearest ep
        for (int level = _max_level; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto node = _final_graph.mutable_node(currObj, level);
                size_t size = node.size();
                metric_hops++;
                metric_distance_computations += size;

                for (int i = 0; i < size; i++) {
                    location_t cand = node[i];
                    if (cand > _base_option.max_elements)
                        return turbo::InternalError("cand error");
                    auto d = _data_store->get_distance(query_data, cand);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        hnsw_ws->top_candidates.clear();
        hnsw_ws->top_candidates.reserve(hnsw_ws->search_l);
        hnsw_ws->candidate_set.clear();
        hnsw_ws->candidate_set.reserve(2 * hnsw_ws->search_l);
        if (_data_store->deleted_size()) {
            search_base_layer_st<true, true>(currObj, hnsw_ws);
        } else {
            search_base_layer_st<false, true>(currObj, hnsw_ws);
        }
        auto &top_candidates = hnsw_ws->top_candidates;
        while (top_candidates.size() > hnsw_ws->search_l) {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) {
            auto rez = top_candidates.top();
            rez.label = _data_store->get_label(rez.lid).value();
            hnsw_ws->best_l_nodes.insert(rez);
            top_candidates.pop();
        }
        return turbo::OkStatus();
    }

    template<bool has_deletions, bool collect_metrics>
    void HnswEngine::search_base_layer_st(location_t ep_id, HnswWorkSpace *hws) const {
        VisitedList *vl = _visited_list_pool->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        auto ef = hws->search_l;
        auto &top_candidates = hws->top_candidates;
        auto &candidate_set = hws->candidate_set;

        auto isIdAllowed = hws->search_context->is_allowed;
        distance_type lowerBound;
        if ((!has_deletions || !_data_store->is_deleted(ep_id)) &&
            ((!isIdAllowed) || (*isIdAllowed)(_data_store->get_label(ep_id).value()))) {
            auto data_point = to_span<uint8_t>(hws->query_view);
            distance_type dist = _data_store->get_distance(data_point, ep_id);
            lowerBound = dist;
            top_candidates.insert(dist, ep_id);
            candidate_set.insert(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<distance_type>::max();
            candidate_set.insert(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            auto current_node_pair = candidate_set.top();
            // stop search condition
            // 1. current node distance > the largest distance that has got and result size reach limit
            // 2. current node distance > the largest distance that has got and (no filter and no deletions)
            if ((-current_node_pair.distance) > lowerBound &&
                (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
                break;
            }
            candidate_set.pop();

            location_t current_node_id = current_node_pair.lid;
            auto data = _final_graph.const_node(current_node_id, 0);
            size_t size = data.size();
            //bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }
            for (size_t j = 1; j < size; j++) {
                location_t candidate_id = data[j];
                if (visited_array[candidate_id] != visited_array_tag) {
                    visited_array[candidate_id] = visited_array_tag;
                    auto data_point = to_span<uint8_t>(hws->query_view);
                    distance_type dist = _data_store->get_distance(data_point, candidate_id);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.insert(-dist, candidate_id);

                        if ((!has_deletions || !_data_store->is_deleted(candidate_id)) &&
                            ((!isIdAllowed) || (*isIdAllowed)(_data_store->get_label(candidate_id).value())))
                            top_candidates.insert(dist, candidate_id);

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().distance;
                    }
                }
            }
        }

        _visited_list_pool->releaseVisitedList(vl);
    }

    turbo::Status HnswEngine::add_vector_internal(HnswWorkSpace *hws,location_t lid) {

        std::unique_lock<std::mutex> lock_el(_link_list_locks[lid]);
        int cur_level = get_random_level(_mult);

        //_element_levels[cur_c] = cur_level;
        std::unique_lock<std::mutex> templock(_global_lock);
        int max_level_copy = _max_level;
        if (cur_level <= max_level_copy)
            templock.unlock();
        location_t currObj = _enterpoint_node;
        location_t enterpoint_copy = _enterpoint_node;

        auto r = _final_graph.setup_location(lid, cur_level);
        if (!r.ok()) {
            return r;
        }

        // not first one
        if (TURBO_LIKELY(currObj != constants::kUnknownLocation)) {
            if (cur_level < max_level_copy) {
                distance_type curdist = _data_store->get_distance(lid, currObj);
                for (int l_level = max_level_copy; l_level > cur_level; l_level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        std::unique_lock<std::mutex> lock(_link_list_locks[currObj]);
                        auto node = _final_graph.mutable_node(currObj, l_level);
                        size_t size = node.size();
                        TLOG_TRACE("try node {} level {} links {}", currObj, l_level, size);
                        for (int i = 0; i < size; i++) {
                            location_t cand = node[i];
                            if (cand > _base_option.max_elements)
                                return turbo::DataLossError("cand error");
                            auto d = _data_store->get_distance(lid, cand);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = _data_store->is_deleted(enterpoint_copy);
            for (int l_level = std::min(cur_level, max_level_copy); l_level >= 0; l_level--) {
                if (l_level > max_level_copy || l_level < 0)  // possible?
                    return turbo::OutOfRangeError("Level error");
                auto &top_candidates = hws->top_candidates;
                auto &candidate_set = hws->candidate_set;
                candidate_set.clear();
                candidate_set.reserve(_option.ef_construction + 1);
                top_candidates.clear();
                top_candidates.reserve(_option.ef_construction + 1);
                search_base_layer(hws, currObj, lid, l_level);
                if (epDeleted) {
                    top_candidates.insert(_data_store->get_distance(lid, enterpoint_copy), enterpoint_copy);
                }
                auto rs = mutually_connect_new_element(hws, lid, l_level, false);
                if (!rs.ok()) {
                    return rs.status();
                }
                currObj = rs.value();
            }
        } else {
            // Do nothing for the first element
            _enterpoint_node = 0;
            _max_level = cur_level;
        }

        // Releasing lock for the maximum level
        if (cur_level > max_level_copy) {
            _enterpoint_node = lid;
            _max_level = cur_level;
        }
        return turbo::OkStatus();
    }

    int HnswEngine::get_random_level(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(_level_generator)) * reverse_size;
        return (int) r;
    }

    turbo::Status HnswEngine::update_vector_internal(HnswWorkSpace *hws,location_t lid) {
        // update the feature vector associated with existing point with new vector
        int maxLevelCopy = _max_level;
        location_t entryPointCopy = _enterpoint_node;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == lid && _data_store->size() == 1)
            return turbo::OkStatus();

        int elemLevel = _final_graph.level(lid);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<location_t> sCand;
            std::unordered_set<location_t> sNeigh;
            std::vector<location_t> listOneHop = get_connections_with_lock(lid, layer);
            if (listOneHop.empty())
                continue;

            sCand.insert(lid);

            for (auto &&elOneHop: listOneHop) {
                sCand.insert(elOneHop);
                sNeigh.insert(elOneHop);

                std::vector<location_t> listTwoHop = get_connections_with_lock(elOneHop, layer);
                for (auto &&elTwoHop: listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto &&neigh: sNeigh) {
                // if (neigh == internalId)
                //     continue;

                auto &candidates = hws->top_candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() -1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(_option.ef_construction, size);
                candidates.clear();
                candidates.reserve(elementsToKeep);
                for (auto &&cand: sCand) {
                    if (cand == neigh)
                        continue;

                    distance_type distance = _data_store->get_distance(neigh, cand);
                    candidates.insert(distance, cand);
                }

                // Retrieve neighbours using heuristic and set connections.
                get_neighbors_by_heuristic(hws, _final_graph.capacity_for_level(layer));

                {
                    std::unique_lock<std::mutex> lock(_link_list_locks[neigh]);
                    auto ll_cur = _final_graph.mutable_node(neigh, layer);
                    size_t candSize = candidates.size();
                    ll_cur.set_size(candSize);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        ll_cur.set_link(idx, candidates.top().lid);
                        candidates.pop();
                    }
                }
            }
        }

        return repair_connections_for_update(hws, entryPointCopy, lid, elemLevel, maxLevelCopy);
    }

    turbo::Status HnswEngine::repair_connections_for_update(HnswWorkSpace *hws,
            location_t entryPointInternalId,
            location_t dataPointInternalId,
            int dataPointLevel,
            int maxLevel) {
        location_t currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            distance_type curdist = _data_store->get_distance(dataPointInternalId, currObj);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lock(_link_list_locks[currObj]);
                    auto data = _final_graph.mutable_node(currObj, level);
                    size_t size = data.size();
                    for (size_t i = 0; i < size; i++) {
                        location_t cand = data[i];
                        distance_type d = _data_store->get_distance(dataPointInternalId, cand);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            return turbo::InternalError("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            auto &top_candidates = hws->top_candidates;
            auto & candidate_set= hws->candidate_set;
            top_candidates.clear();
            top_candidates.reserve(_option.ef_construction);
            candidate_set.clear();
            candidate_set.reserve(_option.ef_construction);
            search_base_layer(hws, currObj, dataPointInternalId, level);
            candidate_set.swap(top_candidates);
            top_candidates.clear();
            top_candidates.reserve(_option.ef_construction);
            for(size_t i = 0; i < candidate_set.size(); ++i) {
                if(candidate_set[i].lid != dataPointInternalId) {
                    top_candidates.insert(candidate_set[i]);
                }
            }
            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (!top_candidates.empty()) {
                bool epDeleted = _data_store->is_deleted(entryPointInternalId);
                if (epDeleted) {
                    top_candidates.insert(_data_store->get_distance(dataPointInternalId, entryPointInternalId),
                                                  entryPointInternalId);
                }

                auto r = mutually_connect_new_element(hws, dataPointInternalId, level, true);
                if (!r.ok()) {
                    return r.status();
                }
                currObj = r.value();
            }
        }
        return turbo::OkStatus();
    }

    void HnswEngine::search_base_layer(HnswWorkSpace *hws, location_t ep_id, location_t loc, int layer) {
        VisitedList *vl = _visited_list_pool->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        auto &top_candidates = hws->top_candidates;
        auto &candidateSet = hws->candidate_set;
        distance_type lowerBound;
        if (!_data_store->is_deleted(ep_id)) {
            distance_type dist = _data_store->get_distance(loc, ep_id);
            top_candidates.insert(dist, ep_id);
            lowerBound = dist;
            candidateSet.insert(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<distance_type>::max();
            candidateSet.insert(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            auto curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.distance) > lowerBound && top_candidates.size() == _option.ef_construction) {
                break;
            }
            candidateSet.pop();

            location_t curNodeNum = curr_el_pair.lid;

            std::unique_lock<std::mutex> lock(_link_list_locks[curNodeNum]);

            auto data = _final_graph.mutable_node(curNodeNum, layer);

            size_t size = data.size();

            for (size_t j = 0; j < size; j++) {
                location_t candidate_id = data[j];
                if (visited_array[candidate_id] == visited_array_tag) {
                    continue;
                }
                visited_array[candidate_id] = visited_array_tag;

                distance_type dist1 = _data_store->get_distance(loc, candidate_id);
                candidateSet.insert(-dist1, candidate_id);
                if (!_data_store->is_deleted(candidate_id))
                    top_candidates.insert(dist1, candidate_id);
                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().distance;
            }
        }
        _visited_list_pool->releaseVisitedList(vl);
    }


    turbo::ResultStatus<location_t>
    HnswEngine::mutually_connect_new_element(HnswWorkSpace *hws,location_t cur_c, int level,
                                            bool isUpdate) {

        auto node = _final_graph.mutable_node(cur_c, level);
        size_t Mcurmax = node.capacity();
        TLOG_TRACE("mutually_connect_new_element Mcurmax {} {} {} ", Mcurmax, cur_c, level);
        get_neighbors_by_heuristic(hws, _option.m);
        if (hws->top_candidates.size() > _option.m)
            return turbo::OutOfRangeError("Should be not be more than size:{} _M:{} candidates returned by the heuristic", hws->top_candidates.size(), _option.m);


        auto &candidate_set = hws->candidate_set;
        auto &top_candidates = hws->top_candidates;
        candidate_set.swap(top_candidates);
        //
        location_t next_closest_entry_point = candidate_set[0].lid;

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock<std::mutex> lock(_link_list_locks[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }

            if (node.size() && !isUpdate) {
                return turbo::InternalError("The newly inserted element should have blank link list");
            }
            node.set_size(candidate_set.size());
            for (size_t idx = 0; idx < candidate_set.size(); idx++) {
                if (node[idx] && !isUpdate)
                    return turbo::InternalError("Possible memory corruption");
                if (level > node.level())
                    return turbo::InternalError("Trying to make a link on a non-existent level");

                node.set_link(idx, candidate_set[idx].lid);
            }
        }

        auto loop_size = candidate_set.size();
        for (size_t idx = 0; idx < loop_size; idx++) {
            std::unique_lock<std::mutex> lock(_link_list_locks[candidate_set[idx].lid]);

            auto other_node = _final_graph.mutable_node(candidate_set[idx].lid, level);
            TLOG_TRACE("other_node {} {}", candidate_set[idx].lid, level);
            size_t sz_link_list_other = other_node.size();

            if (sz_link_list_other > Mcurmax)
                return turbo::InternalError("Bad value of sz_link_list_other");
            if (candidate_set[idx].lid == cur_c)
                return turbo::InternalError("Trying to connect an element to itself");
            if (level > other_node.level())
                return turbo::InternalError("Trying to make a link on a non-existent level");

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (other_node[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    other_node.at(sz_link_list_other) = cur_c;
                    other_node.set_size(sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    distance_type d_max = _data_store->get_distance(cur_c, candidate_set[idx].lid);
                    // Heuristic:
                    top_candidates.clear();
                    top_candidates.reserve(Mcurmax + 1);
                    top_candidates.insert(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        top_candidates.insert(_data_store->get_distance(other_node[j], candidate_set[idx].lid), other_node[j]);
                    }

                    get_neighbors_by_heuristic(hws, Mcurmax);

                    int indx = 0;
                    while (!top_candidates.empty()) {
                        other_node.set_link(indx, top_candidates.top().lid);
                        top_candidates.pop();
                        indx++;
                    }

                    other_node.set_size(indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        distance_type d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    std::vector<location_t> HnswEngine::get_connections_with_lock(location_t lid, int level) {
        std::unique_lock<std::mutex> lock(_link_list_locks[lid]);
        auto node = _final_graph.mutable_node(lid, level);
        size_t size = node.size();
        std::vector<location_t> result(size);
        auto list = node.links();
        memcpy(result.data(), list.data(), size * sizeof(location_t));
        return result;
    }
    void HnswEngine::get_neighbors_by_heuristic(HnswWorkSpace *ws, const size_t M) {
        auto &top_candidates = ws->top_candidates;
        if (top_candidates.size() < M) {
            return;
        }

        //auto &candidate_set = ws->candidate_set;
        auto &return_list= ws->return_list;
        return_list.clear();

        auto size = top_candidates.size();
        for(size_t i = 0; i < size; ++i) {
            if (return_list.size() >= M)
                break;
            auto curent_pair = top_candidates[i];
            distance_type dist_to_query = curent_pair.distance;
            bool good = true;

            for (auto second_pair : return_list) {
                distance_type curdist = _data_store->get_distance(second_pair.second, curent_pair.lid);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.emplace_back(curent_pair.distance, curent_pair.lid);
            }
        }
        top_candidates.clear();
        top_candidates.reserve(M);
        for (std::pair<distance_type, location_t> curent_pair: return_list) {
            top_candidates.insert(curent_pair.first, curent_pair.second);
        }
    }

    turbo::Status HnswEngine::save(turbo::SequentialWriteFile *file) {
        /// index status
        auto r = write_binary_pod(*file, _enterpoint_node);
        if (!r.ok()) {
            return r;
        }
        r = write_binary_pod(*file, _max_level);
        if (!r.ok()) {
            return r;
        }

        r = write_binary_pod(*file, _mult);
        if (!r.ok()) {
            return r;
        }

        /// save graph
        r = _final_graph.save(*file);
        if (!r.ok()) {
            return r;
        }

        return turbo::OkStatus();
    }

    turbo::Status HnswEngine::load(turbo::SequentialReadFile *file) {

        /// index status
        auto r = read_binary_pod(*file, _enterpoint_node);
        if (!r.ok()) {
            return r;
        }
        r = read_binary_pod(*file, _max_level);
        if (!r.ok()) {
            return r;
        }

        r = read_binary_pod(*file, _mult);
        if (!r.ok()) {
            return r;
        }
        // graph
        r = _final_graph.load(*file);
        if (!r.ok()) {
            return r;
        }
        return turbo::OkStatus();
    }

    template void
    HnswEngine::search_base_layer_st<true, true>(location_t ep_id, HnswWorkSpace *qctx) const;

    template void
    HnswEngine::search_base_layer_st<false, true>(location_t ep_id, HnswWorkSpace *qctx) const;

}  // namespace tann

tann::Engine* HnswIndexCreator(const std::any &option) {
    auto ptr = new tann::HnswEngine();
    TLOG_INFO(" engine create done {}", turbo::Ptr(ptr));
    return ptr;
}
ENGINE_REGISTER(tann::EngineType::ENGINE_HNSW, HnswIndexCreator);
