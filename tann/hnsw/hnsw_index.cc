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

#include "tann/hnsw/hnsw_index.h"
#include "tann/io/utility.h"

namespace tann {
    turbo::Status HnswIndex::check_option() {
        // check options
        if (_option.dimension == 0) {
            return turbo::InvalidArgumentError("invalid dimension : {}, please set dimension", _option.dimension);
        }
        return turbo::OkStatus();
    }


    turbo::Status HnswIndex::initialize(std::unique_ptr<IndexOption> option) {
        auto *option_ptr = reinterpret_cast<HnswIndexOption *>(option.release());
        _option = *option_ptr;

        // check parameters
        auto r = check_option();
        if (!r.ok()) {
            return r;
        }

        r = _vs.init(_option.dimension, _option.metric, _option.data_type);
        if (!r.ok()) {
            return r;
        }

        r = _data.init(&_vs, _option.batch_size);
        if (!r.ok()) {
            return r;
        }

        _level_generator.seed(_option.random_seed);
        _update_probability_generator.seed(_option.random_seed + 1);

        _maxM = _option.m;

        _index_scratch.resize(_option.max_elements);
        _final_graph.initialize(_option.max_elements, _maxM);
        _visited_list_pool = std::make_unique<VisitedListPool>(1, _option.max_elements);
        std::vector<std::mutex> temp(_option.max_elements);
        _link_list_locks = std::move(temp);
        std::vector<std::mutex> temp1(MAX_LABEL_OPERATION_LOCKS);
        _label_op_locks = std::move(temp1);
        // initializations for special treatment of the first node
        _enterpoint_node = -1;
        _max_level = -1;
        _cur_element_count = 0;

        _mult = 1 / log(1.0 * static_cast<double >(_option.m));
        return turbo::OkStatus();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // label of vector status                                        | operation     | new internal location
    // ---------------------------------------------------------------------------------------------------
    // 1. label exists                                               | update        | no
    // 2. label exists mark delete                                   | update        | no
    // 3. label not exits and not allow replace_deleted              | add new       | yes
    // 4. label not exits and allow replace_deleted and no vacant    | add new       | yes
    // 5. label not exits and allow replace_deleted and has vacant   | update        | no
    // ---------------------------------------------------------------------------------------------------
    //                                     |  |
    //                                     |  |
    //                                    \    /
    //                                     \  /
    //                                      \/
    // ---------------------------------------------------------------------------------------------------
    // label of vector status                                        | operation     | new internal location
    // ---------------------------------------------------------------------------------------------------
    //
    turbo::Status
    HnswIndex::add_vector(const WriteOption &option, turbo::Span<uint8_t> data_point, const label_type &label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(get_label_op_mutex(label));
        if (!option.replace_deleted) {
            auto r = add_vector(option, data_point, label, -1);
            return r.status();
        }
        // check if there is vacant place
        location_t internal_id_replaced ;
        bool is_vacant_place = _data.get_vacant(internal_id_replaced);
        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            auto r = add_vector(option, data_point, label, -1);
            return r.status();
        } else {
            // we assume that there are no concurrent operations on deleted element
            label_type label_replaced = get_external_label(internal_id_replaced);
            set_external_label(internal_id_replaced, label);

            std::unique_lock<std::mutex> lock_table(_label_lookup_lock);
            _label_lookup.erase(label_replaced);
            _label_lookup[label] = internal_id_replaced;
            lock_table.unlock();

            auto r = unmark_deleted_internal(internal_id_replaced);
            if (!r.ok()) {
                return r;
            }
            return update_point(option, data_point, internal_id_replaced, 1.0);
        }
    }

    [[nodiscard]] turbo::Status HnswIndex::search_vector(QueryContext *qctx) {
        std::priority_queue<std::pair<distance_type, label_type >> result;
        if (_cur_element_count == 0) {
            return turbo::OkStatus();
        }

        location_t currObj = _enterpoint_node;
        auto query_data = to_span<uint8_t>(qctx->raw_query);
        distance_type curdist = _data.get_distance(query_data, _enterpoint_node);

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
                    if (cand < 0 || cand > _option.max_elements)
                        return turbo::InternalError("cand error");
                    auto d = _data.get_distance(query_data, cand);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        links_priority_queue top_candidates;
        if (_data.deleted_size()) {
            top_candidates = search_base_layer_st<true, true>(currObj, qctx, std::max(_option.ef, qctx->k));
        } else {
            top_candidates = search_base_layer_st<false, true>(currObj, qctx, std::max(_option.ef, qctx->k));
        }

        while (top_candidates.size() > qctx->k) {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) {
            std::pair<distance_type, location_t> rez = top_candidates.top();
            result.emplace(std::pair<distance_type, location_t>(rez.first, get_external_label(rez.second)));
            top_candidates.pop();
        }
        return turbo::OkStatus();
    }

    template<bool has_deletions, bool collect_metrics>
    links_priority_queue HnswIndex::search_base_layer_st(location_t ep_id, QueryContext *qctx, size_t ef) const {
        VisitedList *vl = _visited_list_pool->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        links_priority_queue top_candidates;
        links_priority_queue candidate_set;

        auto isIdAllowed = qctx->is_allowed;
        distance_type lowerBound;
        if ((!has_deletions || !_data.is_deleted(ep_id)) &&
            ((!isIdAllowed) || (*isIdAllowed)(get_external_label(ep_id)))) {
            auto data_point = to_span<uint8_t>(qctx->raw_query);
            distance_type dist = _data.get_distance(data_point, ep_id);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<distance_type>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<distance_type, location_t> current_node_pair = candidate_set.top();

            if ((-current_node_pair.first) > lowerBound &&
                (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
                break;
            }
            candidate_set.pop();

            location_t current_node_id = current_node_pair.second;
            auto data = _final_graph.const_node(current_node_id, 0);
            size_t size = data.size();
            //bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

            for (size_t j = 1; j <= size; j++) {
                location_t candidate_id = data[j];
                if (visited_array[candidate_id] != visited_array_tag) {
                    visited_array[candidate_id] = visited_array_tag;
                    auto data_point = to_span<uint8_t>(qctx->raw_query);
                    distance_type dist = _data.get_distance(data_point, candidate_id);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);

                        if ((!has_deletions || !_data.is_deleted(candidate_id)) &&
                            ((!isIdAllowed) || (*isIdAllowed)(get_external_label(candidate_id))))
                            top_candidates.emplace(dist, candidate_id);

                        if (top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        _visited_list_pool->releaseVisitedList(vl);
        return top_candidates;
    }

    turbo::Status HnswIndex::remove_vector(const label_type &label) {
        std::unique_lock<std::mutex> lock_label(get_label_op_mutex(label));

        std::unique_lock<std::mutex> lock_table(_label_lookup_lock);
        auto search = _label_lookup.find(label);
        if (search == _label_lookup.end()) {
            return turbo::NotFoundError("Label not found label: {}", label);
        }
        location_t internalId = search->second;
        lock_table.unlock();

        return mark_deleted_internal(internalId);
    }

    turbo::Status HnswIndex::mark_deleted_internal(location_t internalId) {
        assert(internalId < _cur_element_count);
        if (!_data.is_deleted(internalId)) {
            _data.mark_deleted(internalId);
        } else {
            return turbo::AlreadyExistsError("The requested to delete element is already deleted");
        }
        return turbo::OkStatus();
    }

    turbo::ResultStatus<location_t>
    HnswIndex::add_vector(const WriteOption &option, turbo::Span<uint8_t> data_point, label_type label, int level) {
        turbo::Span<uint8_t> vector_data = to_span<uint8_t>(data_point);
        location_t cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock<std::mutex> lock_table(_label_lookup_lock);
            auto search = _label_lookup.find(label);
            if (search != _label_lookup.end()) {
                location_t existingInternalId = search->second;
                if (_data.is_deleted(existingInternalId)) {
                    return turbo::UnavailableError(
                            "Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                }
                lock_table.unlock();

                if (_data.is_deleted(existingInternalId)) {
                    auto r = unmark_deleted_internal(existingInternalId);
                    if (!r.ok()) {
                        return r;
                    }
                }
                auto r = update_point(option, data_point, existingInternalId, 1.0);
                if (!r.ok()) {
                    return r;
                }

                return existingInternalId;
            }

            if (_cur_element_count >= _option.max_elements) {
                return turbo::OutOfRangeError("The number of elements exceeds the specified limit");
            }

            cur_c = _data.prefer_add_vector();
            _cur_element_count++;
            _label_lookup[label] = cur_c;
            TLOG_DEBUG("add vector label: {} location: {}", label, cur_c);
        }

        std::unique_lock<std::mutex> lock_el(_link_list_locks[cur_c]);
        int cur_level = get_random_level(_mult);
        if (level > 0)
            cur_level = level;

        //_element_levels[cur_c] = cur_level;
        std::unique_lock<std::mutex> templock(_global_lock);
        int max_level_copy = _max_level;
        if (cur_level <= max_level_copy)
            templock.unlock();
        location_t currObj = _enterpoint_node;
        location_t enterpoint_copy = _enterpoint_node;
        // Initialisation of the data and label
        set_external_label(cur_c, label);
        // preprocess
        AlignedQuery<uint8_t> aligned_vector;
        if (!option.is_normalized && _vs.distance_factor->preprocessing_required()) {
            vector_data = make_aligned_query(vector_data, aligned_vector);
            _vs.distance_factor->preprocess_base_points(vector_data, _vs.dimension);
        }
        _data.set_vector(cur_c, vector_data);

        auto r = _final_graph.setup_location(cur_c, cur_level);
        if (!r.ok()) {
            return r;
        }

        if ((signed) currObj != -1) {
            if (cur_level < max_level_copy) {
                distance_type curdist = _data.get_distance(cur_c, currObj);
                for (int l_level = max_level_copy; l_level > cur_level; l_level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        std::unique_lock<std::mutex> lock(_link_list_locks[currObj]);
                        auto node = _final_graph.mutable_node(currObj, l_level);
                        size_t size = node.size();

                        for (int i = 0; i < size; i++) {
                            location_t cand = node[i];
                            if (cand > _option.max_elements)
                                return turbo::DataLossError("cand error");
                            auto d = _data.get_distance(cur_c, cand);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = _data.is_deleted(enterpoint_copy);
            for (int l_level = std::min(cur_level, max_level_copy); l_level >= 0; l_level--) {
                if (l_level > max_level_copy || l_level < 0)  // possible?
                    return turbo::OutOfRangeError("Level error");

                links_priority_queue top_candidates = search_base_layer(currObj, cur_c, l_level);
                if (epDeleted) {
                    top_candidates.emplace(_data.get_distance(cur_c, enterpoint_copy), enterpoint_copy);
                    if (top_candidates.size() > _option.ef_construction)
                        top_candidates.pop();
                }
                auto rs = mutually_connect_new_element(cur_c, top_candidates, l_level, false);
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
            _enterpoint_node = cur_c;
            _max_level = cur_level;
        }
        return cur_c;
    }

    links_priority_queue HnswIndex::search_base_layer(location_t ep_id, location_t loc, int layer) {
        VisitedList *vl = _visited_list_pool->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        links_priority_queue top_candidates;
        links_priority_queue candidateSet;

        distance_type lowerBound;
        if (!_data.is_deleted(ep_id)) {
            distance_type dist = _data.get_distance(loc, ep_id);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<distance_type>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<distance_type, location_t> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == _option.ef_construction) {
                break;
            }
            candidateSet.pop();

            location_t curNodeNum = curr_el_pair.second;

            std::unique_lock<std::mutex> lock(_link_list_locks[curNodeNum]);

            auto data = _final_graph.mutable_node(curNodeNum, layer);

            size_t size = data.size();

            for (size_t j = 0; j < size; j++) {
                location_t candidate_id = data[j];
                if (visited_array[candidate_id] == visited_array_tag) {
                    continue;
                }
                visited_array[candidate_id] = visited_array_tag;

                distance_type dist1 = _data.get_distance(loc, candidate_id);
                if (top_candidates.size() < _option.ef_construction || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
                    if (!_data.is_deleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > _option.ef_construction)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        _visited_list_pool->releaseVisitedList(vl);

        return top_candidates;
    }

    turbo::ResultStatus<location_t>
    HnswIndex::mutually_connect_new_element(location_t cur_c, links_priority_queue &top_candidates, int level,
                                            bool isUpdate) {

        auto node = _final_graph.mutable_node(cur_c, level);
        size_t Mcurmax = node.capacity();
        get_neighbors_by_heuristic2(top_candidates, _option.m);
        if (top_candidates.size() > _option.m)
            return turbo::OutOfRangeError("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<location_t> selectedNeighbors;
        selectedNeighbors.reserve(_option.m);
        while (!top_candidates.empty()) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        location_t next_closest_entry_point = selectedNeighbors.back();

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
            node.set_size(selectedNeighbors.size());
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (node[idx] && !isUpdate)
                    return turbo::InternalError("Possible memory corruption");
                if (level > node.level())
                    return turbo::InternalError("Trying to make a link on a non-existent level");

                node[idx] = selectedNeighbors[idx];
            }
        }

        auto loop_size = selectedNeighbors.size();
        for (size_t idx = 0; idx < loop_size; idx++) {
            std::unique_lock<std::mutex> lock(_link_list_locks[selectedNeighbors[idx]]);

            auto other_node = _final_graph.mutable_node(selectedNeighbors[idx], level);

            size_t sz_link_list_other = other_node.size();

            if (sz_link_list_other > Mcurmax)
                return turbo::InternalError("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
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
                    other_node[sz_link_list_other] = cur_c;
                    other_node.set_size(sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    distance_type d_max = _data.get_distance(cur_c, selectedNeighbors[idx]);
                    // Heuristic:
                    std::priority_queue<std::pair<distance_type, location_t>, std::vector<std::pair<distance_type, location_t>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(_data.get_distance(other_node[j], selectedNeighbors[idx]), other_node[j]);
                    }

                    get_neighbors_by_heuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (!candidates.empty()) {
                        other_node[indx] = candidates.top().second;
                        candidates.pop();
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

    turbo::Status
    HnswIndex::update_point(const WriteOption &option, turbo::Span<uint8_t> data_point, location_t internalId,
                            float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        AlignedQuery<uint8_t> aligned_vector;
        auto vector_data = to_span<uint8_t>(data_point);
        if (!option.is_normalized && _vs.distance_factor->preprocessing_required()) {
            vector_data = make_aligned_query(vector_data, aligned_vector);
            _vs.distance_factor->preprocess_base_points(vector_data, _vs.dimension);
        }
        _data.set_vector(internalId, vector_data);
        int maxLevelCopy = _max_level;
        location_t entryPointCopy = _enterpoint_node;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && _cur_element_count == 1)
            return turbo::OkStatus();

        int elemLevel = _final_graph.level(internalId);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<location_t> sCand;
            std::unordered_set<location_t> sNeigh;
            std::vector<location_t> listOneHop = get_connections_with_lock(internalId, layer);
            if (listOneHop.empty())
                continue;

            sCand.insert(internalId);

            for (auto &&elOneHop: listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(_update_probability_generator) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<location_t> listTwoHop = get_connections_with_lock(elOneHop, layer);
                for (auto &&elTwoHop: listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto &&neigh: sNeigh) {
                // if (neigh == internalId)
                //     continue;

                links_priority_queue candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() -
                                                                                1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(_option.ef_construction, size);
                for (auto &&cand: sCand) {
                    if (cand == neigh)
                        continue;

                    distance_type distance = _data.get_distance(neigh, cand);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                get_neighbors_by_heuristic2(candidates, layer == 0 ? _maxM * 2 : _maxM);

                {
                    std::unique_lock<std::mutex> lock(_link_list_locks[neigh]);
                    auto ll_cur = _final_graph.mutable_node(neigh, layer);
                    size_t candSize = candidates.size();
                    ll_cur.set_size(candSize);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        ll_cur[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        return repair_connections_for_update(entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }

    turbo::Status HnswIndex::repair_connections_for_update(
            location_t entryPointInternalId,
            location_t dataPointInternalId,
            int dataPointLevel,
            int maxLevel) {
        location_t currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            distance_type curdist = _data.get_distance(dataPointInternalId, currObj);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lock(_link_list_locks[currObj]);
                    auto data = _final_graph.mutable_node(currObj, level);
                    size_t size = data.size();
                    for (size_t i = 0; i < size; i++) {
                        location_t cand = data[i];
                        distance_type d = _data.get_distance(dataPointInternalId, cand);
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
            links_priority_queue topCandidates = search_base_layer(currObj, dataPointInternalId, level);

            std::priority_queue<std::pair<distance_type, location_t>, std::vector<std::pair<distance_type, location_t>>, CompareByFirst> filteredTopCandidates;
            while (!topCandidates.empty()) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (!filteredTopCandidates.empty()) {
                bool epDeleted = _data.is_deleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(_data.get_distance(dataPointInternalId, entryPointInternalId),
                                                  entryPointInternalId);
                    if (filteredTopCandidates.size() > _option.ef_construction)
                        filteredTopCandidates.pop();
                }

                auto r = mutually_connect_new_element(dataPointInternalId, filteredTopCandidates, level, true);
                if (!r.ok()) {
                    return r.status();
                }
                currObj = r.value();
            }
        }
        return turbo::OkStatus();
    }


    std::vector<location_t> HnswIndex::get_connections_with_lock(location_t internalId, int level) {
        std::unique_lock<std::mutex> lock(_link_list_locks[internalId]);
        auto node = _final_graph.mutable_node(internalId, level);
        size_t size = node.size();
        std::vector<location_t> result(size);
        auto list = node.links();
        memcpy(result.data(), list.data(), size * sizeof(location_t));
        return result;
    }

    void HnswIndex::get_neighbors_by_heuristic2(links_priority_queue &top_candidates, const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<distance_type, location_t>> queue_closest;
        std::vector<std::pair<distance_type, location_t>> return_list;
        while (!top_candidates.empty()) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (!queue_closest.empty()) {
            if (return_list.size() >= M)
                break;
            std::pair<distance_type, location_t> curent_pair = queue_closest.top();
            distance_type dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<distance_type, location_t> second_pair: return_list) {
                distance_type curdist = _data.get_distance(second_pair.second, curent_pair.second);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<distance_type, location_t> curent_pair: return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    turbo::Status HnswIndex::unmark_deleted_internal(location_t internalId) {
        assert(internalId < _cur_element_count);
        if (_data.is_deleted(internalId)) {
            _data.unmark_deleted(internalId);
        } else {
            return turbo::UnavailableError("The requested to undelete element is not deleted");
        }
        return turbo::OkStatus();
    }

    int HnswIndex::get_random_level(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(_level_generator)) * reverse_size;
        return (int) r;
    }

    turbo::Status HnswIndex::save_index(const std::string &path, const SerializeOption &option) {
        turbo::SequentialWriteFile file;
        auto r = file.open(path);
        if (!r.ok()) {
            return r;
        }

        r = write_binary_pod(file, kHnswMagic);
        if (!r.ok()) {
            return r;
        }
        /// write option
        r = write_binary_pod(file, _option);
        if (!r.ok()) {
            return r;
        }
        /// index status
        r = write_binary_pod(file, _enterpoint_node);
        if (!r.ok()) {
            return r;
        }
        r = write_binary_pod(file, _max_level);
        if (!r.ok()) {
            return r;
        }

        r = write_binary_pod(file, _mult);
        if (!r.ok()) {
            return r;
        }
        /// save raw data
        r = _data.save(&file);
        if (!r.ok()) {
            return r;
        }

        /// save graph
        r = _final_graph.save(file);
        if (!r.ok()) {
            return r;
        }

        return turbo::OkStatus();
    }

    turbo::Status HnswIndex::load_index(const std::string &path, const SerializeOption &option) {
        turbo::SequentialReadFile file;
        auto r = file.open(path);
        if (!r.ok()) {
            return r;
        }
        // read and check magic
        size_t magic;
        r = read_binary_pod(file, magic);
        if (!r.ok()) {
            return r;
        }
        if (magic != kHnswMagic) {
            return turbo::UnavailableError("file format error, not hnsw index file");
        }

        /// read option
        HnswIndexOption op;
        r = read_binary_pod(file, op);
        if (!r.ok()) {
            return r;
        }
        // check option
        r = check_load_option(op);
        if (!r.ok()) {
            return r;
        }

        /// index status
        r = read_binary_pod(file, _enterpoint_node);
        if (!r.ok()) {
            return r;
        }
        r = read_binary_pod(file, _max_level);
        if (!r.ok()) {
            return r;
        }

        r = read_binary_pod(file, _mult);
        if (!r.ok()) {
            return r;
        }
        // raw data
        r = _data.load(&file);
        if (!r.ok()) {
            return r;
        }
        // graph
        r = _final_graph.load(file);
        if (!r.ok()) {
            return r;
        }
        return turbo::OkStatus();
    }

    turbo::Status HnswIndex::check_load_option(const HnswIndexOption &op) {
        if (op.dimension != _option.dimension) {
            return turbo::InvalidArgumentError("index option confict dimension: {}, read from index: {}",
                                               _option.dimension, op.dimension);
        }

        if (op.m != _option.m) {
            return turbo::InvalidArgumentError("index option confict m: {}, read from index: {}", _option.m, op.m);
        }
        if (op.batch_size != _option.batch_size) {
            return turbo::InvalidArgumentError("index option confict batch_size: {}, read from index: {}",
                                               _option.batch_size, op.batch_size);
        }
        if (op.metric != _option.metric) {
            return turbo::InvalidArgumentError("index option confict metric: {}, read from index: {}",
                                               turbo::nameof_enum(_option.metric), turbo::nameof_enum(op.metric));
        }

        if (op.data_type != _option.data_type) {
            return turbo::InvalidArgumentError("index option confict data_type: {}, read from index: {}",
                                               turbo::nameof_enum(_option.data_type), turbo::nameof_enum(op.data_type));
        }
        if (op.max_elements != _option.max_elements) {
            return turbo::InvalidArgumentError("index option confict max_elements: {}, read from index: {}",
                                               _option.max_elements, op.max_elements);
        }

        return turbo::OkStatus();
    }

    template links_priority_queue
    HnswIndex::search_base_layer_st<true, true>(location_t ep_id, QueryContext *qctx, size_t ef) const;

    template links_priority_queue
    HnswIndex::search_base_layer_st<false, true>(location_t ep_id, QueryContext *qctx, size_t ef) const;
}  // namespace tann
