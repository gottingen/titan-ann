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
#pragma once

#include <stdint.h>

#include <queue>
#include <string>
#include <utility>
#include <shared_mutex>

#include "tann/index/tann_index.h"

namespace hnswlib {

    template<typename T>
    class SpaceInterface;

    template<typename T>
    class HierarchicalNSW;

}  // namespace hnswlib

namespace tann {


        // max_elements defines the maximum number of elements that can be
        // stored in the structure(can be increased/shrunk).
        // ef_construction defines a construction time/accuracy trade-off.
        // M defines tha maximum number of outgoing connections in the graph.
    struct HnswLibParam {
        size_t m = hnsw::DEFAULT_M;
        size_t ef_construction = hnsw::DEFAULT_EF_CONSTRUCTION;
        size_t ef = hnsw::DEFAULT_EF;
        size_t max_elements = hnsw::DEFAULT_MAX_ELEMENTS;
    };

    class HnswLibSpace;

    class HnswLibIndex : public TannIndex {
    public:
        explicit HnswLibIndex(const IndexOption &index_conf);

        ~HnswLibIndex();

        // initializes the index from with no elements.
        turbo::Status init_new_index(const HnswLibParam &param, const size_t random_seed);

        // set/get the query time accuracy/speed trade-off, defined by the ef parameter
        // Note that the parameter is currently not saved along with the index,
        // so you need to set it manually after loading
        void set_ef(size_t ef);

        size_t get_ef();

        // ef_construction defines a construction time/accuracy trade-off
        size_t get_ef_construction();

        // M defines tha maximum number of outgoing connections in the graph
        size_t get_M();

        // set the default number of cpu threads used during data insertion/querying
        void set_num_threads(int num_threads);

        // inserts the data(numpy array of vectors, shape:N*dim) into the structure
        // labels is an optional N-size numpy array of integer labels for all elements in data.
        // num_threads sets the number of cpu threads to use (-1 means use default).
        // data_labels specifies the labels for the data. If index already has the elements with
        // the same labels, their features will be updated. Note that update procedure is slower
        // than insertion of a new element, but more memory- and query-efficient.
        // Thread-safe with other add_items calls, but not with knn_query.
        void add_item(size_t id, float *vector);

        // get origin vector by it's id
        std::vector<float> get_vector_by_id(size_t id);

        // returns a list of all elements' ids
        std::vector<unsigned int> get_ids_list();

        // make a batch query for k closests elements for each element of the
        // data (shape:N*dim). Returns a numpy array of (shape:N*k).
        // num_threads sets the number of cpu threads to use (-1 means use default).
        // Thread-safe with other knn_query calls, but not with add_items
        std::priority_queue<std::pair<float, size_t>> knn_query(SearchOptions &option, const float *vector, size_t k);

        // marks the element as deleted, so it will be ommited from search results.
        void mark_deleted(size_t label);

        // changes the maximum capacity of the index. Not thread safe with add_items and knn_query
        void resize_index(size_t new_size);

        turbo::Status init();

        size_t size();

        bool support_update() {
            return true;
        }

        bool support_delete() {
            return true;
        }

        void shrink_to_fit();

        void insert(const std::vector<int64_t> &ids, std::vector<float> &vecs);

        void search(SearchOptions &option, std::vector<float> &distances, std::vector<int64_t> &labels);

        void remove(const std::set<uint64_t> &delete_ids);

        void update(const std::vector<int64_t> &ids, std::vector<float> &vecs);

        void clear();

        turbo::Status load(const std::string &file, std::shared_mutex &mutex) override;

        turbo::Status save(const std::string &file) override;

        void init_params(HnswLibParam &param);

    private:
        bool index_inited_ = false;
        bool ep_added_ = false;
        bool normalize_ = false;
        int num_threads_default_ = 0;
        hnswlib::HierarchicalNSW<float> *appr_alg_ = nullptr;
        hnswlib::SpaceInterface<float> *l2space_ = nullptr;
    };

}  // namespace elasticfaiss
