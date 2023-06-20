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
#include "tann/index/hnswlib_index.h"
#include <turbo/log/logging.h>
#include <iostream>
#include <thread>

#include "tann/hnsw/hnswlib.h"
#include "tann/index/error.h"

namespace tann {

    HnswLibIndex::HnswLibIndex(const IndexOption &index_conf) : TannIndex(index_conf) {
    }

    HnswLibIndex::~HnswLibIndex() {
        clear();
    }

    struct HnswSelectorImpl : hnswlib::BaseFilterFunctor {
        explicit HnswSelectorImpl(const SearchOptions &option) {
            bitmap = option.bitmap;
        }

        bool operator()(hnswlib::labeltype id) override {
            if (bitmap) {
                return !bitmap->contains(id);
            }
            return true;
        }

        bluebird::Bitmap *bitmap = nullptr;
    };

    turbo::Status HnswLibIndex::init_new_index(const HnswLibParam &param, const size_t random_seed) {
        if (appr_alg_) {
            TLOG_ERROR("The index is already initiated.");
            return turbo::MakeStatus(kTannModule, kTannAlreadyInitialized);
        }
        try {
            appr_alg_ =
                    new hnswlib::HierarchicalNSW<float>(l2space_, param.max_elements, param.m, param.ef_construction,
                                                        random_seed);
        } catch (std::exception &e) {
            TLOG_ERROR("HNSWLIB exception: {}", e.what());
            return turbo::MakeStatus(kTannModule, kTannHnswError, "exception:{}",  e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
            return turbo::MakeStatus(kTannModule, kTannHnswError, "Unkown exception");
        }
        if (nullptr == appr_alg_) {
            TLOG_ERROR("Fail to create HNSWLIB index.");
            return turbo::MakeStatus(kTannModule, kTannNoIndex);
        }
        index_inited_ = true;
        ep_added_ = false;
        return turbo::OkStatus();
    }

    void HnswLibIndex::init_params(HnswLibParam &param) {
        param.max_elements = index_conf_.hnsw.max_elements;
        param.ef = index_conf_.hnsw.ef;
        param.ef_construction = index_conf_.hnsw.ef_construction;
        param.m = index_conf_.hnsw.m;
    }

    turbo::Status HnswLibIndex::init() {
        normalize_ = false;
        if (index_conf_.metric == MetricType::MetricTypeL2) {
            l2space_ = new hnswlib::L2Space(index_conf_.dimension);
        } else if (index_conf_.metric == MetricType::MetricTypeL1) {
            l2space_ = new hnswlib::InnerProductSpace(index_conf_.dimension);
        } else if (index_conf_.metric == MetricType::MetricTypeCosine) {
            l2space_ = new hnswlib::InnerProductSpace(index_conf_.dimension);
            normalize_ = true;
        }
        if (nullptr == l2space_) {
            return turbo::MakeStatus(kTannModule, kTannCommonError, "alloc l2space fail");
        }

        HnswLibParam param;
        init_params(param);
        appr_alg_ = nullptr;
        ep_added_ = true;
        index_inited_ = false;
        num_threads_default_ = std::thread::hardware_concurrency();
        set_max_elements(param.max_elements);
        auto ret = init_new_index(param, hnsw::DEFAULT_RANDOM_SEED);
        if (!ret.ok()) {
            return ret;
        }
        set_ef(param.ef);
        return turbo::OkStatus();
    }

    size_t HnswLibIndex::size() {
        if (nullptr == appr_alg_) {
            return 0;
        }
        return appr_alg_->cur_element_count;
    }

    void HnswLibIndex::insert(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        if (nullptr == appr_alg_) {
            return;
        }
        for (size_t i = 0; i < ids.size(); i++) {
            add_item(ids[i], vecs.data() + i * index_conf_.dimension);
        }
    }

    void HnswLibIndex::search(SearchOptions &option, std::vector<float> &distances, std::vector<int64_t> &labels) {
        if (0 == size()) {
            return;
        }

        const auto &metric = index_conf_.metric;
        for (int64_t row = 0; row < option.n; row++) {
            std::priority_queue<std::pair<float, size_t>> result =
                    knn_query(option, option.vecs.data() + row * index_conf_.dimension, option.k);
            int offset = row * option.k;
            for (int i = result.size() - 1; i >= 0; i--) {
                auto &result_tuple = result.top();
                int position = offset + i;
                if (metric == MetricType::MetricTypeL1 || metric == MetricType::MetricTypeCosine) {
                    distances[position] = 1.0 - result_tuple.first;
                } else {
                    distances[position] = result_tuple.first;
                }
                labels[position] = result_tuple.second;
                result.pop();
                TLOG_INFO("row:{}, position:{}, labels:{},distances:{}", row, position, labels[position],
                          distances[position]);
            }
        }
    }

    void HnswLibIndex::remove(const std::set<uint64_t> &delete_ids) {
        if (nullptr == appr_alg_) {
            return;
        }
        for (auto id: delete_ids) {
            try {
                appr_alg_->markDelete(id);
            } catch (std::exception &e) {
                TLOG_ERROR("Failed to delete id:{} with err:{}", id, e.what());
            } catch (...) {
                TLOG_ERROR("Unkown exception");
            }
        }
    }

    void HnswLibIndex::clear() {
        if (l2space_) {
            delete l2space_;
            l2space_ = nullptr;
        }
        if (appr_alg_) {
            delete appr_alg_;
            appr_alg_ = nullptr;
        }
    }

    turbo::Status HnswLibIndex::load(const std::string &path_to_index, std::shared_mutex &mutex) {
        try {
            auto appr_alg_tmp = new hnswlib::HierarchicalNSW<float>(l2space_, path_to_index, false, 0);
            if (nullptr == appr_alg_tmp) {
                return turbo::MakeStatus(kTannNoIndex, "");
            }
            TLOG_INFO("Calling load_index for an already inited index. Old index is being deallocated.");
            std::unique_lock guard(mutex);
            if (appr_alg_) {
                delete appr_alg_;
            }
            appr_alg_ = appr_alg_tmp;
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to load index:{}  with err:{}", path_to_index, e.what());
            return turbo::MakeStatus(kTannNoIndex, "Failed to load index:{}  with err:{}", path_to_index, e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
            return turbo::MakeStatus(kTannNoIndex, "Unkown exception");
        }
        return turbo::OkStatus();
    }

    turbo::Status HnswLibIndex::save(const std::string &file) {
        try {
            if (appr_alg_) {
                appr_alg_->saveIndex(file);
            }
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to save index:{} with err:{}", file, e.what());
            return turbo::MakeStatus(kTannNoIndex, "Failed to save index:{} with err:{}", file, e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
            return turbo::MakeStatus(kTannNoIndex, "Unkown exception");
        }
        return turbo::OkStatus();
    }

    void HnswLibIndex::shrink_to_fit() {
        if (nullptr == appr_alg_) {
            return;
        }
        try {
            if (appr_alg_->max_elements_ > size() + 1024) {
                appr_alg_->resizeIndex(size() + 1024);
            }
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to shrink to fit with err:{}", e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
        }
    }

    void HnswLibIndex::set_ef(size_t ef) {
        appr_alg_->ef_ = ef;
    }

    size_t HnswLibIndex::get_ef() {
        return appr_alg_->ef_;
    }

    size_t HnswLibIndex::get_ef_construction() {
        return appr_alg_->ef_construction_;
    }

    size_t HnswLibIndex::get_M() {
        return appr_alg_->M_;
    }

    void HnswLibIndex::set_num_threads(int num_threads) {
        this->num_threads_default_ = num_threads;
    }

    void HnswLibIndex::add_item(size_t id, float *vector) {
        try {
            if (size() >= appr_alg_->max_elements_) {
                appr_alg_->resizeIndex(appr_alg_->max_elements_ * 2);
            }
            appr_alg_->addPoint(reinterpret_cast<void *>(vector), id);
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to add id:{}, err:{}", id, e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
        }
    }

    std::vector<float> HnswLibIndex::get_vector_by_id(size_t id) {
        try {
            return appr_alg_->template getDataByLabel<float>(id);
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to get vector by id:{}, err:{}", id, e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
        }
        return {
        };
    }

    std::vector<unsigned int> HnswLibIndex::get_ids_list() {
        std::vector<unsigned int> ids;
        for (auto kv: appr_alg_->label_lookup_) {
            ids.push_back(kv.first);
        }
        return ids;
    }

    std::priority_queue<std::pair<float, size_t>> HnswLibIndex::knn_query(SearchOptions &option, const float *vector,
                                                                          size_t k) {
        try {
            HnswSelectorImpl selector(option);
            return appr_alg_->searchKnn(static_cast<const void *>(vector), k, &selector);
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to search index:{}, error:{}", index_conf_.index_name, e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
        }
        return {};
    }

    void HnswLibIndex::mark_deleted(size_t label) {
        try {
            appr_alg_->markDelete(label);
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to delete label:{}, error:{}", label, e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
        }
    }

    void HnswLibIndex::resize_index(size_t new_size) {
        try {
            appr_alg_->resizeIndex(new_size);
        } catch (std::exception &e) {
            TLOG_ERROR("Failed to resize index to:{} err:{}", new_size, e.what());
        } catch (...) {
            TLOG_ERROR("Unkown exception");
        }
    }

    void HnswLibIndex::update(const std::vector<int64_t> &ids, std::vector<float> &vecs) {
        insert(ids, vecs);
    }

}  // namespace tann
