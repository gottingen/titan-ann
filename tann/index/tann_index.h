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
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <shared_mutex>
#include "turbo/base/status.h"
#include "tann/config/check.h"
#include "tann/index/option.h"
#include "bluebird/bits/bitmap.h"
#include "tann/index/constants.h"

namespace tann {


    struct SearchOptions {
        int64_t n = 0;
        int64_t k = 0;
        std::vector<float> vecs;
        bluebird::Bitmap *bitmap{nullptr};
        uint32_t nprobe = 0;
        float radius = 0;
    };

    class TannIndex;

    using TannIndexPtr = std::shared_ptr<TannIndex>;

    class TannIndex {
    public:
        explicit TannIndex(const IndexOption &index_conf);

        virtual ~TannIndex();

        TannIndexPtr create_ann_index(const IndexOption &index_conf);

        virtual turbo::Status init() = 0;

        virtual size_t size() = 0;

        virtual bool support_update() = 0;

        virtual bool support_delete() = 0;

        virtual void insert(const std::vector<int64_t> &ids, std::vector<float> &vecs) = 0;

        virtual void search(SearchOptions &option, std::vector<float> &distances, std::vector<int64_t> &labels) = 0;

        virtual void remove(const std::set<uint64_t> &delete_ids) = 0;

        virtual void update(const std::vector<int64_t> &ids, std::vector<float> &vecs) = 0;

        virtual void clear() = 0;

        virtual turbo::Status load(const std::string &file, std::shared_mutex &mutex) = 0;

        virtual turbo::Status save(const std::string &file) = 0;

        void set_max_elements(size_t max_elements) {
            if (max_elements > max_elements_) {
                max_elements_ = max_elements;
            }
        }

        virtual void shrink_to_fit() {
        }

        virtual bool build_with_id() {
            return true;
        }

        virtual void print_stats() {
        }

    protected:
        IndexOption index_conf_;
        size_t max_elements_{kDefaultMaxItem};
    };


};  // namespace tann
