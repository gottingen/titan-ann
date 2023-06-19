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

#ifndef TANN_INDEX_TANN_INDEX_H_
#define TANN_INDEX_TANN_INDEX_H_

#include <type_traits>
#include <memory>
#include "turbo/base/status.h"
#include "tann/index/tann_index_config.h"

namespace tann {

    struct TannSearchOption {

    };

    struct IDIterator {
        virtual ~IDIterator() = default;

        /// if Next returns std::numeric<int64_t>::max()
        /// iterator will stop
        virtual int64_t Next() = 0;
    };

    ///
    /// TannIndex is the interface
    class TannIndex {
    public:

        explicit TannIndex(std::shared_ptr<TannIndexConfig> &config);

        virtual ~TannIndex() = default;

        /// \brief initialize index;
        /// \return
        [[nodiscard]] virtual turbo::Status init_index() = 0;

        [[nodiscard]] virtual size_t size() const noexcept = 0;

        [[nodiscard]] virtual bool is_realtime() const noexcept = 0;

        [[nodiscard]] virtual bool is_trained() const noexcept = 0;


        virtual void add_vector(const std::vector<uint64_t> &ids, std::vector<float> &vecs) = 0;

        virtual void update_vector(const std::vector<uint64_t> &ids, std::vector<float> &vecs) = 0;

        virtual void remove(const IDIterator &ids) = 0;

        virtual void clear() = 0;

        virtual void search(TannSearchOption &option, std::vector<float> &distances, std::vector<int64_t> &labels) = 0;

        virtual int load(const std::string &file) = 0;

        virtual int save(const std::string &file) = 0;

        virtual int train(const std::string &file) = 0;

        virtual int64_t  snapshot_id() = 0;

    protected:
        std::shared_ptr<TannIndexConfig> _config;
    };
}  // namespace tann

#endif  // TANN_INDEX_TANN_INDEX_H_
