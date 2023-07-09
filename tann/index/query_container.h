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

#ifndef TANN_INDEX_QUERY_CONTAINER_H_
#define TANN_INDEX_QUERY_CONTAINER_H_

#include <vector>
#include "tann/common/config.h"
#include "turbo/format/format.h"
#include "tann/common/exception.h"

namespace tann {
    class QueryContainer {
    public:
        template<typename QTYPE>
        explicit QueryContainer(const std::vector<QTYPE> &q) { setQuery(q); }

        ~QueryContainer() { deleteQuery(); }

        template<typename QTYPE>
        void setQuery(const std::vector<QTYPE> &q) {
            if (query != nullptr) {
                deleteQuery();
            }
            queryType = &typeid(QTYPE);
            if (*queryType != typeid(float) && *queryType != typeid(double) && *queryType != typeid(uint8_t)) {
                query = nullptr;
                queryType = nullptr;
                TANN_THROW("tann::SearchQuery: Invalid query type!");
            }
            query = new std::vector<QTYPE>(q);
        }

        void *getQuery() { return query; }

        const std::type_info &getQueryType() { return *queryType; }

    private:
        void deleteQuery() {
            if (query == nullptr) {
                return;
            }
            if (*queryType == typeid(float)) {
                delete static_cast<std::vector<float> *>(query);
            } else if (*queryType == typeid(double)) {
                delete static_cast<std::vector<double> *>(query);
            } else if (*queryType == typeid(uint8_t)) {
                delete static_cast<std::vector<uint8_t> *>(query);
#ifdef TANN_ENABLE_HALF_FLOAT
            } else if (*queryType == typeid(float16)) {
                delete static_cast<std::vector<float16> *>(query);
#endif
            }
            query = nullptr;
            queryType = nullptr;
        }

        void *query{nullptr};
        const std::type_info *queryType{nullptr};
    };
}
#endif  // TANN_INDEX_QUERY_CONTAINER_H_
