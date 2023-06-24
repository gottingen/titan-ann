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


#ifndef TANN_INDEX_DML_QUERY_H_
#define TANN_INDEX_DML_QUERY_H_

#include "tann/index/container.h"

namespace tann {
    class InsertContainer : public Container {
    public:
        InsertContainer(Object &f, ObjectID i) : Container(f, i) {}
    };

}
#endif  // TANN_INDEX_DML_QUERY_H_
