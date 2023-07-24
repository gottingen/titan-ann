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
#ifndef TANN_CORE_SERIALIZE_OPTION_H_
#define TANN_CORE_SERIALIZE_OPTION_H_

#include "tann/core/types.h"
#include "tann/common/constants.h"

namespace tann {

    ///////////////////////////////////////////////////
    // data_type and dimension is must It must be filled
    // in accurately, n_vectors is determined according
    // to different formats and situations.
    struct SerializeOption {
        DataType data_type;
        std::size_t n_vectors{constants::kUnknownSize};
        std::size_t dimension;
    };

}  // namespace tann

#endif  // TANN_CORE_SERIALIZE_OPTION_H_
