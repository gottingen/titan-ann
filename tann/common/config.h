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

#ifndef TANN_COMMON_CONFIG_H_
#define TANN_COMMON_CONFIG_H_

#include "tann/common/defines.h"

#ifdef TANN_ENABLE_HALF_FLOAT
    #include "tann/common/half.hpp"
#endif

namespace tann {
    typedef unsigned int ObjectID;
    typedef float Distance;
#ifdef TANN_ENABLE_HALF_FLOAT
    typedef half_float::half float16;
#endif

}
#endif  // TANN_COMMON_CONFIG_H_
