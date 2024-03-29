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


#ifndef TANN_TEST_UTIL_H
#define TANN_TEST_UTIL_H
#include "tann/common/config.h"

#define TEST_NORMAL_TYPES uint8_t, int32_t, int64_t, float

#define SIMD_TEST_TYPES uint8_t, uint16_t, uint32_t, uint64_t, float, double

#define TEST_TYPES uint8_t, tann::float16, int32_t, int64_t, float

#define TEST_HM_TYPES uint8_t, uint32_t, uint64_t

#define TEST_NORM_TYPES tann::float16, float

#endif //TANN_TEST_UTIL_H
