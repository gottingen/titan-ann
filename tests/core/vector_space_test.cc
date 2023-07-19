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
#include "turbo/simd/simd.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "turbo/format/print.h"
#include "turbo/meta/reflect.h"
#include "tann/core/vector_space.h"

TEST_CASE("alignment") {
    tann::VectorSpace vs;
    auto rs = vs.init(256, tann::METRIC_L2, tann::DataType::DT_FLOAT16);
    CHECK(rs.ok());
    CHECK_EQ(vs.alignment_bytes, 32);
    CHECK_EQ(vs.alignment_dim, 16);
    CHECK_EQ(vs.dimension, 256);
    CHECK_EQ(vs.type_size, 2);
    CHECK_EQ(vs.vector_byte_size, 512);
}