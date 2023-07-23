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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "tann/store/vector_batch.h"
#include "turbo/format/print.h"
#include "tann/core/vector_space.h"

TEST_CASE("vector batch") {
    tann::VectorSpace vs;
    auto rs = vs.init(256, tann::METRIC_L2, tann::DataType::DT_FLOAT16);
    tann::VectorBatch vb;
    turbo::Println("alloc size:{}", 256 * 256 * 2);
    //turbo::simd::aligned_allocator<uint8_t, tann::VectorSpace::alignment_bytes> alloc;
    //alloc.allocate(256 * 256 * 2);
    auto  r = vb.init(vs.vector_byte_size, 256);
    CHECK(r.ok());
    CHECK_EQ(vb.size(), 0);
    CHECK_EQ(vb.is_empty(), true);
    CHECK_EQ(vb.is_full(), false);
    CHECK_EQ(vb.available(), 256);
    CHECK_EQ(vb.capacity(), 256);
    std::vector<uint8_t> a(vs.vector_byte_size);
    vb.add_vector(tann::to_span<uint8_t>(a));
    CHECK_EQ(vb.size(), 1);
}