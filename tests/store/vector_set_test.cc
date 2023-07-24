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
#include "tann/store/vector_set.h"
#include "turbo/format/print.h"

namespace tann {
    class VectorSetTestFixture {
    public:
        VectorSetTestFixture() {
            auto  r = vs.init(128, MetricType::METRIC_L2, DataType::DT_FLOAT);
            CHECK(r.ok());
            r = vector_set.init(&vs, 128);
            CHECK(r.ok());
            if(turbo::filesystem::exists(vector_file_path)) {
                turbo::filesystem::remove(vector_file_path);
            }
        }

        ~VectorSetTestFixture() {

        }

        VectorSpace vs;
        VectorSet   vector_set;
        std::string vector_file_path = "vset_test.bin";
    };

    TEST_CASE_FIXTURE(VectorSetTestFixture, "init") {
        CHECK_EQ(vector_set.size(),0);
        CHECK_EQ(vector_set.capacity(),0);
        CHECK_EQ(vector_set.get_batch_size(),128);
        CHECK_EQ(vector_set.get_vector_space(), &vs);
    }

    TEST_CASE_FIXTURE(VectorSetTestFixture, "reserve shrink") {
        turbo::Println("{}",__LINE__);
        CHECK_EQ(vector_set.capacity(),0);
        vector_set.reserve(200);
        turbo::Println("{}",__LINE__);
        CHECK_EQ(vector_set.capacity(),256);
        CHECK_EQ(vector_set.size(),0);
        turbo::Println("{}",__LINE__);
        vector_set.shrink();
        turbo::Println("{}",__LINE__);
        CHECK_EQ(vector_set.capacity(),0);
    }

    TEST_CASE_FIXTURE(VectorSetTestFixture, "resize pop") {
        CHECK_EQ(vector_set.capacity(),0);
        CHECK_EQ(vector_set.size(),0);
        vector_set.resize(201);
        CHECK_EQ(vector_set.size(),201);
        CHECK_EQ(vector_set.capacity(),256);

        vector_set.shrink();
        CHECK_EQ(vector_set.capacity(),256);
        vector_set.pop_back();
        CHECK_EQ(vector_set.size(),200);
        vector_set.resize(101);
        CHECK_EQ(vector_set.capacity(),256);
        vector_set.shrink();
        CHECK_EQ(vector_set.capacity(),128);

    }

    TEST_CASE_FIXTURE(VectorSetTestFixture, "mark_delete") {
        CHECK_EQ(vector_set.capacity(),0);
        CHECK_EQ(vector_set.size(),0);
        vector_set.resize(201);
        vector_set.mark_deleted(128);
        vector_set.mark_deleted(167);
        CHECK_EQ(vector_set.deleted_size(), 2);
        CHECK_EQ(true,vector_set.is_deleted(128));
        CHECK_EQ(true,vector_set.is_deleted(167));
        vector_set.unmark_deleted(128);
        vector_set.unmark_deleted(167);
        CHECK_EQ(false,vector_set.is_deleted(128));
        CHECK_EQ(false,vector_set.is_deleted(167));
    }

    TEST_CASE_FIXTURE(VectorSetTestFixture, "save and load") {
        CHECK_EQ(vector_set.capacity(),0);
        CHECK_EQ(vector_set.size(),0);
        vector_set.resize(201);
        vector_set.mark_deleted(128);
        vector_set.mark_deleted(167);
        auto r= vector_set.save("vset_test.bin");
        turbo::Println(r.ToString());
        CHECK_EQ(r.ok(), true);
        VectorSet lset;
        r = lset.init(&vs, 128);
        CHECK_EQ(r.ok(), true);
        r= lset.load("vset_test.bin");
        CHECK_EQ(r.ok(), true);
        CHECK_EQ(lset.deleted_size(), 2);
        CHECK_EQ(true,lset.is_deleted(128));
        CHECK_EQ(true,lset.is_deleted(167));
        lset.unmark_deleted(128);
        lset.unmark_deleted(167);
        CHECK_EQ(false,lset.is_deleted(128));
        CHECK_EQ(false,lset.is_deleted(167));
    }

}  // namespace tann
