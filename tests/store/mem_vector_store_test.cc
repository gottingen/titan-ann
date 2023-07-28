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
#include "tann/store/mem_vector_store.h"
#include "turbo/format/print.h"

namespace tann {
    class VectorSetTestFixture {
    public:
        VectorSetTestFixture() {
            auto  r = vs.init(128, MetricType::METRIC_L2, DataType::DT_FLOAT);
            CHECK(r.ok());
            op.max_elements = 1000;
            op.enable_replace_vacant = true;
            op.batch_size = 256;
            r = vector_set.initialize(&vs, op);
            CHECK(r.ok());
            if(turbo::filesystem::exists(vector_file_path)) {
                turbo::filesystem::remove(vector_file_path);
            }
        }

        ~VectorSetTestFixture() {

        }

        VectorSpace vs;
        VectorStoreOption op;
        MemVectorStore   vector_set;
        std::string vector_file_path = "vset_test.bin";
    };

    TEST_CASE_FIXTURE(VectorSetTestFixture, "init") {
        CHECK_EQ(vector_set.size(),0);
        CHECK_EQ(vector_set.deleted_size(),0);
        CHECK_EQ(vector_set.capacity(), op.max_elements);
        CHECK_EQ(vector_set.get_batch_size(),256);
        CHECK_EQ(vector_set.get_vector_space(), &vs);
    }

    /*
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

        vector_set.reserve(2000);
        CHECK_EQ(vector_set.capacity(),op.max_elements);
    }
*/
    TEST_CASE_FIXTURE(VectorSetTestFixture, "add vector") {

        CHECK_EQ(vector_set.size(),0);
        for(size_t i = 0; i < 300; i++) {
            auto r = vector_set.prefer_add_vector(i);
            CHECK_EQ(r.ok(), true);
        }

        CHECK_EQ(vector_set.size(),300);
        CHECK_EQ(vector_set.deleted_size(),0);

        for(size_t i = 0; i < 100; i++) {
            auto r = vector_set.remove_vector(i);
            CHECK_EQ(r.ok(), true);
        }

        CHECK_EQ(vector_set.size(),200);
        CHECK_EQ(vector_set.deleted_size(),100);
        for(size_t i = 300; i < 400; i++) {
            auto r = vector_set.get_vacant(i);
            CHECK_EQ(r.ok(), true);
        }
        auto r = vector_set.get_vacant(20000);
        CHECK_EQ(r.ok(), false);
        CHECK_EQ(turbo::IsResourceExhausted(r.status()), true);
    }

    TEST_CASE_FIXTURE(VectorSetTestFixture, "mark_delete") {
        vector_set.disable_vacant();
        CHECK_EQ(vector_set.size(),0);
        for(size_t i = 0; i < op.max_elements; i++) {
            auto r = vector_set.prefer_add_vector(i);
            CHECK_EQ(r.ok(), true);
        }

        CHECK_EQ(vector_set.size(),op.max_elements);
        CHECK_EQ(vector_set.deleted_size(),0);
        std::vector<location_t> del_list;
        for(size_t i = 0; i < 100; i++) {
            auto l = vector_set.get_label(i);
            CHECK_EQ(l.ok(), true);
            del_list.push_back(l.value());
            auto r = vector_set.remove_vector(i);
            CHECK_EQ(r.ok(), true);
        }

        CHECK_EQ(vector_set.size(),op.max_elements - 100);
        CHECK_EQ(vector_set.deleted_size(),100);
        for(size_t i = 0; i < del_list.size(); i++) {
            auto r = vector_set.is_deleted(i);
            CHECK_EQ(r, true);
        }
        vector_set.disable_vacant();
        auto r = vector_set.get_vacant(20000);
        CHECK_EQ(r.ok(), false);
        CHECK_EQ(turbo::IsUnavailable(r.status()), true);

        vector_set.enable_vacant();
        r = vector_set.get_vacant(20000);
        CHECK_EQ(r.ok(), true);

        // exist label
        r = vector_set.get_vacant(200);
        CHECK_EQ(r.ok(), false);
        CHECK_EQ(turbo::IsAlreadyExists(r.status()), true);

        vector_set.reset_max_elements(op.max_elements *2);

        r = vector_set.prefer_add_vector(20001);
        CHECK_EQ(r.ok(), true);
        r = vector_set.prefer_add_vector(200);
        CHECK_EQ(r.ok(), false);
        CHECK_EQ(turbo::IsAlreadyExists(r.status()), true);

    }

    TEST_CASE_FIXTURE(VectorSetTestFixture, "save and load") {
        CHECK_EQ(vector_set.size(),0);
        for(size_t i = 0; i < op.max_elements; i++) {
            auto r = vector_set.prefer_add_vector(i);
            CHECK_EQ(r.ok(), true);
        }

        std::vector<location_t> del_list;
        for(size_t i = 0; i < 100; i++) {
            auto l = vector_set.get_label(i);
            CHECK_EQ(l.ok(), true);
            del_list.push_back(l.value());
            auto r = vector_set.remove_vector(i);
            CHECK_EQ(r.ok(), true);
        }

        auto r= vector_set.save(vector_file_path);
        turbo::Println(r.ToString());
        CHECK_EQ(r.ok(), true);
        MemVectorStore lset;
        r = lset.initialize(&vs, op);
        CHECK_EQ(r.ok(), true);
        r= lset.load(vector_file_path);
        CHECK_EQ(r.ok(), true);
        CHECK_EQ(lset.deleted_size(), del_list.size());
        for(size_t i = 0; i < del_list.size(); i++) {
            auto r = vector_set.is_deleted(i);
            CHECK_EQ(r, true);
        }
    }

}  // namespace tann
