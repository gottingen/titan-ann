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

#include "tann/datasets/tsv_vector_io.h"
#include "turbo/files/filesystem.h"
#include "tann/common/vector_utility.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"

class BinSetTest {
public:
    BinSetTest() noexcept {
        tsv_file = "tsv_file.tsv";
        if (turbo::filesystem::exists(tsv_file)) {
            turbo::filesystem::remove(tsv_file);
        }
    }

    ~BinSetTest() {

    }

    std::string tsv_file;
};


TEST_CASE_FIXTURE(BinSetTest, "write read") {
    turbo::SequentialWriteFile file;
    auto r = file.open(tsv_file);
    CHECK(r.ok());
    tann::TsvVectorSetWriter writer;
    tann::WriteOption op;
    op.n_vectors = 100;
    op.dimension = 128;
    op.data_type = tann::DataType::DT_FLOAT;
    r = writer.init(&file, &op);
    CHECK(r.ok());
    std::vector<float> vmem;
    vmem.resize(op.dimension);
    auto vspan = tann::to_span<float>(vmem);
    for (int i = 0; i < 128; i++) {
        vspan[i] = static_cast<float>(i) / 20.8;
    }
    auto span = tann::to_span<uint8_t>(vmem);
    CHECK_EQ(span.size(), 128 * sizeof(float));
    for (size_t i = 0; i < 100; i++) {
        r = writer.write_vector(&span);
        CHECK(r.ok());
    }
    CHECK_EQ(writer.has_write(), 100);
    r = file.flush();
    CHECK(r.ok());
    file.close();

    turbo::SequentialReadFile rfile;
    r = rfile.open(tsv_file);
    CHECK(r.ok());
    tann::ReadOption rop;
    rop.n_vectors = 100;
    rop.dimension = 128;
    rop.data_type = tann::DataType::DT_FLOAT;
    tann::TsvVectorSetReader reader;
    r = reader.init(&rfile, &rop);
    CHECK(r.ok());
    std::vector<float> vrmem;
    vrmem.resize(rop.dimension);
    auto rspan = tann::to_span<uint8_t>(vrmem);
    size_t i = 0;
    auto s = turbo::OkStatus();
    while(s.ok()) {
        s = reader.read_vector(&rspan);
        i++;
        CHECK(r.ok());
        if(!r.ok()) {
            turbo::Println(r.ToString());
        }
    }
    turbo::Println("{}", s.ToString());
    CHECK_EQ(reader.has_read(), 100);
    CHECK_EQ(i, 101);
    CHECK_EQ(turbo::IsReachFileEnd(s), true);
    CHECK_EQ(rfile.is_eof(), true);
}