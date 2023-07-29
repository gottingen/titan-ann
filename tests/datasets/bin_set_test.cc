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

#include "tann/datasets/bin_vector_io.h"
#include "turbo/files/filesystem.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "tann/core/allocator.h"

class BinSetTest {
public:
    BinSetTest() noexcept {
        bin_file = "bin_test.bin";
        if (turbo::filesystem::exists(bin_file)) {
            turbo::filesystem::remove(bin_file);
        }
    }

    ~BinSetTest() {

    }

    std::string bin_file;
};

TEST_CASE_FIXTURE(BinSetTest, "empty read") {
    turbo::SequentialReadFile file;
    auto r = file.open(bin_file);
    file.close();
    CHECK_FALSE(r.ok());
}

TEST_CASE_FIXTURE(BinSetTest, "write read") {
    turbo::SequentialWriteFile file;
    auto r = file.open(bin_file);
    CHECK(r.ok());
    tann::BinaryVectorSetWriter writer;
    tann::SerializeOption op;
    op.n_vectors = 100;
    op.dimension = 128;
    op.data_type = tann::DataType::DT_FLOAT;
    r = writer.initialize(&file, op);
    CHECK(r.ok());
    std::vector<float> vmem;
    vmem.resize(op.dimension);
    for (int i = 0; i < 128; i++) {
        vmem[i] = i % 20;
    }
    auto span = tann::to_span<uint8_t>(vmem);
    CHECK_EQ(span.size(), 128 * sizeof(float));
    for (size_t i = 0; i < 100; i++) {
        r = writer.write_vector(span);
        CHECK(r.ok());
    }
    CHECK_EQ(writer.has_write(), 100);
    r = file.flush();
    CHECK(r.ok());
    file.close();
    auto file_size = turbo::filesystem::file_size(bin_file);
    auto cal_file_size = sizeof(uint32_t) * 2 + op.dimension * tann::data_type_size(op.data_type) * 100;
    CHECK_EQ(cal_file_size, file_size);
    turbo::SequentialReadFile rfile;
    r = rfile.open(bin_file);
    CHECK(r.ok());
    tann::SerializeOption rop;
    rop.n_vectors = 100;
    rop.dimension = 128;
    rop.data_type = tann::DataType::DT_FLOAT;
    tann::BinaryVectorSetReader reader;
    r = reader.initialize(&rfile, rop);
    CHECK(r.ok());
    std::vector<float> vrmem;
    vrmem.resize(rop.dimension);
    auto rspan = tann::to_span<uint8_t>(vrmem);
    size_t i = 0;
    auto s = turbo::OkStatus();
   while(s.ok()) {
        s = reader.read_vector(rspan);
        i++;
        CHECK(r.ok());
        for(size_t j = 0; j < rop.dimension; j++) {
            CHECK_EQ(vmem[i], vrmem[i]);
        }
    }
    CHECK_EQ(reader.has_read(), 100);
    CHECK_EQ(i, 101);
    CHECK_EQ(turbo::IsReachFileEnd(s), true);
    CHECK_EQ(rfile.is_eof(), true);
}