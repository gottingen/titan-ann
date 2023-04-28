// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Copyright 2023 The Tann Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once
#ifndef _WINDOWS

#include "tann/aligned_file_reader.h"

class LinuxAlignedFileReader : public AlignedFileReader {
private:
    uint64_t file_sz;
    FileHandle file_desc;
    io_context_t bad_ctx = (io_context_t) -1;

public:
    LinuxAlignedFileReader();

    ~LinuxAlignedFileReader();

    IOContext &get_ctx();

    // register thread-id for a context
    void register_thread();

    // de-register thread-id for a context
    void deregister_thread();

    void deregister_all_threads();

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname);

    void close();

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
              bool async = false);
};

#endif
