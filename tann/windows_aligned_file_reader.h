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
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
#include <Windows.h>
#include <fcntl.h>
#include <malloc.h>
#include <minwinbase.h>

#include <cstdio>
#include <mutex>
#include <thread>
#include "tann/aligned_file_reader.h"
#include "tsl/robin_map.h"
#include "tann/utils.h"
#include "turbo/platform/port.h"

class WindowsAlignedFileReader : public AlignedFileReader {
 private:
  std::wstring m_filename;

 protected:
  // virtual IOContext createContext();

 public:
  TURBO_DLL WindowsAlignedFileReader(){};
  TURBO_DLL virtual ~WindowsAlignedFileReader(){};

  // Open & close ops
  // Blocking calls
  TURBO_DLL virtual void open(const std::string &fname) override;
  TURBO_DLL virtual void close() override;

  TURBO_DLL virtual void register_thread() override;
  TURBO_DLL virtual void deregister_thread() override {
    // TODO: Needs implementation.
  }
  TURBO_DLL virtual void deregister_all_threads() override {
    // TODO: Needs implementation.
  }
  TURBO_DLL virtual IOContext &get_ctx() override;

  // process batch of aligned requests in parallel
  // NOTE :: blocking call for the calling thread, but can thread-safe
  TURBO_DLL virtual void read(std::vector<AlignedRead> &read_reqs,
                                      IOContext &ctx, bool async) override;
};
#endif  // USE_BING_INFRA
#endif  //_WINDOWS
