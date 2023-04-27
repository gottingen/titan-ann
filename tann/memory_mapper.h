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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#else
#include <Windows.h>
#endif
#include <string>

namespace tann {
  class MemoryMapper {
   private:
#ifndef _WINDOWS
    int _fd;
#else
    HANDLE _bareFile;
    HANDLE _fd;

#endif
    char*       _buf;
    size_t      _fileSize;
    const char* _fileName;

   public:
    MemoryMapper(const char* filename);
    MemoryMapper(const std::string& filename);

    char*  getBuf();
    size_t getFileSize();

    ~MemoryMapper();
  };
}  // namespace tann
