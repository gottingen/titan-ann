// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Copyright 2022 The Tann Authors
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

#include <functional>
#include <iostream>
#include "windows_customizations.h"

namespace tann {
  DISKANN_DLLEXPORT extern std::basic_ostream<char> cout;
  DISKANN_DLLEXPORT extern std::basic_ostream<char> cerr;

  enum class DISKANN_DLLEXPORT LogLevel { LL_Info = 0, LL_Error, LL_Count };

#ifdef EXEC_ENV_OLS
  DISKANN_DLLEXPORT void SetCustomLogger(
      std::function<void(LogLevel, const char*)> logger);
#endif
}  // namespace tann
