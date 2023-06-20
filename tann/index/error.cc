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
#include "tann/index/error.h"
#include "turbo/base/turbo_module.h"
#include "turbo/base/turbo_error.h"

TURBO_REGISTER_MODULE_INDEX(tann::kTannModule, "tann");
TURBO_REGISTER_ERRNO(tann::kTannNoIndex, "no initialize ann index");
TURBO_REGISTER_ERRNO(tann::kTannCommonError, "index common error");
TURBO_REGISTER_ERRNO(tann::kTannAlreadyInitialized, "index is already initiated");
TURBO_REGISTER_ERRNO(tann::kTannHnswError, "hnsw index error");