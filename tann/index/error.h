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

#ifndef TANN_INDEX_ERROR_H_
#define TANN_INDEX_ERROR_H_


namespace tann {
    constexpr int kTannModule = 2;
    constexpr int kTannCommonError = 11000;
    constexpr int kTannAlreadyInitialized = 11001;
    constexpr int kTannNoIndex = 11002;
    constexpr int kTannHnswError = 11003;
}  // namespace tann

#endif  // TANN_INDEX_ERROR_H_
