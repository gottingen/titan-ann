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
#ifndef TANN_VAMANA_CONSTANTS_H_
#define TANN_VAMANA_CONSTANTS_H_

namespace tann::constants {

    static constexpr float kGraphSlackFactor = 1.3f;
    static constexpr int kMaxGraphDegree = 512;
    static constexpr int kMaxQuantizeChunk = 512;

    static constexpr size_t kMaxPointsForUsingBitset = 10000000;
    static constexpr int kNumQuantizeBits = 8;
    static constexpr int kNumQuantizeCentroids = 8;
    static constexpr size_t kQuantizeBlockSize = 5000000;
    static constexpr int kNumKmeansRepsQuantize = 12;
    static constexpr int kMaxOpqItems = 20;
}  // namespace tann::constants

#endif  // TANN_VAMANA_CONSTANTS_H_
