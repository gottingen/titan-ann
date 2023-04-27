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

#include <chrono>

namespace tann {
  class Timer {
    typedef std::chrono::high_resolution_clock _clock;
    std::chrono::time_point<_clock>            check_point;

   public:
    Timer() : check_point(_clock::now()) {
    }

    void reset() {
      check_point = _clock::now();
    }

    long long elapsed() const {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 _clock::now() - check_point)
          .count();
    }

    float elapsed_seconds() const {
      return (float) elapsed() / 1000000.0;
    }

    std::string elapsed_seconds_for_step(const std::string& step) const {
      return std::string("Time for ") + step + std::string(": ") +
             std::to_string(elapsed_seconds()) + std::string(" seconds");
    }
  };
}  // namespace tann
