#
# Copyright 2023 The Turbo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

include_guard()

include(utils)

FetchContent_DeclareGitHubWithMirror(gtest
        google/googletest v1.13.0
        MD5=a1279c6fb5bf7d4a5e0d0b2a4adb39ac
        )

FetchContent_MakeAvailableWithArgs(gtest
        BUILD_GMOCK=ON
        INSTALL_GTEST=OFF
        )
