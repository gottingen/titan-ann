#
# Copyright 2023 The Carbin Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if (CARBIN_BUILD_TEST)
    enable_testing()
    include(require_gtest)
    include(require_gmock)
    find_package(boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIR})
    link_directories(${CONDA_PREFIX}/lib)
endif (CARBIN_BUILD_TEST)

set(CARBIN_SYSTEM_DYLINK)
if (APPLE)
    find_library(CoreFoundation CoreFoundation)
    list(APPEND CARBIN_SYSTEM_DYLINK ${CoreFoundation} pthread)
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND CARBIN_SYSTEM_DYLINK rt dl pthread)
endif ()

include(require_turbo)
find_package(MKL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(AIO REQUIRED)
set(CARBIN_DEPS_LINK
        ${TURBO_LIB}
        ${CARBIN_SYSTEM_DYLINK}
        ${MKL_LIBRARIES}
        ${AIO_LIBRARIES}
        OpenMP::OpenMP_CXX
        )
if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND CARBIN_DEPS_LINK
            CUDA::cudart_static
            CUDA::cuda_driver
            )
endif ()






