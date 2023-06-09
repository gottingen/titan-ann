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

#include(FindArm)
include(FindSSE)
include(FindAvx)

if(CXX_AVX2_FOUND)
    message(STATUS "AVX2 SUPPORTED for CXX")
    set(AVX2_SUPPORTED true)
    set(HIGHEST_SIMD_SUPPORTED "AVX2")
    list(APPEND SSE_SUPPORTED_LIST  ${AVX2_SUPPORTED})
else()
    set(AVX2_SUPPORTED false)
endif()

if(CXX_AVX512_FOUND)
    message(STATUS "AVX512 SUPPORTED for C and CXX")
    set(AVX512_SUPPORTED true)
    set(HIGHEST_SIMD_SUPPORTED "AVX512")
    list(APPEND SSE_SUPPORTED_LIST  ${AVX512_SUPPORTED})
else()
    set(AVX512_SUPPORTED false)
endif()

set(CARBIN_SIMD_FLAGS)

if(CARBIN_USE_SSE1)
    message(STATUS "CARBIN SSE1 SELECTED")
    list(APPEND CARBIN_SSE1_SIMD_FLAGS  "-msse")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_SSE1_SIMD_FLAGS})
endif()

if(CXX_SSE2_FOUND)
    message(STATUS "CARBIN SSE2 SELECTED")
    list(APPEND CARBIN_SSE2_SIMD_FLAGS "-msse2")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_SSE2_SIMD_FLAGS})
endif()

if(CXX_SSE3_FOUND)
    message(STATUS "CARBIN SSE3 SELECTED")
    list(APPEND CARBIN_SSE3_SIMD_FLAGS "-msse3")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_SSE3_SIMD_FLAGS})
endif()

if(CXX_SSSE3_FOUND)
    message(STATUS "CARBIN SSSE3 SELECTED")
    list(APPEND CARBIN_SSSE3_SIMD_FLAGS "-mssse3")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_SSSE3_SIMD_FLAGS})
endif()

if(CXX_SSE4_1_FOUND)
    message(STATUS "CARBIN SSE4_1 SELECTED")
    list(APPEND CARBIN_SSE4_1_SIMD_FLAGS "-msse4.1")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_SSE4_1_SIMD_FLAGS})
endif()

if(CXX_SSE4_2_FOUND)
    message(STATUS "CARBIN SSE4_2 SELECTED")
    list(APPEND CARBIN_SSE4_2_SIMD_FLAGS "-msse4.2")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_SSE4_2_SIMD_FLAGS})
endif()

if(CXX_AVX_FOUND)
    message(STATUS "CARBIN AVX SELECTED")
    list(APPEND CARBIN_AVX_SIMD_FLAGS "-mavx")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_AVX_SIMD_FLAGS})
endif()

if(CXX_AVX2_FOUND)
    message(STATUS "CARBIN AVX2 SELECTED")
    list(APPEND CARBIN_AVX2_SIMD_FLAGS "-mavx2"  "-mfma")
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_AVX2_SIMD_FLAGS})
endif()

if(CXX_AVX512_FOUND)
    message(STATUS "CARBIN AVX512 SELECTED")
    list(APPEND CARBIN_AVX512_SIMD_FLAGS "-mavx512f" "-mfma") # note that this is a bit platform specific
    list(APPEND CARBIN_SIMD_FLAGS ${CARBIN_AVX512_SIMD_FLAGS}) # note that this is a bit platform specific
endif()
MESSAGE(STATUS "CARBIN SIMD FLAGS ${CARBIN_SIMD_FLAGS}")