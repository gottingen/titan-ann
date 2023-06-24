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

#ifndef TANN_COMMON_CPU_INFO_H_
#define TANN_COMMON_CPU_INFO_H_

#include <iostream>

namespace tann {

    class CpuInfo {
    public:
        enum SimdType {
            SimdTypeAVX = 0,
            SimdTypeAVX2 = 1,
            SimdTypeAVX512F = 2,
            SimdTypeAVX512VL = 3,
            SimdTypeAVX512BW = 4,
            SimdTypeAVX512DQ = 5,
            SimdTypeAVX512CD = 6,
            SimdTypeAVX512ER = 7,
            SimdTypeAVX512PF = 8,
            SimdTypeAVX512VBMI = 9,
            SimdTypeAVX512IFMA = 10,
            SimdTypeAVX5124VNNIW = 11,
            SimdTypeAVX5124FMAPS = 12,
            SimdTypeAVX512VPOPCNTDQ = 13,
            SimdTypeAVX512VBMI2 = 14,
            SimdTypeAVX512VNNI = 15
        };

        CpuInfo() {}

        static bool is(SimdType type) {
            switch (type) {
#if defined(__AVX__)
                case SimdTypeAVX:
                    return __builtin_cpu_supports("avx") > 0;
                    break;
#endif
#if defined(__AVX2__)
                case SimdTypeAVX2:
                    return __builtin_cpu_supports("avx2") > 0;
                    break;
#endif
#if defined(__AVX512F__)
                    case SimdTypeAVX512F: return __builtin_cpu_supports("avx512f") > 0; break;
#endif
#if defined(__AVX512VL__)
                    case SimdTypeAVX512VL: return __builtin_cpu_supports("avx512vl") > 0; break;
#endif
#if defined(__AVX512BW__)
                    case SimdTypeAVX512BW: return __builtin_cpu_supports("avx512bw") > 0; break;
#endif
#if defined(__AVX512DQ__)
                    case SimdTypeAVX512DQ: return __builtin_cpu_supports("avx512dq") > 0; break;
#endif
#if defined(__AVX512CD__)
                    case SimdTypeAVX512CD: return __builtin_cpu_supports("avx512cd") > 0; break;
#endif
#if defined(__AVX512ER__)
                    case SimdTypeAVX512ER: return __builtin_cpu_supports("avx512er") > 0; break;
#endif
#if defined(__AVX512PF__)
                    case SimdTypeAVX512PF: return __builtin_cpu_supports("avx512pf") > 0; break;
#endif
#if defined(__AVX512VBMI__)
                    case SimdTypeAVX512VBMI: return __builtin_cpu_supports("avx512vbmi") > 0; break;
#endif
#if defined(__AVX512IFMA__)
                    case SimdTypeAVX512IFMA: return __builtin_cpu_supports("avx512ifma") > 0; break;
#endif
#if defined(__AVX5124VNNIW__)
                    case SimdTypeAVX5124VNNIW: return __builtin_cpu_supports("avx5124vnniw") > 0; break;
#endif
#if defined(__AVX5124FMAPS__)
                    case SimdTypeAVX5124FMAPS: return __builtin_cpu_supports("avx5124fmaps") > 0; break;
#endif
#if defined(__AVX512VPOPCNTDQ__)
                    case SimdTypeAVX512VPOPCNTDQ: return __builtin_cpu_supports("avx512vpopcntdq") > 0; break;
#endif
#if defined(__AVX512VBMI2__)
                    case SimdTypeAVX512VBMI2: return __builtin_cpu_supports("avx512vbmi2") > 0; break;
#endif
#if defined(__AVX512VNNI__)
                    case SimdTypeAVX512VNNI: return __builtin_cpu_supports("avx512vnni") > 0; break;
#endif
                default:
                    break;
            }
            return false;
        }

        static bool isAVX512() { return is(SimdTypeAVX512F); };

        static bool isAVX2() { return is(SimdTypeAVX2); };

        static void showSimdTypes() {
            std::cout << getSupportedSimdTypes() << std::endl;
        }

        static std::string getSupportedSimdTypes() {
            static constexpr char const *simdTypes[] = {"avx", "avx2", "avx512f", "avx512vl",
                                                        "avx512bw", "avx512dq", "avx512cd",
                                                        "avx512er", "avx512pf", "avx512vbmi",
                                                        "avx512ifma", "avx5124vnniw",
                                                        "avx5124fmaps", "avx512vpopcntdq",
                                                        "avx512vbmi2", "avx512vnni"};
            std::string types;
            int size = sizeof(simdTypes) / sizeof(simdTypes[0]);
            for (int i = 0; i <= size; i++) {
                if (is(static_cast<SimdType>(i))) {
                    types += simdTypes[i];
                }
                if (i != size) {
                    types += " ";
                }
            }
            return types;
        }
    };

}  // namespace tann

#endif  // TANN_COMMON_CPU_INFO_H_
