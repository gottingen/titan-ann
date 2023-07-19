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

#include "turbo/simd/simd.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"
#include "../distance/test_util.h"
#include "turbo/format/print.h"
#include "turbo/meta/reflect.h"

struct check_supported {
    template<class Arch>
    void operator()(Arch) const {
        turbo::Println("arch name:{}", Arch::name());
        turbo::Println("alignment:{}", Arch::alignment());
        turbo::Println("requires_alignment:{}", Arch::requires_alignment());
    }
};

template<typename T>
struct check_type_supported {
    template<class Arch>
    void operator()(Arch) const {
        turbo::Println("type: {} arch: {} batch size: {}", turbo::nameof_type<T>(), Arch::name(),
                       turbo::simd::batch<T, Arch>::size);
    }
};


TEST_CASE("alignment") {
    turbo::simd::supported_architectures::for_each(check_supported());

}

TEST_CASE_TEMPLATE("alignment", T, SIMD_TEST_TYPES) {
    turbo::simd::supported_architectures::for_each(check_type_supported<T>());

}

TEST_CASE_TEMPLATE("alignment", T, SIMD_TEST_TYPES) {
    turbo::Println("type: {} arch: {} alignment:{} batch size: {}", turbo::nameof_type<T>(),
                   turbo::simd::default_arch::name(),
                   turbo::simd::default_arch::alignment(), turbo::simd::batch<T, turbo::simd::default_arch>::size);

}