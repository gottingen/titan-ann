//
// Created by jeff on 23-6-20.
//

//#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <iostream>
//#include <doctest/doctest.h>
#include "turbo/simd/simd.h"
#include "turbo/times/stop_watcher.h"
#include "turbo/log/logging.h"
#include <cstdlib>

#define TLOG_CRASH_ON

void simple_test(int t) {
    std::vector<float> a(128, 2.3);
    std::vector<float> b(128, 0.0);
    for (int j = 0; j < t; ++j) {
        for (int i = 0; i < a.size(); ++i) {
            b[i] = a[i] * a[i];
        }
    }

}

void simd_test(int t) {
    std::vector<float> src(128, 2.3);
    constexpr float sd = 2.3 * 2.3;
    for (int k = 0; k < t; k++) {
        turbo::simd::batch<float> a = turbo::simd::load_unaligned(static_cast<float *>(src.data()));
        auto c = turbo::simd::mul(a, a);
        //std::cout << c.get(0) - sd << std::endl;

    }
}

int main() {
    turbo::StopWatcher sw1;
    simple_test(20000000);
    std::cout << sw1.elapsed_micro() << std::endl;

    turbo::StopWatcher sw;
    simd_test(20000000);
    std::cout << sw1.elapsed_micro() << std::endl;
}

