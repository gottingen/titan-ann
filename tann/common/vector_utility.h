//
// Created by jeff on 23-7-21.
//

#ifndef TANN_COMMON_VECTOR_UTILITY_H_
#define TANN_COMMON_VECTOR_UTILITY_H_

#include "turbo/random/random.h"

namespace tann {

    template <typename T>
    void random_vector_fill(turbo::Span<T> span, T min, T max) {
        turbo::BitGen  gen;
        for(auto &s : span) {
            s = turbo::Uniform(gen, min, max);
        }
    }
}  // namespace tann

#endif // TANN_COMMON_VECTOR_UTILITY_H_
