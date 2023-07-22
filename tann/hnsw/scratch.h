//
// Created by jeff on 23-7-22.
//

#ifndef TANN_HNSW_SCRATCH_H_
#define TANN_HNSW_SCRATCH_H_

#include "tann/core/types.h"

namespace tann {

    struct HnswScratch {
        bool is_deleted{false};
        label_type label{0};
    };
}  // namespace tann
#endif  // TANN_HNSW_SCRATCH_H_
