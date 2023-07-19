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
#ifndef TANN_DISTANCE_DISTANCE_FACTORY_H_
#define TANN_DISTANCE_DISTANCE_FACTORY_H_

#include <type_traits>
#include "tann/distance/primitive_distance.h"
#include "tann/core/types.h"
#include "turbo/base/result_status.h"
#include "turbo/meta/reflect.h"
#include "turbo/log/logging.h"

namespace tann {

    class DistanceFactory {
    public:
        static inline turbo::ResultStatus<DistanceBase*> create_distance_factor(MetricType m, DataType dt) {
            if (dt == DataType::DT_FLOAT16) {
                switch (m) {
                    case METRIC_L1:{
                        return new PrimDistanceL1Float16();
                    }
                    case METRIC_L2: {
                        return new PrimDistanceL2Float16();
                    }
                    case METRIC_IP: {
                        return new PrimDistanceIPFloat16();
                    }
                    case METRIC_HAMMING:
                    case METRIC_JACCARD:{
                        return turbo::UnavailableError("float type do support {} type metric", turbo::nameof_enum(m));
                    }
                    case METRIC_COSINE: {
                        return new PrimDistanceCosineFloat16();
                    }
                    case METRIC_ANGLE: {
                        return new PrimDistanceAngleFloat16();
                    }
                    case METRIC_NORMALIZED_COSINE: {
                        return new PrimDistanceNormalizedCosineFloat16();
                    }
                    case METRIC_NORMALIZED_ANGLE: {
                        return new PrimDistanceNormalizedAngleFloat16();
                    }
                    case METRIC_NORMALIZED_L2: {
                        return new PrimDistanceNormalizedL2Float16();
                    }
                    case METRIC_POINCARE: {
                        return new PrimDistancePoincareFloat16();
                    }
                    case METRIC_LORENTZ: {
                        return new PrimDistanceLorentzFloat16();
                    }
                    default:
                        return turbo::UnknownError("unknown distance type: {}", static_cast<int>(m));
                }
            } else if (dt == DataType::DT_FLOAT) {
                switch (m) {
                    case METRIC_L1:{
                        return new PrimDistanceL1Float();
                    }
                    case METRIC_L2: {
                        return new PrimDistanceL2Float();
                    }
                    case METRIC_IP: {
                        return new PrimDistanceIPFloat();
                    }
                    case METRIC_HAMMING:
                    case METRIC_JACCARD:{
                        return turbo::UnavailableError("float type do support {} type metric", turbo::nameof_enum(m));
                    }
                    case METRIC_COSINE: {
                        return new PrimDistanceCosineFloat();
                    }
                    case METRIC_ANGLE: {
                        return new PrimDistanceAngleFloat();
                    }
                    case METRIC_NORMALIZED_COSINE: {
                        return new PrimDistanceNormalizedCosineFloat();
                    }
                    case METRIC_NORMALIZED_ANGLE: {
                        return new PrimDistanceNormalizedAngleFloat();
                    }
                    case METRIC_NORMALIZED_L2: {
                        return new PrimDistanceNormalizedL2Float();
                    }
                    case METRIC_POINCARE: {
                        return new PrimDistancePoincareFloat();
                    }
                    case METRIC_LORENTZ: {
                        return new PrimDistanceLorentzFloat();
                    }
                    default:
                        return turbo::UnknownError("unknown distance type: {}", static_cast<int>(m));
                }
            } else if (dt == DataType::DT_UINT8) {
                switch (m) {
                    case METRIC_HAMMING:{
                        return new PrimDistanceHammingUint8();
                    }
                    case METRIC_JACCARD:{
                        return new PrimDistanceJaccardUint8();
                    }
                    case METRIC_L1:{
                        return new PrimDistanceL1Uint8();
                    }
                    case METRIC_L2: {
                        return new PrimDistanceL2Uint8();
                    }
                    case METRIC_IP: {
                        return new PrimDistanceIPUint8();
                    }
                    case METRIC_COSINE: {
                        TLOG_WARN("Although tann supports the {} measurement"
                                  "method of this data type: uint8_t, it is not recommended"
                                  "to use this type of integer data for this type of "
                                  "measurement. It is recommended to use the floating "
                                  "point data type to avoid the loss of numerical accuracy.",
                                  turbo::nameof_enum(m));
                        return new PrimDistanceCosineUint8();
                    }
                    case METRIC_ANGLE: {
                        TLOG_WARN("Although tann supports the {} measurement"
                                  "method of this data type: uint8_t, it is not recommended"
                                  "to use this type of integer data for this type of "
                                  "measurement. It is recommended to use the floating "
                                  "point data type to avoid the loss of numerical accuracy.",
                                  turbo::nameof_enum(m));
                        return new PrimDistanceAngleUint8();
                    }
                    case METRIC_NORMALIZED_COSINE:
                    case METRIC_NORMALIZED_ANGLE:
                    case METRIC_NORMALIZED_L2:
                    case METRIC_POINCARE:
                    case METRIC_LORENTZ: {
                        return turbo::UnavailableError("float type do support {} type metric", turbo::nameof_enum(m));
                    }
                    default:
                        return turbo::UnknownError("unknown distance type: {}", static_cast<int>(m));
                }
            }
            return nullptr;
        }
    };
}  // namespace tann

#endif  // TANN_DISTANCE_DISTANCE_FACTORY_H_
