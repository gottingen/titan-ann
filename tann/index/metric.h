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
#ifndef TANN_INDEX_METRIC_H_
#define TANN_INDEX_METRIC_H_

namespace tann {
    
    enum class wMetricType {
        INDEX_METRIC_L2,
        INDEX_METRIC_INNER_PRODUCT,
        INDEX_METRIC_COSINE,
    };
    enum class MetricType {
        MetricTypeNone = -1,
        /// d = 1.0 - sum(Ai*Bi)
        MetricTypeL1 = 0,
        /// d = sum((Ai-Bi)^2)
        MetricTypeL2 = 1,
        MetricTypeHamming = 2,
        MetricTypeAngle = 3,
        /// d = 1.0 - sum(Ai*Bi) / sqrt(sum(Ai*Ai) * sum(Bi*Bi))
        MetricTypeCosine = 4,
        MetricTypeNormalizedAngle = 5,
        /// as the data is normalized
        /// dot product is ok.
        MetricTypeNormalizedCosine = 6,
        MetricTypeJaccard = 7,
        MetricTypeSparseJaccard = 8,
        MetricTypeNormalizedL2 = 9,
        MetricTypePoincare = 100,  // added by Nyapicom
        MetricTypeLorentz = 101  // added by Nyapicom
    };
}
#endif  // TANN_INDEX_METRIC_H_
