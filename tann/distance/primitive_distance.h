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
#ifndef TANN_DISTANCE_PRIMITIVE_DISTANCE_H_
#define TANN_DISTANCE_PRIMITIVE_DISTANCE_H_

#include "tann/distance/utility.h"
#include "turbo/simd/arch/generic.h"
#include "turbo/meta/span.h"
#include "turbo/log/logging.h"
#include "tann/distance/primitive_comparator.h"
#include "tann/core/allocator.h"
#include "tann/core/types.h"

namespace tann {
    class DistanceBase {
    public:
        TURBO_DLL explicit DistanceBase(tann::MetricType dist_metric) : _distance_metric(dist_metric) {
        }

        // distance comparison function
        TURBO_DLL [[nodiscard]] virtual double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const = 0;

        // For MIPS, normalization adds an extra dimension to the vectors.
        // This function lets callers know if the normalization process
        // changes the dimension.
        TURBO_DLL [[nodiscard]] virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const {
            return orig_dimension;
        }

        TURBO_DLL [[nodiscard]] virtual tann::MetricType get_metric() const {
            return _distance_metric;
        }

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL [[nodiscard]] virtual bool preprocessing_required() const {
            return false;
        }

        // Check the preprocessing_required() function before calling this.
        // Clients can call the function like this:
        //
        //  if (metric->preprocessing_required()){
        //     T* normalized_data_batch;
        //      Split data into batches of batch_size and for each, call:
        //       metric->preprocess_base_points(data_batch, batch_size);
        //
        //  TODO: This does not take into account the case for SSD inner product
        //  where the dimensions change after normalization.
        TURBO_DLL virtual void preprocess_base_points(turbo::Span<uint8_t> original_data, size_t dim) {

        }

        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        TURBO_DLL virtual void
        preprocess_query(turbo::Span<uint8_t> query_vec, turbo::Span<uint8_t> scratch_query) {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            std::memcpy(scratch_query.data(), query_vec.data(), query_vec.size() * sizeof(uint8_t));
        }

        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL virtual ~DistanceBase() = default;

    protected:
        tann::MetricType _distance_metric;
    };

    class PrimDistanceL1Uint8 : public DistanceBase {
    public:
        PrimDistanceL1Uint8() : DistanceBase(tann::MetricType::METRIC_L1) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_l1(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL1Uint8() override = default;

    };

    class PrimDistanceL1Float16 : public DistanceBase {
    public:
        PrimDistanceL1Float16() : DistanceBase(tann::MetricType::METRIC_L1) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_l1(to_span<tann::float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL1Float16() override = default;

    };

    class PrimDistanceL1Float : public DistanceBase {
    public:
        PrimDistanceL1Float() : DistanceBase(tann::MetricType::METRIC_L1) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_l1(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL1Float() override = default;

    };

    class PrimDistanceL1Uint32 : public DistanceBase {
    public:
        PrimDistanceL1Uint32() : DistanceBase(tann::MetricType::METRIC_L1) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_l1(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL1Uint32() override = default;

    };

    class PrimDistanceL2Uint8 : public DistanceBase {
    public:
        PrimDistanceL2Uint8() : DistanceBase(tann::MetricType::METRIC_L2) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_l2(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL2Uint8() override = default;
    };

    class PrimDistanceL2Float16 : public DistanceBase {
    public:
        PrimDistanceL2Float16() : DistanceBase(tann::MetricType::METRIC_L2) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_l2(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL2Float16() override = default;
    };

    class PrimDistanceL2Float : public DistanceBase {
    public:
        PrimDistanceL2Float() : DistanceBase(tann::MetricType::METRIC_L2) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_l2(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL2Float() override = default;
    };

    class PrimDistanceHammingUint8 : public DistanceBase {
    public:
        PrimDistanceHammingUint8() : DistanceBase(tann::MetricType::METRIC_HAMMING) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_hamming(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceHammingUint8() override = default;

    };

    class PrimDistanceJaccardUint8 : public DistanceBase {
    public:
        PrimDistanceJaccardUint8() : DistanceBase(tann::MetricType::METRIC_JACCARD) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_jaccard(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceJaccardUint8() override = default;

    };

    class PrimDistanceCosineUint8 : public DistanceBase {
    public:
        PrimDistanceCosineUint8() : DistanceBase(tann::MetricType::METRIC_COSINE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_cosine(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceCosineUint8() override = default;

    };

    class PrimDistanceCosineFloat16 : public DistanceBase {
    public:
        PrimDistanceCosineFloat16() : DistanceBase(tann::MetricType::METRIC_COSINE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_cosine(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceCosineFloat16() override = default;

    };

    class PrimDistanceCosineFloat : public DistanceBase {
    public:
        PrimDistanceCosineFloat() : DistanceBase(tann::MetricType::METRIC_COSINE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_cosine(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceCosineFloat() override = default;

    };

    class PrimDistanceAngleUint8 : public DistanceBase {
    public:
        PrimDistanceAngleUint8() : DistanceBase(tann::MetricType::METRIC_ANGLE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_angle(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceAngleUint8() override = default;
    };

    class PrimDistanceAngleFloat16 : public DistanceBase {
    public:
        PrimDistanceAngleFloat16() : DistanceBase(tann::MetricType::METRIC_ANGLE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_angle(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceAngleFloat16() override = default;
    };

    class PrimDistanceAngleFloat : public DistanceBase {
    public:
        PrimDistanceAngleFloat() : DistanceBase(tann::MetricType::METRIC_ANGLE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_angle(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceAngleFloat() override = default;
    };

    class PrimDistanceIPUint8 : public DistanceBase {
    public:
        PrimDistanceIPUint8() : DistanceBase(tann::MetricType::METRIC_IP) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_inner_product(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceIPUint8() override = default;

    };

    class PrimDistanceIPFloat16 : public DistanceBase {
    public:
        PrimDistanceIPFloat16() : DistanceBase(tann::MetricType::METRIC_IP) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_inner_product(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceIPFloat16() override = default;

    };

    class PrimDistanceIPFloat : public DistanceBase {
    public:
        PrimDistanceIPFloat() : DistanceBase(tann::MetricType::METRIC_IP) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_inner_product(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceIPFloat() override = default;

    };

    class PrimDistanceNormalizedCosineFloat16 : public DistanceBase {
    public:
        PrimDistanceNormalizedCosineFloat16() : DistanceBase(tann::MetricType::METRIC_NORMALIZED_COSINE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_normalized_cosine(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedCosineFloat16() override = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        [[nodiscard]] TURBO_DLL bool preprocessing_required() const override {
            return true;
        }

        // Check the preprocessing_required() function before calling this.
        // Clients can call the function like this:
        //
        //  if (metric->preprocessing_required()){
        //     T* normalized_data_batch;
        //      Split data into batches of batch_size and for each, call:
        //       metric->preprocess_base_points(data_batch, batch_size);
        //
        //  TODO: This does not take into account the case for SSD inner product
        //  where the dimensions change after normalization.
        TURBO_DLL void preprocess_base_points(turbo::Span<uint8_t> arr, size_t dim) override {
            size_t nvec = arr.size() / (dim * sizeof(float16));
            for (size_t i = 0; i < nvec; ++i) {
                auto one_v = to_span<float16>(arr);
                l2_norm(one_v);
            }
        }

        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        TURBO_DLL void
        preprocess_query(turbo::Span<uint8_t> query_vec, turbo::Span<uint8_t> scratch_query) override {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(to_span<float16>(query_vec), to_span<float16>(scratch_query));
        }


    };

    class PrimDistanceNormalizedCosineFloat : public DistanceBase {
    public:
        PrimDistanceNormalizedCosineFloat() : DistanceBase(tann::MetricType::METRIC_NORMALIZED_COSINE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_normalized_cosine(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedCosineFloat() override = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        [[nodiscard]] TURBO_DLL bool preprocessing_required() const override {
            return true;
        }

        // Check the preprocessing_required() function before calling this.
        // Clients can call the function like this:
        //
        //  if (metric->preprocessing_required()){
        //     T* normalized_data_batch;
        //      Split data into batches of batch_size and for each, call:
        //       metric->preprocess_base_points(data_batch, batch_size);
        //
        //  TODO: This does not take into account the case for SSD inner product
        //  where the dimensions change after normalization.
        TURBO_DLL void preprocess_base_points(turbo::Span<uint8_t> arr, size_t dim) override {
            size_t nvec = arr.size() / (dim * sizeof(float));
            for (size_t i = 0; i < nvec; ++i) {
                turbo::Span<float> one_v = to_span<float>(arr);
                l2_norm(one_v);
            }
        }

        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        TURBO_DLL void
        preprocess_query(turbo::Span<uint8_t> query_vec, turbo::Span<uint8_t> scratch_query) override {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            auto ad = to_span<float>(scratch_query);
            l2_norm(to_span<float>(query_vec), ad);
        }


    };

    class PrimDistanceNormalizedAngleFloat16 : public DistanceBase {
    public:
        PrimDistanceNormalizedAngleFloat16() : DistanceBase(tann::MetricType::METRIC_NORMALIZED_ANGLE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_normalized_angle(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedAngleFloat16() override = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL [[nodiscard]] bool preprocessing_required() const override {
            return true;
        }

        TURBO_DLL void preprocess_base_points(turbo::Span<uint8_t> arr, size_t dim) override {
            size_t nvec = arr.size() / (dim * sizeof(float16));
            for (size_t i = 0; i < nvec; ++i) {
                auto one_v = to_span<float16>(arr);
                l2_norm(one_v);
            }
        }

        TURBO_DLL void
        preprocess_query(turbo::Span<uint8_t> query_vec, turbo::Span<uint8_t> scratch_query) override {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(to_span<float16>(query_vec), to_span<float16>(scratch_query));
        }

    };

    class PrimDistanceNormalizedAngleFloat : public DistanceBase {
    public:
        PrimDistanceNormalizedAngleFloat() : DistanceBase(tann::MetricType::METRIC_NORMALIZED_ANGLE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_normalized_angle(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedAngleFloat() override = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL [[nodiscard]] bool preprocessing_required() const override {
            return true;
        }

        TURBO_DLL void preprocess_base_points(turbo::Span<uint8_t> arr, size_t dim) override {
            size_t nvec = arr.size() / (dim * sizeof(float));
            for (size_t i = 0; i < nvec; ++i) {
                auto one_v = to_span<float>(arr);
                l2_norm(one_v);
            }
        }

        TURBO_DLL void
        preprocess_query(turbo::Span<uint8_t> query_vec, turbo::Span<uint8_t> scratch_query) override {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(to_span<float>(query_vec), to_span<float>(scratch_query));
        }

    };


    class PrimDistanceNormalizedL2Float16 : public DistanceBase {
    public:
        PrimDistanceNormalizedL2Float16() : DistanceBase(tann::MetricType::METRIC_NORMALIZED_L2) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_normalized_l2(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedL2Float16() override = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL [[nodiscard]] bool preprocessing_required() const override {
            return true;
        }

        TURBO_DLL void preprocess_base_points(turbo::Span<uint8_t> arr, size_t dim) override {
            size_t nvec = arr.size() / (dim * sizeof(float16));
            for (size_t i = 0; i < nvec; ++i) {
                auto one_v = to_span<float16>(arr);
                l2_norm(one_v);
            }
        }

        TURBO_DLL void
        preprocess_query(turbo::Span<uint8_t> query_vec, turbo::Span<uint8_t> scratch_query) override {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(to_span<float16>(query_vec), to_span<float16>(scratch_query));
        }

    };

    class PrimDistanceNormalizedL2Float : public DistanceBase {
    public:
        PrimDistanceNormalizedL2Float() : DistanceBase(tann::MetricType::METRIC_NORMALIZED_L2) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_normalized_l2(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedL2Float() override = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL [[nodiscard]] bool preprocessing_required() const override {
            return true;
        }

        TURBO_DLL void preprocess_base_points(turbo::Span<uint8_t> arr, size_t dim) override {
            size_t nvec = arr.size() / (dim * sizeof(float));
            for (size_t i = 0; i < nvec; ++i) {
                auto one_v = to_span<float>(arr);
                l2_norm(one_v);
            }
        }

        TURBO_DLL void
        preprocess_query(turbo::Span<uint8_t> query_vec, turbo::Span<uint8_t> scratch_query) override {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(to_span<float>(query_vec), to_span<float>(scratch_query));
        }

    };

    class PrimDistancePoincareFloat16 : public DistanceBase {
    public:
        PrimDistancePoincareFloat16() : DistanceBase(tann::MetricType::METRIC_POINCARE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_poincare(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistancePoincareFloat16() override = default;
    };

    class PrimDistancePoincareFloat : public DistanceBase {
    public:
        PrimDistancePoincareFloat() : DistanceBase(tann::MetricType::METRIC_POINCARE) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_poincare(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistancePoincareFloat() override = default;
    };

    class PrimDistanceLorentzFloat16 : public DistanceBase {
    public:
        PrimDistanceLorentzFloat16() : DistanceBase(tann::MetricType::METRIC_LORENTZ) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_lorentz(to_span<float16>(a), to_span<float16>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceLorentzFloat16() override = default;
    };

    class PrimDistanceLorentzFloat : public DistanceBase {
    public:
        PrimDistanceLorentzFloat() : DistanceBase(tann::MetricType::METRIC_LORENTZ) {}
        // distance comparison function
        TURBO_DLL [[nodiscard]] double
        compare(turbo::Span<uint8_t> a, turbo::Span<uint8_t> b) const override {
            return PrimComparator::compare_lorentz(to_span<float>(a), to_span<float>(b));
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceLorentzFloat() override = default;
    };

}
#endif // TANN_DISTANCE_PRIMITIVE_DISTANCE_H_
