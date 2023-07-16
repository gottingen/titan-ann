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
#include "turbo/container/array_view.h"
#include "turbo/log/logging.h"
#include "tann/distance/primitive_comparator.h"

namespace tann {

    enum MetricType {
        UNDEFINED = 0,
        METRIC_L1,
        METRIC_L2,
        METRIC_IP,
        METRIC_HAMMING,
        METRIC_JACCARD,
        METRIC_COSINE,
        METRIC_ANGLE,
        METRIC_NORMALIZED_COSINE,
        METRIC_NORMALIZED_ANGLE,
        METRIC_NORMALIZED_L2,
        METRIC_POINCARE,
        METRIC_LORENTZ,
    };

    template<typename T>
    class DistanceBase {
    public:
        static constexpr std::size_t requires_alignment = turbo::simd::default_arch::requires_alignment();
        // If an algorithm has a requirement that some data be aligned to a certain
        // boundary it can use this function to indicate that requirement. Currently,
        // we are setting it to  Arch::alignment() for bytes alignment, and the dim alignment is set
        // to alignment_bytes/sizeof(T)
        static constexpr std::size_t alignment_bytes = turbo::simd::default_arch::alignment();
        static constexpr std::size_t alignment = alignment_bytes / sizeof(T);

        static_assert(alignment_bytes % sizeof(T) == 0, "type T is not aligned with alignment_bytes");

        TURBO_DLL DistanceBase(tann::MetricType dist_metric) : _distance_metric(dist_metric) {
        }

        // distance comparison function
        TURBO_DLL virtual float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const = 0;

        // For MIPS, normalization adds an extra dimension to the vectors.
        // This function lets callers know if the normalization process
        // changes the dimension.
        TURBO_DLL virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const {
            return orig_dimension;
        }

        TURBO_DLL virtual tann::MetricType get_metric() const {
            return _distance_metric;
        }

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL virtual bool preprocessing_required() const {
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
        TURBO_DLL virtual void preprocess_base_points(turbo::array_view<T> &original_data) {

        }

        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        TURBO_DLL virtual void
        preprocess_query(const turbo::array_view<T> &query_vec, turbo::array_view<T> &scratch_query) {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            std::memcpy(scratch_query.data(), query_vec.data(), query_vec.size() * sizeof(T));
        }

        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL virtual ~DistanceBase() = default;

    protected:
        tann::MetricType _distance_metric;
    };

    template<typename T>
    class PrimDistanceL1 : public DistanceBase<T> {
    public:
        PrimDistanceL1() : DistanceBase<T>(tann::MetricType::METRIC_L1) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_l1(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL1() = default;

    };

    template<typename T>
    class PrimDistanceL2 : public DistanceBase<T> {
    public:
        PrimDistanceL2() : DistanceBase<T>(tann::MetricType::METRIC_L2) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_l2(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceL2() = default;

    };

    template<typename T>
    class PrimDistanceHamming : public DistanceBase<T> {
    public:
        PrimDistanceHamming() : DistanceBase<T>(tann::MetricType::METRIC_HAMMING) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_hamming(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceHamming() = default;

    };

    template<typename T>
    class PrimDistanceJaccard : public DistanceBase<T> {
    public:
        PrimDistanceJaccard() : DistanceBase<T>(tann::MetricType::METRIC_JACCARD) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_jaccard(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceJaccard() = default;

    };

    template<typename T>
    class PrimDistanceCosine : public DistanceBase<T> {
    public:
        PrimDistanceCosine() : DistanceBase<T>(tann::MetricType::METRIC_COSINE) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_cosine(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceCosine() = default;

    };

    template<typename T>
    class PrimDistanceAngle : public DistanceBase<T> {
    public:
        PrimDistanceAngle() : DistanceBase<T>(tann::MetricType::METRIC_ANGLE) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_angle(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceAngle() = default;

    };

    template<typename T>
    class PrimDistanceIP : public DistanceBase<T> {
    public:
        PrimDistanceIP() : DistanceBase<T>(tann::MetricType::METRIC_IP) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_inner_product(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceIP() = default;

    };

    template<typename T>
    class PrimDistanceNormalizedCosine : public DistanceBase<T> {
    public:
        PrimDistanceNormalizedCosine() : DistanceBase<T>(tann::MetricType::METRIC_NORMALIZED_COSINE) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_normalized_cosine(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedCosine() = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL bool preprocessing_required() const override {
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
        TURBO_DLL void preprocess_base_points(turbo::array_view<T> &arr) override {
            l2_norm(arr);
        }

        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        TURBO_DLL virtual void
        preprocess_query(const turbo::array_view<T> &query_vec, turbo::array_view<T> &scratch_query) {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(query_vec, scratch_query);
        }


    };

    template<typename T>
    class PrimDistanceNormalizedAngle : public DistanceBase<T> {
    public:
        PrimDistanceNormalizedAngle() : DistanceBase<T>(tann::MetricType::METRIC_NORMALIZED_ANGLE) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_normalized_angle(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedAngle() = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL virtual bool preprocessing_required() const override {
            return true;
        }

        TURBO_DLL void preprocess_base_points(turbo::array_view<T> &arr) override {
            l2_norm(arr);
        }

        TURBO_DLL virtual void
        preprocess_query(const turbo::array_view<T> &query_vec, turbo::array_view<T> &scratch_query) {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(query_vec, scratch_query);
        }

    };

    template<typename T>
    class PrimDistanceNormalizedL2 : public DistanceBase<T> {
    public:
        PrimDistanceNormalizedL2() : DistanceBase<T>(tann::MetricType::METRIC_NORMALIZED_L2) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_normalized_l2(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceNormalizedL2() = default;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL virtual bool preprocessing_required() const override {
            return true;
        }

        TURBO_DLL void preprocess_base_points(turbo::array_view<T> &arr) override {
            l2_norm(arr);
        }

        TURBO_DLL virtual void
        preprocess_query(const turbo::array_view<T> &query_vec, turbo::array_view<T> &scratch_query) {
            TLOG_CHECK_LE(query_vec.size(), scratch_query.size(),
                          "input query vector size must little equal than des vector size.");
            l2_norm(query_vec, scratch_query);
        }

    };

    template<typename T>
    class PrimDistancePoincare : public DistanceBase<T> {
    public:
        PrimDistancePoincare() : DistanceBase<T>(tann::MetricType::METRIC_POINCARE) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_poincare(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistancePoincare() = default;
    };

    template<typename T>
    class PrimDistanceLorentz : public DistanceBase<T> {
    public:
        PrimDistanceLorentz() : DistanceBase<T>(tann::MetricType::METRIC_LORENTZ) {}
        // distance comparison function
        TURBO_DLL float compare(const turbo::array_view<T> &a, const turbo::array_view<T> &b) const override {
            return PrimComparator::compare_lorentz(a, b);
        }
        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL ~PrimDistanceLorentz() = default;
    };

}
#endif // TANN_DISTANCE_PRIMITIVE_DISTANCE_H_
