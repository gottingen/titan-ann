#pragma once

#include "turbo/platform/port.h"
#include <cstring>

namespace tann {
    enum Metric {
        UN_DEFINE = 0,
        L1,
        L2,
        INNER_PRODUCT ,
        COSINE,
        FAST_L2
    };

    template<typename T>
    class Distance {
    public:
        TURBO_DLL Distance(tann::Metric dist_metric) : _distance_metric(dist_metric) {
        }

        // distance comparison function
        TURBO_DLL virtual float compare(const T *a, const T *b, uint32_t length) const = 0;

        // Needed only for COSINE-BYTE and INNER_PRODUCT-BYTE
        TURBO_DLL virtual float compare(const T *a, const T *b, const float normA, const float normB,
                                        uint32_t length) const;

        // For MIPS, normalization adds an extra dimension to the vectors.
        // This function lets callers know if the normalization process
        // changes the dimension.
        TURBO_DLL virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const;

        TURBO_DLL virtual tann::Metric get_metric() const;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        TURBO_DLL virtual bool preprocessing_required() const;

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
        TURBO_DLL virtual void preprocess_base_points(T *original_data, const size_t orig_dim,
                                                      const size_t num_points);

        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        TURBO_DLL virtual void preprocess_query(const T *query_vec, const size_t query_dim, T *scratch_query);

        // If an algorithm has a requirement that some data be aligned to a certain
        // boundary it can use this function to indicate that requirement. Currently,
        // we are setting it to 8 because that works well for AVX2. If we have AVX512
        // implementations of distance algos, they might have to set this to 16
        // (depending on how they are implemented)
        TURBO_DLL virtual size_t get_required_alignment() const;

        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        TURBO_DLL virtual ~Distance();

    protected:
        tann::Metric _distance_metric;
        size_t _alignment_factor = 8;
    };

    class DistanceCosineInt8 : public Distance<int8_t> {
    public:
        DistanceCosineInt8() : Distance<int8_t>(tann::Metric::COSINE) {
        }

        TURBO_DLL virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
    };

    class DistanceL2Int8 : public Distance<int8_t> {
    public:
        DistanceL2Int8() : Distance<int8_t>(tann::Metric::L2) {
        }

        TURBO_DLL virtual float compare(const int8_t *a, const int8_t *b, uint32_t size) const;
    };

// AVX implementations. Borrowed from HNSW code.
    class AVXDistanceL2Int8 : public Distance<int8_t> {
    public:
        AVXDistanceL2Int8() : Distance<int8_t>(tann::Metric::L2) {
        }

        TURBO_DLL virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
    };

    class DistanceCosineFloat : public Distance<float> {
    public:
        DistanceCosineFloat() : Distance<float>(tann::Metric::COSINE) {
        }

        TURBO_DLL virtual float compare(const float *a, const float *b, uint32_t length) const;
    };

    class DistanceL2Float : public Distance<float> {
    public:
        DistanceL2Float() : Distance<float>(tann::Metric::L2) {
        }

#ifdef _WINDOWS
        TURBO_DLL virtual float compare(const float *a, const float *b, uint32_t size) const;
#else
        TURBO_DLL virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));

#endif
    };

    class AVXDistanceL2Float : public Distance<float> {
    public:
        AVXDistanceL2Float() : Distance<float>(tann::Metric::L2) {
        }

        TURBO_DLL virtual float compare(const float *a, const float *b, uint32_t length) const;
    };

    template<typename T>
    class SlowDistanceL2 : public Distance<T> {
    public:
        SlowDistanceL2() : Distance<T>(tann::Metric::L2) {
        }

        TURBO_DLL virtual float compare(const T *a, const T *b, uint32_t length) const;
    };

    class SlowDistanceCosineUInt8 : public Distance<uint8_t> {
    public:
        SlowDistanceCosineUInt8() : Distance<uint8_t>(tann::Metric::COSINE) {
        }

        TURBO_DLL virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t length) const;
    };

    class DistanceL2UInt8 : public Distance<uint8_t> {
    public:
        DistanceL2UInt8() : Distance<uint8_t>(tann::Metric::L2) {
        }

        TURBO_DLL virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const;
    };

    template<typename T>
    class DistanceInnerProduct : public Distance<T> {
    public:
        DistanceInnerProduct() : Distance<T>(tann::Metric::INNER_PRODUCT) {
        }

        DistanceInnerProduct(tann::Metric metric) : Distance<T>(metric) {
        }

        inline float inner_product(const T *a, const T *b, unsigned size) const;

        inline float compare(const T *a, const T *b, unsigned size) const {
            float result = inner_product(a, b, size);
            //      if (result < 0)
            //      return std::numeric_limits<float>::max();
            //      else
            return -result;
        }
    };

    template<typename T>
    class DistanceFastL2 : public DistanceInnerProduct<T> {
        // currently defined only for float.
        // templated for future use.
    public:
        DistanceFastL2() : DistanceInnerProduct<T>(tann::Metric::FAST_L2) {
        }

        float norm(const T *a, unsigned size) const;

        float compare(const T *a, const T *b, float norm, unsigned size) const;
    };

    class AVXDistanceInnerProductFloat : public Distance<float> {
    public:
        AVXDistanceInnerProductFloat() : Distance<float>(tann::Metric::INNER_PRODUCT) {
        }

        TURBO_DLL virtual float compare(const float *a, const float *b, uint32_t length) const;
    };

    class AVXNormalizedCosineDistanceFloat : public Distance<float> {
    private:
        AVXDistanceInnerProductFloat _innerProduct;

    protected:
        void normalize_and_copy(const float *a, uint32_t length, float *a_norm) const;

    public:
        AVXNormalizedCosineDistanceFloat() : Distance<float>(tann::Metric::COSINE) {
        }

        TURBO_DLL virtual float compare(const float *a, const float *b, uint32_t length) const {
            // Inner product returns negative values to indicate distance.
            // This will ensure that cosine is between -1 and 1.
            return 1.0f + _innerProduct.compare(a, b, length);
        }

        TURBO_DLL virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const override;

        TURBO_DLL virtual bool preprocessing_required() const;

        TURBO_DLL virtual void preprocess_base_points(float *original_data, const size_t orig_dim,
                                                      const size_t num_points) override;

        TURBO_DLL virtual void preprocess_query(const float *query_vec, const size_t query_dim,
                                                float *scratch_query_vector) override;
    };

    template<typename T>
    Distance<T> *get_distance_function(Metric m);

} // namespace tann
