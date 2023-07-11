// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TANN_VECTOR_VECTOR_SET_H_
#define TANN_VECTOR_VECTOR_SET_H_

#include "tann/vector/array.h"

namespace tann {

    class VectorSet {
    public:
        VectorSet();

        virtual ~VectorSet();

        [[nodiscard]] virtual VectorValueType GetValueType() const = 0;

        [[nodiscard]] virtual void *GetVector(SizeType p_vectorID) const = 0;

        [[nodiscard]] virtual void *GetData() const = 0;

        [[nodiscard]] virtual DimensionType Dimension() const = 0;

        [[nodiscard]] virtual SizeType Count() const = 0;

        [[nodiscard]] virtual bool Available() const = 0;

        [[nodiscard]] virtual ErrorCode Save(const std::string &p_vectorFile) const = 0;

        [[nodiscard]] virtual ErrorCode AppendSave(const std::string &p_vectorFile) const = 0;

        [[nodiscard]] virtual SizeType PerVectorDataSize() const = 0;

        virtual void Normalize(int p_threads) = 0;
    };


    class BasicVectorSet : public VectorSet {
    public:
        BasicVectorSet(const ByteArray &p_bytesArray,
                       VectorValueType p_valueType,
                       DimensionType p_dimension,
                       SizeType p_vectorCount);

        ~BasicVectorSet() override;

        [[nodiscard]] VectorValueType GetValueType() const override;

        [[nodiscard]] void *GetVector(SizeType p_vectorID) const override;

        [[nodiscard]] void *GetData() const override;

        [[nodiscard]] DimensionType Dimension() const override;

        [[nodiscard]] SizeType Count() const override;

        [[nodiscard]] bool Available() const override;

        [[nodiscard]] ErrorCode Save(const std::string &p_vectorFile) const override;

        [[nodiscard]] ErrorCode AppendSave(const std::string &p_vectorFile) const override;

        [[nodiscard]] SizeType PerVectorDataSize() const override;

        void Normalize(int p_threads) override;

    private:
        ByteArray m_data;

        VectorValueType m_valueType;

        DimensionType m_dimension;

        SizeType m_vectorCount;

        size_t m_perVectorDataSize;
    };

} // namespace tann

#endif // TANN_VECTOR_VECTOR_SET_H_
