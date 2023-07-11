// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tann/io/xvec_reader.h"
#include "tann/utility/CommonHelper.h"

#include <time.h>

using namespace tann;
using namespace tann::Helper;

XvecVectorReader::XvecVectorReader(std::shared_ptr<ReaderOptions> p_options)
        : VectorSetReader(p_options) {
    std::string tempFolder("tempfolder");
    if (!direxists(tempFolder.c_str())) {
        mkdir(tempFolder.c_str());
    }
    std::srand(clock());
    m_vectorOutput = tempFolder + FolderSep + "vectorset.bin." + std::to_string(std::rand());
}


XvecVectorReader::~XvecVectorReader() {
    if (fileexists(m_vectorOutput.c_str())) {
        remove(m_vectorOutput.c_str());
    }
}


ErrorCode
XvecVectorReader::LoadFile(const std::string &p_filePaths) {
    const auto &files = Helper::StrUtils::SplitString(p_filePaths, ",");
    auto fp = f_createIO();
    if (fp == nullptr || !fp->Initialize(m_vectorOutput.c_str(), std::ios::binary | std::ios::out)) {
        TLOG_ERROR("Failed to write file: {} ", m_vectorOutput.c_str());
        return ErrorCode::FailedCreateFile;
    }
    SizeType vectorCount = 0;
    IOBINARY(fp, WriteBinary, sizeof(vectorCount), (char *) &vectorCount);
    IOBINARY(fp, WriteBinary, sizeof(m_options->m_dimension), (char *) &(m_options->m_dimension));

    size_t vectorDataSize = GetValueTypeSize(m_options->m_inputValueType) * m_options->m_dimension;
    std::unique_ptr<char[]> buffer(new char[vectorDataSize]);
    for (std::string file: files) {
        auto ptr = f_createIO();
        if (ptr == nullptr || !ptr->Initialize(file.c_str(), std::ios::binary | std::ios::in)) {
            TLOG_ERROR("Failed to read file: {} ", file.c_str());
            return ErrorCode::FailedOpenFile;
        }
        while (true) {
            DimensionType dim;
            if (ptr->ReadBinary(sizeof(DimensionType), (char *) &dim) == 0) break;

            if (dim != m_options->m_dimension) {
                TLOG_ERROR(
                             "Xvec file {} has No.{} vector whose dims are not as many as expected. Expected: {}, Fact: {}",
                             file.c_str(), vectorCount, m_options->m_dimension, dim);
                return ErrorCode::DimensionSizeMismatch;
            }
            IOBINARY(ptr, ReadBinary, vectorDataSize, buffer.get());
            IOBINARY(fp, WriteBinary, vectorDataSize, buffer.get());
            vectorCount++;
        }
    }
    IOBINARY(fp, WriteBinary, sizeof(vectorCount), (char *) &vectorCount, 0);
    return ErrorCode::Success;
}


std::shared_ptr<VectorSet>
XvecVectorReader::GetVectorSet(SizeType start, SizeType end) const {
    auto ptr = f_createIO();
    if (ptr == nullptr || !ptr->Initialize(m_vectorOutput.c_str(), std::ios::binary | std::ios::in)) {
        TLOG_ERROR("Failed to read file {}.", m_vectorOutput.c_str());
        throw std::runtime_error("Failed read file");
    }

    SizeType row;
    DimensionType col;
    if (ptr->ReadBinary(sizeof(SizeType), (char *) &row) != sizeof(SizeType)) {
        TLOG_ERROR("Failed to read VectorSet!");
        throw std::runtime_error("Failed read file");
    }
    if (ptr->ReadBinary(sizeof(DimensionType), (char *) &col) != sizeof(DimensionType)) {
        TLOG_ERROR("Failed to read VectorSet!");
        throw std::runtime_error("Failed read file");
    }

    if (start > row) start = row;
    if (end < 0 || end > row) end = row;
    std::uint64_t totalRecordVectorBytes =
            ((std::uint64_t) GetValueTypeSize(m_options->m_inputValueType)) * (end - start) * col;
    ByteArray vectorSet;
    if (totalRecordVectorBytes > 0) {
        vectorSet = ByteArray::Alloc(totalRecordVectorBytes);
        char *vecBuf = reinterpret_cast<char *>(vectorSet.Data());
        std::uint64_t offset =
                ((std::uint64_t) GetValueTypeSize(m_options->m_inputValueType)) * start * col + +sizeof(SizeType) +
                sizeof(DimensionType);
        if (ptr->ReadBinary(totalRecordVectorBytes, vecBuf, offset) != totalRecordVectorBytes) {
            TLOG_ERROR("Failed to read VectorSet!");
            throw std::runtime_error("Failed read file");
        }
    }
    return std::shared_ptr<VectorSet>(new BasicVectorSet(vectorSet,
                                                         m_options->m_inputValueType,
                                                         col,
                                                         end - start));
}


std::shared_ptr<MetadataSet>
XvecVectorReader::GetMetadataSet() const {
    return nullptr;
}
