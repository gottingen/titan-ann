// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TANN_IO_XVEC_READER_H_
#define TANN_IO_XVEC_READER_H_

#include "tann/io/vector_set_reader.h"

namespace tann
{
namespace Helper
{

class XvecVectorReader : public VectorSetReader
{
public:
    XvecVectorReader(std::shared_ptr<ReaderOptions> p_options);

    virtual ~XvecVectorReader();

    virtual ErrorCode LoadFile(const std::string& p_filePaths);

    virtual std::shared_ptr<VectorSet> GetVectorSet(SizeType start = 0, SizeType end = -1) const;

    virtual std::shared_ptr<MetadataSet> GetMetadataSet() const;

private:
    std::string m_vectorOutput;
};



} // namespace Helper
} // namespace tann

#endif // TANN_IO_XVEC_READER_H_
