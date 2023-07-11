//
// Created by jeff on 23-7-12.
//

#ifndef TANN_CORE_ERROR_CODE_H_
#define TANN_CORE_ERROR_CODE_H_

namespace tann {

    static const int FailedOpenFile = 200;
    static const int ParamNotFound = 201;
    static const int FailedParseValue = 202;
    static const int MemoryOverFlow = 203;
    static const int LackOfInputs = 204;
    static const int VectorNotFound = 205;
    static const int EmptyIndex = 206;
    static const int EmptyData = 207;
    static const int DimensionSizeMismatch = 208;
    static const int ExternalAbort = 209;
    static const int EmptyDiskIO = 210;
    static const int DiskIOFail = 211;

    static const int ReadIni_FailedParseSection = 0x3000;
    static const int ReadIni_FailedParseParam = 0x3001;
    static const int ReadIni_DuplicatedSection = 0x3002;
    static const int ReadIni_DuplicatedParam = 0x3003;


}  // namespace tann

#endif  // TANN_CORE_ERROR_CODE_H_
