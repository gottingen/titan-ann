// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <string>
#include <stdexcept>
#include <system_error>
#include "turbo/platform/port.h"

#ifndef _WINDOWS
#define __FUNCSIG__ __PRETTY_FUNCTION__
#endif

namespace tann {

    class ANNException : public std::runtime_error {
    public:
        TURBO_DLL ANNException(const std::string &message, int errorCode);

        TURBO_DLL ANNException(const std::string &message, int errorCode, const std::string &funcSig,
                               const std::string &fileName, uint32_t lineNum);

    private:
        int _errorCode;
    };

    class FileException : public ANNException {
    public:
        TURBO_DLL FileException(const std::string &filename, std::system_error &e, const std::string &funcSig,
                                const std::string &fileName, uint32_t lineNum);
    };
} // namespace tann
