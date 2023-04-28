// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Copyright 2023 The Tann Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

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

        TURBO_DLL ANNException(const std::string &message, int errorCode,
                               const std::string &funcSig,
                               const std::string &fileName,
                               unsigned int lineNum);

    private:
        int _errorCode;
    };

    class FileException : public ANNException {
    public:
        TURBO_DLL FileException(const std::string &filename,
                                std::system_error &e,
                                const std::string &funcSig,
                                const std::string &fileName,
                                unsigned int lineNum);
    };
}  // namespace tann
