// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <functional>
#include <iostream>
#include "turbo/platform/port.h"

#ifdef EXEC_ENV_OLS
#ifndef ENABLE_CUSTOM_LOGGER
#define ENABLE_CUSTOM_LOGGER
#endif // !ENABLE_CUSTOM_LOGGER
#endif // EXEC_ENV_OLS

namespace tann {
    TURBO_DLL extern std::basic_ostream<char> cout;
    TURBO_DLL extern std::basic_ostream<char> cerr;

    enum class TURBO_DLL LogLevel {
        LL_Info = 0,
        LL_Error,
        LL_Count
    };

#ifdef ENABLE_CUSTOM_LOGGER
    TURBO_DLL void SetCustomLogger(std::function<void(LogLevel, const char *)> logger);
#endif
} // namespace tann
