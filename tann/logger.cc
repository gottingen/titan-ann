// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iostream>

#include "logger_impl.h"
#include "turbo/platform/port.h"

namespace tann {

    TURBO_DLL ANNStreamBuf coutBuff(stdout);
    TURBO_DLL ANNStreamBuf cerrBuff(stderr);

    TURBO_DLL std::basic_ostream<char> cout(&coutBuff);
    TURBO_DLL std::basic_ostream<char> cerr(&cerrBuff);

#ifdef ENABLE_CUSTOM_LOGGER
    std::function<void(LogLevel, const char *)> g_logger;

    void SetCustomLogger(std::function<void(LogLevel, const char *)> logger)
    {
        g_logger = logger;
        tann::cout << "Set Custom Logger" << std::endl;
    }
#endif

    ANNStreamBuf::ANNStreamBuf(FILE *fp) {
        if (fp == nullptr) {
            throw tann::ANNException("File pointer passed to ANNStreamBuf() cannot be null", -1);
        }
        if (fp != stdout && fp != stderr) {
            throw tann::ANNException("The custom logger only supports stdout and stderr.", -1);
        }
        _fp = fp;
        _logLevel = (_fp == stdout) ? LogLevel::LL_Info : LogLevel::LL_Error;
#ifdef ENABLE_CUSTOM_LOGGER
        _buf = new char[BUFFER_SIZE + 1]; // See comment in the header
#else
        _buf = new char[BUFFER_SIZE]; // See comment in the header
#endif

        std::memset(_buf, 0, (BUFFER_SIZE) * sizeof(char));
        setp(_buf, _buf + BUFFER_SIZE - 1);
    }

    ANNStreamBuf::~ANNStreamBuf() {
        sync();
        _fp = nullptr; // we'll not close because we can't.
        delete[] _buf;
    }

    int ANNStreamBuf::overflow(int c) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (c != EOF) {
            *pptr() = (char) c;
            pbump(1);
        }
        flush();
        return c;
    }

    int ANNStreamBuf::sync() {
        std::lock_guard<std::mutex> lock(_mutex);
        flush();
        return 0;
    }

    int ANNStreamBuf::underflow() {
        throw tann::ANNException("Attempt to read on streambuf meant only for writing.", -1);
    }

    int ANNStreamBuf::flush() {
        const int num = (int) (pptr() - pbase());
        logImpl(pbase(), num);
        pbump(-num);
        return num;
    }

    void ANNStreamBuf::logImpl(char *str, int num) {
#ifdef ENABLE_CUSTOM_LOGGER
        str[num] = '\0'; // Safe. See the c'tor.
        // Invoke the OLS custom logging function.
        if (g_logger)
        {
            g_logger(_logLevel, str);
        }
#else
        fwrite(str, sizeof(char), num, _fp);
        fflush(_fp);
#endif
    }

} // namespace tann
