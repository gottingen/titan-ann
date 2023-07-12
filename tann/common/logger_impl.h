// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <sstream>
#include <mutex>

#include "tann/common/ann_exception.h"
#include "tann/common/logger.h"

namespace tann {
    class ANNStreamBuf : public std::basic_streambuf<char> {
    public:
        TURBO_DLL explicit ANNStreamBuf(FILE *fp);

        TURBO_DLL ~ANNStreamBuf();

        TURBO_DLL bool is_open() const {
            return true; // because stdout and stderr are always open.
        }

        TURBO_DLL void close();

        TURBO_DLL virtual int underflow();

        TURBO_DLL virtual int overflow(int c);

        TURBO_DLL virtual int sync();

    private:
        FILE *_fp;
        char *_buf;
        int _bufIndex;
        std::mutex _mutex;
        LogLevel _logLevel;

        int flush();

        void logImpl(char *str, int numchars);

// Why the two buffer-sizes? If we are running normally, we are basically
// interacting with a character output system, so we short-circuit the
// output process by keeping an empty buffer and writing each character
// to stdout/stderr. But if we are running in OLS, we have to take all
// the text that is written to tann::cout/tann:cerr, consolidate it
// and push it out in one-shot, because the OLS infra does not give us
// character based output. Therefore, we use a larger buffer that is large
// enough to store the longest message, and continuously add characters
// to it. When the calling code outputs a std::endl or std::flush, sync()
// will be called and will output a log level, component name, and the text
// that has been collected. (sync() is also called if the buffer is full, so
// overflows/missing text are not a concern).
// This implies calling code _must_ either print std::endl or std::flush
// to ensure that the message is written immediately.
#ifdef ENABLE_CUSTOM_LOGGER
        static const int BUFFER_SIZE = 1024;
#else
        // Allocating an arbitrarily small buffer here because the overflow() and
        // other function implementations push the BUFFER_SIZE chars into the
        // buffer before flushing to fwrite.
        static const int BUFFER_SIZE = 4;
#endif

        ANNStreamBuf(const ANNStreamBuf &);

        ANNStreamBuf &operator=(const ANNStreamBuf &);
    };
} // namespace tann
