// Copyright 2023 The titan-search Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef TANN_COMMON_STD_OSTREAM_REDIRECTOR_H_
#define TANN_COMMON_STD_OSTREAM_REDIRECTOR_H_

#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <climits>
#include <iomanip>
#include <algorithm>
#include <typeinfo>

#include <sys/time.h>
#include <fcntl.h>

namespace tann {

    class StdOstreamRedirector {
    public:
        StdOstreamRedirector(bool e = false, const std::string path = "/dev/null",
                             mode_t m = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, int f = 2) {
            logFilePath = path;
            mode = m;
            logFD = -1;
            fdNo = f;
            enabled = e;
        }

        ~StdOstreamRedirector() { end(); }

        void enable() { enabled = true; }

        void disable() { enabled = false; }

        void set(bool e) { enabled = e; }

        void bgin(bool e) {
            set(e);
            begin();
        }

        void begin() {
            if (!enabled) {
                return;
            }
            if (logFilePath == "/dev/null") {
                logFD = open(logFilePath.c_str(), O_WRONLY | O_APPEND, mode);
            } else {
                logFD = open(logFilePath.c_str(), O_CREAT | O_WRONLY | O_APPEND, mode);
            }
            if (logFD < 0) {
                std::cerr << "Logger: Cannot begin logging." << std::endl;
                logFD = -1;
                return;
            }
            savedFdNo = dup(fdNo);
            std::cerr << std::flush;
            dup2(logFD, fdNo);
        }

        void end() {
            if (logFD < 0) {
                return;
            }
            std::cerr << std::flush;
            dup2(savedFdNo, fdNo);
            savedFdNo = -1;
        }

        std::string logFilePath;
        mode_t mode;
        int logFD;
        int savedFdNo;
        int fdNo;
        bool enabled;
    };


}  // namespace tann

#endif  // TANN_COMMON_STD_OSTREAM_REDIRECTOR_H_
