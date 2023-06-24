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

#ifndef TANN_COMMON_UTILITY_H_
#define TANN_COMMON_UTILITY_H_

#include <string_view>
#include <string>
#include <vector>
#include <sstream>
#include "tann/common/exception.h"
#include <iostream>
#include <sys/time.h>
#include <fcntl.h>
#include <cstdlib>
#include <unistd.h>

namespace tann {

    class Common {
    public:
        static void tokenize(const std::string &str, std::vector<std::string> &token, const std::string seps) {
            std::string::size_type current = 0;
            std::string::size_type next;
            while ((next = str.find_first_of(seps, current)) != std::string::npos) {
                token.push_back(str.substr(current, next - current));
                current = next + 1;
            }
            std::string t = str.substr(current);
            token.push_back(t);
        }

        static double strtod(const std::string &str) {
            char *e;
            double val = std::strtod(str.c_str(), &e);
            if (*e != 0) {
                std::stringstream msg;
                msg << "Invalid string. " << e;
                TANN_THROW(msg);
            }
            return val;
        }

        static float strtof(const std::string &str) {
            char *e;
            double val = std::strtof(str.c_str(), &e);
            if (*e != 0) {
                std::stringstream msg;
                msg << "Invalid string. " << e;
                TANN_THROW(msg);
            }
            return val;
        }

        static long strtol(const std::string &str, int base = 10) {
            char *e;
            long val = std::strtol(str.c_str(), &e, base);
            if (*e != 0) {
                std::stringstream msg;
                msg << "Invalid string. " << e;
                TANN_THROW(msg);
            }
            return val;
        }


        template<typename T>
        static void extractVector(const std::string &textLine, const std::string &sep, T &object) {
            std::vector<std::string> tokens;
            tann::Common::tokenize(textLine, tokens, sep);
            size_t idx;
            for (idx = 0; idx < tokens.size(); idx++) {
                if (tokens[idx].size() == 0) {
                    std::stringstream msg;
                    msg << "Common::extractVecotFromText: No data. " << textLine;
                    TANN_THROW(msg);
                }
                char *e;
                double v = ::strtod(tokens[idx].c_str(), &e);
                if (*e != 0) {
                    std::cerr << "VectorSpace::readText: Warning! Not numerical value. [" << e << "]" << std::endl;
                    break;
                }
                object.push_back(v);
            }
        }


        static std::string getProcessStatus(const std::string &stat) {
            pid_t pid = getpid();
            std::stringstream str;
            str << "/proc/" << pid << "/status";
            std::ifstream procStatus(str.str());
            if (!procStatus.fail()) {
                std::string line;
                while (getline(procStatus, line)) {
                    std::vector<std::string> tokens;
                    tann::Common::tokenize(line, tokens, ": \t");
                    if (tokens[0] == stat) {
                        for (size_t i = 1; i < tokens.size(); i++) {
                            if (tokens[i].empty()) {
                                continue;
                            }
                            return tokens[i];
                        }
                    }
                }
            }
            return "-1";
        }

        // size unit is kbyte
        static int getProcessVmSize() { return strtol(getProcessStatus("VmSize")); }

        static int getProcessVmPeak() { return strtol(getProcessStatus("VmPeak")); }

        static int getProcessVmRSS() { return strtol(getProcessStatus("VmRSS")); }

        static std::string sizeToString(float size) {
            char unit = 'K';
            if (size > 1024) {
                size /= 1024;
                unit = 'M';
            }
            if (size > 1024) {
                size /= 1024;
                unit = 'G';
            }
            size = round(size * 100) / 100;
            std::stringstream str;
            str << size << unit;
            return str.str();
        }

        static std::string getProcessVmSizeStr() { return sizeToString(getProcessVmSize()); }

        static std::string getProcessVmPeakStr() { return sizeToString(getProcessVmPeak()); }

        static std::string getProcessVmRSSStr() { return sizeToString(getProcessVmRSS()); }
    };

}  // namespace

#endif  // TANN_COMMON_UTILITY_H_
