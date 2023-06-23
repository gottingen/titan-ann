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

#ifndef TANN_COMMON_ARGS_H_
#define TANN_COMMON_ARGS_H_

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <set>
#include <sstream>
#include "tann/common/exception.h"

namespace tann {

    class Args : public std::map<std::string, std::string> {
    public:
        Args() {}

        Args(int argc, char **argv, std::string noVal = "") {
            argC = argc;
            argV = argv;
            parse(noVal);
        }

        void parse(std::string noVal = "") {
            auto argc = argC;
            auto argv = argV;
            clear();
            std::vector<std::string> opts;
            int optcount = 0;
            insert(std::make_pair(std::string("#-"), std::string(argv[0])));
            noVal += "h";
            for (int i = 1; i < argc; ++i) {
                opts.push_back(std::string(argv[i]));
                if ((argv[i][0] == '-') && (noVal.find(argv[i][1]) != std::string::npos)) {
                    opts.push_back("t");
                }
            }
            for (auto i = opts.begin(); i != opts.end(); ++i) {
                std::string &opt = *i;
                std::string key, value;
                if (opt.size() > 2 && opt.substr(0, 2) == "--") {
                    auto pos = opt.find('=');
                    if (pos == std::string::npos) {
                        key = opt.substr(2);
                        value = "";
                    } else {
                        key = opt.substr(2, pos - 2);
                        value = opt.substr(++pos);
                    }
                } else if (opt.size() > 1 && opt[0] == '-') {
                    if (opt.size() == 2) {
                        key = opt[1];
                        ++i;
                        if (i != opts.end()) {
                            value = *i;
                        } else {
                            value = "";
                            --i;
                        }
                    } else {
                        key = opt[1];
                        value = opt.substr(2);
                    }
                } else {
                    key = "#" + std::to_string(optcount++);
                    value = opt;
                }
                auto status = insert(std::make_pair(key, value));
                if (!status.second) {
                    std::cerr << "Args: Duplicated options. [" << opt << "]" << std::endl;
                }
            }
        }

        std::set<std::string> getUnusedOptions() {
            std::set<std::string> o;
            for (auto i = begin(); i != end(); ++i) {
                o.insert((*i).first);
            }
            for (auto i = usedOptions.begin(); i != usedOptions.end(); ++i) {
                o.erase(*i);
            }
            return o;
        }

        std::string checkUnusedOptions() {
            auto uopt = getUnusedOptions();
            std::stringstream msg;
            if (!uopt.empty()) {
                msg << "Unused options: ";
                for (auto i = uopt.begin(); i != uopt.end(); ++i) {
                    msg << *i << " ";
                }
            }
            return msg.str();
        }

        std::string &find(const char *s) { return get(s); }

        char getChar(const char *s, char v) {
            try {
                return get(s)[0];
            } catch (...) {
                return v;
            }
        }

        std::string getString(const char *s, const char *v) {
            try {
                return get(s);
            } catch (...) {
                return v;
            }
        }

        std::string &get(const char *s) {
            Args::iterator ai;
            ai = map<std::string, std::string>::find(std::string(s));
            if (ai == this->end()) {
                std::stringstream msg;
                msg << s << ": Not specified" << std::endl;
                TANN_THROW(msg.str());
            }
            usedOptions.insert(ai->first);
            return ai->second;
        }

        bool getBool(const char *s) {
            try {
                get(s);
            } catch (...) {
                return false;
            }
            return true;
        }

        long getl(const char *s, long v) {
            char *e;
            long val;
            try {
                val = strtol(get(s).c_str(), &e, 10);
            } catch (...) {
                return v;
            }
            if (*e != 0) {
                std::stringstream msg;
                msg << "ARGS::getl: Illegal string. Option=-" << s << " Specified value=" << get(s)
                    << " Illegal string=" << e << std::endl;
                TANN_THROW(msg.str());
            }
            return val;
        }

        float getf(const char *s, float v) {
            char *e;
            float val;
            try {
                val = strtof(get(s).c_str(), &e);
            } catch (...) {
                return v;
            }
            if (*e != 0) {
                std::stringstream msg;
                msg << "ARGS::getf: Illegal string. Option=-" << s << " Specified value=" << get(s)
                    << " Illegal string=" << e << std::endl;
                TANN_THROW(msg.str());
            }
            return val;
        }

        std::set<std::string> usedOptions;
        int argC;
        char **argV;
    };


}  // namespace tann

#endif  // TANN_COMMON_ARGS_H_
