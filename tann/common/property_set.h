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
#ifndef TANN_COMMON_PROPERTY_SET_H_
#define TANN_COMMON_PROPERTY_SET_H_

#include <map>
#include "turbo/strings/str_split.h"

namespace tann {

    class PropertySet : public std::map<std::string, std::string> {
    public:
        void set(const std::string &key, const std::string &value) {
            iterator it = find(key);
            if (it == end()) {
                insert(std::pair<std::string, std::string>(key, value));
            } else {
                (*it).second = value;
            }
        }

        template<class VALUE_TYPE>
        void set(const std::string &key, VALUE_TYPE value) {
            std::stringstream vstr;
            vstr << value;
            iterator it = find(key);
            if (it == end()) {
                insert(std::pair<std::string, std::string>(key, vstr.str()));
            } else {
                (*it).second = vstr.str();
            }
        }

        std::string get(const std::string &key) {
            iterator it = find(key);
            if (it != end()) {
                return it->second;
            }
            return "";
        }

        float getf(const std::string &key, float defvalue) {
            iterator it = find(key);
            if (it != end()) {
                char *e = 0;
                float val = strtof(it->second.c_str(), &e);
                if (*e != 0) {
                    std::cerr << "Warning: Illegal property. " << key << ":" << it->second << " (" << e << ")"
                              << std::endl;
                    return defvalue;
                }
                return val;
            }
            return defvalue;
        }

        void updateAndInsert(PropertySet &prop) {
            for (std::map<std::string, std::string>::iterator i = prop.begin(); i != prop.end(); ++i) {
                set((*i).first, (*i).second);
            }
        }

        long getl(const std::string &key, long defvalue) {
            iterator it = find(key);
            if (it != end()) {
                char *e = 0;
                float val = strtol(it->second.c_str(), &e, 10);
                if (*e != 0) {
                    std::cerr << "Warning: Illegal property. " << key << ":" << it->second << " (" << e << ")"
                              << std::endl;
                }
                return val;
            }
            return defvalue;
        }

        void load(const std::string &f) {
            std::ifstream st(f);
            if (!st) {
                std::stringstream msg;
                msg << "PropertySet::load: Cannot load the property file " << f << ".";
                TANN_THROW(msg);
            }
            load(st);
        }

        void save(const std::string &f) {
            std::ofstream st(f);
            if (!st) {
                std::stringstream msg;
                msg << "PropertySet::save: Cannot save. " << f << std::endl;
                TANN_THROW(msg);
            }
            save(st);
        }

        void save(std::ofstream &os) {
            for (std::map<std::string, std::string>::iterator i = this->begin(); i != this->end(); i++) {
                os << i->first << "\t" << i->second << std::endl;
            }
        }

        void load(std::ifstream &is) {
            std::string line;
            while (getline(is, line)) {
                std::vector<std::string> tokens = turbo::StrSplit(line, "\t");
                if (tokens.size() != 2) {
                    std::cerr << "Property file is illegal. " << line << std::endl;
                    continue;
                }
                set(tokens[0], tokens[1]);
            }
        }
    };

}  // namespace tann

#endif  // TANN_COMMON_PROPERTY_SET_H_
