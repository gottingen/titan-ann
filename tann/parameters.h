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
#include <sstream>
#include <typeinfo>
#include <unordered_map>

namespace tann {

  class Parameters {
   public:
    Parameters() {
      int *p = new int;
      *p = 0;
      params["num_threads"] = p;
    }

    template<typename ParamType>
    inline void Set(const std::string &name, const ParamType &value) {
      //      ParamType *ptr = (ParamType *) malloc(sizeof(ParamType));
      if (params.find(name) != params.end()) {
        free(params[name]);
      }
      ParamType *ptr = new ParamType;
      *ptr = value;
      params[name] = (void *) ptr;
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name) const {
      auto item = params.find(name);
      if (item == params.end()) {
        throw std::invalid_argument("Invalid parameter name.");
      } else {
        // return ConvertStrToValue<ParamType>(item->second);
        if (item->second == nullptr) {
          throw std::invalid_argument(std::string("Parameter ") + name +
                                      " has value null.");
        } else {
          return *(static_cast<ParamType *>(item->second));
        }
      }
    }

    template<typename ParamType>
    inline ParamType Get(const std::string &name,
                         const ParamType   &default_value) {
      try {
        return Get<ParamType>(name);
      } catch (std::invalid_argument e) {
        return default_value;
      }
    }

    ~Parameters() {
      for (auto iter = params.begin(); iter != params.end(); iter++) {
        if (iter->second != nullptr)
          free(iter->second);
        // delete iter->second;
      }
    }

   private:
    std::unordered_map<std::string, void *> params;

    Parameters(const Parameters &);
    Parameters &operator=(const Parameters &);

    template<typename ParamType>
    inline ParamType ConvertStrToValue(const std::string &str) const {
      std::stringstream sstream(str);
      ParamType         value;
      if (!(sstream >> value) || !sstream.eof()) {
        std::stringstream err;
        err << "Failed to convert value '" << str
            << "' to type: " << typeid(value).name();
        throw std::runtime_error(err.str());
      }
      return value;
    }
  };
}  // namespace tann
