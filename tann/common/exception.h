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

#ifndef TANN_COMMON_EXCEPTION_H_
#define TANN_COMMON_EXCEPTION_H_

#include <exception>
#include <string>
#include <string_view>
#include "turbo/format/str_format.h"

namespace tann {

#define    TANN_THROW(MESSAGE)            throw tann::Exception(__FILE__, __FUNCTION__, (size_t)__LINE__, MESSAGE)
#define    TANN_THROW_SPEC(MESSAGE, TYPE)    throw tann::TYPE(__FILE__, __FUNCTION__, (size_t)__LINE__, MESSAGE)

    class Exception : public std::exception {
    public:
        Exception() : message("No message") {}

        Exception(const Exception &) = default;

        Exception(const std::string_view &file, const std::string_view &function, size_t line,
                  const std::string_view &m) {
            set(file, function, line, m);
        }

        Exception(const std::string_view &file, const std::string_view &function, size_t line,
                  const std::stringstream &m) {
            set(file, function, line, m.str());
        }

        void
        set(const std::string_view &file, const std::string_view &function, size_t line, const std::string_view &m) {
            message = turbo::Format("{}:{}:{}: {}", file, function, line, m);
        }

        ~Exception() throw() {}

        Exception &operator=(const Exception &e) {
            message = e.message;
            return *this;
        }

        virtual const char *what() const throw() {
            return message.c_str();
        }

        std::string &getMessage() { return message; }

    protected:
        std::string message;
    };
}
#endif  // TANN_COMMON_EXCEPTION_H_
