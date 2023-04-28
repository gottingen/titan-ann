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

#include "tann/ann_exception.h"
#include <sstream>
#include <string>

namespace tann {
  ANNException::ANNException(const std::string& message, int errorCode)
      : std::runtime_error(message), _errorCode(errorCode) {
  }

  std::string package_string(const std::string& item_name,
                             const std::string& item_val) {
    return std::string("[") + item_name + ": " + std::string(item_val) +
           std::string("]");
  }

  ANNException::ANNException(const std::string& message, int errorCode,
                             const std::string& funcSig,
                             const std::string& fileName, unsigned lineNum)
      : ANNException(
            package_string(std::string("FUNC"), funcSig) +
                package_string(std::string("FILE"), fileName) +
                package_string(std::string("LINE"), std::to_string(lineNum)) +
                "  " + message,
            errorCode) {
  }

  FileException::FileException(const std::string& filename,
                               std::system_error& e, const std::string& funcSig,
                               const std::string& fileName,
                               unsigned int       lineNum)
      : ANNException(std::string(" While opening file \'") + filename +
                         std::string("\', error code: ") +
                         std::to_string(e.code().value()) + "  " +
                         e.code().message(),
                     e.code().value(), funcSig, fileName, lineNum) {
  }

}  // namespace tann