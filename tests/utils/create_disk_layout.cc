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

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "tann/utils.h"
#include "tann/disk_utils.h"
#include "tann/cached_io.h"

template<typename T>
int create_disk_layout(char **argv) {
  std::string base_file(argv[2]);
  std::string vamana_file(argv[3]);
  std::string output_file(argv[4]);
  tann::create_disk_layout<T>(base_file, vamana_file, output_file);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << argv[0]
              << " data_type <float/int8/uint8> data_bin "
                 "vamana_index_file output_tann_index_file"
              << std::endl;
    exit(-1);
  }

  int ret_val = -1;
  if (std::string(argv[1]) == std::string("float"))
    ret_val = create_disk_layout<float>(argv);
  else if (std::string(argv[1]) == std::string("int8"))
    ret_val = create_disk_layout<int8_t>(argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    ret_val = create_disk_layout<uint8_t>(argv);
  else {
    std::cout << "unsupported type. use int8/uint8/float " << std::endl;
    ret_val = -2;
  }
  return ret_val;
}