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

#include <tann/index.h>
#include <tann/math_utils.h>
#include "tann/cached_io.h"
#include "tann/partition.h"

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cout << "Usage:\n"
              << argv[0]
              << "  datatype<int8/uint8/float>  <data_path>"
                 "  <prefix_path>  <sampling_rate>  "
                 "  <ram_budget(GB)> <graph_degree>  <k_index>"
              << std::endl;
    exit(-1);
  }

  const std::string data_path(argv[2]);
  const std::string prefix_path(argv[3]);
  const float       sampling_rate = atof(argv[4]);
  const double      ram_budget = (double) std::atof(argv[5]);
  const size_t      graph_degree = (size_t) std::atoi(argv[6]);
  const size_t      k_index = (size_t) std::atoi(argv[7]);

  if (std::string(argv[1]) == std::string("float"))
    partition_with_ram_budget<float>(data_path, sampling_rate, ram_budget,
                                     graph_degree, prefix_path, k_index);
  else if (std::string(argv[1]) == std::string("int8"))
    partition_with_ram_budget<int8_t>(data_path, sampling_rate, ram_budget,
                                      graph_degree, prefix_path, k_index);
  else if (std::string(argv[1]) == std::string("uint8"))
    partition_with_ram_budget<uint8_t>(data_path, sampling_rate, ram_budget,
                                       graph_degree, prefix_path, k_index);
  else
    std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
}