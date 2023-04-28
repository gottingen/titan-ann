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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "disk_utils.h"
#include "tann/cached_io.h"
#include "tann/utils.h"

int main(int argc, char **argv) {
  if (argc != 9) {
    std::cout
        << argv[0]
        << " vamana_index_prefix[1] vamana_index_suffix[2] idmaps_prefix[3] "
           "idmaps_suffix[4] n_shards[5] max_degree[6] output_vamana_path[7] "
           "output_medoids_path[8]"
        << std::endl;
    exit(-1);
  }

  std::string vamana_prefix(argv[1]);
  std::string vamana_suffix(argv[2]);
  std::string idmaps_prefix(argv[3]);
  std::string idmaps_suffix(argv[4]);
  _u64        nshards = (_u64) std::atoi(argv[5]);
  _u32        max_degree = (_u64) std::atoi(argv[6]);
  std::string output_index(argv[7]);
  std::string output_medoids(argv[8]);

  return tann::merge_shards(vamana_prefix, vamana_suffix, idmaps_prefix,
                               idmaps_suffix, nshards, max_degree, output_index,
                               output_medoids);
}
