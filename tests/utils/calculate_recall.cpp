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

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "tann/utils.h"
#include "disk_utils.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << argv[0] << " <ground_truth_bin> <our_results_bin>  <r> "
              << std::endl;
    return -1;
  }
  unsigned* gold_std = NULL;
  float*    gs_dist = nullptr;
  unsigned* our_results = NULL;
  float*    or_dist = nullptr;
  size_t    points_num, points_num_gs, points_num_or;
  size_t    dim_gs;
  size_t    dim_or;
  tann::load_truthset(argv[1], gold_std, gs_dist, points_num_gs, dim_gs);
  tann::load_truthset(argv[2], our_results, or_dist, points_num_or, dim_or);

  if (points_num_gs != points_num_or) {
    std::cout
        << "Error. Number of queries mismatch in ground truth and our results"
        << std::endl;
    return -1;
  }
  points_num = points_num_gs;

  uint32_t recall_at = std::atoi(argv[3]);

  if ((dim_or < recall_at) || (recall_at > dim_gs)) {
    std::cout << "ground truth has size " << dim_gs << "; our set has "
              << dim_or << " points. Asking for recall " << recall_at
              << std::endl;
    return -1;
  }
  std::cout << "Calculating recall@" << recall_at << std::endl;
  float recall_val = tann::calculate_recall(
      points_num, gold_std, gs_dist, dim_gs, our_results, dim_or, recall_at);

  //  double avg_recall = (recall*1.0)/(points_num*1.0);
  std::cout << "Avg. recall@" << recall_at << " is " << recall_val << "\n";
}
