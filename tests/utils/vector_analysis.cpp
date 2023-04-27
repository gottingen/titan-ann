// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Copyright 2022 The Tann Authors
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

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <typeinfo>

#include "partition.h"
#include "utils.h"

template<typename T>
int analyze_norm(std::string base_file) {
  std::cout << "Analyzing data norms" << std::endl;
  T*   data;
  _u64 npts, ndims;
  tann::load_bin<T>(base_file, data, npts, ndims);
  std::vector<float> norms(npts, 0);
#pragma omp parallel for schedule(dynamic)
  for (_s64 i = 0; i < (_s64) npts; i++) {
    for (_u32 d = 0; d < ndims; d++)
      norms[i] += data[i * ndims + d] * data[i * ndims + d];
    norms[i] = std::sqrt(norms[i]);
  }
  std::sort(norms.begin(), norms.end());
  for (_u32 p = 0; p < 100; p += 5)
    std::cout << "percentile " << p << ": "
              << norms[std::floor((p / 100.0) * npts)] << std::endl;
  std::cout << "percentile 100"
            << ": " << norms[npts - 1] << std::endl;
  delete[] data;
  return 0;
}

template<typename T>
int normalize_base(std::string base_file, std::string out_file) {
  std::cout << "Normalizing base" << std::endl;
  T*   data;
  _u64 npts, ndims;
  tann::load_bin<T>(base_file, data, npts, ndims);
  //  std::vector<float> norms(npts, 0);
#pragma omp parallel for schedule(dynamic)
  for (_s64 i = 0; i < (_s64) npts; i++) {
    float pt_norm = 0;
    for (_u32 d = 0; d < ndims; d++)
      pt_norm += data[i * ndims + d] * data[i * ndims + d];
    pt_norm = std::sqrt(pt_norm);
    for (_u32 d = 0; d < ndims; d++)
      data[i * ndims + d] = data[i * ndims + d] / pt_norm;
  }
  tann::save_bin<T>(out_file, data, npts, ndims);
  delete[] data;
  return 0;
}

template<typename T>
int augment_base(std::string base_file, std::string out_file,
                 bool prep_base = true) {
  std::cout << "Analyzing data norms" << std::endl;
  T*   data;
  _u64 npts, ndims;
  tann::load_bin<T>(base_file, data, npts, ndims);
  std::vector<float> norms(npts, 0);
  float              max_norm = 0;
#pragma omp parallel for schedule(dynamic)
  for (_s64 i = 0; i < (_s64) npts; i++) {
    for (_u32 d = 0; d < ndims; d++)
      norms[i] += data[i * ndims + d] * data[i * ndims + d];
    max_norm = norms[i] > max_norm ? norms[i] : max_norm;
  }
  //  std::sort(norms.begin(), norms.end());
  max_norm = std::sqrt(max_norm);
  std::cout << "Max norm: " << max_norm << std::endl;
  T*   new_data;
  _u64 newdims = ndims + 1;
  new_data = new T[npts * newdims];
  for (_u64 i = 0; i < npts; i++) {
    if (prep_base) {
      for (_u64 j = 0; j < ndims; j++) {
        new_data[i * newdims + j] = data[i * ndims + j] / max_norm;
      }
      float diff = 1 - (norms[i] / (max_norm * max_norm));
      diff = diff <= 0 ? 0 : std::sqrt(diff);
      new_data[i * newdims + ndims] = diff;
      if (diff <= 0) {
        std::cout << i << " has large max norm, investigate if needed. diff = "
                  << diff << std::endl;
      }
    } else {
      for (_u64 j = 0; j < ndims; j++) {
        new_data[i * newdims + j] = data[i * ndims + j] / std::sqrt(norms[i]);
      }
      new_data[i * newdims + ndims] = 0;
    }
  }
  tann::save_bin<T>(out_file, new_data, npts, newdims);
  delete[] new_data;
  delete[] data;
  return 0;
}

template<typename T>
int aux_main(char** argv) {
  std::string base_file(argv[2]);
  _u32        option = atoi(argv[3]);
  if (option == 1)
    analyze_norm<T>(base_file);
  else if (option == 2)
    augment_base<T>(base_file, std::string(argv[4]), true);
  else if (option == 3)
    augment_base<T>(base_file, std::string(argv[4]), false);
  else if (option == 4)
    normalize_base<T>(base_file, std::string(argv[4]));
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << argv[0]
              << " data_type [float/int8/uint8] base_bin_file "
                 "[option: 1-norm analysis, 2-prep_base_for_mip, "
                 "3-prep_query_for_mip, 4-normalize-vecs] [out_file for "
                 "options 2/3/4]"
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float")) {
    aux_main<float>(argv);
  } else if (std::string(argv[1]) == std::string("int8")) {
    aux_main<int8_t>(argv);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    aux_main<uint8_t>(argv);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8." << std::endl;
  return 0;
}
