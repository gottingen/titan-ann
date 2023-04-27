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

#include <iostream>
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << argv[0] << " input_uint8_bin output_float_bin" << std::endl;
    exit(-1);
  }

  uint8_t* input;
  size_t   npts, nd;
  tann::load_bin<uint8_t>(argv[1], input, npts, nd);
  float* output = new float[npts * nd];
  tann::convert_types<uint8_t, float>(input, output, npts, nd);
  tann::save_bin<float>(argv[2], output, npts, nd);
  delete[] output;
  delete[] input;
}
