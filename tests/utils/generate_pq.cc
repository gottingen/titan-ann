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

#include "tann/math_utils.h"
#include "tann/pq.h"
#include "tann/partition.h"

#define KMEANS_ITERS_FOR_PQ 15

template<typename T>
bool generate_pq(const std::string& data_path,
                 const std::string& index_prefix_path,
                 const size_t num_pq_centers, const size_t num_pq_chunks,
                 const float sampling_rate, const bool opq) {
  std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
  std::string pq_compressed_vectors_path =
      index_prefix_path + "_pq_compressed.bin";

  // generates random sample and sets it to train_data and updates train_size
  size_t train_size, train_dim;
  float* train_data;
  gen_random_slice<T>(data_path, sampling_rate, train_data, train_size,
                      train_dim);
  std::cout << "For computing pivots, loaded sample data of size " << train_size
            << std::endl;

  if (opq) {
    tann::generate_opq_pivots(train_data, train_size, train_dim,
                                 num_pq_centers, num_pq_chunks, pq_pivots_path,
                                 true);

  } else {
    tann::generate_pq_pivots(train_data, train_size, train_dim,
                                num_pq_centers, num_pq_chunks,
                                KMEANS_ITERS_FOR_PQ, pq_pivots_path);
  }
  tann::generate_pq_data_from_pivots<T>(data_path, num_pq_centers,
                                           num_pq_chunks, pq_pivots_path,
                                           pq_compressed_vectors_path, true);

  delete[] train_data;

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cout << "Usage: \n"
              << argv[0]
              << "  <data_type[float/uint8/int8]>   <data_file[.bin]>"
                 "  <PQ_prefix_path>  <target-bytes/data-point>  "
                 "<sampling_rate> <PQ(0)/OPQ(1)>"
              << std::endl;
  } else {
    const std::string data_path(argv[2]);
    const std::string index_prefix_path(argv[3]);
    const size_t      num_pq_centers = 256;
    const size_t      num_pq_chunks = (size_t) atoi(argv[4]);
    const float       sampling_rate = atof(argv[5]);
    const bool        opq = atoi(argv[6]) == 0 ? false : true;

    if (std::string(argv[1]) == std::string("float"))
      generate_pq<float>(data_path, index_prefix_path, num_pq_centers,
                         num_pq_chunks, sampling_rate, opq);
    else if (std::string(argv[1]) == std::string("int8"))
      generate_pq<int8_t>(data_path, index_prefix_path, num_pq_centers,
                          num_pq_chunks, sampling_rate, opq);
    else if (std::string(argv[1]) == std::string("uint8"))
      generate_pq<uint8_t>(data_path, index_prefix_path, num_pq_centers,
                           num_pq_chunks, sampling_rate, opq);
    else
      std::cout << "Error. wrong file type" << std::endl;
  }
}