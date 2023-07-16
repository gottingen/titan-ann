// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "tann/common/math_utils.h"
#include "tann/vamana/pq.h"
#include "tann/vamana/partition.h"
#include "tann_cli.h"

#define KMEANS_ITERS_FOR_PQ 15

namespace detail {
    template<typename T>
    bool generate_pq(const std::string &data_path, const std::string &index_prefix_path, const size_t num_pq_centers,
                     const size_t num_pq_chunks, const float sampling_rate, const bool opq) {
        std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
        std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";

        // generates random sample and sets it to train_data and updates train_size
        size_t train_size, train_dim;
        float *train_data;
        tann::gen_random_slice<T>(data_path, sampling_rate, train_data, train_size, train_dim);
        std::cout << "For computing pivots, loaded sample data of size " << train_size << std::endl;

        if (opq) {
            tann::generate_opq_pivots(train_data, train_size, (uint32_t) train_dim, (uint32_t) num_pq_centers,
                                      (uint32_t) num_pq_chunks, pq_pivots_path, true);
        } else {
            tann::generate_pq_pivots(train_data, train_size, (uint32_t) train_dim, (uint32_t) num_pq_centers,
                                     (uint32_t) num_pq_chunks, KMEANS_ITERS_FOR_PQ, pq_pivots_path);
        }
        tann::generate_pq_data_from_pivots<T>(data_path, (uint32_t) num_pq_centers, (uint32_t) num_pq_chunks,
                                              pq_pivots_path, pq_compressed_vectors_path, true);

        delete[] train_data;

        return 0;
    }

    static std::string data_type;
    static std::string data_path;
    static std::string index_prefix_path;
    static size_t num_pq_chunks;
    static float sampling_rate;
    static bool opq;
}  // namespace detail

namespace tann {
    void set_gen_pq(turbo::App &app) {
        auto *sub = app.add_subcommand("gen_pq", "gen_random_slice");
        sub->add_option("-p, --data_path", detail::data_path, "data_path")->required();
        sub->add_option("-x,--index", detail::index_prefix_path, "index_prefix_path")->required();
        sub->add_option("-R, --rate", detail::sampling_rate, "sampling_rate")->required();
        sub->add_option("-t, --data_type", detail::data_type, "data_type")->required();
        sub->add_option("-n, --num_pq", detail::num_pq_chunks, "data_type")->required();
        sub->add_flag("-q, --opq", detail::opq, "opq or not")->default_val(false);
        sub->callback([]() {
            gen_pq();
        });
    }

    void gen_pq() {

        const size_t num_pq_centers = 256;

        if (detail::data_type == std::string("float"))
            detail::generate_pq<float>(detail::data_path, detail::index_prefix_path, num_pq_centers,
                                       detail::num_pq_chunks, detail::sampling_rate, detail::opq);
        else if (detail::data_type == std::string("int8"))
            detail::generate_pq<int8_t>(detail::data_path, detail::index_prefix_path, num_pq_centers,
                                        detail::num_pq_chunks, detail::sampling_rate, detail::opq);
        else if (detail::data_type == std::string("uint8"))
            detail::generate_pq<uint8_t>(detail::data_path, detail::index_prefix_path, num_pq_centers,
                                         detail::num_pq_chunks, detail::sampling_rate, detail::opq);
        else
            std::cout << "Error. wrong file type" << std::endl;
    }

}   // namespace tann