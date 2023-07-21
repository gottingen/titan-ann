// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <limits>
#include <cstring>
#include <queue>
#include <omp.h>
#include <mkl/mkl.h>
#include <unordered_map>
#include "turbo/container/flat_hash_map.h"
#include "turbo/container/flat_hash_set.h"
#include "tann_cli.h"

#ifdef _WINDOWS
#include <malloc.h>
#else

#include <stdlib.h>

#endif

#include "tann/vamana/filter_utils.h"
#include "tann/common/utils.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

namespace detail {

    template<typename T>
    inline void save_bin(const std::string filename, T *data, size_t npts, size_t ndims) {
        std::ofstream writer;
        writer.exceptions(std::ios::failbit | std::ios::badbit);
        writer.open(filename, std::ios::binary | std::ios::out);
        std::cout << "Writing bin: " << filename << "\n";
        int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
        writer.write((char *) &npts_i32, sizeof(int));
        writer.write((char *) &ndims_i32, sizeof(int));
        std::cout << "bin: #pts = " << npts << ", #dims = " << ndims
                  << ", size = " << npts * ndims * sizeof(T) + 2 * sizeof(int) << "B" << std::endl;

        writer.write((char *) data, npts * ndims * sizeof(T));
        writer.close();
        std::cout << "Finished writing bin" << std::endl;
    }

    inline void save_groundtruth_as_one_file(const std::string filename, int32_t *data, float *distances, size_t npts,
                                             size_t ndims) {
        std::ofstream writer(filename, std::ios::binary | std::ios::out);
        int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
        writer.write((char *) &npts_i32, sizeof(int));
        writer.write((char *) &ndims_i32, sizeof(int));
        std::cout << "Saving truthset in one file (npts, dim, npts*dim id-matrix, "
                     "npts*dim dist-matrix) with npts = "
                  << npts << ", dim = " << ndims << ", size = " << 2 * npts * ndims * sizeof(uint32_t) + 2 * sizeof(int)
                  << "B" << std::endl;

        writer.write((char *) data, npts * ndims * sizeof(uint32_t));
        writer.write((char *) distances, npts * ndims * sizeof(float));
        writer.close();
        std::cout << "Finished writing truthset" << std::endl;
    }

    template<typename T>
    std::vector<std::vector<std::pair<uint32_t, float>>> processUnfilteredParts(const std::string &base_file,
                                                                                size_t &nqueries, size_t &npoints,
                                                                                size_t &dim, size_t &k,
                                                                                float *query_data,
                                                                                const tann::Metric &metric,
                                                                                std::vector<uint32_t> &location_to_tag) {
        float *base_data = nullptr;
        int num_parts = get_num_parts<T>(base_file.c_str());
        std::vector<std::vector<std::pair<uint32_t, float>>> res(nqueries);
        for (int p = 0; p < num_parts; p++) {
            size_t start_id = p * PARTSIZE;
            load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p);

            size_t *closest_points_part = new size_t[nqueries * k];
            float *dist_closest_points_part = new float[nqueries * k];

            auto part_k = k < npoints ? k : npoints;
            exact_knn(dim, part_k, closest_points_part, dist_closest_points_part, npoints, base_data, nqueries,
                      query_data,
                      metric);

            for (size_t i = 0; i < nqueries; i++) {
                for (size_t j = 0; j < part_k; j++) {
                    if (!location_to_tag.empty())
                        if (location_to_tag[closest_points_part[i * k + j] + start_id] == 0)
                            continue;

                    res[i].push_back(std::make_pair((uint32_t) (closest_points_part[i * part_k + j] + start_id),
                                                    dist_closest_points_part[i * part_k + j]));
                }
            }

            delete[] closest_points_part;
            delete[] dist_closest_points_part;

            tann::aligned_free(base_data);
        }
        return res;
    };

    template<typename T>
    int aux_main(const std::string &base_file, const std::string &query_file, const std::string &gt_file, size_t k,
                 const tann::Metric &metric, const std::string &tags_file = std::string("")) {
        size_t npoints, nqueries, dim;

        float *query_data;

        load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);
        if (nqueries > PARTSIZE)
            std::cerr << "WARNING: #Queries provided (" << nqueries << ") is greater than " << PARTSIZE
                      << ". Computing GT only for the first " << PARTSIZE << " queries." << std::endl;

        // load tags
        const bool tags_enabled = tags_file.empty() ? false : true;
        std::vector<uint32_t> location_to_tag = tann::loadTags(tags_file, base_file);

        int *closest_points = new int[nqueries * k];
        float *dist_closest_points = new float[nqueries * k];

        std::vector<std::vector<std::pair<uint32_t, float>>> results =
                processUnfilteredParts<T>(base_file, nqueries, npoints, dim, k, query_data, metric, location_to_tag);

        for (size_t i = 0; i < nqueries; i++) {
            std::vector<std::pair<uint32_t, float>> &cur_res = results[i];
            std::sort(cur_res.begin(), cur_res.end(), custom_dist);
            size_t j = 0;
            for (auto iter: cur_res) {
                if (j == k)
                    break;
                if (tags_enabled) {
                    std::uint32_t index_with_tag = location_to_tag[iter.first];
                    closest_points[i * k + j] = (int32_t) index_with_tag;
                } else {
                    closest_points[i * k + j] = (int32_t) iter.first;
                }

                if (metric == tann::Metric::INNER_PRODUCT)
                    dist_closest_points[i * k + j] = -iter.second;
                else
                    dist_closest_points[i * k + j] = iter.second;

                ++j;
            }
            if (j < k)
                std::cout << "WARNING: found less than k GT entries for query " << i << std::endl;
        }

        save_groundtruth_as_one_file(gt_file, closest_points, dist_closest_points, nqueries, k);
        delete[] closest_points;
        delete[] dist_closest_points;
        tann::aligned_free(query_data);

        return 0;
    }

    static std::string dist_fn;
    static std::string base_file;
    static std::string query_file;
    static std::string gt_file;
    static std::string tags_file;
    static std::string data_type;
    static uint64_t   K;
}  // namespace detail
namespace tann {
    void set_compute_groundtruth(turbo::App &app) {
        auto *cg =app.add_subcommand("compute_groundtruth", "compute_groundtruth");
        cg->add_option("-t, --data_type", detail::data_type, "data type <int8/uint8/float>")->required();
        cg->add_option("-f, --dist_fn", detail::dist_fn, "distance function <l2/mips>")->required();
        cg->add_option("-b, --base_file", detail::base_file,"File containing the base vectors in binary format")->required();
        cg->add_option("-q, --query_file", detail::query_file,"File containing the query vectors in binary format")->required();
        cg->add_option("-g, --gt_file", detail::gt_file,
                       "File name for the writing ground truth in binary "
                       "format, please don' append .bin at end if "
                       "no filter_label or filter_label_file is provided it "
                       "will save the file with '.bin' at end."
                       "else it will save the file as filename_label.bin")->required();
        cg->add_option("--K", detail::K,
                       "Number of ground truth nearest neighbors to compute")->required();
        cg->add_option("-T, --tags_file", detail::tags_file,
                       "File containing the tags in binary format")->default_val(std::string());
        cg->callback([](){
            compute_groundtruth();
        });
    }
    void compute_groundtruth() {

        if (detail::data_type != std::string("float") && detail::data_type != std::string("int8") &&
                detail::data_type != std::string("uint8")) {
            std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
        }

        tann::Metric metric;
        if (detail::dist_fn == std::string("l2")) {
            metric = tann::Metric::L2;
        } else if (detail::dist_fn == std::string("mips")) {
            metric = tann::Metric::INNER_PRODUCT;
        } else if (detail::dist_fn == std::string("cosine")) {
            metric = tann::Metric::COSINE;
        } else {
            std::cerr << "Unsupported distance function. Use l2/mips/cosine." << std::endl;
        }

        try {
            if (detail::data_type == std::string("float"))
                detail::aux_main<float>(detail::base_file, detail::query_file, detail::gt_file, detail::K, metric, detail::tags_file);
            if (detail::data_type == std::string("int8"))
                detail::aux_main<int8_t>(detail::base_file, detail::query_file, detail::gt_file, detail::K, metric, detail::tags_file);
            if (detail::data_type == std::string("uint8"))
                detail::aux_main<uint8_t>(detail::base_file, detail::query_file, detail::gt_file, detail::K, metric, detail::tags_file);
        }
        catch (const std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            tann::cerr << "Compute GT failed." << std::endl;
        }
    }
}