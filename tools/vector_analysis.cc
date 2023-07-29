// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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

#include "tann/diskann/partition.h"
#include "tann/common/utils.h"
#include "tann_cli.h"

namespace detail {
    template<typename T>
    int analyze_norm(std::string base_file) {
        std::cout << "Analyzing data norms" << std::endl;
        T *data;
        size_t npts, ndims;
        tann::load_bin<T>(base_file, data, npts, ndims);
        std::vector<float> norms(npts, 0);
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            for (size_t d = 0; d < ndims; d++)
                norms[i] += data[i * ndims + d] * data[i * ndims + d];
            norms[i] = std::sqrt(norms[i]);
        }
        std::sort(norms.begin(), norms.end());
        for (int p = 0; p < 100; p += 5)
            std::cout << "percentile " << p << ": " << norms[(uint64_t) (std::floor((p / 100.0) * npts))] << std::endl;
        std::cout << "percentile 100"
                  << ": " << norms[npts - 1] << std::endl;
        delete[] data;
        return 0;
    }

    template<typename T>
    int normalize_base(std::string base_file, std::string out_file) {
        std::cout << "Normalizing base" << std::endl;
        T *data;
        size_t npts, ndims;
        tann::load_bin<T>(base_file, data, npts, ndims);
        //  std::vector<float> norms(npts, 0);
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            float pt_norm = 0;
            for (size_t d = 0; d < ndims; d++)
                pt_norm += data[i * ndims + d] * data[i * ndims + d];
            pt_norm = std::sqrt(pt_norm);
            for (size_t d = 0; d < ndims; d++)
                data[i * ndims + d] = static_cast<T>(data[i * ndims + d] / pt_norm);
        }
        tann::save_bin<T>(out_file, data, npts, ndims);
        delete[] data;
        return 0;
    }

    template<typename T>
    int augment_base(std::string base_file, std::string out_file, bool prep_base = true) {
        std::cout << "Analyzing data norms" << std::endl;
        T *data;
        size_t npts, ndims;
        tann::load_bin<T>(base_file, data, npts, ndims);
        std::vector<float> norms(npts, 0);
        float max_norm = 0;
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            for (size_t d = 0; d < ndims; d++)
                norms[i] += data[i * ndims + d] * data[i * ndims + d];
            max_norm = norms[i] > max_norm ? norms[i] : max_norm;
        }
        //  std::sort(norms.begin(), norms.end());
        max_norm = std::sqrt(max_norm);
        std::cout << "Max norm: " << max_norm << std::endl;
        T *new_data;
        size_t newdims = ndims + 1;
        new_data = new T[npts * newdims];
        for (size_t i = 0; i < npts; i++) {
            if (prep_base) {
                for (size_t j = 0; j < ndims; j++) {
                    new_data[i * newdims + j] = static_cast<T>(data[i * ndims + j] / max_norm);
                }
                float diff = 1 - (norms[i] / (max_norm * max_norm));
                diff = diff <= 0 ? 0 : std::sqrt(diff);
                new_data[i * newdims + ndims] = static_cast<T>(diff);
                if (diff <= 0) {
                    std::cout << i << " has large max norm, investigate if needed. diff = " << diff << std::endl;
                }
            } else {
                for (size_t j = 0; j < ndims; j++) {
                    new_data[i * newdims + j] = static_cast<T>(data[i * ndims + j] / std::sqrt(norms[i]));
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
    int aux_main(const std::string &base_file, uint32_t option, const std::string &args) {
        if (option == 1)
            analyze_norm<T>(base_file);
        else if (option == 2)
            augment_base<T>(base_file, std::string(), true);
        else if (option == 3)
            augment_base<T>(base_file, args, false);
        else if (option == 4)
            normalize_base<T>(base_file, args);
        return 0;
    }
    static std::string base_file;
    static std::string data_type;
    static std::string args;
    static uint32_t option;
}

namespace tann {
    void set_vector_analysis(turbo::App &app) {
        auto *sub = app.add_subcommand("vector_analysis", "vector_analysis");
        sub->add_option("-t, --data_type", detail::data_type, "data type <int8/uint8/float>")->required();
        sub->add_option("b, --base_file", detail::base_file, "distance function <l2/mips>")->required();
        sub->add_option("-O, --option", detail::option,"File containing the base vectors in binary format")->required();
        sub->add_option("-A, --arg", detail::args,"File containing the base vectors in binary format")->default_val("");
        sub->callback([]() {
            vector_analysis();
        });
    }
    void vector_analysis() {
        if (std::string(detail::data_type) == std::string("float")) {
            detail::aux_main<float>(detail::base_file, detail::option, detail::args);
        } else if (std::string(detail::data_type) == std::string("int8")) {
            detail::aux_main<int8_t>(detail::base_file, detail::option, detail::args);
        } else if (std::string(detail::data_type) == std::string("uint8")) {
            detail::aux_main<uint8_t>(detail::base_file, detail::option, detail::args);
        } else
            std::cout << "Unsupported type. Use float/int8/uint8." << std::endl;
        return ;
    }
}