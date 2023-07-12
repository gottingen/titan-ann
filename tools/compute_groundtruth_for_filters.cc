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
#include "tann/vamana/tsl/robin_map.h"
#include "tann/vamana/tsl/robin_set.h"
#include "tann_cli.h"

#ifdef _WINDOWS
#include <malloc.h>
#else

#include <stdlib.h>

#endif

#include "tann/vamana/filter_utils.h"
#include "tann/vamana/utils.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

#define PARTSIZE 10000000
#define ALIGNMENT 512

namespace detail {


    template<typename T>
    inline std::vector<size_t>
    load_filtered_bin_as_float(const char *filename, float *&data, size_t &npts, size_t &ndims,
                               int part_num, const char *label_file,
                               const std::string &filter_label,
                               const std::string &universal_label, size_t &npoints_filt,
                               std::vector<std::vector<std::string>> &pts_to_labels) {
        std::ifstream reader(filename, std::ios::binary);
        if (reader.fail()) {
            throw tann::ANNException(std::string("Failed to open file ") + filename, -1);
        }

        std::cout << "Reading bin file " << filename << " ...\n";
        int npts_i32, ndims_i32;
        std::vector<size_t> rev_map;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &ndims_i32, sizeof(int));
        uint64_t start_id = part_num * PARTSIZE;
        uint64_t end_id = (std::min)(start_id + PARTSIZE, (uint64_t) npts_i32);
        npts = end_id - start_id;
        ndims = (uint32_t) ndims_i32;
        uint64_t nptsuint64_t = (uint64_t) npts;
        uint64_t ndimsuint64_t = (uint64_t) ndims;
        npoints_filt = 0;
        std::cout << "#pts in part = " << npts << ", #dims = " << ndims
                  << ", size = " << nptsuint64_t * ndimsuint64_t * sizeof(T) << "B" << std::endl;
        std::cout << "start and end ids: " << start_id << ", " << end_id << std::endl;
        reader.seekg(start_id * ndims * sizeof(T) + 2 * sizeof(uint32_t), std::ios::beg);

        T *data_T = new T[nptsuint64_t * ndimsuint64_t];
        reader.read((char *) data_T, sizeof(T) * nptsuint64_t * ndimsuint64_t);
        std::cout << "Finished reading part of the bin file." << std::endl;
        reader.close();

        data = aligned_malloc<float>(nptsuint64_t * ndimsuint64_t, ALIGNMENT);

        for (int64_t i = 0; i < (int64_t) nptsuint64_t; i++) {
            if (std::find(pts_to_labels[start_id + i].begin(), pts_to_labels[start_id + i].end(), filter_label) !=
                pts_to_labels[start_id + i].end() ||
                std::find(pts_to_labels[start_id + i].begin(), pts_to_labels[start_id + i].end(), universal_label) !=
                pts_to_labels[start_id + i].end()) {
                rev_map.push_back(start_id + i);
                for (int64_t j = 0; j < (int64_t) ndimsuint64_t; j++) {
                    float cur_val_float = (float) data_T[i * ndimsuint64_t + j];
                    std::memcpy((char *) (data + npoints_filt * ndimsuint64_t + j), (char *) &cur_val_float,
                                sizeof(float));
                }
                npoints_filt++;
            }
        }
        delete[] data_T;
        std::cout << "Finished converting part data to float.. identified " << npoints_filt
                  << " points matching the filter." << std::endl;
        return rev_map;
    }

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

    inline void parse_label_file_into_vec(size_t &line_cnt, const std::string &map_file,
                                          std::vector<std::vector<std::string>> &pts_to_labels) {
        std::ifstream infile(map_file);
        std::string line, token;
        std::set<std::string> labels;
        infile.clear();
        infile.seekg(0, std::ios::beg);
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            std::vector<std::string> lbls(0);

            getline(iss, token, '\t');
            std::istringstream new_iss(token);
            while (getline(new_iss, token, ',')) {
                token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
                token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
                lbls.push_back(token);
                labels.insert(token);
            }
            if (lbls.size() <= 0) {
                std::cout << "No label found";
                exit(-1);
            }
            std::sort(lbls.begin(), lbls.end());
            pts_to_labels.push_back(lbls);
        }
        std::cout << "Identified " << labels.size() << " distinct label(s), and populated labels for "
                  << pts_to_labels.size() << " points" << std::endl;
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
                for (uint64_t j = 0; j < part_k; j++) {
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
    std::vector<std::vector<std::pair<uint32_t, float>>> processFilteredParts(
            const std::string &base_file, const std::string &label_file, const std::string &filter_label,
            const std::string &universal_label, size_t &nqueries, size_t &npoints, size_t &dim, size_t &k,
            float *query_data,
            const tann::Metric &metric, std::vector<uint32_t> &location_to_tag) {
        size_t npoints_filt = 0;
        float *base_data = nullptr;
        std::vector<std::vector<std::pair<uint32_t, float>>> res(nqueries);
        int num_parts = get_num_parts<T>(base_file.c_str());

        std::vector<std::vector<std::string>> pts_to_labels;
        if (filter_label != "")
            parse_label_file_into_vec(npoints, label_file, pts_to_labels);

        for (int p = 0; p < num_parts; p++) {
            size_t start_id = p * PARTSIZE;
            std::vector<size_t> rev_map;
            if (filter_label != "")
                rev_map = load_filtered_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p,
                                                        label_file.c_str(),
                                                        filter_label, universal_label, npoints_filt, pts_to_labels);
            size_t *closest_points_part = new size_t[nqueries * k];
            float *dist_closest_points_part = new float[nqueries * k];

            auto part_k = k < npoints_filt ? k : npoints_filt;
            if (npoints_filt > 0) {
                exact_knn(dim, part_k, closest_points_part, dist_closest_points_part, npoints_filt, base_data, nqueries,
                          query_data, metric);
            }

            for (size_t i = 0; i < nqueries; i++) {
                for (uint64_t j = 0; j < part_k; j++) {
                    if (!location_to_tag.empty())
                        if (location_to_tag[closest_points_part[i * k + j] + start_id] == 0)
                            continue;

                    res[i].push_back(std::make_pair((uint32_t) (rev_map[closest_points_part[i * part_k + j]]),
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
    int aux_main(const std::string &base_file, const std::string &label_file, const std::string &query_file,
                 const std::string &gt_file, size_t k, const std::string &universal_label, const tann::Metric &metric,
                 const std::string &filter_label, const std::string &tags_file = std::string("")) {
        size_t npoints, nqueries, dim;

        float *query_data = nullptr;

        load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, 0);
        if (nqueries > PARTSIZE)
            std::cerr << "WARNING: #Queries provided (" << nqueries << ") is greater than " << PARTSIZE
                      << ". Computing GT only for the first " << PARTSIZE << " queries." << std::endl;

        // load tags
        const bool tags_enabled = tags_file.empty() ? false : true;
        std::vector<uint32_t> location_to_tag = tann::loadTags(tags_file, base_file);

        int *closest_points = new int[nqueries * k];
        float *dist_closest_points = new float[nqueries * k];

        std::vector<std::vector<std::pair<uint32_t, float>>> results;
        if (filter_label == "") {
            results = processUnfilteredParts<T>(base_file, nqueries, npoints, dim, k, query_data, metric,
                                                location_to_tag);
        } else {
            results = processFilteredParts<T>(base_file, label_file, filter_label, universal_label, nqueries, npoints,
                                              dim,
                                              k, query_data, metric, location_to_tag);
        }

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
    static std::string label_file;
    static std::string filter_label;
    static std::string universal_label;
    static std::string filter_label_file;
    uint64_t K;
}  // namespace detail

namespace tann {
    void set_compute_groundtruth_filter(turbo::App &app) {
        auto *cg = app.add_subcommand("compute_groundtruth_for_filter", "compute_groundtruth");
        cg->add_option("-t, --data_type", detail::data_type, "data type <int8/uint8/float>")->required();
        cg->add_option("-f, --dist_fn", detail::dist_fn, "distance function <l2/mips>")->required();
        cg->add_option("-b, --base_file", detail::base_file, "File containing the base vectors in binary format")->required();
        cg->add_option("-q, --query_file", detail::query_file,
                       "File containing the query vectors in binary format")->required();
        cg->add_option("-l, --label_file", detail::label_file, "Input labels file in txt format if present")->default_val("");
        cg->add_option("-F, --filter_label", detail::filter_label,
                       "Input filter label if doing filtered groundtruth")->default_val("");
        cg->add_option("-u, --universal_label", detail::universal_label,
                       "Universal label, if using it, only in conjunction with label_file")->default_val("");
        cg->add_option("-g, --gt_file", detail::gt_file,
                "File name for the writing ground truth in binary "
                "format, please don' append .bin at end if "
                "no filter_label or filter_label_file is provided it "
                "will save the file with '.bin' at end."
                "else it will save the file as filename_label.bin")->required();
        cg->add_option("--K", detail::K,
                       "Number of ground truth nearest neighbors to compute")->required();
        cg->add_option("-T, --tags_file", detail::tags_file,
                       "File containing the tags in binary format")->default_val("");
        cg->add_option("--filter_label_file",
                       detail::filter_label_file,
                       "Filter file for Queries for Filtered Search ")->default_val("");
        cg->callback([](){
            compute_groundtruth_filter();
        });
    }

    void compute_groundtruth_filter() {

        if (detail::data_type != std::string("float") && detail::data_type != std::string("int8") &&
                detail::data_type != std::string("uint8")) {
            std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
            return;
        }

        if (detail::filter_label != "" && detail::filter_label_file != "") {
            std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
            return;
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
            return;
        }

        std::vector<std::string> filter_labels;
        if (detail::filter_label != "") {
            filter_labels.push_back(detail::filter_label);
        } else if (detail::filter_label_file != "") {
            filter_labels = read_file_to_vector_of_strings(detail::filter_label_file, false);
        }

        // only when there is no filter label or 1 filter label for all queries
        if (filter_labels.size() == 1) {
            try {
                if (detail::data_type == std::string("float"))
                    detail::aux_main<float>(detail::base_file, detail::label_file, detail::query_file, detail::gt_file, detail::K, detail::universal_label, metric,
                                    filter_labels[0], detail::tags_file);
                if (detail::data_type == std::string("int8"))
                    detail::aux_main<int8_t>(detail::base_file, detail::label_file, detail::query_file, detail::gt_file, detail::K, detail::universal_label, metric,
                                     filter_labels[0], detail::tags_file);
                if (detail::data_type == std::string("uint8"))
                    detail::aux_main<uint8_t>(detail::base_file, detail::label_file, detail::query_file, detail::gt_file, detail::K, detail::universal_label, metric,
                                      filter_labels[0], detail::tags_file);
            }
            catch (const std::exception &e) {
                std::cout << std::string(e.what()) << std::endl;
                tann::cerr << "Compute GT failed." << std::endl;
                return ;
            }
        } else { // Each query has its own filter label
            // Split up data and query bins into label specific ones
            tsl::robin_map<std::string, uint32_t> labels_to_number_of_points;
            tsl::robin_map<std::string, uint32_t> labels_to_number_of_queries;

            label_set all_labels;
            for (size_t i = 0; i < filter_labels.size(); i++) {
                std::string label = filter_labels[i];
                all_labels.insert(label);

                if (labels_to_number_of_queries.find(label) == labels_to_number_of_queries.end()) {
                    labels_to_number_of_queries[label] = 0;
                }
                labels_to_number_of_queries[label] += 1;
            }

            size_t npoints;
            std::vector<std::vector<std::string>> point_to_labels;
            detail::parse_label_file_into_vec(npoints, detail::label_file, point_to_labels);
            std::vector<label_set> point_ids_to_labels(point_to_labels.size());
            std::vector<label_set> query_ids_to_labels(filter_labels.size());

            for (size_t i = 0; i < point_to_labels.size(); i++) {
                for (size_t j = 0; j < point_to_labels[i].size(); j++) {
                    std::string label = point_to_labels[i][j];
                    if (all_labels.find(label) != all_labels.end()) {
                        point_ids_to_labels[i].insert(point_to_labels[i][j]);
                        if (labels_to_number_of_points.find(label) == labels_to_number_of_points.end()) {
                            labels_to_number_of_points[label] = 0;
                        }
                        labels_to_number_of_points[label] += 1;
                    }
                }
            }

            for (size_t i = 0; i < filter_labels.size(); i++) {
                query_ids_to_labels[i].insert(filter_labels[i]);
            }

            tsl::robin_map<std::string, std::vector<uint32_t>> label_id_to_orig_id;
            tsl::robin_map<std::string, std::vector<uint32_t>> label_query_id_to_orig_id;

            if (detail::data_type == std::string("float")) {
                label_id_to_orig_id = tann::generate_label_specific_vector_files_compat<float>(
                        detail::base_file, labels_to_number_of_points, point_ids_to_labels, all_labels);

                label_query_id_to_orig_id = tann::generate_label_specific_vector_files_compat<float>(
                        detail::query_file, labels_to_number_of_queries, query_ids_to_labels,
                        all_labels); // query_filters acts like query_ids_to_labels
            } else if (detail::data_type == std::string("int8")) {
                label_id_to_orig_id = tann::generate_label_specific_vector_files_compat<int8_t>(
                        detail::base_file, labels_to_number_of_points, point_ids_to_labels, all_labels);

                label_query_id_to_orig_id = tann::generate_label_specific_vector_files_compat<int8_t>(
                        detail::query_file, labels_to_number_of_queries, query_ids_to_labels,
                        all_labels); // query_filters acts like query_ids_to_labels
            } else if (detail::data_type == std::string("uint8")) {
                label_id_to_orig_id = tann::generate_label_specific_vector_files_compat<uint8_t>(
                        detail::base_file, labels_to_number_of_points, point_ids_to_labels, all_labels);

                label_query_id_to_orig_id = tann::generate_label_specific_vector_files_compat<uint8_t>(
                        detail::query_file, labels_to_number_of_queries, query_ids_to_labels,
                        all_labels); // query_filters acts like query_ids_to_labels
            } else {
                tann::cerr << "Invalid data type" << std::endl;
                return ;
            }

            // Generate label specific ground truths

            try {
                for (const auto &label: all_labels) {
                    std::string filtered_base_file = detail::base_file + "_" + label;
                    std::string filtered_query_file = detail::query_file + "_" + label;
                    std::string filtered_gt_file = detail::gt_file + "_" + label;
                    if (detail::data_type == std::string("float"))
                        detail::aux_main<float>(filtered_base_file, "", filtered_query_file, filtered_gt_file, detail::K, "", metric,
                                        "");
                    if (detail::data_type == std::string("int8"))
                        detail::aux_main<int8_t>(filtered_base_file, "", filtered_query_file, filtered_gt_file, detail::K, "", metric,
                                         "");
                    if (detail::data_type == std::string("uint8"))
                        detail::aux_main<uint8_t>(filtered_base_file, "", filtered_query_file, filtered_gt_file,detail::K, "", metric,
                                          "");
                }
            }
            catch (const std::exception &e) {
                std::cout << std::string(e.what()) << std::endl;
                tann::cerr << "Compute GT failed." << std::endl;
                return ;
            }

            // Combine the label specific ground truths to produce a single GT file

            uint32_t *gt_ids = nullptr;
            float *gt_dists = nullptr;
            size_t gt_num, gt_dim;

            std::vector<std::vector<int32_t>> final_gt_ids;
            std::vector<std::vector<float>> final_gt_dists;

            uint32_t query_num = 0;
            for (const auto &lbl: all_labels) {
                query_num += labels_to_number_of_queries[lbl];
            }

            for (uint32_t i = 0; i < query_num; i++) {
                final_gt_ids.push_back(std::vector<int32_t>(detail::K));
                final_gt_dists.push_back(std::vector<float>(detail::K));
            }

            for (const auto &lbl: all_labels) {
                std::string filtered_gt_file = detail::gt_file + "_" + lbl;
                load_truthset(filtered_gt_file, gt_ids, gt_dists, gt_num, gt_dim);

                for (uint32_t i = 0; i < labels_to_number_of_queries[lbl]; i++) {
                    uint32_t orig_query_id = label_query_id_to_orig_id[lbl][i];
                    for (uint64_t j = 0; j < detail::K; j++) {
                        final_gt_ids[orig_query_id][j] = label_id_to_orig_id[lbl][gt_ids[i * detail::K + j]];
                        final_gt_dists[orig_query_id][j] = gt_dists[i *detail:: K + j];
                    }
                }
            }

            int32_t *closest_points = new int32_t[query_num * detail::K];
            float *dist_closest_points = new float[query_num * detail::K];

            for (uint32_t i = 0; i < query_num; i++) {
                for (uint32_t j = 0; j < detail::K; j++) {
                    closest_points[i * detail::K + j] = final_gt_ids[i][j];
                    dist_closest_points[i * detail::K + j] = final_gt_dists[i][j];
                }
            }

            detail::save_groundtruth_as_one_file(detail::gt_file, closest_points, dist_closest_points, query_num, detail::K);

            // cleanup artifacts
            std::cout << "Cleaning up artifacts..." << std::endl;
            tsl::robin_set<std::string> paths_to_clean{detail::gt_file, detail::base_file, detail::query_file};
            clean_up_artifacts(paths_to_clean, all_labels);
        }
    }
}