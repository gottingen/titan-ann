// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>

#include "tann/common/utils.h"
#include "tann/diskann/disk_utils.h"
#include "tann/common/math_utils.h"
#include "tann/diskann/index.h"
#include "tann/diskann/partition.h"
#include "turbo/flags/flags.h"


int main(int argc, char **argv) {
    std::string data_type, dist_fn, data_path, index_path_prefix, codebook_prefix, label_file, universal_label,
            label_type;
    uint32_t num_threads, R, L, disk_PQ, build_PQ, QD, Lf, filter_threshold;
    float B, M;
    bool append_reorder_data = false;
    bool use_opq = false;

    turbo::App app("build disk index");

    app.add_option("-t, --data_type", data_type, "data type <int8/uint8/float>")->required();
    app.add_option("-F, --dist_fn", dist_fn, "distance function <l2/mips>")->required();
    app.add_option("-p, --data_path", data_path, "Input data file in bin format")->required();
    app.add_option("-x, --index_path_prefix", index_path_prefix, "Path prefix for saving index file components")->required();
    app.add_option("--max_degree,-R", R, "Maximum graph degree")->default_val(64);
    app.add_option("--Lbuild,-L", L, "Build complexity, higher value results in better graphs")->default_val(100);
    app.add_option("--search_DRAM_budget,-B", B,
                   "DRAM budget in GB for searching the index to set the "
                   "compressed level for data while search happens")->required();
    app.add_option("--build_DRAM_budget,-M", M,
                   "DRAM budget in GB for building the index")->required();
    app.add_option("--num_threads,-T", num_threads,
                   "Number of threads used for building index (defaults to "
                   "omp_get_num_procs())")->default_val(omp_get_num_procs());
    app.add_option("--QD", QD, " Quantized Dimension for compression")->default_val(0);
    app.add_option("--codebook_prefix", codebook_prefix,
                   "Path prefix for pre-trained codebook")->default_val("");
    app.add_option("--PQ_disk_bytes", disk_PQ,
                   "Number of bytes to which vectors should be compressed "
                   "on SSD; 0 for no compression")->default_val(0);
    app.add_option("--append_reorder_data", append_reorder_data,
                   "Include full precision data in the index. Use only in "
                   "conjuction with compressed data on SSD.")->default_val(false);
    app.add_option("--build_PQ_bytes", build_PQ,
                   "Number of PQ bytes to build the index; 0 for full "
                   "precision build")->default_val(0);
    app.add_option("--use_opq", use_opq,
                   "Use Optimized Product Quantization (OPQ).")->default_val(false);
    app.add_option("--label_file", label_file,
                   "Input label file in txt format for Filtered Index build ."
                   "The file should contain comma separated filters for each node "
                   "with each line corresponding to a graph node")->default_val("");
    app.add_option("--universal_label", universal_label,
                   "Universal label, Use only in conjuction with label file for "
                   "filtered "
                   "index build. If a graph node has all the labels against it, we "
                   "can "
                   "assign a special universal filter to the point instead of comma "
                   "separated filters for that point")->default_val("");
    app.add_option("--FilteredLbuild", Lf,
                   "Build complexity for filtered points, higher value "
                   "results in better graphs")->default_val(0);
    app.add_option("--filter_threshold", filter_threshold,
                   "Threshold to break up the existing nodes to generate new graph "
                   "internally where each node has a maximum F labels.")->default_val(0);
    app.add_option("--label_type", label_type,
                   "Storage type of Labels <uint/ushort>, default value is uint which "
                   "will consume memory 4 bytes per filter")->default_val("uint");

    TURBO_FLAGS_PARSE(app, argc, argv);

    bool use_filters = false;
    if (label_file != "") {
        use_filters = true;
    }

    tann::Metric metric;
    if (dist_fn == std::string("l2"))
        metric = tann::Metric::L2;
    else if (dist_fn == std::string("mips"))
        metric = tann::Metric::INNER_PRODUCT;
    else {
        std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
        return -1;
    }

    if (append_reorder_data) {
        if (disk_PQ == 0) {
            std::cout << "Error: It is not necessary to append data for reordering "
                         "when vectors are not compressed on disk."
                      << std::endl;
            return -1;
        }
        if (data_type != std::string("float")) {
            std::cout << "Error: Appending data for reordering currently only "
                         "supported for float data type."
                      << std::endl;
            return -1;
        }
    }

    std::string params = std::string(std::to_string(R)) + " " + std::string(std::to_string(L)) + " " +
                         std::string(std::to_string(B)) + " " + std::string(std::to_string(M)) + " " +
                         std::string(std::to_string(num_threads)) + " " + std::string(std::to_string(disk_PQ)) + " " +
                         std::string(std::to_string(append_reorder_data)) + " " +
                         std::string(std::to_string(build_PQ)) + " " + std::string(std::to_string(QD));

    try {
        if (label_file != "" && label_type == "ushort") {
            if (data_type == std::string("int8"))
                return tann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                      metric, use_opq, codebook_prefix, use_filters, label_file,
                                                      universal_label, filter_threshold, Lf);
            else if (data_type == std::string("uint8"))
                return tann::build_disk_index<uint8_t, uint16_t>(
                        data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
                        use_filters, label_file, universal_label, filter_threshold, Lf);
            else if (data_type == std::string("float"))
                return tann::build_disk_index<float, uint16_t>(
                        data_path.c_str(), index_path_prefix.c_str(), params.c_str(), metric, use_opq, codebook_prefix,
                        use_filters, label_file, universal_label, filter_threshold, Lf);
            else {
                tann::cerr << "Error. Unsupported data type" << std::endl;
                return -1;
            }
        } else {
            if (data_type == std::string("int8"))
                return tann::build_disk_index<int8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                      metric, use_opq, codebook_prefix, use_filters, label_file,
                                                      universal_label, filter_threshold, Lf);
            else if (data_type == std::string("uint8"))
                return tann::build_disk_index<uint8_t>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                       metric, use_opq, codebook_prefix, use_filters, label_file,
                                                       universal_label, filter_threshold, Lf);
            else if (data_type == std::string("float"))
                return tann::build_disk_index<float>(data_path.c_str(), index_path_prefix.c_str(), params.c_str(),
                                                     metric, use_opq, codebook_prefix, use_filters, label_file,
                                                     universal_label, filter_threshold, Lf);
            else {
                tann::cerr << "Error. Unsupported data type" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e) {
        std::cout << std::string(e.what()) << std::endl;
        tann::cerr << "Index build failed." << std::endl;
        return -1;
    }
}
