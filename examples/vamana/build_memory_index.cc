// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <omp.h>
#include <cstring>
#include "tann/vamana/index.h"
#include "tann/common/utils.h"

#ifndef _WINDOWS

#include <sys/mman.h>
#include <unistd.h>

#else
#include <Windows.h>
#endif

#include "tann/io/memory_mapper.h"
#include "tann/common/ann_exception.h"
#include "turbo/flags/flags.h"

template<typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
int build_in_memory_index(const tann::Metric &metric, const std::string &data_path, const uint32_t R,
                          const uint32_t L, const float alpha, const std::string &save_path, const uint32_t num_threads,
                          const bool use_pq_build, const size_t num_pq_bytes, const bool use_opq,
                          const std::string &label_file, const std::string &universal_label, const uint32_t Lf) {
    tann::IndexWriteParameters paras = tann::IndexWriteParametersBuilder(L, R)
            .with_filter_list_size(Lf)
            .with_alpha(alpha)
            .with_saturate_graph(false)
            .with_num_threads(num_threads)
            .build();
    std::string labels_file_to_use = save_path + "_label_formatted.txt";
    std::string mem_labels_int_map_file = save_path + "_labels_map.txt";

    size_t data_num, data_dim;
    tann::get_bin_metadata(data_path, data_num, data_dim);

    tann::Index<T, TagT, LabelT> index(metric, data_dim, data_num, false, false, false, use_pq_build, num_pq_bytes,
                                       use_opq);
    auto s = std::chrono::high_resolution_clock::now();
    if (label_file == "") {
        index.build(data_path.c_str(), data_num, paras);
    } else {
        convert_labels_string_to_int(label_file, labels_file_to_use, mem_labels_int_map_file, universal_label);
        if (universal_label != "") {
            LabelT unv_label_as_num = 0;
            index.set_universal_label(unv_label_as_num);
        }
        index.build_filtered_index(data_path.c_str(), labels_file_to_use, data_num, paras);
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    std::cout << "Indexing time: " << diff.count() << "\n";
    index.save(save_path.c_str());
    if (label_file != "")
        std::remove(labels_file_to_use.c_str());
    return 0;
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, data_path, index_path_prefix, label_file, universal_label, label_type;
    uint32_t num_threads, R, L, Lf, build_PQ_bytes;
    float alpha;
    bool use_pq_build, use_opq;

    turbo::App app("build memory index");

    app.add_option("-t, --data_type", data_type, "data type <int8/uint8/float>")->required();
    app.add_option("-F, --dist_fn", dist_fn,
                   "distance function <l2/mips/cosine>")->required();
    app.add_option("-p, --data_path", data_path,
                   "Input data file in bin format")->required();
    app.add_option("-x, --index_path_prefix", index_path_prefix,
                   "Path prefix for saving index file components")->required();
    app.add_option("--max_degree,-R", R, "Maximum graph degree")->default_val(64);
    app.add_option("--Lbuild,-L", L,
                   "Build complexity, higher value results in better graphs")->default_val(100);
    app.add_option("-a, --alpha", alpha,
                   "alpha controls density and diameter of graph, set "
                   "1 for sparse graph, "
                   "1.2 or 1.4 for denser graphs with lower diameter")->default_val(1.2f);
    app.add_option("--num_threads,-T", num_threads,
                   "Number of threads used for building index (defaults to "
                   "omp_get_num_procs())")->default_val(omp_get_num_procs());
    app.add_option("-q, --build_PQ_bytes", build_PQ_bytes,
                   "Number of PQ bytes to build the index; 0 for full precision "
                   "build")->default_val(0);
    app.add_option("--use_opq", use_opq,
                   "Set true for OPQ compression while using PQ "
                   "distance comparisons for "
                   "building the index, and false for PQ compression")->default_val(false);
    app.add_option("--use_pq_build", use_pq_build,
                   "Set true for PQ compression while using PQ "
                   "distance comparisons for "
                   "building the index, and false for PQ compression")->default_val(false);
    app.add_option("-l, --label_file", label_file,
                   "Input label file in txt format for Filtered Index search. "
                   "The file should contain comma separated filters for each node "
                   "with each line corresponding to a graph node")->default_val("");
    app.add_option("-u, --universal_label", universal_label,
                   "Universal label, if using it, only in conjunction with "
                   "labels_file")->default_val("");
    app.add_option("--FilteredLbuild,-f", Lf,
                   "Build complexity for filtered points, higher value "
                   "results in better graphs")->default_val(0);
    app.add_option("-z, --label_type", label_type,
                   "Storage type of Labels <uint/ushort>, default value is uint which "
                   "will consume memory 4 bytes per filter")->default_val("uint");

    TURBO_FLAGS_PARSE(app, argc, argv);

    tann::Metric metric;
    if (dist_fn == std::string("mips")) {
        metric = tann::Metric::INNER_PRODUCT;
    } else if (dist_fn == std::string("l2")) {
        metric = tann::Metric::L2;
    } else if (dist_fn == std::string("cosine")) {
        metric = tann::Metric::COSINE;
    } else {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    try {
        tann::cout << "Starting index build with R: " << R << "  Lbuild: " << L << "  alpha: " << alpha
                   << "  #threads: " << num_threads << std::endl;
        if (label_file != "" && label_type == "ushort") {
            if (data_type == std::string("int8"))
                return build_in_memory_index<int8_t, uint32_t, uint16_t>(
                        metric, data_path, R, L, alpha, index_path_prefix, num_threads, use_pq_build, build_PQ_bytes,
                        use_opq, label_file, universal_label, Lf);
            else if (data_type == std::string("uint8"))
                return build_in_memory_index<uint8_t, uint32_t, uint16_t>(
                        metric, data_path, R, L, alpha, index_path_prefix, num_threads, use_pq_build, build_PQ_bytes,
                        use_opq, label_file, universal_label, Lf);
            else if (data_type == std::string("float"))
                return build_in_memory_index<float, uint32_t, uint16_t>(
                        metric, data_path, R, L, alpha, index_path_prefix, num_threads, use_pq_build, build_PQ_bytes,
                        use_opq, label_file, universal_label, Lf);
            else {
                std::cout << "Unsupported type. Use one of int8, uint8 or float." << std::endl;
                return -1;
            }
        } else {
            if (data_type == std::string("int8"))
                return build_in_memory_index<int8_t>(metric, data_path, R, L, alpha, index_path_prefix, num_threads,
                                                     use_pq_build, build_PQ_bytes, use_opq, label_file, universal_label,
                                                     Lf);
            else if (data_type == std::string("uint8"))
                return build_in_memory_index<uint8_t>(metric, data_path, R, L, alpha, index_path_prefix, num_threads,
                                                      use_pq_build, build_PQ_bytes, use_opq, label_file,
                                                      universal_label, Lf);
            else if (data_type == std::string("float"))
                return build_in_memory_index<float>(metric, data_path, R, L, alpha, index_path_prefix, num_threads,
                                                    use_pq_build, build_PQ_bytes, use_opq, label_file, universal_label,
                                                    Lf);
            else {
                std::cout << "Unsupported type. Use one of int8, uint8 or float." << std::endl;
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
