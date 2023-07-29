// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "tann/diskann/index.h"
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include "tann/diskann/timer.h"
#include <future>

#include "tann/common/utils.h"
#include "turbo/flags/flags.h"

#ifndef _WINDOWS

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

#include "tann/io/memory_mapper.h"


// load_aligned_bin modified to read pieces of the file, but using ifstream
// instead of cached_ifstream.
template<typename T>
inline void load_aligned_bin_part(const std::string &bin_file, T *data, size_t offset_points, size_t points_to_read) {
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    int npts_i32, dim_i32;
    reader.read((char *) &npts_i32, sizeof(int));
    reader.read((char *) &dim_i32, sizeof(int));
    size_t npts = (uint32_t) npts_i32;
    size_t dim = (uint32_t) dim_i32;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size) {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is  "
               << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " size of <T>= " << sizeof(T)
               << std::endl;
        std::cout << stream.str();
        throw tann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (offset_points + points_to_read > npts) {
        std::stringstream stream;
        stream << "Error. Not enough points in file. Requested " << offset_points << "  offset and " << points_to_read
               << " points, but have only " << npts << " points" << std::endl;
        std::cout << stream.str();
        throw tann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

    const size_t rounded_dim = ROUND_UP(dim, 8);

    for (size_t i = 0; i < points_to_read; i++) {
        reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    reader.close();
}

std::string get_save_filename(const std::string &save_path, size_t active_window, size_t consolidate_interval,
                              size_t max_points_to_insert) {
    std::string final_path = save_path;
    final_path += "act" + std::to_string(active_window) + "-";
    final_path += "cons" + std::to_string(consolidate_interval) + "-";
    final_path += "max" + std::to_string(max_points_to_insert);
    return final_path;
}

template<typename T, typename TagT, typename LabelT>
void insert_next_batch(tann::Index<T, TagT, LabelT> &index, size_t start, size_t end, size_t insert_threads, T *data,
                       size_t aligned_dim) {
    try {
        tann::Timer insert_timer;
        std::cout << std::endl << "Inserting from " << start << " to " << end << std::endl;

        size_t num_failed = 0;
#pragma omp parallel for num_threads((int32_t)insert_threads) schedule(dynamic) reduction(+ : num_failed)
        for (int64_t j = start; j < (int64_t) end; j++) {
            if (index.insert_point(&data[(j - start) * aligned_dim], 1 + static_cast<TagT>(j)) != 0) {
                std::cerr << "Insert failed " << j << std::endl;
                num_failed++;
            }
        }
        const double elapsedSeconds = insert_timer.elapsed() / 1000000.0;
        std::cout << "Insertion time " << elapsedSeconds << " seconds (" << (end - start) / elapsedSeconds
                  << " points/second overall, " << (end - start) / elapsedSeconds / insert_threads << " per thread)"
                  << std::endl;
        if (num_failed > 0)
            std::cout << num_failed << " of " << end - start << "inserts failed" << std::endl;
    }
    catch (std::system_error &e) {
        std::cout << "Exiting after catching exception in insertion task: " << e.what() << std::endl;
    }
}

template<typename T, typename TagT, typename LabelT>
void delete_and_consolidate(tann::Index<T, TagT, LabelT> &index, tann::IndexWriteParameters &delete_params,
                            size_t start, size_t end) {
    try {
        std::cout << std::endl << "Lazy deleting points " << start << " to " << end << "... ";
        for (size_t i = start; i < end; ++i)
            index.lazy_delete(static_cast<TagT>(1 + i));
        std::cout << "lazy delete done." << std::endl;

        auto report = index.consolidate_deletes(delete_params);
        while (report._status != tann::consolidation_report::status_code::SUCCESS) {
            int wait_time = 5;
            if (report._status == tann::consolidation_report::status_code::LOCK_FAIL) {
                tann::cerr << "Unable to acquire consolidate delete lock after "
                           << "deleting points " << start << " to " << end << ". Will retry in " << wait_time
                           << "seconds." << std::endl;
            } else if (report._status == tann::consolidation_report::status_code::INCONSISTENT_COUNT_ERROR) {
                tann::cerr << "Inconsistent counts in data structure. "
                           << "Will retry in " << wait_time << "seconds." << std::endl;
            } else {
                std::cerr << "Exiting after unknown error in consolidate delete" << std::endl;
                exit(-1);
            }
            std::this_thread::sleep_for(std::chrono::seconds(wait_time));
            report = index.consolidate_deletes(delete_params);
        }
        auto points_processed = report._active_points + report._slots_released;
        auto deletion_rate = points_processed / report._time;
        std::cout << "#active points: " << report._active_points << std::endl
                  << "max points: " << report._max_points << std::endl
                  << "empty slots: " << report._empty_slots << std::endl
                  << "deletes processed: " << report._slots_released << std::endl
                  << "latest delete size: " << report._delete_set_size << std::endl
                  << "Deletion rate: " << deletion_rate << "/sec   "
                  << "Deletion rate: " << deletion_rate / delete_params.num_threads << "/thread/sec   " << std::endl;
    }
    catch (std::system_error &e) {
        std::cerr << "Exiting after catching exception in deletion task: " << e.what() << std::endl;
        exit(-1);
    }
}

template<typename T>
void build_incremental_index(const std::string &data_path, const uint32_t L, const uint32_t R, const float alpha,
                             const uint32_t insert_threads, const uint32_t consolidate_threads,
                             size_t max_points_to_insert, size_t active_window, size_t consolidate_interval,
                             const float start_point_norm, uint32_t num_start_pts, const std::string &save_path) {
    const uint32_t C = 500;
    const bool saturate_graph = false;

    tann::IndexWriteParameters params = tann::IndexWriteParametersBuilder(L, R)
            .with_max_occlusion_size(C)
            .with_alpha(alpha)
            .with_saturate_graph(saturate_graph)
            .with_num_threads(insert_threads)
            .with_num_frozen_points(num_start_pts)
            .build();

    tann::IndexWriteParameters delete_params = tann::IndexWriteParametersBuilder(L, R)
            .with_max_occlusion_size(C)
            .with_alpha(alpha)
            .with_saturate_graph(saturate_graph)
            .with_num_threads(consolidate_threads)
            .build();

    size_t dim, aligned_dim;
    size_t num_points;

    tann::get_bin_metadata(data_path, num_points, dim);
    tann::cout << "metadata: file " << data_path << " has " << num_points << " points in " << dim << " dims"
               << std::endl;
    aligned_dim = ROUND_UP(dim, 8);

    if (max_points_to_insert == 0) {
        max_points_to_insert = num_points;
    }

    if (num_points < max_points_to_insert)
        throw tann::ANNException(std::string("num_points(") + std::to_string(num_points) +
                                 ") < max_points_to_insert(" + std::to_string(max_points_to_insert) + ")",
                                 -1, __FUNCSIG__, __FILE__, __LINE__);

    if (max_points_to_insert < active_window + consolidate_interval)
        throw tann::ANNException("ERROR: max_points_to_insert < "
                                 "active_window + consolidate_interval",
                                 -1, __FUNCSIG__, __FILE__, __LINE__);

    if (consolidate_interval < max_points_to_insert / 1000)
        throw tann::ANNException("ERROR: consolidate_interval is too small", -1, __FUNCSIG__, __FILE__, __LINE__);

    using TagT = uint32_t;
    using LabelT = uint32_t;
    const bool enable_tags = true;

    tann::Index<T, TagT, LabelT> index(tann::L2, dim, active_window + 4 * consolidate_interval, true, params, L,
                                       insert_threads, enable_tags, true);
    index.set_start_points_at_random(static_cast<T>(start_point_norm));
    index.enable_delete();

    T *data = nullptr;
    tann::alloc_aligned((void **) &data, std::max(consolidate_interval, active_window) * aligned_dim * sizeof(T),
                        8 * sizeof(T));

    std::vector<TagT> tags(max_points_to_insert);
    std::iota(tags.begin(), tags.end(), static_cast<TagT>(0));

    tann::Timer timer;

    std::vector<std::future<void>> delete_tasks;

    auto insert_task = std::async(std::launch::async, [&]() {
        load_aligned_bin_part(data_path, data, 0, active_window);
        insert_next_batch(index, 0, active_window, insert_threads, data, aligned_dim);
    });
    insert_task.wait();

    for (size_t start = active_window; start + consolidate_interval <= max_points_to_insert;
         start += consolidate_interval) {
        auto end = std::min(start + consolidate_interval, max_points_to_insert);
        auto insert_task = std::async(std::launch::async, [&]() {
            load_aligned_bin_part(data_path, data, start, end - start);
            insert_next_batch(index, start, end, insert_threads, data, aligned_dim);
        });
        insert_task.wait();

        if (delete_tasks.size() > 0)
            delete_tasks[delete_tasks.size() - 1].wait();
        if (start >= active_window + consolidate_interval) {
            auto start_del = start - active_window - consolidate_interval;
            auto end_del = start - active_window;

            delete_tasks.emplace_back(std::async(
                    std::launch::async, [&]() { delete_and_consolidate(index, delete_params, start_del, end_del); }));
        }
    }
    if (delete_tasks.size() > 0)
        delete_tasks[delete_tasks.size() - 1].wait();

    std::cout << "Time Elapsed " << timer.elapsed() / 1000 << "ms\n";
    const auto save_path_inc =
            get_save_filename(save_path + ".after-streaming-", active_window, consolidate_interval,
                              max_points_to_insert);
    index.save(save_path_inc.c_str(), true);

    tann::aligned_free(data);
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, data_path, index_path_prefix;
    uint32_t insert_threads, consolidate_threads;
    uint32_t R, L, num_start_pts;
    float alpha, start_point_norm;
    size_t max_points_to_insert, active_window, consolidate_interval;

    turbo::App app("test_streaming_scenario");

    app.add_option("-t, --data_type", data_type, "data type <int8/uint8/float>")->required();
    app.add_option("-F, --dist_fn", dist_fn, "distance function <l2/mips>")->required();
    app.add_option("-P, --data_path", data_path,
                   "Input data file in bin format")->required();
    app.add_option("-x, --index_path_prefix", index_path_prefix,
                   "Path prefix for saving index file components")->required();
    app.add_option("--max_degree,-R", R, "Maximum graph degree")->default_val(64);
    app.add_option("--Lbuild,-L", L,
                   "Build complexity, higher value results in better graphs")->default_val(100);
    app.add_option("-A, --alpha", alpha,
                   "alpha controls density and diameter of graph, set "
                   "1 for sparse graph, "
                   "1.2 or 1.4 for denser graphs with lower diameter")->default_val(1.2f);
    app.add_option("--insert_threads",
                   insert_threads,
                   "Number of threads used for inserting into the index (defaults to "
                   "omp_get_num_procs()/2)")->default_val(omp_get_num_procs() / 2);
    app.add_option("--consolidate_threads", consolidate_threads,
                   "Number of threads used for consolidating deletes to "
                   "the index (defaults to omp_get_num_procs()/2)")->default_val(omp_get_num_procs() / 2);

    app.add_option("--max_points_to_insert", max_points_to_insert,
                   "The number of points from the file that the program streams "
                   "over ")->default_val(0);
    app.add_option("-w, --active_window", active_window,
                   "Program maintains an index over an active window of "
                   "this size that slides through the data")->required();
    app.add_option("--consolidate_interval", consolidate_interval,
                   "The program simultaneously adds this number of points to the "
                   "right of "
                   "the window while deleting the same number from the left")->required();
    app.add_option("--start_point_norm", start_point_norm,
                   "Set the start point to a random point on a sphere of this radius")->required();
    app.add_option(
            "--num_start_points", num_start_pts,
            "Set the number of random start (frozen) points to use when "
            "inserting and searching")->default_val(tann::defaults::NUM_FROZEN_POINTS_DYNAMIC);

    TURBO_FLAGS_PARSE(app, argc, argv);
    try {
        if (data_type == std::string("int8"))
            build_incremental_index<int8_t>(data_path, L, R, alpha, insert_threads, consolidate_threads,
                                            max_points_to_insert, active_window, consolidate_interval, start_point_norm,
                                            num_start_pts, index_path_prefix);
        else if (data_type == std::string("uint8"))
            build_incremental_index<uint8_t>(data_path, L, R, alpha, insert_threads, consolidate_threads,
                                             max_points_to_insert, active_window, consolidate_interval,
                                             start_point_norm, num_start_pts, index_path_prefix);
        else if (data_type == std::string("float"))
            build_incremental_index<float>(data_path, L, R, alpha, insert_threads, consolidate_threads,
                                           max_points_to_insert, active_window, consolidate_interval, start_point_norm,
                                           num_start_pts, index_path_prefix);
        else
            std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
    }
    catch (const std::exception &e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        exit(-1);
    }
    catch (...) {
        std::cerr << "Caught unknown exception" << std::endl;
        exit(-1);
    }

    return 0;
}
