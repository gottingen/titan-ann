// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "tann/common/utils.h"

const uint32_t MAX_REQUEST_SIZE = 1024 * 1024 * 1024; // 64MB
const uint32_t MAX_SIMULTANEOUS_READ_REQUESTS = 128;

#ifdef _WINDOWS
#include <intrin.h>

// Taken from:
// https://insufficientlycomplicated.wordpress.com/2011/11/07/detecting-intel-advanced-vector-extensions-avx-in-visual-studio/
bool cpuHasAvxSupport()
{
    bool avxSupported = false;

    // Checking for AVX requires 3 things:
    // 1) CPUID indicates that the OS uses XSAVE and XRSTORE
    //     instructions (allowing saving YMM registers on context
    //     switch)
    // 2) CPUID indicates support for AVX
    // 3) XGETBV indicates the AVX registers will be saved and
    //     restored on context switch
    //
    // Note that XGETBV is only available on 686 or later CPUs, so
    // the instruction needs to be conditionally run.
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);

    bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
    bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;

    if (osUsesXSAVE_XRSTORE && cpuAVXSuport)
    {
        // Check if the OS will save the YMM registers
        unsigned long long xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) || false;
    }

    return avxSupported;
}

bool cpuHasAvx2Support()
{
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    int n = cpuInfo[0];
    if (n >= 7)
    {
        __cpuidex(cpuInfo, 7, 0);
        static int avx2Mask = 0x20;
        return (cpuInfo[1] & avx2Mask) > 0;
    }
    return false;
}

bool AvxSupportedCPU = cpuHasAvxSupport();
bool Avx2SupportedCPU = cpuHasAvx2Support();

#else

bool Avx2SupportedCPU = true;
bool AvxSupportedCPU = false;
#endif

namespace tann {

    void block_convert(std::ofstream &writr, std::ifstream &readr, float *read_buf, size_t npts, size_t ndims) {
        readr.read((char *) read_buf, npts * ndims * sizeof(float));
        uint32_t ndims_u32 = (uint32_t) ndims;
#pragma omp parallel for
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            float norm_pt = std::numeric_limits<float>::epsilon();
            for (uint32_t dim = 0; dim < ndims_u32; dim++) {
                norm_pt += *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
            }
            norm_pt = std::sqrt(norm_pt);
            for (uint32_t dim = 0; dim < ndims_u32; dim++) {
                *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
            }
        }
        writr.write((char *) read_buf, npts * ndims * sizeof(float));
    }

    void normalize_data_file(const std::string &inFileName, const std::string &outFileName) {
        std::ifstream readr(inFileName, std::ios::binary);
        std::ofstream writr(outFileName, std::ios::binary);

        int npts_s32, ndims_s32;
        readr.read((char *) &npts_s32, sizeof(int32_t));
        readr.read((char *) &ndims_s32, sizeof(int32_t));

        writr.write((char *) &npts_s32, sizeof(int32_t));
        writr.write((char *) &ndims_s32, sizeof(int32_t));

        size_t npts = (size_t) npts_s32;
        size_t ndims = (size_t) ndims_s32;
        tann::cout << "Normalizing FLOAT vectors in file: " << inFileName << std::endl;
        tann::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        tann::cout << "# blks: " << nblks << std::endl;

        float *read_buf = new float[npts * ndims];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert(writr, readr, read_buf, cblk_size, ndims);
        }
        delete[] read_buf;

        tann::cout << "Wrote normalized points to file: " << outFileName << std::endl;
    }

    double calculate_recall(uint32_t num_queries, uint32_t *gold_std, float *gs_dist, uint32_t dim_gs,
                            uint32_t *our_results, uint32_t dim_or, uint32_t recall_at) {
        double total_recall = 0;
        std::set<uint32_t> gt, res;

        for (size_t i = 0; i < num_queries; i++) {
            gt.clear();
            res.clear();
            uint32_t *gt_vec = gold_std + dim_gs * i;
            uint32_t *res_vec = our_results + dim_or * i;
            size_t tie_breaker = recall_at;
            if (gs_dist != nullptr) {
                tie_breaker = recall_at - 1;
                float *gt_dist_vec = gs_dist + dim_gs * i;
                while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
                    tie_breaker++;
            }

            gt.insert(gt_vec, gt_vec + tie_breaker);
            res.insert(res_vec,
                       res_vec + recall_at); // change to recall_at for recall k@k
            // or dim_or for k@dim_or
            uint32_t cur_recall = 0;
            for (auto &v: gt) {
                if (res.find(v) != res.end()) {
                    cur_recall++;
                }
            }
            total_recall += cur_recall;
        }
        return total_recall / (num_queries) * (100.0 / recall_at);
    }

    double calculate_recall(uint32_t num_queries, uint32_t *gold_std, float *gs_dist, uint32_t dim_gs,
                            uint32_t *our_results, uint32_t dim_or, uint32_t recall_at,
                            const turbo::flat_hash_set<uint32_t> &active_tags) {
        double total_recall = 0;
        std::set<uint32_t> gt, res;
        bool printed = false;
        for (size_t i = 0; i < num_queries; i++) {
            gt.clear();
            res.clear();
            uint32_t *gt_vec = gold_std + dim_gs * i;
            uint32_t *res_vec = our_results + dim_or * i;
            size_t tie_breaker = recall_at;
            uint32_t active_points_count = 0;
            uint32_t cur_counter = 0;
            while (active_points_count < recall_at && cur_counter < dim_gs) {
                if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
                    active_points_count++;
                }
                cur_counter++;
            }
            if (active_tags.empty())
                cur_counter = recall_at;

            if ((active_points_count < recall_at && !active_tags.empty()) && !printed) {
                tann::cout << "Warning: Couldn't find enough closest neighbors " << active_points_count << "/"
                           << recall_at
                           << " from "
                              "truthset for query # "
                           << i << ". Will result in under-reported value of recall." << std::endl;
                printed = true;
            }
            if (gs_dist != nullptr) {
                tie_breaker = cur_counter - 1;
                float *gt_dist_vec = gs_dist + dim_gs * i;
                while (tie_breaker < dim_gs && gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
                    tie_breaker++;
            }

            gt.insert(gt_vec, gt_vec + tie_breaker);
            res.insert(res_vec, res_vec + recall_at);
            uint32_t cur_recall = 0;
            for (auto &v: res) {
                if (gt.find(v) != gt.end()) {
                    cur_recall++;
                }
            }
            total_recall += cur_recall;
        }
        return ((double) (total_recall / (num_queries))) * ((double) (100.0 / recall_at));
    }

    double calculate_range_search_recall(uint32_t num_queries, std::vector<std::vector<uint32_t>> &groundtruth,
                                         std::vector<std::vector<uint32_t>> &our_results) {
        double total_recall = 0;
        std::set<uint32_t> gt, res;

        for (size_t i = 0; i < num_queries; i++) {
            gt.clear();
            res.clear();

            gt.insert(groundtruth[i].begin(), groundtruth[i].end());
            res.insert(our_results[i].begin(), our_results[i].end());
            uint32_t cur_recall = 0;
            for (auto &v: gt) {
                if (res.find(v) != res.end()) {
                    cur_recall++;
                }
            }
            if (gt.size() != 0)
                total_recall += ((100.0 * cur_recall) / gt.size());
            else
                total_recall += 100;
        }
        return total_recall / (num_queries);
    }


} // namespace tann
