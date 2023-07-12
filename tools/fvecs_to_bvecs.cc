// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "tann/vamana/utils.h"
#include "tann_cli.h"
namespace detail {
    void block_convert(std::ifstream &reader, std::ofstream &writer, float *read_buf, uint8_t *write_buf, size_t npts,
                       size_t ndims) {
        reader.read((char *) read_buf, npts * (ndims * sizeof(float) + sizeof(uint32_t)));
        for (size_t i = 0; i < npts; i++) {
            memcpy(write_buf + i * (ndims + 4), read_buf + i * (ndims + 1), sizeof(uint32_t));
            for (size_t d = 0; d < ndims; d++)
                write_buf[i * (ndims + 4) + 4 + d] = (uint8_t) read_buf[i * (ndims + 1) + 1 + d];
        }
        writer.write((char *) write_buf, npts * (ndims * 1 + 4));
    }

    static std::string input;
    static std::string output;
}
namespace tann {
    void set_fvec_to_bvec(turbo::App &app) {
        auto *sub = app.add_subcommand("fvec_to_bvec", "float vector to bin vector");
        sub->add_option("-i,--input", detail::input, "input data path")->required();
        sub->add_option("-o,--output", detail::output, "path to output")->required();
        sub->callback([]() {
            fvec_to_bvec();
        });
    }
    void fvec_to_bvec();
    void fvec_to_bvec() {
        std::ifstream reader(detail::input, std::ios::binary | std::ios::ate);
        size_t fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);

        uint32_t ndims_u32;
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        reader.seekg(0, std::ios::beg);
        size_t ndims = (size_t) ndims_u32;
        size_t npts = fsize / ((ndims + 1) * sizeof(float));
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(detail::output, std::ios::binary);
        auto read_buf = new float[npts * (ndims + 1)];
        auto write_buf = new uint8_t[npts * (ndims + 4)];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            detail::block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        reader.close();
        writer.close();
    }
}