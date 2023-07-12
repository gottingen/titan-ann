// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "tann/vamana/utils.h"
#include "tann_cli.h"

namespace detail {
// Convert float types
    void
    block_convert_float(std::ifstream &reader, std::ofstream &writer, float *read_buf, float *write_buf, size_t npts,
                        size_t ndims) {
        reader.read((char *) read_buf, npts * (ndims * sizeof(float) + sizeof(uint32_t)));
        for (size_t i = 0; i < npts; i++) {
            memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1, ndims * sizeof(float));
        }
        writer.write((char *) write_buf, npts * ndims * sizeof(float));
    }

// Convert byte types
    void block_convert_byte(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf,
                            size_t npts, size_t ndims) {
        reader.read((char *) read_buf, npts * (ndims * sizeof(uint8_t) + sizeof(uint32_t)));
        for (size_t i = 0; i < npts; i++) {
            memcpy(write_buf + i * ndims, (read_buf + i * (ndims + sizeof(uint32_t))) + sizeof(uint32_t),
                   ndims * sizeof(uint8_t));
        }
        writer.write((char *) write_buf, npts * ndims * sizeof(uint8_t));
    }

    static std::string input;
    static std::string output;
    static std::string data_type;
}

namespace tann {
    void set_fvec_to_bin(turbo::App &app) {
        auto *sub = app.add_subcommand("fvec_to_bin", "float vector to bin");
        sub->add_option("-i,--input", detail::input, "input data path")->required();
        sub->add_option("-o,--output", detail::output, "path to output")->required();
        sub->add_option("-t, --data_type", detail::data_type, "bias")->required();
        sub->callback([]() {
            fvec_to_bin();
        });
    }

    void fvec_to_bin() {

        int datasize = sizeof(float);

        if (strcmp(detail::data_type.c_str(), "uint8") == 0 || strcmp(detail::data_type.c_str(), "int8") == 0) {
            datasize = sizeof(uint8_t);
        } else if (strcmp(detail::data_type.c_str(), "float") != 0) {
            std::cout << "Error: type not supported. Use float/int8/uint8" << std::endl;
            exit(-1);
        }

        std::ifstream reader(detail::input, std::ios::binary | std::ios::ate);
        size_t fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);

        uint32_t ndims_u32;
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        reader.seekg(0, std::ios::beg);
        size_t ndims = (size_t) ndims_u32;
        size_t npts = fsize / ((ndims * datasize) + sizeof(uint32_t));
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(detail::output, std::ios::binary);
        int32_t npts_s32 = (int32_t) npts;
        int32_t ndims_s32 = (int32_t) ndims;
        writer.write((char *) &npts_s32, sizeof(int32_t));
        writer.write((char *) &ndims_s32, sizeof(int32_t));

        size_t chunknpts = std::min(npts, blk_size);
        uint8_t *read_buf = new uint8_t[chunknpts * ((ndims * datasize) + sizeof(uint32_t))];
        uint8_t *write_buf = new uint8_t[chunknpts * ndims * datasize];

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            if (datasize == sizeof(float)) {
                detail::block_convert_float(reader, writer, (float *) read_buf, (float *) write_buf, cblk_size, ndims);
            } else {
                detail::block_convert_byte(reader, writer, read_buf, write_buf, cblk_size, ndims);
            }
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        reader.close();
        writer.close();
    }
}