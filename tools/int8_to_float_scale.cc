// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "tann/common/utils.h"
#include "tann_cli.h"
namespace detail {
    void block_convert(std::ofstream &writer, float *write_buf, std::ifstream &reader, int8_t *read_buf, size_t npts,
                       size_t ndims, float bias, float scale) {
        reader.read((char *) read_buf, npts * ndims * sizeof(int8_t));

        for (size_t i = 0; i < npts; i++) {
            for (size_t d = 0; d < ndims; d++) {
                write_buf[d + i * ndims] = (((float) read_buf[d + i * ndims] - bias) * scale);
            }
        }
        writer.write((char *) write_buf, npts * ndims * sizeof(float));
    }
    static std::string input;
    static std::string output;
    static float bias;
    static float scale;
}
namespace tann {

    void set_int8_to_float_scale(turbo::App &app) {
        auto *sub = app.add_subcommand("int8_to_float_scale", "int8_to_float_scale");
        sub->add_option("-i, --input", detail::input, "input data path")->required();
        sub->add_option("-o, --output", detail::output, "path to output")->required();
        sub->add_option("-B, --bias", detail::bias, "bias")->required();
        sub->add_option("-s, --scale", detail::scale, "scale")->required();
        sub->callback([]() {
            int8_to_float_scale();
        });
    }
    void int8_to_float_scale() {
        std::ifstream reader(detail::input, std::ios::binary);
        uint32_t npts_u32;
        uint32_t ndims_u32;
        reader.read((char *) &npts_u32, sizeof(uint32_t));
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        size_t npts = npts_u32;
        size_t ndims = ndims_u32;
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;

        std::ofstream writer(detail::output, std::ios::binary);
        auto read_buf = new int8_t[blk_size * ndims];
        auto write_buf = new float[blk_size * ndims];
        float bias = detail::bias;
        float scale = (float) detail::scale;

        writer.write((char *) (&npts_u32), sizeof(uint32_t));
        writer.write((char *) (&ndims_u32), sizeof(uint32_t));

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            detail::block_convert(writer, write_buf, reader, read_buf, cblk_size, ndims, bias, scale);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        writer.close();
        reader.close();
    }
}