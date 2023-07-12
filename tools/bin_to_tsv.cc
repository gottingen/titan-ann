// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "tann/vamana/utils.h"
#include "tann_cli.h"

namespace detail {
    template<class T>
    void block_convert(std::ofstream &writer, std::ifstream &reader, T *read_buf, size_t npts, size_t ndims) {
        reader.read((char *) read_buf, npts * ndims * sizeof(float));

        for (size_t i = 0; i < npts; i++) {
            for (size_t d = 0; d < ndims; d++) {
                writer << read_buf[d + i * ndims];
                if (d < ndims - 1)
                    writer << "\t";
                else
                    writer << "\n";
            }
        }
    }
    static std::string input;
    static std::string output;
    static std::string data_type;
} // namespace detail

namespace tann {
    void set_bin_to_tsv(turbo::App &app) {
        auto *btt =app.add_subcommand("bin_to_tsv", "binary to tsv");
        btt->add_option("-i, --input",detail::input, "input file")->required();
        btt->add_option("-o, --output",detail::output, "output file")->required();
        btt->add_option("-t, --data_type",detail::data_type, "format file")->required();
        btt->callback([](){
            bin_to_tsv();
        });
    }
    void bin_to_tsv() {
        std::string type_string(detail::data_type);
        if ((type_string != std::string("float")) && (type_string != std::string("int8")) &&
            (type_string != std::string("uin8"))) {
            std::cerr << "Error: type not supported. Use float/int8/uint8" << std::endl;
        }

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

        std::ofstream writer(detail::output);
        char *read_buf = new char[blk_size * ndims * 4];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            if (type_string == std::string("float"))
                detail::block_convert<float>(writer, reader, (float *) read_buf, cblk_size, ndims);
            else if (type_string == std::string("int8"))
                detail::block_convert<int8_t>(writer, reader, (int8_t *) read_buf, cblk_size, ndims);
            else if (type_string == std::string("uint8"))
                detail::block_convert<uint8_t>(writer, reader, (uint8_t *) read_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;

        writer.close();
        reader.close();
    }
}  // namespace tann
