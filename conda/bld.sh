#!/bin/bash
set -e

cmake -B build -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DCMAKE_BUILD_TYPE=Release \
        -DCARBIN_BUILD_TEST=OFF \
        -DCARBIN_BUILD_BENCHMARK=OFF \
        -DCARBIN_BUILD_EXAMPLES=OFF \
        -DCARBIN_USE_CXX11_ABI=ON \
        -DBUILD_SHARED_LIBRARY=ON \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_STATIC_LIBRARY=OFF \
        -DENABLE_PYTHON=OFF \
        .

cmake --build build
cmake --build build --target install
cmake --install build --prefix _libtann_stage/