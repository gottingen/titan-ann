#!/bin/bash
set -e

mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DCMAKE_BUILD_TYPE=Release \
        -DCARBIN_BUILD_TEST=OFF \
        -DCARBIN_BUILD_BENCHMARK=OFF \
        -DCARBIN_BUILD_EXAMPLES=OFF \
        -DCARBIN_USE_CXX11_ABI=ON \
        -DBUILD_SHARED_LIBRARY=ON \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_STATIC_LIBRARY=OFF

cmake --build .
cmake --build . --target install