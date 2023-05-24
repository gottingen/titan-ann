#!/bin/bash
set -e

cmake -B buildpy_${PY_VER} -DCMAKE_BUILD_TYPE=Release \
                           -DCARBIN_BUILD_TEST=OFF \
                           -DCARBIN_BUILD_BENCHMARK=OFF \
                           -DCARBIN_BUILD_EXAMPLES=OFF \
                           -DCARBIN_USE_CXX11_ABI=ON \
                           -DBUILD_SHARED_LIBRARY=ON \
                           -DCMAKE_INSTALL_LIBDIR=lib \
                           -DBUILD_STATIC_LIBRARY=OFF \
                           -DPython_EXECUTABLE=$PYTHON python

cmake --build buildpy_${PY_VER}
cp -r python/tannpy buildpy_${PY_VER}/python
cp -r python/setup.py buildpy_${PY_VER}/python

cd buildpy_${PY_VER}/python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX