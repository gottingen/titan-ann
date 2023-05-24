#!/bin/bash
set -e

cmake -B buildpy_${PY_VER} -DCMAKE_BUILD_TYPE=Release \
                           -DPYTHON_EXECUTABLE=$PYTHON \
                           -Dtann_ROOT=_libtann_stage/
                           -DENABLE_PYTHON=OFF .

cmake --build buildpy_${PY_VER}
cp -r python/tannpy buildpy_${PY_VER}/output
cp -r python/setup.py buildpy_${PY_VER}/output

cd buildpy_${PY_VER}/output
cp buildpy_${PY_VER}/python/*.so tannpy
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX