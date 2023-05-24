#!/bin/bash
set -e

cmake -B buildpy_${PY_VER} -DCMAKE_BUILD_TYPE=Release \
                           -DPYTHON_EXECUTABLE=$PYTHON \
                           -Dtann_ROOT=_libtann_stage/ \
                           -DENABLE_PYTHON=OFF python

cmake --build buildpy_${PY_VER}
nkdir -p buildpy_${PY_VER}/output/tannpy
cp -r python/tannpy/* buildpy_${PY_VER}/output/tannpy/
cp -r python/setup.py buildpy_${PY_VER}/output/setup.py
cp buildpy_${PY_VER}/*.so buildpy_${PY_VER}/output/tannpy/
cd buildpy_${PY_VER}/output
find . -name "__pycache__" -type d -print
find . -name "__pycache__" -type d -print | xargs rm -rf
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX