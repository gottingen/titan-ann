#!/bin/bash
set -e

cmake -B buildpy_${PY_VER}  python

cmake --build buildpy_${PY_VER}
cp -r python/tannpy buildpy_${PY_VER}/python
cp -r python/setup.py buildpy_${PY_VER}/python

cd buildpy_${PY_VER}/python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt --prefix=$PREFIX