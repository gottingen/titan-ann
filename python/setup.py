# Copyright (c) titan-search Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
import os
import shutil
import platform

# make the tann python package dir
shutil.rmtree("tannpy", ignore_errors=True)
os.mkdir("tannpy")
shutil.copytree("tannpy", "tannpy")

long_description="""
tann is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
 up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning.
"""
setup(
    name='tannpy',
    version='0.3.0',
    description='A library for efficient similarity search and clustering of dense vectors',
    long_description=long_description,
    url='https://github.com/gottingen/titan-ann',
    author='Jeff.li',
    author_email='bbohuli2048@gmail.com',
    license='Apache 2',
    keywords='search nearest neighbors',

    install_requires=['numpy'],
    packages=['tannpy'],
    package_data={
        'tannpy': ['*.so', '*.pyd'],
    },
    zip_safe=False,
)