titan-ann
===========

ann engine

# dependencies

see [carbin_deps.txt](carbin_deps.txt)
* [turbo](https://github.com/gottingen/turbo)
* [bluebird](https://github.com/gottingen/bluebird)
* [doctest](https://github.com/doctest/doctest) for test only
* [carbin](https://github.com/gottingen/carbin) tools to manage package that deps, not necessary

# build

```shell
git clone https://github.com/gottingen/titan-ann.git
cd titan-ann
pip install carbin
carbin install 
mkdir build
cd build
cmake ..
make
make test
```

# examples
* [memory](examples/vamana/in_memory_index.md)
* [memory filter](examples/vamana/filtered_in_memory.md)
* [ssd](examples/vamana/ssd_index.md)
* [ssd filter](examples/vamana/filtered_ssd_index.md)
* [real time index](examples/vamana/dynamic_index.md)

# engine type
* flat
* hnsw
# metric types

* METRIC_L1,
* METRIC_L2,
* METRIC_IP,
* METRIC_HAMMING,
* METRIC_JACCARD,
* METRIC_COSINE,
* METRIC_ANGLE,
* METRIC_NORMALIZED_COSINE,
* METRIC_NORMALIZED_ANGLE,
* METRIC_NORMALIZED_L2,
* METRIC_POINCARE,
* METRIC_LORENTZ,

# benchmarks
* [sift](http://corpus-texmex.irisa.fr/)
* [ann bench](https://github.com/erikbern/ann-benchmarks#evaluated)