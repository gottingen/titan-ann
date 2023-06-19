# titan-ann
ann engine

# install

```shell
conda install tannpy
```


run test
```shell
cd test/python
python -m unittest
```
# test data source
* [sift](http://corpus-texmex.irisa.fr/)

```shell
mkdir build
 
cd build
cmake ..
make -j 4
```
* [ssd test](docs/ssd_index.md)
* [memory example](docs/in_memory_index.md)

[ann benchmark](https://github.com/erikbern/ann-benchmarks#evaluated)