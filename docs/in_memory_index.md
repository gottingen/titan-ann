**Usage for in-memory indices**
================================

To generate index, use the `tests/build_memory_index` program.
--------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported.
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--index_path_prefix**: The constructed index components will be saved to this path prefix.
5. **-R (--max_degree)** (default is 64): the degree of the graph index, typically between 32 and 150. Larger R will result in larger indices and longer indexing times, but might yield better search quality.
6. **-L (--Lbuild)** (default is 100): the size of search list we maintain during index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality.
7. **--alpha** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs.
8. **T (--num_threads)** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
9. **--build_PQ_bytes** (default is 0): Set to a positive value less than the dimensionality of the data to enable faster index build with PQ based distance comparisons. Defaults to using full precision vectors for distance comparisons.
   10.**--use_opq**: use the flag to use OPQ rather than PQ compression. OPQ is more space efficient for some high dimensional datasets, but also needs a bit more build time.


To search the generated index, use the `tests/search_memory_index` program:
---------------------------------------------------------------------------


The arguments are as follows:

1. **data_type**: The type of dataset you built the index on. float(32 bit), signed int8 and unsigned uint8 are supported. Use the same data type as in arg (1) above used in building the index.
2. **dist_fn**: There are two distance functions supported: l2 and mips. There is an additional *fast_l2* implementation that could provide faster results for small (about a million-sized) indices. Use the same distance as in arg (2) above used in building the index.
3. **memory_index_path**: index built above in argument (4).
4. **T**: The number of threads used for searching. Threads run in parallel and one thread handles one query at a time. More threads will result in higher aggregate query throughput, but may lead to higher per-query latency, especially if the DRAM bandwidth is a bottleneck. So find the balance depending on throughput and latency required for your application.
5. **query_bin**: The queries to be searched on in same binary file format as the data file (ii) above. The query file must be the same type as in argument (1).
6. **truthset.bin**: The ground truth file for the queries in arg (7) and data file used in index construction.  The binary file must start with *n*, the number of queries (4 bytes), followed by *d*, the number of ground truth elements per query (4 bytes), followed by `n*d` entries per query representing the d closest IDs per query in integer format,  followed by `n*d` entries representing the corresponding distances (float). Total file size is `8 + 4*n*d + 4*n*d` bytes. The groundtruth file, if not available, can be calculated using the program `tests/utils/compute_groundtruth`. Use "null" if you do not have this file and if you do not want to compute recall.
7. **K**: search for *K* neighbors and measure *K*-recall@*K*, meaning the intersection between the retrieved top-*K* nearest neighbors and ground truth *K* nearest neighbors.
8. **result_output_prefix**: search results will be stored in files, one per L value (see next arg), with specified prefix, in binary format.
9. **-L (--search_list)**: A list of search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be atleast the value of *K* in (7).


Example with BIGANN:
--------------------

This example demonstrates the use of the commands above on a 100K slice of the [BIGANN dataset](http://corpus-texmex.irisa.fr/) with 128 dimensional SIFT descriptors applied to images.

Download the base and query set and convert the data to binary format
```bash
mkdir -p data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd ..
./bin/fvecs_to_bin float data/sift/sift_learn.fvecs data/sift/sift_learn.fbin
./bin/fvecs_to_bin float data/sift/sift_query.fvecs data/sift/sift_query.fbin
```

Now build and search the index and measure the recall using ground truth computed using brutefoce.
```bash
./bin/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/sift/sift_learn.fbin --query_file  data/sift/sift_query.fbin --gt_file data/sift/sift_query_learn_gt100 --K 100
./bin/build_memory_index  --data_type float --dist_fn l2 --data_path data/sift/sift_learn.fbin --index_path_prefix data/sift/index_sift_learn_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2
 ./bin/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/index_sift_learn_R32_L50_A1.2 --query_file data/sift/sift_query.fbin  --gt_file data/sift/sift_query_learn_gt100 -K 10 -L 10 20 30 40 50 100 --result_path data/sift/res
 ```


The output of search lists the throughput (Queries/sec) as well as mean and 99.9 latency in microseconds for each `L` parameter provided. (We measured on a 6-core 12-thread)
 ```
  Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
=================================================================================
  10    58113.21            345.38              191.19        1829.31       95.00
  20    40523.40            517.11              295.62        2004.24       97.65
  30    30982.41            674.75              386.68        2094.17       98.58
  40    25105.23            822.76              476.21        2225.31       99.02
  50    21329.67            963.39              561.82        2306.73       99.31
 100    12612.72           1587.02              950.61        3113.84       99.81
 
 add -Ofast flag
 
   Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
=================================================================================
  10   203895.12            345.81               58.49        1335.75       95.11
  20   131318.45            517.10               91.02        3031.84       97.71
  30    97275.94            674.66              123.18        3336.60       98.61
  40    78014.63            822.80              153.61        3387.55       99.08
  50    64533.26            963.42              185.08        3507.63       99.32
 100    38374.44           1586.78              312.33        3647.76       99.81


 ```

