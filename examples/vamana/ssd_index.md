**Usage for SSD-based indices**
===============================

To generate an SSD-friendly index, use the `tests/build_disk_index` program.
----------------------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported.
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as an integer. The next 4 bytes represent the dimension of data as an integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. `sizeof(T)` is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--index_path_prefix**: the index will span a few files, all beginning with the specified prefix path. For example, if you provide `~/index_test` as the prefix path, build  generates files such as `~/index_test_pq_pivots.bin, ~/index_test_pq_compressed.bin, ~/index_test_disk.index, ...`. There may be between 8 and 10 files generated with this prefix depending on how the index is constructed.
5. **-R (--max_degree)**  (default is 64): the degree of the graph index, typically between 60 and 150. Larger R will result in larger indices and longer indexing times, but better search quality.
6. **-L (--Lbuild)**  (default is 100): the size of search listduring index build. Typical values are between 75 to 200. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Use a value for L value that is at least the value of R unless you need to build indices really quickly and can somewhat compromise on quality.
7. **-B (--search_DRAM_budget)**: bound on the memory footprint of the index at search time in GB. Once built, the index will use up only the specified RAM limit, the rest will reside on disk. This will dictate how aggressively we compress the data vectors to store in memory. Larger will yield better performance at search time. For an n point index, to use b byte PQ compressed representation in memory, use `B = ((n * b) / 2^30  + (250000*(4*R + sizeof(T)*ndim)) / 2^30)`. The second term in the summation is to allow some buffer for caching about 250,000 nodes from the graph in memory while serving.  If you are not sure about this term, add 0.25GB to the first term.
8. **-M (--build_DRAM_budget)**: Limit on the memory allowed for building the index in GB. If you specify a value less than what is required to build the index in one pass, the index is  built using a divide and conquer approach so that  sub-graphs will fit in the RAM budget. The sub-graphs are overlayed to build the overall index. This approach can be upto 1.5 times slower than building the index in one shot. Allocate as much memory as your RAM allows.
9. **-T (--num_threads)** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
10. **--PQ_disk_bytes**  (default is 0): Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100% recall. If your vectors are too large to store in SSD, this parameter provides the option to compress the vectors using PQ for storing on SSD. This will trade off recall. You would also want this to be greater than the number of bytes used for the PQ compressed data stored in-memory
11. **--build_PQ_bytes** (default is 0): Set to a positive value less than the dimensionality of the data to enable faster index build with PQ based distance comparisons.
12. **--use_opq**: use the flag to use OPQ rather than PQ compression. OPQ is more space efficient for some high dimensional datasets, but also needs a bit more build time.

To search the SSD-index, use the `tests/search_disk_index` program.
-------------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. Use the same data type as in arg (1) above used in building the index.
2.  **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips). Use the same distance as in arg (2) above used in building the index.
3. **--index_path_prefix**: same as the prefix used in building the index (see arg 4 above).
4. **--num_nodes_to_cache** (default is 0): While serving the index, the entire graph is stored on SSD. For faster search performance, you can cache a few frequently accessed nodes in memory.
5. **-T (--num_threads)** (default is to get_omp_num_procs()): The number of threads used for searching. Threads run in parallel and one thread handles one query at a time. More threads will result in higher aggregate query throughput, but will also use more IOs/second across the system, which may lead to higher per-query latency. So find the balance depending on the maximum number of IOPs supported by the SSD.
6. **-W (--beamwidth)** (default is 2): The beamwidth to be used for search. This is the maximum number of IO requests each query will issue per iteration of search code. Larger beamwidth will result in fewer IO round-trips per query, but might result in slightly higher total number of IO requests to SSD per query. For the highest query throughput with a fixed SSD IOps rating, use `W=1`. For best latency, use `W=4,8` or higher complexity search. Specifying 0 will optimize the beamwidth depending on the number of threads performing search, but will involve some tuning overhead.
7. **--query_file**: The queries to be searched on in same binary file format as the data file in arg (2) above. The query file must be the same type as argument (1).
8. **--gt_file**: The ground truth file for the queries in arg (7) and data file used in index construction.  The binary file must start with *n*, the number of queries (4 bytes), followed by *d*, the number of ground truth elements per query (4 bytes), followed by `n*d` entries per query representing the d closest IDs per query in integer format,  followed by `n*d` entries representing the corresponding distances (float). Total file size is `8 + 4*n*d + 4*n*d` bytes. The groundtruth file, if not available, can be calculated using the program `tests/utils/compute_groundtruth`. Use "null" if you do not have this file and if you do not want to compute recall.
9. **K**: search for *K* neighbors and measure *K*-recall@*K*, meaning the intersection between the retrieved top-*K* nearest neighbors and ground truth *K* nearest neighbors.
10. **result_output_prefix**: Search results will be stored in files with specified prefix, in bin format.
11. **-L (--search_list)**: A list of search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be atleast the value of *K* in arg (9).


Example with BIGANN:
--------------------

This example demonstrates the use of the commands above on a 100K slice of the [BIGANN dataset](http://corpus-texmex.irisa.fr/) with 128 dimensional SIFT descriptors applied to images.

Download the base and query set and convert the data to binary format
pwd = tann/build
```bash
mkdir -p data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd ..
./tools/tann fvec_to_bin -t float -i data/sift/sift_learn.fvecs -o data/sift/sift_learn.fbin
./tools/tann fvec_to_bin -t float -i data/sift/sift_query.fvecs -o data/sift/sift_query.fbin
```

Now build and search the index and measure the recall using ground truth computed using brutefoce.
```bash
./bin/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/sift/sift_learn.fbin --query_file  data/sift/sift_query.fbin --gt_file data/sift/sift_query_learn_gt100 --K 100
# Using 0.003GB search memory budget for 100K vectors implies 32 byte PQ compression
./bin/build_disk_index --data_type float --dist_fn l2 --data_path data/sift/sift_learn.fbin --index_path_prefix data/sift/disk_index_sift_learn_R32_L50_A1.2 -R 32 -L50 -B 0.003 -M 1
 ./bin/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/disk_index_sift_learn_R32_L50_A1.2 --query_file data/sift/sift_query.fbin  --gt_file data/sift/sift_query_learn_gt100 -K 10 -L 10 20 30 40 50 100 --result_path data/sift/res --num_nodes_to_cache 10000
 ```

The search might be slower on machine with remote SSDs. The output lists the quer throughput, the mean and 99.9pc latency in microseconds and mean number of 4KB IOs to disk for each `L` parameter provided.

```
     L   Beamwidth             QPS    Mean Latency    99.9 Latency        Mean IOs         CPU (s)       Recall@10
======================================================================================================================
    10           2        10550.63         1038.11         4477.00            9.08          239.12           80.57
    20           2         6797.39         1665.23         4121.00           16.36          376.49           95.52
    30           2         4745.44         2418.18         5350.00           23.89          581.71           98.23
    40           2         3628.36         3184.25         5861.00           31.52          797.77           99.05
    50           2         2899.26         4003.10         9782.00           39.20         1041.01           99.38
   100           2         1481.24         7928.84        13138.00           78.21         2173.38           99.84
        L   Beamwidth             QPS    Mean Latency    99.9 Latency        Mean IOs         CPU (s)       Recall@10
======================================================================================================================
    10           2         9823.95         1201.47         3832.00            9.07          204.69           80.64
    20           2         6071.49         1962.46         4774.00           16.39          314.12           95.64
    30           2         4291.78         2777.17         5828.00           23.95          429.16           98.30
    40           2         3296.13         3624.07         6451.00           31.59          589.05           99.07
    50           2         2800.71         4254.43        10799.00           39.27          541.35           99.43
   100           2         1451.30         8244.98        12612.00           78.27         1018.31           99.84

```