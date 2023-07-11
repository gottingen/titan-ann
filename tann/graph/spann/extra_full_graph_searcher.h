// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASEARCHER_H_
#define _SPTAG_SPANN_EXTRASEARCHER_H_

#include "tann/io/vector_set_reader.h"
#include "tann/io/async_file_reader.h"
#include "i_extra_searcher.h"
#include "tann/utility/truth_set.h"
#include "compressor.h"

#include <map>
#include <cmath>
#include <climits>
#include <future>
#include <numeric>

namespace tann
{
    namespace SPANN
    {
        extern std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO;

        struct Selection {
            std::string m_tmpfile;
            size_t m_totalsize;
            size_t m_start;
            size_t m_end;
            std::vector<Edge> m_selections;
            static EdgeCompare g_edgeComparer;

            Selection(size_t totalsize, std::string tmpdir) : m_tmpfile(tmpdir + FolderSep + "selection_tmp"), m_totalsize(totalsize), m_start(0), m_end(totalsize) { remove(m_tmpfile.c_str()); m_selections.resize(totalsize); }

            ErrorCode SaveBatch()
            {
                auto f_out = f_createIO();
                if (f_out == nullptr || !f_out->Initialize(m_tmpfile.c_str(), std::ios::out | std::ios::binary | (fileexists(m_tmpfile.c_str()) ? std::ios::in : 0))) {
                    TLOG_ERROR("Cannot open {} to save selection for batching!", m_tmpfile.c_str());
                    return ErrorCode::FailedOpenFile;
                }
                if (f_out->WriteBinary(sizeof(Edge) * (m_end - m_start), (const char*)m_selections.data(), sizeof(Edge) * m_start) != sizeof(Edge) * (m_end - m_start)) {
                    TLOG_ERROR("Cannot write to {}!", m_tmpfile.c_str());
                    return ErrorCode::DiskIOFail;
                }
                std::vector<Edge> batch_selection;
                m_selections.swap(batch_selection);
                m_start = m_end = 0;
                return ErrorCode::Success;
            }

            ErrorCode LoadBatch(size_t start, size_t end)
            {
                auto f_in = f_createIO();
                if (f_in == nullptr || !f_in->Initialize(m_tmpfile.c_str(), std::ios::in | std::ios::binary)) {
                    TLOG_ERROR("Cannot open {} to load selection batch!", m_tmpfile.c_str());
                    return ErrorCode::FailedOpenFile;
                }

                size_t readsize = end - start;
                m_selections.resize(readsize);
                if (f_in->ReadBinary(readsize * sizeof(Edge), (char*)m_selections.data(), start * sizeof(Edge)) != readsize * sizeof(Edge)) {
                    TLOG_ERROR("Cannot read from {}! start:{} size:{}", m_tmpfile.c_str(), start, readsize);
                    return ErrorCode::DiskIOFail;
                }
                m_start = start;
                m_end = end;
                return ErrorCode::Success;
            }

            size_t lower_bound(SizeType node)
            {
                auto ptr = std::lower_bound(m_selections.begin(), m_selections.end(), node, g_edgeComparer);
                return m_start + (ptr - m_selections.begin());
            }

            Edge& operator[](size_t offset)
            {
                if (offset < m_start || offset >= m_end) {
                    TLOG_ERROR("Error read offset in selections:{}", offset);
                }
                return m_selections[offset - m_start];
            }
        };

#define DecompressPosting(){\
        p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer(); \
        if (listInfo->listEleCount != 0) { \
            std::size_t sizePostingListFullData;\
            try {\
                sizePostingListFullData = m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);\
            }\
            catch (std::runtime_error& err) {\
                TLOG_ERROR("Decompress postingList {}  failed! {}, ", listInfo - m_listInfos.data(), err.what());\
                return;\
            }\
            if (sizePostingListFullData != listInfo->listEleCount * m_vectorInfoSize) {\
                TLOG_ERROR("PostingList {} decompressed size not match! {}, {}, ", listInfo - m_listInfos.data(), sizePostingListFullData, listInfo->listEleCount * m_vectorInfoSize);\
                return;\
            }\
        }\
}\

#define ProcessPosting() \
        for (int i = 0; i < listInfo->listEleCount; i++) { \
            uint64_t offsetVectorID, offsetVector;\
            (this->*m_parsePosting)(offsetVectorID, offsetVector, i, listInfo->listEleCount);\
            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID));\
            if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue; \
            (this->*m_parseEncoding)(p_index, listInfo, (ValueType*)(p_postingListFullData + offsetVector));\
            auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), p_postingListFullData + offsetVector); \
            queryResults.AddPoint(vectorID, distance2leaf); \
        } \

        template <typename ValueType>
        class ExtraFullGraphSearcher : public IExtraSearcher
        {
        public:
            ExtraFullGraphSearcher()
            {
                m_enableDeltaEncoding = false;
                m_enablePostingListRearrange = false;
                m_enableDataCompression = false;
                m_enableDictTraining = true;
            }

            virtual ~ExtraFullGraphSearcher()
            {
            }

            virtual bool LoadIndex(Options& p_opt) {
                m_extraFullGraphFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                std::string curFile = m_extraFullGraphFile;
                do {
                    auto curIndexFile = f_createAsyncIO();
                    if (curIndexFile == nullptr || !curIndexFile->Initialize(curFile.c_str(), std::ios::binary | std::ios::in, 
#ifndef _MSC_VER
#ifdef BATCH_READ
                        p_opt.m_searchInternalResultNum, 2, 2, p_opt.m_iSSDNumberOfThreads
#else
                        p_opt.m_searchInternalResultNum * p_opt.m_iSSDNumberOfThreads / p_opt.m_ioThreads + 1, 2, 2, p_opt.m_ioThreads
#endif
/*
#ifdef BATCH_READ
                        max(p_opt.m_searchInternalResultNum*m_vectorInfoSize, 1 << 12), 2, 2, p_opt.m_iSSDNumberOfThreads
#else
                        p_opt.m_searchInternalResultNum* p_opt.m_iSSDNumberOfThreads / p_opt.m_ioThreads + 1, 2, 2, p_opt.m_ioThreads
#endif
*/
#else
                        (p_opt.m_searchPostingPageLimit + 1) * PageSize, 2, 2, (std::uint16_t)p_opt.m_ioThreads
#endif
                    )) {
                        TLOG_ERROR("Cannot open file:{}!", curFile.c_str());
                        return false;
                    }

                    m_indexFiles.emplace_back(curIndexFile);
                    try {
                        m_totalListCount += LoadingHeadInfo(curFile, p_opt.m_searchPostingPageLimit, m_listInfos);
                    } 
                    catch (std::exception& e)
                    {
                        TLOG_ERROR("Error occurs when loading HeadInfo:{}", e.what());
                        return false;
                    }

                    curFile = m_extraFullGraphFile + "_" + std::to_string(m_indexFiles.size());
                } while (fileexists(curFile.c_str()));
                m_oneContext = (m_indexFiles.size() == 1);

                m_enableDeltaEncoding = p_opt.m_enableDeltaEncoding;
                m_enablePostingListRearrange = p_opt.m_enablePostingListRearrange;
                m_enableDataCompression = p_opt.m_enableDataCompression;
                m_enableDictTraining = p_opt.m_enableDictTraining;

                if (m_enablePostingListRearrange) m_parsePosting = &ExtraFullGraphSearcher<ValueType>::ParsePostingListRearrange;
                else m_parsePosting = &ExtraFullGraphSearcher<ValueType>::ParsePostingList;
                if (m_enableDeltaEncoding) m_parseEncoding = &ExtraFullGraphSearcher<ValueType>::ParseDeltaEncoding;
                else m_parseEncoding = &ExtraFullGraphSearcher<ValueType>::ParseEncoding;
                
                m_listPerFile = static_cast<int>((m_totalListCount + m_indexFiles.size() - 1) / m_indexFiles.size());

#ifndef _MSC_VER
                Helper::AIOTimeout.tv_nsec = p_opt.m_iotimeout * 1000;
#endif
                return true;
            }

            virtual void SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index,
                SearchStats* p_stats,
                std::set<int>* truth, std::map<int, std::set<int>>* found)
            {
                const uint32_t postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

                COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);
 
                int diskRead = 0;
                int diskIO = 0;
                int listElements = 0;

#if defined(ASYNC_READ) && !defined(BATCH_READ)
                int unprocessed = 0;
#endif

                for (uint32_t pi = 0; pi < postingListCount; ++pi)
                {
                    auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                    ListInfo* listInfo = &(m_listInfos[curPostingID]);
                    int fileid = m_oneContext? 0: curPostingID / m_listPerFile;

#ifndef BATCH_READ
                    Helper::DiskIO* indexFile = m_indexFiles[fileid].get();
#endif

                    diskRead += listInfo->listPageCount;
                    diskIO += 1;
                    listElements += listInfo->listEleCount;

                    size_t totalBytes = (static_cast<size_t>(listInfo->listPageCount) << PageSizeEx);
                    char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

#ifdef ASYNC_READ       
                    auto& request = p_exWorkSpace->m_diskRequests[pi];
                    request.m_offset = listInfo->listOffset;
                    request.m_readSize = totalBytes;
                    request.m_buffer = buffer;
                    request.m_status = (fileid << 16) | p_exWorkSpace->m_spaceID;
                    request.m_payload = (void*)listInfo; 
                    request.m_success = false;

#ifdef BATCH_READ // async batch read
                    request.m_callback = [&p_exWorkSpace, &queryResults, &p_index, &request, this](bool success)
                    {
                        char* buffer = request.m_buffer;
                        ListInfo* listInfo = (ListInfo*)(request.m_payload);

                        // decompress posting list
                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            DecompressPosting();
                        }

                        ProcessPosting();
                    };
#else // async read
                    request.m_callback = [&p_exWorkSpace, &request](bool success)
                    {
                        p_exWorkSpace->m_processIocp.push(&request);
                    };

                    ++unprocessed;
                    if (!(indexFile->ReadFileAsync(request)))
                    {
                        TLOG_ERROR("Failed to read file!");
                        unprocessed--;
                    }
#endif
#else // sync read
                    auto numRead = indexFile->ReadBinary(totalBytes, buffer, listInfo->listOffset);
                    if (numRead != totalBytes) {
                        TLOG_ERROR("File {} read bytes, expected: {}, acutal: {}.", m_extraFullGraphFile.c_str(), totalBytes, numRead);
                        throw std::runtime_error("File read mismatch");
                    }
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
#endif
                }

#ifdef ASYNC_READ
#ifdef BATCH_READ
                BatchReadFileAsync(m_indexFiles, (p_exWorkSpace->m_diskRequests).data(), postingListCount);
#else
                while (unprocessed > 0)
                {
                    Helper::AsyncReadRequest* request;
                    if (!(p_exWorkSpace->m_processIocp.pop(request))) break;

                    --unprocessed;
                    char* buffer = request->m_buffer;
                    ListInfo* listInfo = static_cast<ListInfo*>(request->m_payload);
                    // decompress posting list
                    char* p_postingListFullData = buffer + listInfo->pageOffset;
                    if (m_enableDataCompression)
                    {
                        DecompressPosting();
                    }

                    ProcessPosting();
                }
#endif
#endif
                if (truth) {
                    for (uint32_t pi = 0; pi < postingListCount; ++pi)
                    {
                        auto curPostingID = p_exWorkSpace->m_postingIDs[pi];

                        ListInfo* listInfo = &(m_listInfos[curPostingID]);
                        char* buffer = (char*)((p_exWorkSpace->m_pageBuffers[pi]).GetBuffer());

                        char* p_postingListFullData = buffer + listInfo->pageOffset;
                        if (m_enableDataCompression)
                        {
                            p_postingListFullData = (char*)p_exWorkSpace->m_decompressBuffer.GetBuffer();
                            if (listInfo->listEleCount != 0)
                            {
                                try {
                                    m_pCompressor->Decompress(buffer + listInfo->pageOffset, listInfo->listTotalBytes, p_postingListFullData, listInfo->listEleCount * m_vectorInfoSize, m_enableDictTraining);
                                }
                                catch (std::runtime_error& err) {
                                    TLOG_ERROR("Decompress postingList {}  failed! {}, ", curPostingID, err.what());
                                    continue;
                                }
                            }
                        }

                        for (size_t i = 0; i < listInfo->listEleCount; ++i) {
                            uint64_t offsetVectorID = m_enablePostingListRearrange ? (m_vectorInfoSize - sizeof(int)) * listInfo->listEleCount + sizeof(int) * i : m_vectorInfoSize * i; \
                            int vectorID = *(reinterpret_cast<int*>(p_postingListFullData + offsetVectorID)); \
                            if (truth && truth->count(vectorID)) (*found)[curPostingID].insert(vectorID);
                        }
                    }
                }

                if (p_stats) 
                {
                    p_stats->m_totalListElementsCount = listElements;
                    p_stats->m_diskIOCount = diskIO;
                    p_stats->m_diskAccessCount = diskRead;
                }
            }

            std::string GetPostingListFullData(
                int postingListId,
                size_t p_postingListSize,
                Selection &p_selections,
                std::shared_ptr<VectorSet> p_fullVectors,
                bool p_enableDeltaEncoding = false,
                bool p_enablePostingListRearrange = false,
                const ValueType *headVector = nullptr)
            {
                std::string postingListFullData("");
                std::string vectors("");
                std::string vectorIDs("");
                size_t selectIdx = p_selections.lower_bound(postingListId);
                // iterate over all the vectors in the posting list
                for (int i = 0; i < p_postingListSize; ++i)
                {
                    if (p_selections[selectIdx].node != postingListId)
                    {
                        TLOG_ERROR("Selection ID NOT MATCH! node:{} offset:{}", postingListId, selectIdx);
                        throw std::runtime_error("Selection ID mismatch");
                    }
                    std::string vectorID("");
                    std::string vector("");

                    int vid = p_selections[selectIdx++].tonode;
                    vectorID.append(reinterpret_cast<char *>(&vid), sizeof(int));

                    ValueType *p_vector = reinterpret_cast<ValueType *>(p_fullVectors->GetVector(vid));
                    if (p_enableDeltaEncoding)
                    {
                        DimensionType n = p_fullVectors->Dimension();
                        std::vector<ValueType> p_vector_delta(n);
                        for (auto j = 0; j < n; j++)
                        {
                            p_vector_delta[j] = p_vector[j] - headVector[j];
                        }
                        vector.append(reinterpret_cast<char *>(&p_vector_delta[0]), p_fullVectors->PerVectorDataSize());
                    }
                    else
                    {
                        vector.append(reinterpret_cast<char *>(p_vector), p_fullVectors->PerVectorDataSize());
                    }

                    if (p_enablePostingListRearrange)
                    {
                        vectorIDs += vectorID;
                        vectors += vector;
                    }
                    else
                    {
                        postingListFullData += (vectorID + vector);
                    }
                }
                if (p_enablePostingListRearrange)
                {
                    return vectors + vectorIDs;
                }
                return postingListFullData;
            }

            bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt) {
                std::string outputFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
                if (outputFile.empty())
                {
                    TLOG_ERROR("Output file can't be empty!");
                    return false;
                }

                int numThreads = p_opt.m_iSSDNumberOfThreads;
                int candidateNum = p_opt.m_internalResultNum;
                std::unordered_set<SizeType> headVectorIDS;
                if (p_opt.m_headIDFile.empty()) {
                    TLOG_ERROR("Not found VectorIDTranslate!");
                    return false;
                }

                {
                    auto ptr = tann::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize((p_opt.m_indexDirectory + FolderSep +  p_opt.m_headIDFile).c_str(), std::ios::binary | std::ios::in)) {
                        TLOG_ERROR("failed open VectorIDTranslate: {}", p_opt.m_headIDFile.c_str());
                        return false;
                    }

                    std::uint64_t vid;
                    while (ptr->ReadBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) == sizeof(vid))
                    {
                        headVectorIDS.insert(static_cast<SizeType>(vid));
                    }
                    TLOG_INFO("Loaded {} Vector IDs", static_cast<uint32_t>(headVectorIDS.size()));
                }

                SizeType fullCount = 0;
                size_t vectorInfoSize = 0;
                {
                    auto fullVectors = p_reader->GetVectorSet();
                    fullCount = fullVectors->Count();
                    vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int);
                }

                Selection selections(static_cast<size_t>(fullCount) * p_opt.m_replicaCount, p_opt.m_tmpdir);
                TLOG_INFO("Full vector count:{} Edge bytes:{} selection size:{}, capacity size:{}", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
                std::vector<std::atomic_int> replicaCount(fullCount);
                std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
                for (auto& pls : postingListSize) pls = 0;
                std::unordered_set<SizeType> emptySet;
                SizeType batchSize = (fullCount + p_opt.m_batches - 1) / p_opt.m_batches;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (p_opt.m_batches > 1)
                {
                    if (selections.SaveBatch() != ErrorCode::Success)
                    {
                        return false;
                    }
                }
                {
                    TLOG_INFO("Preparation done, start candidate searching.");
                    SizeType sampleSize = p_opt.m_samples;
                    std::vector<SizeType> samples(sampleSize, 0);
                    for (int i = 0; i < p_opt.m_batches; i++) {
                        SizeType start = i * batchSize;
                        SizeType end = min(start + batchSize, fullCount);
                        auto fullVectors = p_reader->GetVectorSet(start, end);
                        if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                        if (p_opt.m_batches > 1) {
                            if (selections.LoadBatch(static_cast<size_t>(start) * p_opt.m_replicaCount, static_cast<size_t>(end) * p_opt.m_replicaCount) != ErrorCode::Success)
                            {
                                return false;
                            }
                            emptySet.clear();
                            for (auto vid : headVectorIDS) {
                                if (vid >= start && vid < end) emptySet.insert(vid - start);
                            }
                        }
                        else {
                            emptySet = headVectorIDS;
                        }

                        int sampleNum = 0;
                        for (int j = start; j < end && sampleNum < sampleSize; j++)
                        {
                            if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
                        }

                        float acc = 0;
#pragma omp parallel for schedule(dynamic)
                        for (int j = 0; j < sampleNum; j++)
                        {
                            COMMON::Utils::atomic_float_add(&acc, COMMON::TruthSet::CalculateRecall(p_headIndex.get(), fullVectors->GetVector(samples[j]), candidateNum));
                        }
                        acc = acc / sampleNum;
                        TLOG_INFO("Batch {} vector({},{}) loaded with {} vectors ({}) HeadIndex acc @{}:%f.", i, start, end, fullVectors->Count(), selections.m_selections.size(), candidateNum, acc);

                        p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opt.m_replicaCount, numThreads, p_opt.m_gpuSSDNumTrees, p_opt.m_gpuSSDLeafSize, p_opt.m_rngFactor, p_opt.m_numGPUs);
                        TLOG_INFO("Batch {} finished!", i);

                        for (SizeType j = start; j < end; j++) {
                            replicaCount[j] = 0;
                            size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                            if (headVectorIDS.count(j) == 0) {
                                for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                                    ++postingListSize[selections[vecOffset + resNum].node];
                                    selections[vecOffset + resNum].tonode = j;
                                    ++replicaCount[j];
                                }
                            }
                        }

                        if (p_opt.m_batches > 1)
                        {
                            if (selections.SaveBatch() != ErrorCode::Success)
                            {
                                return false;
                            }
                        }
                    }
                }
                auto t2 = std::chrono::high_resolution_clock::now();
                TLOG_INFO("Searching replicas ended. Search Time: {} mins", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

                if (p_opt.m_batches > 1)
                {
                    if (selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount) != ErrorCode::Success)
                    {
                        return false;
                    }
                }

                // Sort results either in CPU or GPU
                VectorIndex::SortSelections(&selections.m_selections);

                auto t3 = std::chrono::high_resolution_clock::now();
                TLOG_INFO("Time to sort selections:{} sec.", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);

                int postingSizeLimit = INT_MAX;
                if (p_opt.m_postingPageLimit > 0)
                {
                    postingSizeLimit = static_cast<int>(p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
                }

                TLOG_INFO("Posting size limit: {}", postingSizeLimit);

                {
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0) continue;
                        ++replicaCountDist[replicaCount[i]];
                    }

                    TLOG_INFO("Before Posting Cut:");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        TLOG_INFO("Replica Count Dist: {}, {}", i, replicaCountDist[i]);
                    }
                }

#pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < postingListSize.size(); ++i)
                {
                    if (postingListSize[i] <= postingSizeLimit) continue;

                    std::size_t selectIdx = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i, Selection::g_edgeComparer) - selections.m_selections.begin();

                    for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                    {
                        int tonode = selections.m_selections[selectIdx + dropID].tonode;
                        --replicaCount[tonode];
                    }
                    postingListSize[i] = postingSizeLimit;
                }

                if (p_opt.m_outputEmptyReplicaID)
                {
                    std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                    auto ptr = tann::f_createIO();
                    if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                        TLOG_ERROR("Fail to create EmptyReplicaID.bin!");
                        return false;
                    }
                    for (int i = 0; i < replicaCount.size(); ++i)
                    {
                        if (headVectorIDS.count(i) > 0) continue;

                        ++replicaCountDist[replicaCount[i]];

                        if (replicaCount[i] < 2)
                        {
                            long long vid = i;
                            if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                                TLOG_ERROR("Failt to write EmptyReplicaID.bin!");
                                return false;
                            }
                        }
                    }

                    TLOG_INFO("After Posting Cut:");
                    for (int i = 0; i < replicaCountDist.size(); ++i)
                    {
                        TLOG_INFO("Replica Count Dist: {}, {}", i, replicaCountDist[i]);
                    }
                }

                auto t4 = std::chrono::high_resolution_clock::now();
                TLOG_INFO("Time to perform posting cut:{} sec.", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

                // number of posting lists per file
                size_t postingFileSize = (postingListSize.size() + p_opt.m_ssdIndexFileNum - 1) / p_opt.m_ssdIndexFileNum;
                std::vector<size_t> selectionsBatchOffset(p_opt.m_ssdIndexFileNum + 1, 0);
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    selectionsBatchOffset[i + 1] = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), (SizeType)curPostingListEnd, Selection::g_edgeComparer) - selections.m_selections.begin();
                }

                if (p_opt.m_ssdIndexFileNum > 1)
                {
                    if (selections.SaveBatch() != ErrorCode::Success)
                    {
                        return false;
                    }
                }

                auto fullVectors = p_reader->GetVectorSet();
                if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                // iterate over files
                for (int i = 0; i < p_opt.m_ssdIndexFileNum; i++) {
                    size_t curPostingListOffSet = i * postingFileSize;
                    size_t curPostingListEnd = min(postingListSize.size(), (i + 1) * postingFileSize);
                    // postingListSize: number of vectors in the posting list, type vector<int>
                    std::vector<int> curPostingListSizes(
                        postingListSize.begin() + curPostingListOffSet,
                        postingListSize.begin() + curPostingListEnd);

                    std::vector<size_t> curPostingListBytes(curPostingListSizes.size());
                    
                    if (p_opt.m_ssdIndexFileNum > 1)
                    {
                        if (selections.LoadBatch(selectionsBatchOffset[i], selectionsBatchOffset[i + 1]) != ErrorCode::Success)
                        {
                            return false;
                        }
                    }
                    // create compressor
                    if (p_opt.m_enableDataCompression && i == 0)
                    {
                        m_pCompressor = std::make_unique<Compressor>(p_opt.m_zstdCompressLevel, p_opt.m_dictBufferCapacity);
                        // train dict
                        if (p_opt.m_enableDictTraining) {
                            TLOG_INFO("Training dictionary...");
                            std::string samplesBuffer("");
                            std::vector<size_t> samplesSizes;
                            for (int j = 0; j < curPostingListSizes.size(); j++) {
                                if (curPostingListSizes[j] == 0) {
                                    continue;
                                }
                                ValueType* headVector = nullptr;
                                if (p_opt.m_enableDeltaEncoding)
                                {
                                    headVector = (ValueType*)p_headIndex->GetSample(j);
                                }
                                std::string postingListFullData = GetPostingListFullData(
                                    j, curPostingListSizes[j], selections, fullVectors, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);

                                samplesBuffer += postingListFullData;
                                samplesSizes.push_back(postingListFullData.size());
                                if (samplesBuffer.size() > p_opt.m_minDictTraingBufferSize) break;
                            }
                            TLOG_INFO("Using the first {} postingLists to train dictionary... ", samplesSizes.size());
                            std::size_t dictSize = m_pCompressor->TrainDict(samplesBuffer, &samplesSizes[0], (unsigned int)samplesSizes.size());
                            TLOG_INFO("Dictionary trained, dictionary size: {} ", dictSize);
                        }
                    }

                    if (p_opt.m_enableDataCompression) {
                        TLOG_INFO("Getting compressed size of each posting list...");
#pragma omp parallel for schedule(dynamic)
                        for (int j = 0; j < curPostingListSizes.size(); j++) 
                        {
                            SizeType postingListId = j + (SizeType)curPostingListOffSet;
                            // do not compress if no data
                            if (postingListSize[postingListId] == 0) {
                                curPostingListBytes[j] = 0;
                                continue;
                            }
                            ValueType* headVector = nullptr;
                            if (p_opt.m_enableDeltaEncoding)
                            {
                                headVector = (ValueType*)p_headIndex->GetSample(postingListId);
                            }
                            std::string postingListFullData = GetPostingListFullData(
                                postingListId, postingListSize[postingListId], selections, fullVectors, p_opt.m_enableDeltaEncoding, p_opt.m_enablePostingListRearrange, headVector);
                            size_t sizeToCompress = postingListSize[postingListId] * vectorInfoSize;
                            if (sizeToCompress != postingListFullData.size()) {
                                TLOG_ERROR("Size to compress NOT MATCH! PostingListFullData size: {} sizeToCompress: {} ", postingListFullData.size(), sizeToCompress);
                            }
                            curPostingListBytes[j] = m_pCompressor->GetCompressedSize(postingListFullData, p_opt.m_enableDictTraining);
                            if (postingListId % 10000 == 0 || curPostingListBytes[j] > static_cast<uint64_t>(p_opt.m_postingPageLimit) * PageSize) {
                                TLOG_INFO("Posting list {}/{}, compressed size: {}, compression ratio: {}", postingListId, postingListSize.size(), curPostingListBytes[j], curPostingListBytes[j] / float(sizeToCompress));
                            }
                        }
                        TLOG_INFO("Getted compressed size for all the {} posting lists in SSD Index file {}.", curPostingListBytes.size(), i);
                        TLOG_INFO("Mean compressed size: {} ", std::accumulate(curPostingListBytes.begin(), curPostingListBytes.end(), 0.0) / curPostingListBytes.size());
                        TLOG_INFO("Mean compression ratio: {} ", std::accumulate(curPostingListBytes.begin(), curPostingListBytes.end(), 0.0) / (std::accumulate(curPostingListSizes.begin(), curPostingListSizes.end(), 0.0) * vectorInfoSize));
                    }
                    else {
                        for (int j = 0; j < curPostingListSizes.size(); j++)
                        {
                            curPostingListBytes[j] = curPostingListSizes[j] * vectorInfoSize;
                        }
                    }

                    std::unique_ptr<int[]> postPageNum;
                    std::unique_ptr<std::uint16_t[]> postPageOffset;
                    std::vector<int> postingOrderInIndex;
                    SelectPostingOffset(curPostingListBytes, postPageNum, postPageOffset, postingOrderInIndex);

                    OutputSSDIndexFile((i == 0) ? outputFile : outputFile + "_" + std::to_string(i),
                        p_opt.m_enableDeltaEncoding,
                        p_opt.m_enablePostingListRearrange,
                        p_opt.m_enableDataCompression,
                        p_opt.m_enableDictTraining,
                        vectorInfoSize,
                        curPostingListSizes,
                        curPostingListBytes,
                        p_headIndex,
                        selections,
                        postPageNum,
                        postPageOffset,
                        postingOrderInIndex,
                        fullVectors,
                        curPostingListOffSet);
                }

                auto t5 = std::chrono::high_resolution_clock::now();
                auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
                TLOG_INFO("Total used time: {} minutes (about {} hours).", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
             
                return true;
            }

            virtual bool CheckValidPosting(SizeType postingID)
            {
                return m_listInfos[postingID].listEleCount != 0;
            }

        private:
            struct ListInfo
            {
                std::size_t listTotalBytes = 0;
                
                int listEleCount = 0;

                std::uint16_t listPageCount = 0;

                std::uint64_t listOffset = 0;

                std::uint16_t pageOffset = 0;
            };

            int LoadingHeadInfo(const std::string& p_file, int p_postingPageLimit, std::vector<ListInfo>& p_listInfos)
            {
                auto ptr = tann::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                    TLOG_ERROR("Failed to open file: {}", p_file.c_str());
                    throw std::runtime_error("Failed open file in LoadingHeadInfo");
                }
                m_pCompressor = std::make_unique<Compressor>(); // no need compress level to decompress

                int m_listCount;
                int m_totalDocumentCount;
                int m_listPageOffset;

                if (ptr->ReadBinary(sizeof(m_listCount), reinterpret_cast<char*>(&m_listCount)) != sizeof(m_listCount)) {
                    TLOG_ERROR("Failed to read head info file!");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(m_totalDocumentCount), reinterpret_cast<char*>(&m_totalDocumentCount)) != sizeof(m_totalDocumentCount)) {
                    TLOG_ERROR("Failed to read head info file!");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(m_iDataDimension), reinterpret_cast<char*>(&m_iDataDimension)) != sizeof(m_iDataDimension)) {
                    TLOG_ERROR("Failed to read head info file!");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }
                if (ptr->ReadBinary(sizeof(m_listPageOffset), reinterpret_cast<char*>(&m_listPageOffset)) != sizeof(m_listPageOffset)) {
                    TLOG_ERROR("Failed to read head info file!");
                    throw std::runtime_error("Failed read file in LoadingHeadInfo");
                }

                if (m_vectorInfoSize == 0) m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
                else if (m_vectorInfoSize != m_iDataDimension * sizeof(ValueType) + sizeof(int)) {
                    TLOG_ERROR("Failed to read head info file! DataDimension and ValueType are not match!");
                    throw std::runtime_error("DataDimension and ValueType don't match in LoadingHeadInfo");
                }

                size_t totalListCount = p_listInfos.size();
                p_listInfos.resize(totalListCount + m_listCount);

                size_t totalListElementCount = 0;

                std::map<int, int> pageCountDist;

                size_t biglistCount = 0;
                size_t biglistElementCount = 0;
                int pageNum;
                for (int i = 0; i < m_listCount; ++i)
                {
                    ListInfo* listInfo = &(p_listInfos[totalListCount + i]);

                    if (m_enableDataCompression)
                    {
                        if (ptr->ReadBinary(sizeof(listInfo->listTotalBytes), reinterpret_cast<char*>(&(listInfo->listTotalBytes))) != sizeof(listInfo->listTotalBytes)) {
                            TLOG_ERROR("Failed to read head info file!");
                            throw std::runtime_error("Failed read file in LoadingHeadInfo");
                        }
                    }
                    if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char*>(&(pageNum))) != sizeof(pageNum)) {
                        TLOG_ERROR("Failed to read head info file!");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->pageOffset), reinterpret_cast<char*>(&(listInfo->pageOffset))) != sizeof(listInfo->pageOffset)) {
                        TLOG_ERROR("Failed to read head info file!");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->listEleCount), reinterpret_cast<char*>(&(listInfo->listEleCount))) != sizeof(listInfo->listEleCount)) {
                        TLOG_ERROR("Failed to read head info file!");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    if (ptr->ReadBinary(sizeof(listInfo->listPageCount), reinterpret_cast<char*>(&(listInfo->listPageCount))) != sizeof(listInfo->listPageCount)) {
                        TLOG_ERROR("Failed to read head info file!");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    listInfo->listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);
                    if (!m_enableDataCompression)
                    {
                        listInfo->listTotalBytes = listInfo->listEleCount * m_vectorInfoSize;
                        listInfo->listEleCount = min(listInfo->listEleCount, (min(static_cast<int>(listInfo->listPageCount), p_postingPageLimit) << PageSizeEx) / m_vectorInfoSize);
                        listInfo->listPageCount = static_cast<std::uint16_t>(ceil((m_vectorInfoSize * listInfo->listEleCount + listInfo->pageOffset) * 1.0 / (1 << PageSizeEx)));
                    }
                    totalListElementCount += listInfo->listEleCount;
                    int pageCount = listInfo->listPageCount;

                    if (pageCount > 1)
                    {
                        ++biglistCount;
                        biglistElementCount += listInfo->listEleCount;
                    }

                    if (pageCountDist.count(pageCount) == 0)
                    {
                        pageCountDist[pageCount] = 1;
                    }
                    else
                    {
                        pageCountDist[pageCount] += 1;
                    }
                }

                if (m_enableDataCompression && m_enableDictTraining)
                {
                    size_t dictBufferSize;
                    if (ptr->ReadBinary(sizeof(size_t), reinterpret_cast<char*>(&dictBufferSize)) != sizeof(dictBufferSize)) {
                        TLOG_ERROR("Failed to read head info file!");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    char* dictBuffer = new char[dictBufferSize];
                    if (ptr->ReadBinary(dictBufferSize, dictBuffer) != dictBufferSize) {
                        TLOG_ERROR("Failed to read head info file!");
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    try {
                        m_pCompressor->SetDictBuffer(std::string(dictBuffer, dictBufferSize));
                    }
                    catch (std::runtime_error& err) {
                        TLOG_ERROR("Failed to read head info file: {} ", err.what());
                        throw std::runtime_error("Failed read file in LoadingHeadInfo");
                    }
                    delete[] dictBuffer;
                }

                TLOG_INFO(
                    "Finish reading header info, list count {}, total doc count {}, dimension {}, list page offset {}.",
                    m_listCount,
                    m_totalDocumentCount,
                    m_iDataDimension,
                    m_listPageOffset);

                TLOG_INFO(
                    "Big page (>4K): list count {}, total element count {}.",
                    biglistCount,
                    biglistElementCount);

                TLOG_INFO("Total Element Count: {}", totalListElementCount);

                for (auto& ele : pageCountDist)
                {
                    TLOG_INFO("Page Count Dist: {} {}", ele.first, ele.second);
                }

                return m_listCount;
            }

            inline void ParsePostingListRearrange(uint64_t& offsetVectorID, uint64_t& offsetVector, int i, int eleCount)
            {
                offsetVectorID = (m_vectorInfoSize - sizeof(int)) * eleCount + sizeof(int) * i;
                offsetVector = (m_vectorInfoSize - sizeof(int)) * i;
            }

            inline void ParsePostingList(uint64_t& offsetVectorID, uint64_t& offsetVector, int i, int eleCount)
            {
                offsetVectorID = m_vectorInfoSize * i;
                offsetVector = offsetVectorID + sizeof(int);
            }

            inline void ParseDeltaEncoding(std::shared_ptr<VectorIndex>& p_index, ListInfo* p_info, ValueType* vector)
            {
                ValueType* headVector = (ValueType*)p_index->GetSample((SizeType)(p_info - m_listInfos.data()));
                COMMON::SIMDUtils::ComputeSum(vector, headVector, m_iDataDimension);
            }

            inline void ParseEncoding(std::shared_ptr<VectorIndex>& p_index, ListInfo* p_info, ValueType* vector) { }

            void SelectPostingOffset(
                const std::vector<size_t>& p_postingListBytes,
                std::unique_ptr<int[]>& p_postPageNum,
                std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                std::vector<int>& p_postingOrderInIndex)
            {
                p_postPageNum.reset(new int[p_postingListBytes.size()]);
                p_postPageOffset.reset(new std::uint16_t[p_postingListBytes.size()]);

                struct PageModWithID
                {
                    int id;

                    std::uint16_t rest;
                };

                struct PageModeWithIDCmp
                {
                    bool operator()(const PageModWithID& a, const PageModWithID& b) const
                    {
                        return a.rest == b.rest ? a.id < b.id : a.rest > b.rest;
                    }
                };

                std::set<PageModWithID, PageModeWithIDCmp> listRestSize;

                p_postingOrderInIndex.clear();
                p_postingOrderInIndex.reserve(p_postingListBytes.size());

                PageModWithID listInfo;
                for (size_t i = 0; i < p_postingListBytes.size(); ++i)
                {
                    if (p_postingListBytes[i] == 0)
                    {
                        continue;
                    }

                    listInfo.id = static_cast<int>(i);
                    listInfo.rest = static_cast<std::uint16_t>(p_postingListBytes[i] % PageSize);

                    listRestSize.insert(listInfo);
                }

                listInfo.id = -1;

                int currPageNum = 0;
                std::uint16_t currOffset = 0;

                while (!listRestSize.empty())
                {
                    listInfo.rest = PageSize - currOffset;
                    auto iter = listRestSize.lower_bound(listInfo); // avoid page-crossing
                    if (iter == listRestSize.end() || (listInfo.rest != PageSize && iter->rest == 0))
                    {
                        ++currPageNum;
                        currOffset = 0;
                    }
                    else
                    {
                        p_postPageNum[iter->id] = currPageNum;
                        p_postPageOffset[iter->id] = currOffset;

                        p_postingOrderInIndex.push_back(iter->id);

                        currOffset += iter->rest;
                        if (currOffset > PageSize)
                        {
                            TLOG_ERROR("Crossing extra pages");
                            throw std::runtime_error("Read too many pages");
                        }

                        if (currOffset == PageSize)
                        {
                            ++currPageNum;
                            currOffset = 0;
                        }

                        currPageNum += static_cast<int>(p_postingListBytes[iter->id] / PageSize);

                        listRestSize.erase(iter);
                    }
                }

                TLOG_INFO("TotalPageNumbers: {}, IndexSize: {}", currPageNum, static_cast<uint64_t>(currPageNum) * PageSize + currOffset);
            }

            void OutputSSDIndexFile(const std::string& p_outputFile,
                bool p_enableDeltaEncoding,
                bool p_enablePostingListRearrange,
                bool p_enableDataCompression,
                bool p_enableDictTraining,
                size_t p_spacePerVector,
                const std::vector<int>& p_postingListSizes,
                const std::vector<size_t>& p_postingListBytes,
                std::shared_ptr<VectorIndex> p_headIndex,
                Selection& p_postingSelections,
                const std::unique_ptr<int[]>& p_postPageNum,
                const std::unique_ptr<std::uint16_t[]>& p_postPageOffset,
                const std::vector<int>& p_postingOrderInIndex,
                std::shared_ptr<VectorSet> p_fullVectors,
                size_t p_postingListOffset)
            {
                TLOG_INFO("Start output...");

                auto t1 = std::chrono::high_resolution_clock::now();

                auto ptr = tann::f_createIO();
                int retry = 3;
                // open file 
                while (retry > 0 && (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)))
                {
                    TLOG_ERROR("Failed open file {}, retrying...", p_outputFile.c_str());
                    retry--;
                }

                if (ptr == nullptr || !ptr->Initialize(p_outputFile.c_str(), std::ios::binary | std::ios::out)) {
                    TLOG_ERROR("Failed open file {}", p_outputFile.c_str());
                    throw std::runtime_error("Failed to open file for SSD index save");
                }
                // meta size of global info
                std::uint64_t listOffset = sizeof(int) * 4;
                // meta size of the posting lists
                listOffset += (sizeof(int) + sizeof(std::uint16_t) + sizeof(int) + sizeof(std::uint16_t)) * p_postingListSizes.size();
                // write listTotalBytes only when enabled data compression
                if (p_enableDataCompression)
                {
                    listOffset += sizeof(size_t) * p_postingListSizes.size();
                }

                // compression dict
                if (p_enableDataCompression && p_enableDictTraining)
                {
                    listOffset += sizeof(size_t);
                    listOffset += m_pCompressor->GetDictBuffer().size();
                }

                std::unique_ptr<char[]> paddingVals(new char[PageSize]);
                memset(paddingVals.get(), 0, sizeof(char) * PageSize);
                // paddingSize: bytes left in the last page
                std::uint64_t paddingSize = PageSize - (listOffset % PageSize);
                if (paddingSize == PageSize)
                {
                    paddingSize = 0;
                }
                else
                {
                    listOffset += paddingSize;
                }

                // Number of posting lists
                int i32Val = static_cast<int>(p_postingListSizes.size());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    TLOG_ERROR("Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }

                // Number of vectors
                i32Val = static_cast<int>(p_fullVectors->Count());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    TLOG_ERROR("Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }

                // Vector dimension
                i32Val = static_cast<int>(p_fullVectors->Dimension());
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    TLOG_ERROR("Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }

                // Page offset of list content section
                i32Val = static_cast<int>(listOffset / PageSize);
                if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                    TLOG_ERROR("Failed to write SSDIndex File!");
                    throw std::runtime_error("Failed to write SSDIndex File");
                }

                // Meta of each posting list
                for (int i = 0; i < p_postingListSizes.size(); ++i)
                {
                    size_t postingListByte = 0;
                    int pageNum = 0; // starting page number
                    std::uint16_t pageOffset = 0;
                    int listEleCount = 0;
                    std::uint16_t listPageCount = 0;

                    if (p_postingListSizes[i] > 0)
                    {
                        pageNum = p_postPageNum[i];
                        pageOffset = static_cast<std::uint16_t>(p_postPageOffset[i]);
                        listEleCount = static_cast<int>(p_postingListSizes[i]);
                        postingListByte = p_postingListBytes[i];
                        listPageCount = static_cast<std::uint16_t>(postingListByte / PageSize);
                        if (0 != (postingListByte % PageSize))
                        {
                            ++listPageCount;
                        }
                    }
                    // Total bytes of the posting list, write only when enabled data compression
                    if (p_enableDataCompression && ptr->WriteBinary(sizeof(postingListByte), reinterpret_cast<char*>(&postingListByte)) != sizeof(postingListByte)) {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page number of the posting list
                    if (ptr->WriteBinary(sizeof(pageNum), reinterpret_cast<char*>(&pageNum)) != sizeof(pageNum)) {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page offset
                    if (ptr->WriteBinary(sizeof(pageOffset), reinterpret_cast<char*>(&pageOffset)) != sizeof(pageOffset)) {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Number of vectors in the posting list
                    if (ptr->WriteBinary(sizeof(listEleCount), reinterpret_cast<char*>(&listEleCount)) != sizeof(listEleCount)) {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // Page count of the posting list
                    if (ptr->WriteBinary(sizeof(listPageCount), reinterpret_cast<char*>(&listPageCount)) != sizeof(listPageCount)) {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }
                // compression dict
                if (p_enableDataCompression && p_enableDictTraining)
                {
                    std::string dictBuffer = m_pCompressor->GetDictBuffer();
                    // dict size
                    size_t dictBufferSize = dictBuffer.size();
                    if (ptr->WriteBinary(sizeof(size_t), reinterpret_cast<char *>(&dictBufferSize)) != sizeof(dictBufferSize))
                    {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                    // dict
                    if (ptr->WriteBinary(dictBuffer.size(), const_cast<char *>(dictBuffer.data())) != dictBuffer.size())
                    {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                // Write padding vals
                if (paddingSize > 0)
                {
                    if (ptr->WriteBinary(paddingSize, reinterpret_cast<char*>(paddingVals.get())) != paddingSize) {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                if (static_cast<uint64_t>(ptr->TellP()) != listOffset)
                {
                    TLOG_INFO("List offset not match!");
                    throw std::runtime_error("List offset mismatch");
                }

                TLOG_INFO("SubIndex Size: {} bytes, {} MBytes", listOffset, listOffset >> 20);

                listOffset = 0;

                std::uint64_t paddedSize = 0;
                // iterate over all the posting lists
                for (auto id : p_postingOrderInIndex)
                {
                    std::uint64_t targetOffset = static_cast<uint64_t>(p_postPageNum[id]) * PageSize + p_postPageOffset[id];
                    if (targetOffset < listOffset)
                    {
                        TLOG_INFO("List offset not match, targetOffset < listOffset!");
                        throw std::runtime_error("List offset mismatch");
                    }
                    // write padding vals before the posting list
                    if (targetOffset > listOffset)
                    {
                        if (targetOffset - listOffset > PageSize)
                        {
                            TLOG_ERROR("Padding size greater than page size!");
                            throw std::runtime_error("Padding size mismatch with page size");
                        }

                        if (ptr->WriteBinary(targetOffset - listOffset, reinterpret_cast<char*>(paddingVals.get())) != targetOffset - listOffset) {
                            TLOG_ERROR("Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }

                        paddedSize += targetOffset - listOffset;

                        listOffset = targetOffset;
                    }

                    if (p_postingListSizes[id] == 0)
                    {
                        continue;
                    }
                    int postingListId = id + (int)p_postingListOffset;
                    // get posting list full content and write it at once
                    ValueType *headVector = nullptr;
                    if (p_enableDeltaEncoding)
                    {
                        headVector = (ValueType *)p_headIndex->GetSample(postingListId);
                    }
                    std::string postingListFullData = GetPostingListFullData(
                        postingListId, p_postingListSizes[id], p_postingSelections, p_fullVectors, p_enableDeltaEncoding, p_enablePostingListRearrange, headVector);
                    size_t postingListFullSize = p_postingListSizes[id] * p_spacePerVector;
                    if (postingListFullSize != postingListFullData.size())
                    {
                        TLOG_ERROR("posting list full data size NOT MATCH! postingListFullData.size(): {} postingListFullSize: {} ", postingListFullData.size(), postingListFullSize);
                        throw std::runtime_error("Posting list full size mismatch");
                    }
                    if (p_enableDataCompression)
                    {
                        std::string compressedData = m_pCompressor->Compress(postingListFullData, p_enableDictTraining);
                        size_t compressedSize = compressedData.size();
                        if (compressedSize != p_postingListBytes[id])
                        {
                            TLOG_ERROR("Compressed size NOT MATCH! compressed size:{}, pre-calculated compressed size:{}", compressedSize, p_postingListBytes[id]);
                            throw std::runtime_error("Compression size mismatch");
                        }
                        if (ptr->WriteBinary(compressedSize, compressedData.data()) != compressedSize)
                        {
                            TLOG_ERROR("Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }
                        listOffset += compressedSize;
                    }
                    else
                    {
                        if (ptr->WriteBinary(postingListFullSize, postingListFullData.data()) != postingListFullSize)
                        {
                            TLOG_ERROR("Failed to write SSDIndex File!");
                            throw std::runtime_error("Failed to write SSDIndex File");
                        }
                        listOffset += postingListFullSize;
                    }
                }

                paddingSize = PageSize - (listOffset % PageSize);
                if (paddingSize == PageSize)
                {
                    paddingSize = 0;
                }
                else
                {
                    listOffset += paddingSize;
                    paddedSize += paddingSize;
                }

                if (paddingSize > 0)
                {
                    if (ptr->WriteBinary(paddingSize, reinterpret_cast<char *>(paddingVals.get())) != paddingSize)
                    {
                        TLOG_ERROR("Failed to write SSDIndex File!");
                        throw std::runtime_error("Failed to write SSDIndex File");
                    }
                }

                TLOG_INFO("Padded Size: {}, final total size: {}.", paddedSize, listOffset);

                TLOG_INFO("Output done...");
                auto t2 = std::chrono::high_resolution_clock::now();
                TLOG_INFO("Time to write results:{} sec.", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000);
            }

        private:
            
            std::string m_extraFullGraphFile;

            std::vector<ListInfo> m_listInfos;
            bool m_oneContext;

            std::vector<std::shared_ptr<Helper::DiskIO>> m_indexFiles;
            std::unique_ptr<Compressor> m_pCompressor;
            bool m_enableDeltaEncoding;
            bool m_enablePostingListRearrange;
            bool m_enableDataCompression;
            bool m_enableDictTraining;

            void (ExtraFullGraphSearcher<ValueType>::*m_parsePosting)(uint64_t&, uint64_t&, int, int);
            void (ExtraFullGraphSearcher<ValueType>::*m_parseEncoding)(std::shared_ptr<VectorIndex>&, ListInfo*, ValueType*);

            int m_vectorInfoSize = 0;
            int m_iDataDimension = 0;

            int m_totalListCount = 0;

            int m_listPerFile = 0;
        };
    } // namespace SPANN
} // namespace tann

#endif // _SPTAG_SPANN_EXTRASEARCHER_H_
