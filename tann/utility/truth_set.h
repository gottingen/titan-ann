// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TANN_UTILITY_TRUTH_SET_H_
#define TANN_UTILITY_TRUTH_SET_H_

#include "tann/core/vector_index.h"
#include "tann/core/query_result_set.h"

namespace tann
{
    namespace COMMON
    {
        class TruthSet {
        public:
            static void LoadTruthTXT(std::shared_ptr<tann::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, int K, int& originalK, SizeType& p_iTruthNumber)
            {
                std::size_t lineBufferSize = 20;
                std::unique_ptr<char[]> currentLine(new char[lineBufferSize]);
                truth.clear();
                truth.resize(p_iTruthNumber);
                for (int i = 0; i < p_iTruthNumber; ++i)
                {
                    truth[i].clear();
                    if (ptr->ReadString(lineBufferSize, currentLine, '\n') == 0) {
                        TLOG_ERROR("Truth number({}) and query number({}) are not match!", i, p_iTruthNumber);
                        exit(1);
                    }
                    char* tmp = strtok(currentLine.get(), " ");
                    for (int j = 0; j < K; ++j)
                    {
                        if (tmp == nullptr) {
                            TLOG_ERROR("Truth number({}, {}) and query number({}) are not match!", i, j, p_iTruthNumber);
                            exit(1);
                        }
                        int vid = std::atoi(tmp);
                        if (vid >= 0) truth[i].insert(vid);
                        tmp = strtok(nullptr, " ");
                    }
                }
            }

            static void LoadTruthXVEC(std::shared_ptr<tann::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, int K, int& originalK, SizeType& p_iTruthNumber)
            {
                truth.clear();
                truth.resize(p_iTruthNumber);
                std::vector<int> vec(K);
                for (int i = 0; i < p_iTruthNumber; i++) {
                    if (ptr->ReadBinary(4, (char*)&originalK) != 4 || originalK < K) {
                        TLOG_ERROR("Error: Xvec file has No.{} vector whose dims are fewer than expected. Expected: {}, Fact: {}", i, K, originalK);
                        exit(1);
                    }
                    if (originalK > K) vec.resize(originalK);
                    if (ptr->ReadBinary(originalK * 4, (char*)vec.data()) != originalK * 4) {
                        TLOG_ERROR("Truth number({}) and query number({}) are not match!", i, p_iTruthNumber);
                        exit(1);
                    }
                    truth[i].insert(vec.begin(), vec.begin() + K);
                }
            }

            static void LoadTruthDefault(std::shared_ptr<tann::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, int K, int& originalK, SizeType& p_iTruthNumber) {
                if (ptr->TellP() == 0) {
                    int row;
                    if (ptr->ReadBinary(4, (char*)&row) != 4 || ptr->ReadBinary(4, (char*)&originalK) != 4) {
                        TLOG_ERROR("Fail to read truth file!");
                        exit(1);
                    }
                }
                truth.clear();
                truth.resize(p_iTruthNumber);
                std::vector<int> vec(originalK);
                for (int i = 0; i < p_iTruthNumber; i++)
                {
                    if (ptr->ReadBinary(4 * originalK, (char*)vec.data()) != 4 * originalK) {
                        TLOG_ERROR("Truth number({}) and query number({}) are not match!", i, p_iTruthNumber);
                        exit(1);
                    }
                    truth[i].insert(vec.begin(), vec.begin() + K);
                }
            }

            static void LoadTruth(std::shared_ptr<tann::Helper::DiskIO>& ptr, std::vector<std::set<SizeType>>& truth, SizeType& NumQuerys, int& originalK, int K, TruthFileType type)
            {
                if (type == TruthFileType::TXT)
                {
                    LoadTruthTXT(ptr, truth, K, originalK, NumQuerys);
                }
                else if (type == TruthFileType::XVEC)
                {
                    LoadTruthXVEC(ptr, truth, K, originalK, NumQuerys);
                }
                else if (type == TruthFileType::DEFAULT) {
                    LoadTruthDefault(ptr, truth, K, originalK, NumQuerys);
                }
                else
                {
                    TLOG_ERROR("TruthFileType Unsupported.");
                    exit(1);
                }
            }

            static void writeTruthFile(const std::string truthFile, SizeType queryNumber, const int K, std::vector<std::vector<tann::SizeType>>& truthset, std::vector<std::vector<float>>& distset, tann::TruthFileType TFT) {
                auto ptr = tann::f_createIO();
                if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::out | std::ios::binary)) {
                    TLOG_ERROR("Fail to create the file:{}", truthFile.c_str());
                    exit(1);
                }

                if (TFT == tann::TruthFileType::TXT)
                {
                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        for (int k = 0; k < K; k++)
                        {
                            if (ptr->WriteString((std::to_string(truthset[i][k]) + " ").c_str()) == 0) {
                                TLOG_ERROR("Fail to write the truth file!");
                                exit(1);
                            }
                        }
                        if (ptr->WriteString("") == 0) {
                            TLOG_ERROR("Fail to write the truth file!");
                            exit(1);
                        }
                    }
                }
                else if (TFT == tann::TruthFileType::XVEC)
                {
                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        if (ptr->WriteBinary(sizeof(K), (char*)&K) != sizeof(K) || ptr->WriteBinary(K * 4, (char*)(truthset[i].data())) != K * 4) {
                            TLOG_ERROR("Fail to write the truth file!");
                            exit(1);
                        }
                    }
                }
                else if (TFT == tann::TruthFileType::DEFAULT) {
                    ptr->WriteBinary(4, (char*)&queryNumber);
                    ptr->WriteBinary(4, (char*)&K);

                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        if (ptr->WriteBinary(K * 4, (char*)(truthset[i].data())) != K * 4) {
                            TLOG_ERROR("Fail to write the truth file!");
                            exit(1);
                        }
                    }
                    for (SizeType i = 0; i < queryNumber; i++)
                    {
                        if (ptr->WriteBinary(K * 4, (char*)(distset[i].data())) != K * 4) {
                            TLOG_ERROR("Fail to write the truth file!");
                            exit(1);
                        }
                    }
                }
                else {
                    TLOG_ERROR("Found unsupported file type for generating truth.");
                    exit(-1);
                }
            }

            template<typename T>
            static void GenerateTruth(std::shared_ptr<VectorSet> querySet, std::shared_ptr<VectorSet> vectorSet, const std::string truthFile,
                const tann::DistCalcMethod distMethod, const int K, const tann::TruthFileType p_truthFileType, const std::shared_ptr<IQuantizer>& quantizer);

            template <typename T>
            static float CalculateRecall(VectorIndex* index, std::vector<QueryResult>& results, const std::vector<std::set<SizeType>>& truth, int K, int truthK, std::shared_ptr<tann::VectorSet> querySet, std::shared_ptr<tann::VectorSet> vectorSet, SizeType NumQuerys, std::ofstream* log = nullptr, bool debug = false, float* MRR = nullptr)
            {
                float meanrecall = 0, minrecall = MaxDist, maxrecall = 0, stdrecall = 0, meanmrr = 0;
                std::vector<float> thisrecall(NumQuerys, 0);
                std::unique_ptr<bool[]> visited(new bool[K]);
                for (SizeType i = 0; i < NumQuerys; i++)
                {
                    int minpos = K;
                    memset(visited.get(), 0, K * sizeof(bool));
                    for (SizeType id : truth[i])
                    {
                        for (int j = 0; j < K; j++)
                        {
                            if (visited[j] || results[i].GetResult(j)->VID < 0) continue;

                            if (results[i].GetResult(j)->VID == id)
                            {
                                thisrecall[i] += 1;
                                visited[j] = true;
                                if (j < minpos) minpos = j;
                                break;
                            }
                            else if (vectorSet != nullptr) {
                                float dist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(results[i].GetResult(j)->VID), vectorSet->Dimension(), index->GetDistCalcMethod());
                                float truthDist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(id), vectorSet->Dimension(), index->GetDistCalcMethod());
                                if (index->GetDistCalcMethod() == tann::DistCalcMethod::Cosine && fabs(dist - truthDist) < Epsilon) {
                                    thisrecall[i] += 1;
                                    visited[j] = true;
                                    break;
                                }
                                else if (index->GetDistCalcMethod() == tann::DistCalcMethod::L2 && fabs(dist - truthDist) < Epsilon * (dist + Epsilon)) {
                                    thisrecall[i] += 1;
                                    visited[j] = true;
                                    break;
                                }
                            }
                        }
                    }
                    thisrecall[i] /= truth[i].size();
                    meanrecall += thisrecall[i];
                    if (thisrecall[i] < minrecall) minrecall = thisrecall[i];
                    if (thisrecall[i] > maxrecall) maxrecall = thisrecall[i];
                    if (minpos < K) meanmrr += 1.0f / (minpos + 1);

                    if (debug) {
                        std::string ll("recall:" + std::to_string(thisrecall[i]) + "\ngroundtruth:");
                        std::vector<NodeDistPair> truthvec;
                        for (SizeType id : truth[i]) {
                            float truthDist = 0.0;
                            if (vectorSet != nullptr) {
                                truthDist = COMMON::DistanceUtils::ComputeDistance((const T*)querySet->GetVector(i), (const T*)vectorSet->GetVector(id), querySet->Dimension(), index->GetDistCalcMethod());
                            }
                            truthvec.emplace_back(id, truthDist);
                        }
                        std::sort(truthvec.begin(), truthvec.end());
                        for (int j = 0; j < truthvec.size(); j++)
                            ll += std::to_string(truthvec[j].node) + "@" + std::to_string(truthvec[j].distance) + ",";
                        TLOG_INFO("{}", ll.c_str());
                        ll = "ann:";
                        for (int j = 0; j < K; j++)
                            ll += std::to_string(results[i].GetResult(j)->VID) + "@" + std::to_string(results[i].GetResult(j)->Dist) + ",";
                        TLOG_INFO("{}", ll.c_str());
                    }
                }
                meanrecall /= NumQuerys;
                for (SizeType i = 0; i < NumQuerys; i++)
                {
                    stdrecall += (thisrecall[i] - meanrecall) * (thisrecall[i] - meanrecall);
                }
                stdrecall = std::sqrt(stdrecall / NumQuerys);
                if (log) (*log) << meanrecall << " " << stdrecall << " " << minrecall << " " << maxrecall << std::endl;
                if (MRR) *MRR = meanmrr / NumQuerys;
                return meanrecall;
            }

            template <typename T>
            static float CalculateRecall(VectorIndex* index, T* query, int K) {
                COMMON::QueryResultSet<void> sampleANN(query, K);
                COMMON::QueryResultSet<void> sampleTruth(query, K);
                void* reconstructVector = nullptr;
                if (index->m_pQuantizer)
                {
                    reconstructVector = ALIGN_ALLOC(index->m_pQuantizer->ReconstructSize());
                    index->m_pQuantizer->ReconstructVector((const uint8_t*)query, reconstructVector);
                    sampleANN.SetTarget(reconstructVector, index->m_pQuantizer);
                    sampleTruth.SetTarget(reconstructVector, index->m_pQuantizer);
                }

                index->SearchIndex(sampleANN);
                for (SizeType y = 0; y < index->GetNumSamples(); y++)
                {
                    float dist = index->ComputeDistance(sampleTruth.GetQuantizedTarget(), index->GetSample(y));
                    sampleTruth.AddPoint(y, dist);
                }
                sampleTruth.SortResult();

                float recalls = 0;
                std::vector<bool> visited(K, false);
                for (SizeType y = 0; y < K; y++)
                {
                    for (SizeType z = 0; z < K; z++)
                    {
                        if (visited[z]) continue;

                        if (fabs(sampleANN.GetResult(z)->Dist - sampleTruth.GetResult(y)->Dist) < Epsilon)
                        {
                            recalls += 1;
                            visited[z] = true;
                            break;
                        }
                    }
                }
                if (reconstructVector)
                {
                    ALIGN_FREE(reconstructVector);
                }

                return recalls / K;
            }
        };
    }
}


#endif // TANN_UTILITY_TRUTH_SET_H_
