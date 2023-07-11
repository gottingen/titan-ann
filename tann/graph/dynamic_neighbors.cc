#include "tann/graph/dynamic_neighbors.h"
#include "tann/utility/common.h"

using namespace tann::Helper;

DynamicNeighbors::DynamicNeighbors(const int *p_data, const int p_length)
        : c_data(p_data),
          c_length(p_length) {
}


DynamicNeighbors::~DynamicNeighbors() {
}


int
DynamicNeighbors::operator[](const int p_id) const {
    if (p_id < c_length && p_id >= 0) {
        return c_data[p_id];
    }

    return -1;
}


int
DynamicNeighbors::Size() const {
    return c_length;
}


DynamicNeighborsSet::DynamicNeighborsSet(const char *p_filePath) {
    auto fp = f_createIO();
    if (fp == nullptr || !fp->Initialize(p_filePath, std::ios::binary | std::ios::in)) {
        TLOG_ERROR("Failed open graph file: {}", p_filePath);
        throw std::runtime_error("Opening graph file failed");
    }

    if (fp->ReadBinary(sizeof(m_vectorCount), (char *) &m_vectorCount) != sizeof(m_vectorCount)) {
        TLOG_ERROR("Failed to read DynamicNeighborsSet!");
        throw std::runtime_error("reading DynamicNeighborsSet failed");
    }

    m_neighborOffset.reset(new int[m_vectorCount + 1]);
    m_neighborOffset[0] = 0;
    if (fp->ReadBinary(m_vectorCount * sizeof(int), (char *) (m_neighborOffset.get() + 1)) !=
        m_vectorCount * sizeof(int)) {
        TLOG_ERROR("Failed to read DynamicNeighborsSet!");
        throw std::runtime_error("reading DynamicNeighborsSet failed");
    }

    size_t graphSize = static_cast<size_t>(m_neighborOffset[m_vectorCount]);
    TLOG_ERROR("Vector count: {}, Graph size: {}", m_vectorCount, graphSize);

    m_data.reset(new int[graphSize]);
    auto readSize = fp->ReadBinary(graphSize * sizeof(int), (char *) (m_data.get()));
    if (readSize != graphSize * sizeof(int)) {
        TLOG_ERROR("Failed read graph: size not match, expected {}, actually {}",
                     static_cast<size_t>(graphSize * sizeof(int)),
                     static_cast<size_t>(readSize));
        throw std::runtime_error("Graph size doesn't match expected");
    }
}


DynamicNeighborsSet::~DynamicNeighborsSet() {
}


DynamicNeighbors DynamicNeighborsSet::operator[](const int p_id) const {
    if (p_id >= m_vectorCount) {
        return DynamicNeighbors(nullptr, 0);
    }

    return DynamicNeighbors(m_data.get() + static_cast<uint64_t>(m_neighborOffset[p_id]),
                            m_neighborOffset[p_id + 1] - m_neighborOffset[p_id]);
}
