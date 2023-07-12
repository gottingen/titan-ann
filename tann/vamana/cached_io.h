// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "logger.h"
#include "tann/vamana/ann_exception.h"

namespace tann {
// sequential cached reads
    class cached_ifstream {
    public:
        cached_ifstream() {
        }

        cached_ifstream(const std::string &filename, uint64_t cacheSize) : _cache_size(cacheSize), _cur_off(0) {
            _reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            this->open(filename, _cache_size);
        }

        ~cached_ifstream() {
            delete[] _cache_buf;
            _reader.close();
        }

        void open(const std::string &filename, uint64_t cacheSize) {
            this->_cur_off = 0;

            try {
                _reader.open(filename, std::ios::binary | std::ios::ate);
                _fsize = _reader.tellg();
                _reader.seekg(0, std::ios::beg);
                assert(_reader.is_open());
                assert(cacheSize > 0);
                cacheSize = (std::min)(cacheSize, _fsize);
                this->_cache_size = cacheSize;
                _cache_buf = new char[cacheSize];
                _reader.read(_cache_buf, cacheSize);
                tann::cout << "Opened: " << filename.c_str() << ", size: " << _fsize << ", _cache_size: " << cacheSize
                           << std::endl;
            }
            catch (std::system_error &e) {
                throw tann::FileException(filename, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }

        size_t get_file_size() {
            return _fsize;
        }

        void read(char *read_buf, uint64_t n_bytes) {
            assert(_cache_buf != nullptr);
            assert(read_buf != nullptr);

            if (n_bytes <= (_cache_size - _cur_off)) {
                // case 1: cache contains all data
                memcpy(read_buf, _cache_buf + _cur_off, n_bytes);
                _cur_off += n_bytes;
            } else {
                // case 2: cache contains some data
                uint64_t cached_bytes = _cache_size - _cur_off;
                if (n_bytes - cached_bytes > _fsize - _reader.tellg()) {
                    std::stringstream stream;
                    stream << "Reading beyond end of file" << std::endl;
                    stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes << " _fsize: " << _fsize
                           << " current pos:" << _reader.tellg() << std::endl;
                    tann::cout << stream.str() << std::endl;
                    throw tann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
                }
                memcpy(read_buf, _cache_buf + _cur_off, cached_bytes);

                // go to disk and fetch more data
                _reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
                // reset cur off
                _cur_off = _cache_size;

                uint64_t size_left = _fsize - _reader.tellg();

                if (size_left >= _cache_size) {
                    _reader.read(_cache_buf, _cache_size);
                    _cur_off = 0;
                }
                // note that if size_left < _cache_size, then _cur_off = _cache_size,
                // so subsequent reads will all be directly from file
            }
        }

    private:
        // underlying ifstream
        std::ifstream _reader;
        // # bytes to cache in one shot read
        uint64_t _cache_size = 0;
        // underlying buf for cache
        char *_cache_buf = nullptr;
        // offset into _cache_buf for cur_pos
        uint64_t _cur_off = 0;
        // file size
        uint64_t _fsize = 0;
    };

// sequential cached writes
    class cached_ofstream {
    public:
        cached_ofstream(const std::string &filename, uint64_t _cache_size) : _cache_size(_cache_size), _cur_off(0) {
            _writer.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            try {
                _writer.open(filename, std::ios::binary);
                assert(_writer.is_open());
                assert(_cache_size > 0);
                _cache_buf = new char[_cache_size];
                tann::cout << "Opened: " << filename.c_str() << ", _cache_size: " << _cache_size << std::endl;
            }
            catch (std::system_error &e) {
                throw tann::FileException(filename, e, __FUNCSIG__, __FILE__, __LINE__);
            }
        }

        ~cached_ofstream() {
            this->close();
        }

        void close() {
            // dump any remaining data in memory
            if (_cur_off > 0) {
                this->flush_cache();
            }

            if (_cache_buf != nullptr) {
                delete[] _cache_buf;
                _cache_buf = nullptr;
            }

            if (_writer.is_open())
                _writer.close();
            tann::cout << "Finished writing " << _fsize << "B" << std::endl;
        }

        size_t get_file_size() {
            return _fsize;
        }

        // writes n_bytes from write_buf to the underlying ofstream/cache
        void write(char *write_buf, uint64_t n_bytes) {
            assert(_cache_buf != nullptr);
            if (n_bytes <= (_cache_size - _cur_off)) {
                // case 1: cache can take all data
                memcpy(_cache_buf + _cur_off, write_buf, n_bytes);
                _cur_off += n_bytes;
            } else {
                // case 2: cache cant take all data
                // go to disk and write existing cache data
                _writer.write(_cache_buf, _cur_off);
                _fsize += _cur_off;
                // write the new data to disk
                _writer.write(write_buf, n_bytes);
                _fsize += n_bytes;
                // memset all cache data and reset _cur_off
                memset(_cache_buf, 0, _cache_size);
                _cur_off = 0;
            }
        }

        void flush_cache() {
            assert(_cache_buf != nullptr);
            _writer.write(_cache_buf, _cur_off);
            _fsize += _cur_off;
            memset(_cache_buf, 0, _cache_size);
            _cur_off = 0;
        }

        void reset() {
            flush_cache();
            _writer.seekp(0);
        }

    private:
        // underlying ofstream
        std::ofstream _writer;
        // # bytes to cache for one shot write
        uint64_t _cache_size = 0;
        // underlying buf for cache
        char *_cache_buf = nullptr;
        // offset into _cache_buf for cur_pos
        uint64_t _cur_off = 0;

        // file size
        uint64_t _fsize = 0;
    };
}  // namespace tann
