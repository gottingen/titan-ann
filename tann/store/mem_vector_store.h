// Copyright 2023 The titan-search Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//


#ifndef TANN_MEM_STORE_VECTOR_STORE_H_
#define TANN_MEM_STORE_VECTOR_STORE_H_

#include <vector>
#include <string_view>
#include <shared_mutex>
#include "tann/core/vector_space.h"
#include "tann/core/vector_store_option.h"
#include "tann/store/vector_batch.h"
#include "turbo/files/sequential_write_file.h"
#include "turbo/files/sequential_read_file.h"
#include "bluebird/bits/bitmap.h"
#include "turbo/container/flat_hash_map.h"
#include "turbo/concurrent/hash_lock.h"

namespace tann {

    class MemVectorStore {
    public:
        MemVectorStore() = default;

        ~MemVectorStore() = default;

        turbo::Status initialize(VectorSpace *vp, VectorStoreOption op);

        [[nodiscard]] const VectorSpace *get_vector_space() const;

        [[nodiscard]] const std::vector<VectorBatch> &vector_batch() const;

        [[nodiscard]] std::vector<VectorBatch> &vector_batch();

        [[nodiscard]] uint32_t get_batch_size() const;

        void set_vector(location_t i, turbo::Span<uint8_t> vector);

        [[nodiscard]] turbo::Span<uint8_t> get_vector(location_t i) const;

        void copy_vector(location_t, turbo::Span<uint8_t> &des) const;

        void enable_vacant();

        void disable_vacant();

        void reset_max_elements(uint32_t max_size);

        [[nodiscard]] double get_distance(location_t l1, location_t l2) const;

        [[nodiscard]] double get_distance(turbo::Span<uint8_t> vector, location_t l1) const;

        void get_distance(turbo::Span<uint8_t> vector, turbo::Span<std::size_t> ls,
                          turbo::Span<double> ds) const;

        turbo::ResultStatus<location_t> add_vector(label_type label, const turbo::Span<uint8_t> &vector);

        turbo::ResultStatus<location_t> prefer_add_vector(label_type label);

        turbo::ResultStatus<location_t> remove_vector(label_type label);

        [[nodiscard]] std::size_t size() const;

        [[nodiscard]] std::size_t deleted_size() const;

        [[nodiscard]] std::size_t current_index() const;

        [[nodiscard]] std::size_t capacity() const;

        [[nodiscard]] std::size_t available() const;

        turbo::ResultStatus<label_type> get_label(location_t loc);

        [[nodiscard]] bool exists_label(label_type label) const;

        [[nodiscard]] bool is_deleted(location_t loc) const;

        [[nodiscard]] turbo::ResultStatus<location_t> get_vacant(label_type label);

        turbo::Status load(std::string_view path);

        turbo::Status save(std::string_view path);

        turbo::Status load(turbo::SequentialReadFile *file);

        turbo::Status save(turbo::SequentialWriteFile *file);

        inline std::shared_mutex *get_label_op_mutex(label_type label) const {
            return _label_op_lock.get_lock(label);
        }
        std::shared_mutex* get_update_lock() const {
            return &_data_lock;
        }
    private:
        void move_vector(location_t from, location_t to);
        void reserve(std::size_t n);

        void shrink();

        void pop_back(std::size_t n = 1);

        void resize(std::size_t n);
    private:
        void expend();

        [[nodiscard]] std::size_t capacity_impl() const;

        void resize_impl(std::size_t n);

        void reserve_impl(std::size_t n);

        turbo::Span<uint8_t> get_vector_internal(location_t i) const;

    private:
        VectorSpace *_vs{nullptr};
        bool _is_available{false};
        VectorStoreOption _option;
        // guard by _data_lock
        std::atomic<std::size_t> _current_idx{0};
        // guard by _meta_lock
        std::atomic<std::size_t> _deleted_size{0};

        mutable std::shared_mutex _meta_lock;
        // guard by _meta_lock
        bluebird::Bitmap _deleted_map;

        // guard for labels option. this may multi
        // function span, so user should use LabelLockGuard/LabelSharedLockGuard
        // lock it outsize this scope
        turbo::HashLock<label_type> _label_op_lock;
        // guard by _meta_lock
        std::vector<label_type> _lid_to_label;
        mutable std::shared_mutex _label_map_lock;  // lock for _label_map_lock
        // guard by _label_map_lock
        turbo::flat_hash_map<label_type, location_t> _label_map;
        //
        mutable std::shared_mutex _data_lock;
        // guard by _data_lock
        std::vector<VectorBatch> _data;
    };

    class UpdateLockGuard {
    public:
        explicit UpdateLockGuard(MemVectorStore *mvs) : _mutex(mvs->get_update_lock()) {
            _mutex->lock();
        }

        ~UpdateLockGuard() {
            _mutex->unlock();
        }

    private:
        std::shared_mutex *_mutex{nullptr};
    };

    class UpdateSharedLockGuard {
    public:
        explicit UpdateSharedLockGuard(MemVectorStore *mvs) : _mutex(mvs->get_update_lock()) {
            _mutex->lock_shared();
        }

        ~UpdateSharedLockGuard() {
            _mutex->unlock_shared();
        }

    private:
        std::shared_mutex *_mutex{nullptr};
    };

    class LabelLockGuard {
    public:
        explicit LabelLockGuard(MemVectorStore *mvs, label_type label) : _mutex(mvs->get_label_op_mutex(label)) {
            _mutex->lock();
        }

        ~LabelLockGuard() {
            _mutex->unlock();
        }

    private:
        std::shared_mutex *_mutex{nullptr};
    };

    class LabelSharedLockGuard {
    public:
        explicit LabelSharedLockGuard(MemVectorStore *mvs, label_type label) : _mutex(mvs->get_label_op_mutex(label)) {
            _mutex->lock_shared();
        }

        ~LabelSharedLockGuard() {
            _mutex->unlock_shared();
        }

    private:
        std::shared_mutex *_mutex{nullptr};
    };
}  // namespace tann

#endif  // TANN_MEM_STORE_VECTOR_STORE_H_
