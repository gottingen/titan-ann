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
#ifndef TANN_CORE_WORKER_SPACE_H_
#define TANN_CORE_WORKER_SPACE_H_

#include "tann/core/search_context.h"
#include "tann/core/neighbor_queue.h"
#include "tann/common/concurrent_queue.h"
#include "turbo/times/stop_watcher.h"

namespace tann {

    struct WorkSpace {
        virtual ~WorkSpace() = default;
        SearchContext *search_context{nullptr};
        NeighborQueue best_l_nodes;
        turbo::Span<uint8_t> query_view;
        WriteOption write_option;
        bool        is_update{false};
        turbo::StopWatcher timer;

        void set_up(SearchContext *sc) {
            timer.reset();
            search_context = sc;
            make_aligned_query(sc->original_query, raw_query);
            query_view = to_span<uint8_t>(raw_query);
            best_l_nodes.clear();
            best_l_nodes.reserve(sc->k);
        }

        // for write
        void set_up(const WriteOption &option, turbo::Span<uint8_t> query) {
            search_context = nullptr;
            make_aligned_query(query, raw_query);
            query_view = to_span<uint8_t>(raw_query);
            write_option = option;
        }

        void clear() {
            search_context = nullptr;
            best_l_nodes.clear();
            is_update = false;
            clear_sub();
        }
    protected:
        virtual void clear_sub() = 0;
    protected:
        AlignedQuery<uint8_t> raw_query;
    };

    template<typename T>
    class WorkSpaceGuard {
    public:
        WorkSpaceGuard(ConcurrentQueue<T *> &query_scratch) : _space_pool(query_scratch) {
            _space = query_scratch.pop();
            while (_space == nullptr) {
                query_scratch.wait_for_push_notify();
                _space = query_scratch.pop();
            }
        }

        T *work_space() {
            return _space;
        }

        ~WorkSpaceGuard() {
            _space->clear();
            _space_pool.push(_space);
            _space_pool.push_notify_all();
        }

        void destroy() {
            while (!_space_pool.empty()) {
                auto scratch = _space_pool.pop();
                while (scratch == nullptr) {
                    _space_pool.wait_for_push_notify();
                    scratch = _space_pool.pop();
                }
                delete scratch;
            }
        }

    private:
        T *_space;
        ConcurrentQueue<T *> &_space_pool;

        WorkSpaceGuard(const WorkSpaceGuard<T> &) = delete;

        WorkSpaceGuard &operator=(const WorkSpaceGuard<T> &) = delete;
    };
}  // namespace tann

#endif  // TANN_CORE_WORKER_SPACE_H_
