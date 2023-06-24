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
#ifndef TANN_COMMON_TIMER_H_
#define TANN_COMMON_TIMER_H_

#include <iostream>
#include <iomanip>
#include <ctime>

namespace tann {
    class Timer {
    public:
        Timer() : time(0) {}

        void reset() {
            time = 0;
            ntime = 0;
        }

        void start() {
            struct timespec res;
            clock_getres(CLOCK_REALTIME, &res);
            reset();
            clock_gettime(CLOCK_REALTIME, &startTime);
        }

        void restart() {
            clock_gettime(CLOCK_REALTIME, &startTime);
        }

        void stop() {
            clock_gettime(CLOCK_REALTIME, &stopTime);
            sec = stopTime.tv_sec - startTime.tv_sec;
            nsec = stopTime.tv_nsec - startTime.tv_nsec;
            if (nsec < 0) {
                sec -= 1;
                nsec += 1000000000L;
            }
            time += (double) sec + (double) nsec / 1000000000.0;
            ntime += sec * 1000000000L + nsec;
        }

        friend std::ostream &operator<<(std::ostream &os, Timer &t) {
            auto time = t.time;
            if (time < 1.0) {
                time *= 1000.0;
                os << std::setprecision(6) << time << " (ms)";
                return os;
            }
            if (time < 60.0) {
                os << std::setprecision(6) << time << " (s)";
                return os;
            }
            time /= 60.0;
            if (time < 60.0) {
                os << std::setprecision(6) << time << " (m)";
                return os;
            }
            time /= 60.0;
            os << std::setprecision(6) << time << " (h)";
            return os;
        }

        struct timespec startTime;
        struct timespec stopTime;

        int64_t sec;
        int64_t nsec;
        int64_t ntime;    // nano second
        double time;    // second
    };
}
#endif  // TANN_COMMON_TIMER_H_
