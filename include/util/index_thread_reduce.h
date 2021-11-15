/**
* This file is part of Direct Sparse Localization (DSL).
*
* Copyright (C) 2021 Haoyang Ye <hy.ye at connect dot ust dot hk>,
* and Huaiyang Huang <hhuangat at connect dot use dot hk>,
* Robotics and Multiperception Lab (RAM-LAB <https://ram-lab.com>),
* The Hong Kong University of Science and Technology
*
* For more information please see <https://github.com/hyye/dsl>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
*
* DSL is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSL is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSL.  If not, see <http://www.gnu.org/licenses/>.
*/
//
// Created by hyye on 11/6/19.
//

#ifndef DSL_INDEX_THREAD_REDUCE_H_
#define DSL_INDEX_THREAD_REDUCE_H_

#include "util_common.h"
#include "num_type.h"
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace dsl {

template <typename Running>
class IndexThreadReduce {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline IndexThreadReduce() {
    next_index_ = 0;
    end_index_ = 0;
    step_size_ = 1;
    CallPerIndex = std::bind(&IndexThreadReduce::CallPerIndexDefault, this,
                             std::placeholders::_1, std::placeholders::_2,
                             std::placeholders::_3, std::placeholders::_4);

    running_ = true;
    todo_ready_ = false;
    proc_ready_ = false;
    for (int i = 0; i < NUM_THREADS; ++i) {
      is_done_[i] = false;
      got_one_[i] = true;
      worker_thread_[i] = std::thread(&IndexThreadReduce::WorkerLoop, this, i);
    }
  }

  inline ~IndexThreadReduce() {
    {
      std::unique_lock<std::mutex> loop_lock(loop_mutex_);
      running_ = false;
    }

    {
      std::unique_lock<std::mutex> lock(ex_mutex_);

      todo_ready_ = true;  // NOTE: necessary fake todo_ready_;
      cv_todo_.notify_all();
    }

    for (int i = 0; i < NUM_THREADS; ++i) {
      worker_thread_[i].join();
    }

    LOG(INFO) << "Destroyed ThreadReduce";
  }

  // start (0), end (size)
  inline void Reduce(
      std::function<void(int, int, Running &, int)> CallPerIndexIn, int first,
      int end, int step_size = 0) {
    memset(&stats_, 0, sizeof(Running));

    if (step_size == 0) {
      step_size = (end - first + NUM_THREADS - 1) / NUM_THREADS;
    }

    {
      std::unique_lock<std::mutex> lock(ex_mutex_);
      this->CallPerIndex = CallPerIndexIn;
      this->next_index_ = first;
      this->end_index_ = end;
      this->step_size_ = step_size;

      for (int i = 0; i < NUM_THREADS; ++i) {
        is_done_[i] = false;
        got_one_[i] = false;
      }
      proc_ready_ = false;

      todo_ready_ = true;
      cv_todo_.notify_all();

      cv_done_.wait(lock, [&] {
        proc_ready_ = true;
        for (int i = 0; i < NUM_THREADS; ++i) {
          proc_ready_ = proc_ready_ && is_done_[i];
        }
        return proc_ready_;
      });

      // DLOG(INFO) << "Reduce finished...";

      this->next_index_ = 0;
      this->end_index_ = 0;
      this->step_size_ = 1;
      this->CallPerIndex = 0;
      // this->CallPerIndex = std::bind(
      //     &IndexThreadReduce::CallPerIndexDefault, this,
      //     std::placeholders::_1, std::placeholders::_2,
      //     std::placeholders::_3, std::placeholders::_4);
    }
  }

  inline bool GetStats(Running &stats) {
    std::unique_lock<std::mutex> lock(ex_mutex_);
    stats = this->stats_;
    if (proc_ready_) {
      return true;
    }
    assert(proc_ready_);
    return false;
  }

  Running stats_;

private:
  std::thread worker_thread_[NUM_THREADS];
  bool is_done_[NUM_THREADS];
  bool got_one_[NUM_THREADS];

  std::mutex ex_mutex_;
  std::mutex loop_mutex_;
  std::condition_variable cv_todo_;
  std::condition_variable cv_done_;

  int next_index_ = 0;
  int end_index_ = 0;
  int step_size_ = 1;

  bool running_ = false;
  bool todo_ready_ = false;
  bool proc_ready_ = false;

  std::function<void(int, int, Running &, int)> CallPerIndex;

  void CallPerIndexDefault(int i, int j, Running &k, int tid) {
    std::stringstream ss;
    ss << "ERROR: " << __func__ << " should never be called!!!";
    LOG(ERROR) << ss.str();
    assert(false);
  }

  void WorkerLoop(int idx) {
    while ([&] {
      std::unique_lock<std::mutex> loop_lock(loop_mutex_);
      return running_;
    }()) {
      int todo = 0;
      int end_todo = 0;
      bool got_something = false;
      {  // prepare the # of tasks
        std::unique_lock<std::mutex> lock(ex_mutex_);

        if (next_index_ < end_index_) {
          todo = next_index_;
          end_todo = todo + step_size_;
          next_index_ += step_size_;
          got_something = true;

          assert(this->CallPerIndex != 0);
        }

        todo_ready_ = (next_index_ < end_index_);  // avoid deadlock
      }

      // std::this_thread::sleep_for(
      //     std::chrono::microseconds(1000));

      if (got_something) {

        assert(this->CallPerIndex != 0);

        Running s;
        memset(&s, 0, sizeof(Running));
        CallPerIndex(todo, std::min(end_todo, end_index_), s, idx);
        got_one_[idx] = true;

        std::unique_lock<std::mutex> lock(ex_mutex_);

        stats_ += s;
      } else {
        if (!got_one_[idx]) {
          Running s;
          memset(&s, 0, sizeof(Running));
          got_one_[idx] = true;

          std::unique_lock<std::mutex> lock(ex_mutex_);
          stats_ += s;
        }

        std::unique_lock<std::mutex> wait_lock(ex_mutex_);
        is_done_[idx] = true;

        cv_done_.notify_all();

        cv_todo_.wait(wait_lock, [&] {
          // WARNING: avoid deadlock, better solution???
          if (!todo_ready_ && !is_done_[idx]) {
            is_done_[idx] = true;
            cv_done_.notify_all();
          }
          return todo_ready_;
        });
      }
    }
  }
};

}

#endif // DSL_INDEX_THREAD_REDUCE_H_
