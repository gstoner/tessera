#include "tile_scheduler.h"
using namespace tsr;

ThreadPool::ThreadPool(unsigned nthreads) {
  if (nthreads == 0) nthreads = 1;
  workers_.reserve(nthreads);
  for (unsigned i = 0; i < nthreads; ++i) {
    workers_.emplace_back([this]{ WorkerLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::Enqueue(std::function<void()> fn) {
  {
    std::unique_lock<std::mutex> lk(mu_);
    q_.push(std::move(fn));
  }
  cv_.notify_one();
}

void ThreadPool::WaitIdle() {
  std::unique_lock<std::mutex> lk(mu_);
  cv_idle_.wait(lk, [this]{ return q_.empty() && active_.load() == 0; });
}

void ThreadPool::WorkerLoop() {
  for (;;) {
    std::function<void()> fn;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this]{ return stop_.load() || !q_.empty(); });
      if (stop_.load() && q_.empty()) return;
      fn = std::move(q_.front());
      q_.pop();
      active_.fetch_add(1);
    }
    fn();
    {
      std::unique_lock<std::mutex> lk(mu_);
      active_.fetch_sub(1);
      if (q_.empty() && active_.load() == 0) cv_idle_.notify_all();
    }
  }
}
