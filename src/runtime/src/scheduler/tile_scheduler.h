#pragma once
#include <functional>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace tsr {

class ThreadPool {
 public:
  explicit ThreadPool(unsigned nthreads = std::thread::hardware_concurrency());
  ~ThreadPool();

  void Enqueue(std::function<void()> fn);
  void WaitIdle();

 private:
  void WorkerLoop();
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> q_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable cv_idle_;
  std::atomic<bool> stop_{false};
  std::atomic<int> active_{0};
};

} // namespace tsr
