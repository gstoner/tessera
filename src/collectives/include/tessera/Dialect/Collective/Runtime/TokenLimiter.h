#pragma once
#include <atomic>
#include <thread>
#include <condition_variable>
#include <mutex>

namespace tessera { namespace collective {

class TokenLimiter {
public:
  explicit TokenLimiter(int tokens) : max_(tokens) {}
  // Set the concurrency ceiling. Tracking max_ + inflight_ separately (rather
  // than a single counter that release() blindly increments) means shrinking
  // the limit below the in-flight count can't over-credit it: acquire() simply
  // blocks until inflight_ drops back below the new ceiling.
  void set(int tokens) {
    std::lock_guard<std::mutex> g(mu_); max_ = tokens; cv_.notify_all();
  }
  void acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{ return inflight_ < max_; });
    ++inflight_;
  }
  void release() {
    std::lock_guard<std::mutex> g(mu_);
    if (inflight_ > 0) --inflight_;
    cv_.notify_one();
  }
private:
  int max_;
  int inflight_ = 0;
  std::mutex mu_;
  std::condition_variable cv_;
};

}} // ns
