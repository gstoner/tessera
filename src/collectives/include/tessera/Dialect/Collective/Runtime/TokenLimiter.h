#pragma once
#include <atomic>
#include <thread>
#include <condition_variable>
#include <mutex>

namespace tessera { namespace collective {

class TokenLimiter {
public:
  explicit TokenLimiter(int tokens) : tokens_(tokens) {}
  void set(int tokens) {
    std::lock_guard<std::mutex> g(mu_); tokens_ = tokens; cv_.notify_all();
  }
  void acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{ return tokens_ > 0; });
    --tokens_;
  }
  void release() {
    std::lock_guard<std::mutex> g(mu_); ++tokens_; cv_.notify_one();
  }
private:
  int tokens_;
  std::mutex mu_;
  std::condition_variable cv_;
};

}} // ns
