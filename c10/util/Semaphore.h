#pragma once

#include <version>

/*
  a simple semaphore interface.
*/

// note: __cpp_lib_semaphore will not be defined in some apple platforms
// even if >= C++20.
//
// libstdc++'s __atomic_semaphore has a lost-wakeup bug: _M_release skips
// the futex notify when the counter is already positive, but a concurrent
// _S_do_try_acquire can fail its CAS, see zero, and block — missing the
// wakeup. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98033
#if __has_include(<semaphore>) && defined(__cpp_lib_semaphore) && \
    __cpp_lib_semaphore >= 201907L && !defined(__GLIBCXX__)
#define C10_SEMAPHORE_USE_STL
#endif

#ifdef C10_SEMAPHORE_USE_STL
#include <semaphore>
#else
#include <condition_variable>
#include <mutex>
#endif

namespace c10 {

class Semaphore {
 public:
  Semaphore(int32_t initial_count = 0)
#ifdef C10_SEMAPHORE_USE_STL
      : impl_(initial_count)
#else
      : count_(initial_count)
#endif
  {}

  void release(int32_t n = 1) {
#ifdef C10_SEMAPHORE_USE_STL
    impl_.release(n);
#else
    {
      std::lock_guard<std::mutex> lock(mutex_);
      count_ += n;
    }
    cv_.notify_all();
#endif
  }

  void acquire() {
#ifdef C10_SEMAPHORE_USE_STL
    impl_.acquire();
#else
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return count_ > 0; });
    --count_;
#endif
  }

  bool tryAcquire() {
#ifdef C10_SEMAPHORE_USE_STL
    return impl_.try_acquire();
#else
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ == 0) {
      return false;
    }
    --count_;
    return true;
#endif
  }

 private:
#ifdef C10_SEMAPHORE_USE_STL
  std::counting_semaphore<> impl_;
#else
  std::mutex mutex_;
  std::condition_variable cv_;
  int32_t count_;
#endif
};
} // namespace c10

#undef C10_SEMAPHORE_USE_STL
