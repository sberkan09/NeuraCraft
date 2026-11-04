#include "Parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace nc {

namespace {

thread_local bool tlInsidePool = false;

class ThreadPool {
public:
    explicit ThreadPool(unsigned threads) : size_(std::max(1u, threads)) {
        for (unsigned i = 1; i < size_; ++i) workers_.emplace_back([this] { workerLoop(); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (auto& t : workers_) t.join();
    }

    unsigned size() const { return size_; }

    // Executes task(c) for every c in [0, chunks).
    void run(std::size_t chunks, const std::function<void(std::size_t)>& task) {
        if (chunks == 0) return;
        if (chunks == 1 || workers_.empty() || tlInsidePool) {
            for (std::size_t c = 0; c < chunks; ++c) task(c);
            return;
        }

        std::lock_guard<std::mutex> callerLock(runMutex_);  // one job at a time
        {
            std::lock_guard<std::mutex> lk(mutex_);
            task_ = &task;
            chunks_ = chunks;
            next_.store(0);
            completed_.store(0);
            error_ = nullptr;
            ++generation_;
        }
        wake_.notify_all();

        tlInsidePool = true;
        work(&task, chunks);
        tlInsidePool = false;

        std::unique_lock<std::mutex> lk(mutex_);
        done_.wait(lk, [&] { return completed_.load() == chunks && active_ == 0; });
        task_ = nullptr;  // late-waking workers must not touch the finished job
        if (error_) {
            auto err = error_;
            error_ = nullptr;
            std::rethrow_exception(err);
        }
    }

private:
    void work(const std::function<void(std::size_t)>* task, std::size_t chunks) {
        std::size_t c;
        while ((c = next_.fetch_add(1)) < chunks) {
            try {
                (*task)(c);
            } catch (...) {
                std::lock_guard<std::mutex> lk(mutex_);
                if (!error_) error_ = std::current_exception();
            }
            if (completed_.fetch_add(1) + 1 == chunks) {
                std::lock_guard<std::mutex> lk(mutex_);
                done_.notify_all();
            }
        }
    }

    void workerLoop() {
        tlInsidePool = true;
        unsigned long long seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mutex_);
            wake_.wait(lk, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            const auto* task = task_;
            const std::size_t chunks = chunks_;
            if (!task) continue;
            ++active_;
            lk.unlock();

            work(task, chunks);

            lk.lock();
            if (--active_ == 0) done_.notify_all();
        }
    }

    unsigned size_;
    std::vector<std::thread> workers_;
    std::mutex runMutex_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(std::size_t)>* task_ = nullptr;
    std::size_t chunks_ = 0;
    std::atomic<std::size_t> next_{0};
    std::atomic<std::size_t> completed_{0};
    unsigned active_ = 0;
    unsigned long long generation_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

unsigned defaultThreads() {
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1u : hw;
}

std::mutex& poolMutex() {
    static std::mutex m;
    return m;
}

std::unique_ptr<ThreadPool>& poolStorage() {
    static std::unique_ptr<ThreadPool> pool;
    return pool;
}

ThreadPool& pool() {
    std::lock_guard<std::mutex> lk(poolMutex());
    auto& p = poolStorage();
    if (!p) p = std::make_unique<ThreadPool>(defaultThreads());
    return *p;
}

} // namespace

unsigned threadCount() { return pool().size(); }

void setThreadCount(unsigned threads) {
    if (threads == 0) threads = defaultThreads();
    std::lock_guard<std::mutex> lk(poolMutex());
    auto& p = poolStorage();
    if (p && p->size() == threads) return;
    p.reset();
    p = std::make_unique<ThreadPool>(threads);
}

void parallelFor(std::size_t begin, std::size_t end, std::size_t grain,
                 const std::function<void(std::size_t, std::size_t)>& fn) {
    if (end <= begin) return;
    const std::size_t n = end - begin;
    grain = std::max<std::size_t>(1, grain);
    ThreadPool& p = pool();
    const std::size_t maxChunks = static_cast<std::size_t>(p.size()) * 4;
    const std::size_t chunks = std::max<std::size_t>(1, std::min(maxChunks, n / grain));
    const std::size_t step = (n + chunks - 1) / chunks;
    p.run(chunks, [&](std::size_t c) {
        const std::size_t b = begin + c * step;
        const std::size_t e = std::min(end, b + step);
        if (b < e) fn(b, e);
    });
}

} // namespace nc
