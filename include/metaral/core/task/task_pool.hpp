#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace metaral::core {

// A minimal fixed-size thread pool suitable for long-lived background work.
// Tasks are move-only callables; tasks observe stop via std::stop_token.
class TaskPool {
public:
    using Task = std::function<void(std::stop_token)>;

    TaskPool() = default;
    explicit TaskPool(std::size_t worker_count) { start(worker_count); }
    TaskPool(const TaskPool&) = delete;
    TaskPool& operator=(const TaskPool&) = delete;
    TaskPool(TaskPool&&) = delete;
    TaskPool& operator=(TaskPool&&) = delete;

    ~TaskPool() { shutdown(); }

    void start(std::size_t worker_count);
    void submit(Task&& task);
    void shutdown(); // idempotent

    [[nodiscard]] std::size_t worker_count() const noexcept { return threads_.size(); }
    [[nodiscard]] std::size_t queued_tasks() const noexcept;

private:
    void worker_loop(std::stop_token stop);

    std::vector<std::jthread> threads_;
    std::queue<Task> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable_any cv_;
    bool stopping_ = false;
};

// Retrieve a process-wide pool initialized on first use. The worker count is
// chosen from hardware_concurrency() with a minimum of 1.
[[nodiscard]] inline TaskPool& global_task_pool() {
    static TaskPool pool{};
    return pool;
}

// ---- Inline implementation ----

inline void TaskPool::start(std::size_t worker_count) {
    if (!threads_.empty()) {
        return; // already started
    }
    if (worker_count == 0) {
        const std::size_t hw = std::thread::hardware_concurrency();
        worker_count = hw == 0 ? 1 : hw;
    }

    threads_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        threads_.emplace_back([this](std::stop_token st) { worker_loop(st); });
    }
}

inline void TaskPool::submit(Task&& task) {
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            return;
        }
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

inline void TaskPool::shutdown() {
    {
        std::scoped_lock lock(mutex_);
        if (stopping_) {
            return;
        }
        stopping_ = true;
    }
    cv_.notify_all();
    threads_.clear(); // joins all jthreads
}

inline std::size_t TaskPool::queued_tasks() const noexcept {
    std::scoped_lock lock(mutex_);
    return tasks_.size();
}

inline void TaskPool::worker_loop(std::stop_token stop) {
    while (!stop.stop_requested()) {
        Task task;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, stop, [this]() {
                return stopping_ || !tasks_.empty();
            });
            if (stopping_ && tasks_.empty()) {
                return;
            }
            if (stop.stop_requested()) {
                return;
            }
            if (tasks_.empty()) {
                continue;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }

        if (task) {
            task(stop);
        }
    }
}

} // namespace metaral::core
