#pragma once

#include "metaral/world/chunk.hpp"
#include "metaral/world/world.hpp"

#include <deque>
#include <mutex>
#include <vector>

namespace metaral::world {

// Simple thread-safe queue for passing generated chunks back to the main thread.
class ChunkInbox {
public:
    void push(ChunkData&& chunk) {
        std::lock_guard lock(mutex_);
        queue_.push_back(std::move(chunk));
    }

    // Move all pending chunks into out (clearing the inbox).
    void pop_all(std::vector<ChunkData>& out) {
        std::lock_guard lock(mutex_);
        out.insert(out.end(),
                   std::make_move_iterator(queue_.begin()),
                   std::make_move_iterator(queue_.end()));
        queue_.clear();
    }

    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<ChunkData> queue_;
};

// Drain all chunks from inbox into the world by adopting them on the caller thread.
inline void adopt_all(World& world, ChunkInbox& inbox) {
    std::vector<ChunkData> chunks;
    inbox.pop_all(chunks);
    for (auto& chunk : chunks) {
        world.adopt_chunk(std::move(chunk));
    }
}

} // namespace metaral::world
