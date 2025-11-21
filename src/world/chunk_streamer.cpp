#include "metaral/world/chunk_streamer.hpp"

#include "metaral/core/task/task_pool.hpp"
#include "metaral/world/chunk_inbox.hpp"

#include <cmath>
#include <iostream>

namespace metaral::world {

ChunkStreamer::ChunkStreamer(std::shared_ptr<IChunkProvider> provider,
                             World& world,
                             ChunkInbox& inbox,
                             StreamerConfig cfg)
    : provider_(std::move(provider))
    , world_(world)
    , inbox_(inbox)
    , cfg_(cfg)
{}

bool ChunkStreamer::is_loaded(const core::ChunkCoord& coord) const {
    return world_.find_chunk(coord) != nullptr;
}

bool ChunkStreamer::is_in_flight(const core::ChunkCoord& coord) {
    std::lock_guard lock(in_flight_mutex_);
    return in_flight_.contains(coord);
}

void ChunkStreamer::add_in_flight(const core::ChunkCoord& coord) {
    std::lock_guard lock(in_flight_mutex_);
    in_flight_.insert(coord);
}

void ChunkStreamer::remove_in_flight(const core::ChunkCoord& coord) {
    std::lock_guard lock(in_flight_mutex_);
    in_flight_.erase(coord);
}

void ChunkStreamer::enqueue_missing(const core::ChunkCoord& camera_chunk) {
    const int r = cfg_.load_radius;
    for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
            for (int dz = -r; dz <= r; ++dz) {
                core::ChunkCoord coord{
                    camera_chunk.x + dx,
                    camera_chunk.y + dy,
                    camera_chunk.z + dz,
                };

                if (is_loaded(coord)) {
                    continue;
                }
                if (is_in_flight(coord)) {
                    continue;
                }

                const int dist2 = dx*dx + dy*dy + dz*dz;
                queue_.push(Request{coord, dist2});
                add_in_flight(coord);
            }
        }
    }
}

void ChunkStreamer::unload_far(const core::ChunkCoord& camera_chunk) {
    const int keep = cfg_.keep_radius;
    if (keep <= 0) {
        return;
    }

    auto& chunks = world_.chunks();
    std::vector<core::ChunkCoord> to_remove;
    to_remove.reserve(chunks.size());
    for (const auto& [coord, _] : chunks) {
        const int dx = coord.x - camera_chunk.x;
        const int dy = coord.y - camera_chunk.y;
        const int dz = coord.z - camera_chunk.z;
        if (std::abs(dx) > keep || std::abs(dy) > keep || std::abs(dz) > keep) {
            to_remove.push_back(coord);
        }
    }
    for (const auto& coord : to_remove) {
        chunks.erase(coord);
    }
}

void ChunkStreamer::update(const core::ChunkCoord& camera_chunk) {
    enqueue_missing(camera_chunk);

    auto& pool = core::global_task_pool();
    pool.start(cfg_.worker_count);

    while (!queue_.empty()) {
        Request req = queue_.top();
        queue_.pop();

        pool.submit([this, coord=req.coord](std::stop_token) {
            ChunkData chunk = provider_->get(coord);
            inbox_.push(std::move(chunk));
            remove_in_flight(coord);
        });
    }

    unload_far(camera_chunk);
}

void ChunkStreamer::drain() {
    adopt_all(world_, inbox_);
}

} // namespace metaral::world
