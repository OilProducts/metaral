#pragma once

#include "metaral/core/coords.hpp"
#include "metaral/world/chunk_inbox.hpp"
#include "metaral/world/chunk_provider.hpp"
#include "metaral/world/world.hpp"

#include <queue>
#include <unordered_set>
#include <vector>

namespace metaral::world {

struct StreamerConfig {
    int load_radius = 6;   // chunks
    int keep_radius = 7;   // chunks (unload beyond this)
    std::size_t worker_count = 0; // 0 = auto
};

class ChunkStreamer {
public:
    ChunkStreamer(std::shared_ptr<IChunkProvider> provider,
                  World& world,
                  ChunkInbox& inbox,
                  StreamerConfig cfg = {});

    // Called each frame with the camera chunk coordinate and a heading distance
    // heuristic (squared distance works fine).
    void update(const core::ChunkCoord& camera_chunk);

    // Called on the main thread after worker completion to integrate chunks.
    void drain();

    void set_config(const StreamerConfig& cfg) { cfg_ = cfg; }

private:
    struct Request {
        core::ChunkCoord coord;
        int priority; // lower is higher priority (e.g., distance^2)
    };

    struct RequestCompare {
        bool operator()(const Request& a, const Request& b) const {
            return a.priority > b.priority; // min-heap
        }
    };

    bool is_in_flight(const core::ChunkCoord& coord);
    void add_in_flight(const core::ChunkCoord& coord);
    void remove_in_flight(const core::ChunkCoord& coord);

    void enqueue_missing(const core::ChunkCoord& camera_chunk);
    void unload_far(const core::ChunkCoord& camera_chunk);
    bool is_loaded(const core::ChunkCoord& coord) const;

    std::shared_ptr<IChunkProvider> provider_;
    World& world_;
    ChunkInbox& inbox_;
    StreamerConfig cfg_;

    std::priority_queue<Request, std::vector<Request>, RequestCompare> queue_;
    std::unordered_set<core::ChunkCoord, ChunkKeyHash, ChunkCoordEqual> in_flight_;
    std::mutex in_flight_mutex_;
};

} // namespace metaral::world
