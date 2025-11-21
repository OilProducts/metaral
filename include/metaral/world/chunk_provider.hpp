#pragma once

#include "metaral/world/chunk.hpp"
#include "metaral/world/chunk_store.hpp"

#include <memory>

namespace metaral::world {

namespace terrain {
ChunkData generate_chunk(const core::ChunkCoord& chunk_coord,
                         const core::CoordinateConfig& cfg,
                         MaterialId solid_material,
                         MaterialId empty_material);
}

// Interface for something that can provide a chunk, possibly from cache or generation.
class IChunkProvider {
public:
    virtual ~IChunkProvider() = default;
    virtual ChunkData get(const core::ChunkCoord& coord) = 0;
};

// Wraps a store + generator: tries to load, otherwise generates and saves.
class CachedChunkProvider final : public IChunkProvider {
public:
    CachedChunkProvider(std::shared_ptr<IChunkStore> store,
                        core::CoordinateConfig cfg,
                        MaterialId solid_material = 1,
                        MaterialId empty_material = 0)
        : store_(std::move(store))
        , cfg_(cfg)
        , solid_material_(solid_material)
        , empty_material_(empty_material)
    {}

    ChunkData get(const core::ChunkCoord& coord) override {
        ChunkData chunk;
        if (store_ && store_->load(coord, chunk)) {
            return chunk;
        }
        chunk = terrain::generate_chunk(coord, cfg_, solid_material_, empty_material_);
        if (store_) {
            store_->save(chunk);
        }
        return chunk;
    }

private:
    std::shared_ptr<IChunkStore> store_;
    core::CoordinateConfig cfg_;
    MaterialId solid_material_;
    MaterialId empty_material_;
};

} // namespace metaral::world
