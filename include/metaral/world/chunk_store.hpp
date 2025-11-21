#pragma once

#include "metaral/world/chunk.hpp"
#include "metaral/core/coords.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace metaral::world {

// Abstract chunk persistence API.
class IChunkStore {
public:
    virtual ~IChunkStore() = default;
    virtual bool load(const core::ChunkCoord& coord, ChunkData& out) = 0;
    virtual bool save(const ChunkData& chunk) = 0;
};

// Very simple disk-backed store: one file per chunk, raw materials only.
class FileChunkStore final : public IChunkStore {
public:
    FileChunkStore(std::filesystem::path base_dir,
                   const core::CoordinateConfig& cfg,
                   std::string generator_tag = "default");

    bool load(const core::ChunkCoord& coord, ChunkData& out) override;
    bool save(const ChunkData& chunk) override;

private:
    std::filesystem::path chunk_path(const core::ChunkCoord& coord) const;

    std::filesystem::path root_;
    core::CoordinateConfig cfg_;
    std::string generator_tag_;
};

} // namespace metaral::world
