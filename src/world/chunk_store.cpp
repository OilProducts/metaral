#include "metaral/world/chunk_store.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace metaral::world {

namespace {
constexpr std::uint32_t kChunkMagic   = 0x4d545231; // "MTR1"
constexpr std::uint32_t kChunkVersion = 1;

struct ChunkHeader {
    std::uint32_t magic = kChunkMagic;
    std::uint32_t version = kChunkVersion;
    std::int32_t  chunk_size = 0;
    std::uint32_t voxel_count = 0;
};

std::string make_gen_key(const core::CoordinateConfig& cfg,
                         const std::string& tag)
{
    std::ostringstream oss;
    oss << "cs" << cfg.chunk_size
        << "_vs" << std::setprecision(3) << cfg.voxel_size_m
        << "_pr" << std::setprecision(4) << cfg.planet_radius_m
        << "_" << tag;
    return oss.str();
}

} // namespace

FileChunkStore::FileChunkStore(std::filesystem::path base_dir,
                               const core::CoordinateConfig& cfg,
                               std::string generator_tag)
    : root_(std::move(base_dir))
    , cfg_(cfg)
    , generator_tag_(std::move(generator_tag))
{
    root_ /= make_gen_key(cfg_, generator_tag_);
    std::filesystem::create_directories(root_);
}

std::filesystem::path FileChunkStore::chunk_path(const core::ChunkCoord& coord) const {
    std::ostringstream name;
    name << coord.x << "_" << coord.y << "_" << coord.z << ".bin";
    return root_ / name.str();
}

bool FileChunkStore::save(const ChunkData& chunk) {
    const auto cs = static_cast<std::size_t>(cfg_.chunk_size);
    const auto expected = cs * cs * cs;
    if (chunk.voxels.size() != expected) {
        return false;
    }

    const std::filesystem::path path = chunk_path(chunk.coord);
    std::filesystem::create_directories(path.parent_path());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    ChunkHeader header;
    header.chunk_size = cfg_.chunk_size;
    header.voxel_count = static_cast<std::uint32_t>(expected);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const Voxel& v : chunk.voxels) {
        const std::uint16_t mat = static_cast<std::uint16_t>(v.material);
        out.write(reinterpret_cast<const char*>(&mat), sizeof(mat));
    }

    return static_cast<bool>(out);
}

bool FileChunkStore::load(const core::ChunkCoord& coord, ChunkData& out_chunk) {
    const std::filesystem::path path = chunk_path(coord);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    ChunkHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != kChunkMagic || header.version != kChunkVersion) {
        return false;
    }
    if (header.chunk_size != cfg_.chunk_size) {
        return false;
    }

    const std::size_t expected = static_cast<std::size_t>(header.voxel_count);
    out_chunk.coord = coord;
    out_chunk.voxels.resize(expected);

    for (std::size_t i = 0; i < expected; ++i) {
        std::uint16_t mat = 0;
        in.read(reinterpret_cast<char*>(&mat), sizeof(mat));
        if (!in) {
            return false;
        }
        out_chunk.voxels[i].material = static_cast<MaterialId>(mat);
    }

    return true;
}

} // namespace metaral::world
