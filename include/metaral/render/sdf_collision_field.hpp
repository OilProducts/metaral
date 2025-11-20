// Adapter that lets the fluid simulator query collisions against the render SDF
// without introducing a dependency from sim -> render.

#pragma once

#include "metaral/render/sdf_grid.hpp"
#include "metaral/sim/sph_fluid.hpp"

namespace metaral::render {

class SdfCollisionField : public sim::ICollisionField {
public:
    explicit SdfCollisionField(const SdfGrid* grid) : grid_(grid) {}

    float signed_distance(const core::PlanetPosition& p) const override;
    core::PlanetPosition surface_normal(const core::PlanetPosition& p) const override;

    void set_grid(const SdfGrid* grid) { grid_ = grid; }
    const SdfGrid* grid() const noexcept { return grid_; }

private:
    const SdfGrid* grid_ = nullptr;
};

} // namespace metaral::render
