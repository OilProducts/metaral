add_library(metaral_sim
    sph_fluid.cpp
    density_grid.cpp
    fluid_system.cpp
)

set_target_properties(metaral_sim PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(metaral_sim
    PUBLIC
        metaral_core
        metaral_world
)

target_include_directories(metaral_sim
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
