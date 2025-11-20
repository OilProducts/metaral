add_library(metaral_render
    camera.cpp
    sdf_grid.cpp
    vulkan_renderer.cpp
    sdf_collision_field.cpp
    fluid_compute.cpp
)

set_target_properties(metaral_render PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(metaral_render
    PUBLIC
        metaral_core
        metaral_world
        metaral_sim
)

if(METARAL_ENABLE_VULKAN)
    set(METARAL_SHADER_DIR "${PROJECT_SOURCE_DIR}/shaders")

    set(METARAL_VERT_SHADER "${METARAL_SHADER_DIR}/fullscreen_triangle.vert")
    set(METARAL_FRAG_SHADER "${METARAL_SHADER_DIR}/analytic_sphere.frag")

    # Fluid compute shaders (SPH-inspired pipeline)
    set(METARAL_FLUID_SHADERS
        ${METARAL_SHADER_DIR}/fluid_common.glsl
        ${METARAL_SHADER_DIR}/fluid_external_forces.comp
        ${METARAL_SHADER_DIR}/fluid_spatial_hash.comp
        ${METARAL_SHADER_DIR}/fluid_reorder.comp
        ${METARAL_SHADER_DIR}/fluid_density.comp
        ${METARAL_SHADER_DIR}/fluid_pressure.comp
        ${METARAL_SHADER_DIR}/fluid_viscosity.comp
        ${METARAL_SHADER_DIR}/fluid_integrate.comp
    )

    set(METARAL_VERT_SPV "${CMAKE_BINARY_DIR}/shaders/fullscreen_triangle.vert.spv")
    set(METARAL_FRAG_SPV "${CMAKE_BINARY_DIR}/shaders/analytic_sphere.frag.spv")

    add_custom_command(
        OUTPUT ${METARAL_VERT_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND glslc -fshader-stage=vert ${METARAL_VERT_SHADER} -o ${METARAL_VERT_SPV}
        DEPENDS ${METARAL_VERT_SHADER}
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${METARAL_FRAG_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND glslc -fshader-stage=frag ${METARAL_FRAG_SHADER} -o ${METARAL_FRAG_SPV}
        DEPENDS ${METARAL_FRAG_SHADER}
        VERBATIM
    )

    # Compile compute shaders; glslc deduces stage from extension .comp
    set(METARAL_FLUID_SPV)
    foreach(SRC ${METARAL_FLUID_SHADERS})
        get_filename_component(NAME_WE ${SRC} NAME_WE)
        # Skip the shared include file; no SPIR-V output needed.
        if(NAME_WE STREQUAL "fluid_common")
            continue()
        endif()
        set(OUT_SPV "${CMAKE_BINARY_DIR}/shaders/${NAME_WE}.spv")
        add_custom_command(
            OUTPUT ${OUT_SPV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
            COMMAND glslc -I${METARAL_SHADER_DIR} ${SRC} -o ${OUT_SPV}
            DEPENDS ${SRC} ${METARAL_SHADER_DIR}/fluid_common.glsl
            VERBATIM
        )
        list(APPEND METARAL_FLUID_SPV ${OUT_SPV})
    endforeach()

    add_custom_target(metaral_shaders
        DEPENDS
            ${METARAL_VERT_SPV}
            ${METARAL_FRAG_SPV}
            ${METARAL_FLUID_SPV}
    )

    add_dependencies(metaral_render metaral_shaders)

    target_link_libraries(metaral_render PUBLIC Vulkan::Vulkan)
    target_compile_definitions(metaral_render PUBLIC METARAL_ENABLE_VULKAN=1)
    target_compile_definitions(metaral_render PUBLIC METARAL_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
endif()

target_include_directories(metaral_render PUBLIC ${PROJECT_SOURCE_DIR}/include)
