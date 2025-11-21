add_library(metaral_world
    world.cpp
    terrain.cpp
    edit.cpp
    chunk_store.cpp
    chunk_provider.cpp
)

set_target_properties(metaral_world PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(metaral_world
    PUBLIC
        metaral_core
        Threads::Threads
        stdc++fs
)

target_include_directories(metaral_world PUBLIC ${PROJECT_SOURCE_DIR}/include)
