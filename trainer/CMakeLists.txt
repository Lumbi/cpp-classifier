add_executable(trainer
    src/main.cpp
)

target_include_directories(trainer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(trainer
    PRIVATE classifier
)
