add_executable(trainer
    src/main.cpp
)

target_link_libraries(trainer
    PRIVATE classifier
)
