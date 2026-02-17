add_executable(test_classifier
    src/test_classifier.cpp
)

target_link_libraries(test_classifier
    PRIVATE classifier
)

add_test(NAME test_classifier COMMAND test_classifier)
