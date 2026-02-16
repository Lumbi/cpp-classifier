add_executable(classifier_tests
    src/test_classifier.cpp
)

target_link_libraries(classifier_tests
    PRIVATE classifier
)

target_include_directories(classifier_tests
    PRIVATE ${CMAKE_SOURCE_DIR}/trainer/include
)

add_test(NAME classifier_tests COMMAND classifier_tests)
