add_executable(classifier_tests
    src/test_classifier.cpp
)

target_link_libraries(classifier_tests
    PRIVATE classifier
)

add_test(NAME classifier_tests COMMAND classifier_tests)
