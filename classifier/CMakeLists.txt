add_library(classifier SHARED
    src/classifier.cpp
)

target_include_directories(classifier
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

set_target_properties(classifier PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
