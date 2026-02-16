#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

#include "classifier/model.h"

void test_empty_features() {
    classifier::Model<0> model;
    auto result = model.classify({});
    assert(result.get_prediction() == classifier::Prediction::unknown);
    assert(result.get_confidence() == 0.0f);
    std::cout << "  PASS: test_empty_features\n";
}

void test_positive_classification() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);
    model.set_weight(1, 1.0f);
    model.set_weight(2, 1.0f);
    auto result = model.classify({0.8f, 0.6f, 0.9f});
    assert(result.get_prediction() == classifier::Prediction::positive);
    assert(result.get_confidence() > 0.5f);
    std::cout << "  PASS: test_positive_classification\n";
}

void test_negative_classification() {
    classifier::Model<3> model;
    model.set_weight(0, -1.0f);
    model.set_weight(1, -1.0f);
    model.set_weight(2, -1.0f);
    auto result = model.classify({0.1f, 0.2f, 0.1f});
    assert(result.get_prediction() == classifier::Prediction::negative);
    assert(result.get_confidence() > 0.5f);
    std::cout << "  PASS: test_negative_classification\n";
}

void test_serialize_deserialize() {
    classifier::Model<3> model;
    model.set_weight(0, 1.5f);
    model.set_weight(1, -0.3f);
    model.set_weight(2, 0.7f);
    model.set_bias(0.25f);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    model.serialize(ss);

    classifier::Model<3> loaded;
    loaded.deserialize(ss);

    assert(loaded.weight(0) == 1.5f);
    assert(loaded.weight(1) == -0.3f);
    assert(loaded.weight(2) == 0.7f);
    assert(loaded.bias() == 0.25f);
    std::cout << "  PASS: test_serialize_deserialize\n";
}

void test_serialize_preserves_classification() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);
    model.set_weight(1, 1.0f);
    model.set_weight(2, 1.0f);
    model.set_bias(-1.5f);

    std::array<float, 3> features = {0.8f, 0.6f, 0.9f};
    auto original_result = model.classify(features);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    model.serialize(ss);

    classifier::Model<3> loaded;
    loaded.deserialize(ss);
    auto loaded_result = loaded.classify(features);

    assert(original_result.get_prediction() == loaded_result.get_prediction());
    assert(original_result.get_confidence() == loaded_result.get_confidence());
    std::cout << "  PASS: test_serialize_preserves_classification\n";
}

void test_deserialize_dimension_mismatch() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    model.serialize(ss);

    classifier::Model<2> wrong_size;
    bool caught = false;
    try {
        wrong_size.deserialize(ss);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  PASS: test_deserialize_dimension_mismatch\n";
}

void test_serialize_empty_model() {
    classifier::Model<0> model;

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    model.serialize(ss);

    classifier::Model<0> loaded;
    loaded.deserialize(ss);

    assert(loaded.bias() == 0.0f);
    std::cout << "  PASS: test_serialize_empty_model\n";
}

int main() {
    std::cout << "Running classifier tests...\n";

    test_empty_features();
    test_positive_classification();
    test_negative_classification();
    test_serialize_deserialize();
    test_serialize_preserves_classification();
    test_deserialize_dimension_mismatch();
    test_serialize_empty_model();

    std::cout << "All tests passed.\n";
    return 0;
}
