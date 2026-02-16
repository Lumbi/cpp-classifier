#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

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

int main() {
    std::cout << "Running classifier tests...\n";

    test_empty_features();
    test_positive_classification();
    test_negative_classification();

    std::cout << "All tests passed.\n";
    return 0;
}
