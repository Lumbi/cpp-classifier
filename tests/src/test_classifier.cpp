#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "classifier/classifier.h"

void test_empty_features() {
    classifier::Classifier<0> model;
    auto result = model.classify({});
    assert(result.prediction == classifier::Prediction::unknown);
    assert(result.confidence == 0.0);
    std::cout << "  PASS: test_empty_features\n";
}

void test_positive_classification() {
    classifier::Classifier<3> model;
    auto result = model.classify({0.8, 0.6, 0.9});
    assert(result.prediction == classifier::Prediction::positive);
    assert(result.confidence > 0.5);
    std::cout << "  PASS: test_positive_classification\n";
}

void test_negative_classification() {
    classifier::Classifier<3> model;
    auto result = model.classify({0.1, 0.2, 0.1});
    assert(result.prediction == classifier::Prediction::negative);
    assert(result.confidence > 0.5);
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
