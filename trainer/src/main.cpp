#include <array>
#include <iostream>
#include <string_view>

#include "classifier/classifier.h"

constexpr std::string_view to_string(classifier::Prediction prediction) {
    switch (prediction) {
    case classifier::Prediction::positive: return "positive";
    case classifier::Prediction::negative: return "negative";
    case classifier::Prediction::unknown:  return "unknown";
    }
    return "unknown";
}

int main(int argc, char* argv[]) {
    std::cout << "cpp-classifier trainer\n";

    classifier::Classifier<3> model;

    std::array<double, 3> sample = {0.8, 0.6, 0.9};
    auto result = model.classify(sample);

    std::cout << "Sample classification: " << to_string(result.prediction)
              << " (confidence: " << result.confidence << ")\n";

    return 0;
}
