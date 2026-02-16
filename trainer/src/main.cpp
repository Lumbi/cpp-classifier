#include <array>
#include <iostream>
#include <string_view>

#include "classifier/model.h"

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

    classifier::Model<3> model;
    model.set_weight(0, 1.0f);
    model.set_weight(1, 1.0f);
    model.set_weight(2, 1.0f);

    std::array<float, 3> sample = {0.8f, 0.6f, 0.9f};
    auto result = model.classify(sample);

    std::cout << "Sample classification: " << to_string(result.get_prediction())
              << " (confidence: " << result.get_confidence() << ")\n";

    return 0;
}
