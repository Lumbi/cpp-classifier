#include <array>
#include <iostream>

#include <classifier/model.h>
#include <classifier/trainer.h>

int main() {
    // Train a 2D classifier to distinguish two clusters:
    //   positive: points near (1, 1)
    //   negative: points near (-1, -1)

    using Sample = classifier::Trainer<2>::Sample;

    std::vector<Sample> data = {
        { 1.0f,  1.0f, 1.0f},
        { 0.8f,  1.2f, 1.0f},
        { 1.2f,  0.9f, 1.0f},
        { 0.9f,  0.8f, 1.0f},
        { 1.1f,  1.1f, 1.0f},
        {-1.0f, -1.0f, 0.0f},
        {-0.8f, -1.2f, 0.0f},
        {-1.2f, -0.9f, 0.0f},
        {-0.9f, -0.8f, 0.0f},
        {-1.1f, -1.1f, 0.0f},
    };

    std::cout << "Training a 2D binary classifier...\n";

    classifier::Model<2> model;
    classifier::Trainer<2> t(model);
    if (t.train(data, /*learning_rate=*/0.5f, /*epochs=*/200) != classifier::Error::none) {
        std::cerr << "Training failed\n";
        return 1;
    }

    std::cout << "Learned weights: ["
              << model.weight(0) << ", " << model.weight(1) << "]\n";
    std::cout << "Learned bias:    " << model.bias() << "\n\n";

    // Classify some test points
    struct TestCase {
        std::array<float, 2> features;
        char const* label;
    };

    TestCase tests[] = {
        {{ 0.5f,  0.5f}, "( 0.5,  0.5)"},
        {{ 1.5f,  1.0f}, "( 1.5,  1.0)"},
        {{-0.5f, -0.5f}, "(-0.5, -0.5)"},
        {{-1.5f, -1.0f}, "(-1.5, -1.0)"},
        {{ 0.0f,  0.0f}, "( 0.0,  0.0)"},
    };

    std::cout << "Classification results:\n";
    for (auto const& tc : tests) {
        auto result = model.classify(tc.features);
        char const* pred = (result.prediction == classifier::Prediction::positive)
                               ? "positive"
                               : "negative";
        std::cout << "  " << tc.label << " -> " << pred
                  << " (confidence: " << result.confidence << ")\n";
    }

    return 0;
}
