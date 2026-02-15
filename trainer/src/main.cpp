#include <iostream>
#include <vector>

#include "classifier/classifier.h"

int main(int argc, char* argv[]) {
    std::cout << "cpp-classifier trainer\n";

    classifier::Classifier model;

    std::vector<double> sample = {0.8, 0.6, 0.9};
    auto result = model.classify(sample);

    std::cout << "Sample classification: " << result.label
              << " (confidence: " << result.confidence << ")\n";

    return 0;
}
