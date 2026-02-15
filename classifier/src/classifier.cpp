#include "classifier/classifier.h"

namespace classifier {

Classifier::Classifier() = default;
Classifier::~Classifier() = default;

Result Classifier::classify(const std::vector<double>& features) const {
    if (features.empty()) {
        return {"unknown", 0.0};
    }

    double sum = 0.0;
    for (auto val : features) {
        sum += val;
    }

    double avg = sum / static_cast<double>(features.size());

    if (avg >= 0.5) {
        return {"positive", avg};
    }
    return {"negative", 1.0 - avg};
}

} // namespace classifier
