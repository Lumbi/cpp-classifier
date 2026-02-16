#pragma once

#include <array>
#include <cstddef>

namespace classifier {

enum class Prediction { unknown, positive, negative };

class Result {
public:
    Result(Prediction prediction, double confidence)
        : prediction_(prediction), confidence_(confidence) {}

    [[nodiscard]] Prediction get_prediction() const { return prediction_; }
    [[nodiscard]] double get_confidence() const { return confidence_; }

private:
    Prediction prediction_;
    double confidence_;
};

template <std::size_t N>
class Classifier {
public:
    Classifier() { weights_.fill(1.0); }
    ~Classifier() = default;

    void set_weight(std::size_t index, double weight) {
        weights_.at(index) = weight;
    }

    Result classify(const std::array<double, N>& features) const {
        if constexpr (N == 0) {
            return {Prediction::unknown, 0.0};
        } else {
            double sum = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                sum += features[i] * weights_[i];
            }

            double avg = sum / static_cast<double>(N);

            if (avg >= 0.5) {
                return {Prediction::positive, avg};
            }
            return {Prediction::negative, 1.0 - avg};
        }
    }

private:
    std::array<double, N> weights_;
};

} // namespace classifier
