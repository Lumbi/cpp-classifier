#pragma once

#include <array>
#include <cstddef>

#include "classifier/math.h"

namespace classifier {

enum class Prediction { unknown, positive, negative };

class Result {
public:
    Result(Prediction prediction, float confidence)
        : prediction_(prediction), confidence_(confidence) {}

    [[nodiscard]] Prediction get_prediction() const { return prediction_; }
    [[nodiscard]] float get_confidence() const { return confidence_; }

private:
    Prediction prediction_;
    float confidence_;
};

template <std::size_t N>
class Model {
public:
    Model() : weights_{}, bias_(0.0f) {}

    Result classify(const std::array<float, N>& features) const {
        if constexpr (N == 0) {
            return {Prediction::unknown, 0.0f};
        } else {
            float score = sigmoid(dot(weights_, features) + bias_);
            if (score >= 0.5f) {
                return {Prediction::positive, score};
            }
            return {Prediction::negative, 1.0f - score};
        }
    }

    float weight(std::size_t index) const { return weights_[index]; }
    void set_weight(std::size_t index, float value) { weights_[index] = value; }
    static constexpr std::size_t weight_count() { return N; }

    float bias() const { return bias_; }
    void set_bias(float value) { bias_ = value; }

private:
    std::array<float, N> weights_;
    float bias_;
};

} // namespace classifier
