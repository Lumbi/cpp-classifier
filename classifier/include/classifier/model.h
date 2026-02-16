#pragma once

#include <array>
#include <cstddef>
#include <istream>
#include <ostream>
#include <stdexcept>

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

    void serialize(std::ostream& os) const {
        std::size_t n = N;
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        os.write(reinterpret_cast<const char*>(weights_.data()), sizeof(float) * N);
        os.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));
        if (!os) {
            throw std::runtime_error("failed to serialize model");
        }
    }

    void deserialize(std::istream& is) {
        std::size_t n = 0;
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!is) {
            throw std::runtime_error("failed to read model header");
        }
        if (n != N) {
            throw std::runtime_error(
                "model dimension mismatch: expected " + std::to_string(N) +
                ", got " + std::to_string(n));
        }
        is.read(reinterpret_cast<char*>(weights_.data()), sizeof(float) * N);
        is.read(reinterpret_cast<char*>(&bias_), sizeof(bias_));
        if (!is) {
            throw std::runtime_error("failed to deserialize model");
        }
    }

private:
    std::array<float, N> weights_;
    float bias_;
};

} // namespace classifier
