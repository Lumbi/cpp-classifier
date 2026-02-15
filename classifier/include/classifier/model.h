#pragma once

#include <array>
#include <cstddef>

#include "classifier/math.h"

namespace classifier {

template <std::size_t N>
class Model {
public:
    Model() : weights_{}, bias_(0.0f) {}

    bool classify(const std::array<float, N>& features) const {
        return sigmoid(dot(weights_, features) + bias_) >= 0.5f;
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
