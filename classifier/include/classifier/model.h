#pragma once

#include <array>
#include <cstddef>
#include <istream>
#include <ostream>

#include "classifier/math.h"

namespace classifier {

enum class Prediction { unknown, positive, negative };

struct Result {
    Prediction prediction;
    float confidence;
};

template <std::size_t N>
class Model {
public:
    Model() noexcept : weights_{}, bias_(0.0f) {}

    Result classify(const std::array<float, N>& features) const noexcept {
        if constexpr (N == 0) {
            return {Prediction::unknown, 0.0f};
        } else {
            float score = math::sigmoid(math::dot(weights_, features) + bias_);
            if (score >= 0.5f) {
                return {Prediction::positive, score};
            }
            return {Prediction::negative, 1.0f - score};
        }
    }

    float weight(std::size_t index) const noexcept { return weights_[index]; }
    void set_weight(std::size_t index, float value) noexcept { weights_[index] = value; }
    static constexpr std::size_t weight_count() noexcept { return N; }

    float bias() const noexcept { return bias_; }
    void set_bias(float value) noexcept { bias_ = value; }

    Error serialize(std::ostream& os) const noexcept {
        std::size_t n = N;
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        os.write(reinterpret_cast<const char*>(weights_.data()), sizeof(float) * N);
        os.write(reinterpret_cast<const char*>(&bias_), sizeof(bias_));
        if (!os) {
            return Error::io_failed;
        }
        return Error::none;
    }

    Error deserialize(std::istream& is) noexcept {
        std::size_t n = 0;
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!is) {
            return Error::io_failed;
        }
        if (n != N) {
            return Error::dimension_mismatch;
        }
        is.read(reinterpret_cast<char*>(weights_.data()), sizeof(float) * N);
        is.read(reinterpret_cast<char*>(&bias_), sizeof(bias_));
        if (!is) {
            return Error::io_failed;
        }
        return Error::none;
    }

private:
    std::array<float, N> weights_;
    float bias_;
};

} // namespace classifier
