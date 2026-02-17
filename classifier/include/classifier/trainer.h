#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <istream>
#include <vector>

#include "classifier/math.h"
#include "classifier/model.h"

namespace classifier {

enum class Regularization { none, l1, l2 };

template <std::size_t N>
class Trainer {
public:
    explicit Trainer(Model<N>& model) noexcept : model_(model) {}

    using Sample = std::array<float, N + 1>;
    using TrainingSet = std::vector<Sample>;

    static Error deserialize_training_data(std::istream& is, TrainingSet& out) noexcept {
        std::size_t cols = 0;
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        if (!is) {
            return Error::io_failed;
        }
        if (cols != N + 1) {
            return Error::dimension_mismatch;
        }

        std::size_t rows = 0;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (!is) {
            return Error::io_failed;
        }

        out.resize(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            is.read(reinterpret_cast<char*>(out[r].data()),
                    sizeof(float) * (N + 1));
            if (!is) {
                return Error::io_failed;
            }
        }
        return Error::none;
    }

    Error train(TrainingSet const& data, float learning_rate = 0.1f,
                std::size_t epochs = 100,
                Regularization regularization = Regularization::none,
                float regularization_strength = 0.0f) noexcept {
        if (data.empty()) {
            return Error::empty_training_set;
        }
        if (regularization_strength < 0.0f) {
            return Error::invalid_regularization_strength;
        }

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::array<float, N> weight_gradients{};
            float bias_gradient = 0.0f;

            for (auto const& sample : data) {
                float label = sample[N];

                float z = 0.0f;
                for (std::size_t i = 0; i < N; ++i) {
                    z += model_.weight(i) * sample[i];
                }
                z += model_.bias();

                float prediction = math::sigmoid(z);
                float error = prediction - label;

                for (std::size_t i = 0; i < N; ++i) {
                    weight_gradients[i] += error * sample[i];
                }
                bias_gradient += error;
            }

            float m = static_cast<float>(data.size());
            for (std::size_t i = 0; i < N; ++i) {
                float gradient = weight_gradients[i] / m;

                if (regularization == Regularization::l2) {
                    gradient += regularization_strength * model_.weight(i);
                } else if (regularization == Regularization::l1) {
                    float w = model_.weight(i);
                    if (w > 0.0f) {
                        gradient += regularization_strength;
                    } else if (w < 0.0f) {
                        gradient -= regularization_strength;
                    }
                }

                model_.set_weight(i, model_.weight(i) - learning_rate * gradient);
            }
            model_.set_bias(model_.bias() - learning_rate * bias_gradient / m);
        }
        return Error::none;
    }

private:
    Model<N>& model_;
};

} // namespace classifier
