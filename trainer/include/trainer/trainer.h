#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "classifier/math.h"
#include "classifier/model.h"

namespace trainer {

enum class Regularization { none, l1, l2 };

template <std::size_t N>
class Trainer {
public:
    explicit Trainer(classifier::Model<N>& model) : model_(model) {}

    using Sample = std::pair<std::array<float, N>, float>;
    using TrainingSet = std::vector<Sample>;

    void train(const TrainingSet& data, float learning_rate = 0.1f,
               std::size_t epochs = 100,
               Regularization regularization = Regularization::none,
               float regularization_strength = 0.0f) {
        if (data.empty()) {
            throw std::invalid_argument("training set must not be empty");
        }
        if (regularization_strength < 0.0f) {
            throw std::invalid_argument("regularization strength must be non-negative");
        }

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            std::array<float, N> weight_gradients{};
            float bias_gradient = 0.0f;

            for (const auto& [features, label] : data) {
                float z = 0.0f;
                for (std::size_t i = 0; i < N; ++i) {
                    z += model_.weight(i) * features[i];
                }
                z += model_.bias();

                float prediction = classifier::sigmoid(z);
                float error = prediction - label;

                for (std::size_t i = 0; i < N; ++i) {
                    weight_gradients[i] += error * features[i];
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
    }

private:
    classifier::Model<N>& model_;
};

} // namespace trainer
