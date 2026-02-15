#include "classifier/math.h"
#include "classifier/model.h"

namespace classifier {

Model::Model(std::size_t size)
    : weights_(size, 0.0f), bias_(0.0f) {}

bool Model::classify(const std::vector<float>& features) const {
    return sigmoid(dot(weights_, features) + bias_) >= 0.5f;
}

float Model::weight(std::size_t index) const { return weights_[index]; }

void Model::set_weight(std::size_t index, float value) { weights_[index] = value; }

std::size_t Model::weight_count() const { return weights_.size(); }

float Model::bias() const { return bias_; }

void Model::set_bias(float value) { bias_ = value; }

} // namespace classifier
