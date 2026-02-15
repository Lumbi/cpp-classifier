#include "classifier/math.h"
#include "classifier/model.h"

namespace classifier {

Model::Model(std::size_t size)
    : weights_(size, 0.0f), features_(size, 0.0f), bias_(0.0f) {}

bool Model::classify(const std::vector<float>& features) const {
    return sigmoid(dot(weights_, features) + bias_) >= 0.5f;
}

} // namespace classifier
