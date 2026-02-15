#include "classifier/model.h"

namespace classifier {

Model::Model(std::size_t size)
    : weights_(size, 0.0f), features_(size, 0.0f), bias_(0.0f) {}

} // namespace classifier
