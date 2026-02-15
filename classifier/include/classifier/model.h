#pragma once

#include <vector>

namespace classifier {

class Model {
public:
    std::vector<float> weights;
    std::vector<float> features;
    float bias;
};

} // namespace classifier
