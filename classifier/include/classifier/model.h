#pragma once

#include <cstddef>
#include <vector>

namespace classifier {

class Model {
public:
    explicit Model(std::size_t size);

    bool classify(const std::vector<float>& features) const;

private:
    std::vector<float> weights_;
    float bias_;
};

} // namespace classifier
