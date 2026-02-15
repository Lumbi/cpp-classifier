#pragma once

#include <cstddef>
#include <vector>

namespace classifier {

class Model {
public:
    explicit Model(std::size_t size);

    bool classify(const std::vector<float>& features) const;

    float weight(std::size_t index) const;
    void set_weight(std::size_t index, float value);
    std::size_t weight_count() const;

    float bias() const;
    void set_bias(float value);

private:
    std::vector<float> weights_;
    float bias_;
};

} // namespace classifier
