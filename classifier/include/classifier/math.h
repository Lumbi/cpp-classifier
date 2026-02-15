#pragma once

#include <cmath>
#include <concepts>

namespace classifier {

template <std::floating_point T>
T sigmoid(T x) {
    return T(1) / (T(1) + std::exp(-x));
}

} // namespace classifier
