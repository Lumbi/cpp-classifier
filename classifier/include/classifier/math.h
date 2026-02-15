#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <stdexcept>

namespace classifier {

template <std::floating_point T>
T sigmoid(T x) {
    return T(1) / (T(1) + std::exp(-x));
}

template <std::ranges::sized_range A, std::ranges::sized_range B>
    requires std::floating_point<std::ranges::range_value_t<A>> &&
             std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
auto dot(const A& a, const B& b) -> std::ranges::range_value_t<A> {
    if (std::ranges::size(a) != std::ranges::size(b)) {
        throw std::invalid_argument("dot product requires containers of equal size");
    }
    using T = std::ranges::range_value_t<A>;
    return std::inner_product(std::ranges::begin(a), std::ranges::end(a),
                              std::ranges::begin(b), T(0));
}

template <std::floating_point T, std::size_t N>
T dot(const std::array<T, N>& a, const std::array<T, N>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), T(0));
}

} // namespace classifier
