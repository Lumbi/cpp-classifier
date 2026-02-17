#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <ranges>

namespace classifier {

enum class Error {
    none,
    size_mismatch,
    dimension_mismatch,
    io_failed,
    empty_training_set,
    invalid_regularization_strength,
};

namespace math {

template <std::floating_point T>
T sigmoid(T x) noexcept {
    return T(1) / (T(1) + std::exp(-x));
}

template <std::ranges::sized_range A, std::ranges::sized_range B>
    requires std::floating_point<std::ranges::range_value_t<A>> &&
             std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>
Error dot(A const& a, B const& b, std::ranges::range_value_t<A>& result) noexcept {
    if (std::ranges::size(a) != std::ranges::size(b)) {
        return Error::size_mismatch;
    }
    using T = std::ranges::range_value_t<A>;
    result = std::inner_product(std::ranges::begin(a), std::ranges::end(a),
                                std::ranges::begin(b), T(0));
    return Error::none;
}

template <std::floating_point T, std::size_t N>
T dot(std::array<T, N> const& a, std::array<T, N> const& b) noexcept {
    return std::inner_product(a.begin(), a.end(), b.begin(), T(0));
}

} // namespace math
} // namespace classifier
