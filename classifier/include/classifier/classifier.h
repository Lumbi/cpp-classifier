#pragma once

#include <array>
#include <cstddef>

namespace classifier {

enum class Label { unknown, positive, negative };

struct Result {
    Label label;
    double confidence;
};

template <std::size_t N>
class Classifier {
public:
    Classifier() = default;
    ~Classifier() = default;

    Result classify(const std::array<double, N>& features) const {
        if constexpr (N == 0) {
            return {Label::unknown, 0.0};
        } else {
            double sum = 0.0;
            for (auto val : features) {
                sum += val;
            }

            double avg = sum / static_cast<double>(N);

            if (avg >= 0.5) {
                return {Label::positive, avg};
            }
            return {Label::negative, 1.0 - avg};
        }
    }
};

} // namespace classifier
