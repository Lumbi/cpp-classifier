#pragma once

#include <string>
#include <vector>

namespace classifier {

struct Result {
    std::string label;
    double confidence;
};

class Classifier {
public:
    Classifier();
    ~Classifier();

    Result classify(const std::vector<double>& features) const;
};

} // namespace classifier
