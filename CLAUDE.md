# CLAUDE.md

## Project Overview

C++ header-only library implementing a template-based binary classifier using logistic regression. Built with CMake and C++20. No external dependencies — uses only the C++ standard library.

## Build

```sh
cmake -B build -S .
cmake --build build
```

## Test

```sh
ctest --test-dir build
```

Tests use `<cassert>` with no external testing framework. All tests are in a single executable (`classifier_tests`).

## Project Structure

```
classifier/                     # Header-only library (INTERFACE CMake target)
  include/classifier/
    math.h                      # sigmoid(), dot() — math utilities with C++20 concepts
    model.h                     # Model<N>, Result, Prediction enum — core classifier
    trainer.h                   # Trainer<N>, Regularization enum — gradient descent training
tests/
  src/
    test_classifier.cpp         # All tests (12 test functions, assert-based)
```

## Key Types

- `classifier::Model<N>` — Main classifier template. `N` (compile-time `std::size_t`) sets feature vector dimensionality. Holds `std::array<float, N>` weights and a `float` bias. Classifies via `sigmoid(dot(weights, features) + bias)`.
- `classifier::Result` — Wraps a `Prediction` enum (`positive`, `negative`, `unknown`) and a `float` confidence score.
- `classifier::sigmoid<T>()`, `classifier::dot()` — Math utilities constrained with C++20 concepts (`std::floating_point`, `std::ranges::sized_range`).
- `trainer::Trainer<N>` — Trains a `Model<N>` via gradient descent with logistic loss. Supports `Regularization::none`, `Regularization::l1`, and `Regularization::l2`.
- `trainer::Trainer<N>::Sample` — `std::pair<std::array<float, N>, float>` (features + label).

## Key Conventions

- C++20 standard required, no compiler extensions (`CMAKE_CXX_EXTENSIONS OFF`)
- Minimum CMake version: 3.20
- Template parameter `N` controls feature vector dimensionality at compile time
- All classifier headers are under `classifier/include/classifier/`
- The classifier library is an `INTERFACE` CMake target (header-only)
- Two namespaces: `classifier` (model, math) and `trainer` (training)
- Binary serialization format: `[N (size_t), weights (N × float), bias (float)]`
- `Model::deserialize` validates dimension match and throws `std::runtime_error` on mismatch
- Tests cover: classification, serialization round-trips, regularization effects, edge cases (N=0, negative regularization strength)
