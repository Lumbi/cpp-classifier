# CLAUDE.md

## Project Overview

C++ header-only library implementing a template-based binary classifier using logistic regression. Built with CMake and C++20.

## Build

```sh
cmake -B build -S .
cmake --build build
```

## Test

```sh
ctest --test-dir build
```

## Project Structure

- `classifier/` — Header-only library (`classifier::Classifier<N>`, `Result`, `Prediction`)
- `trainer/` — Training executable
- `tests/` — Tests (no external framework, uses `assert`)

## Key Conventions

- C++20 standard, no compiler extensions
- Template parameter `N` controls feature vector dimensionality at compile time
- All classifier headers are under `classifier/include/classifier/`
- The classifier library is an `INTERFACE` CMake target (header-only)
