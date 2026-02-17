#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

#include "classifier/model.h"
#include "classifier/trainer.h"

void test_empty_features() {
    classifier::Model<0> model;
    auto result = model.classify({});
    assert(result.prediction == classifier::Prediction::unknown);
    assert(result.confidence == 0.0f);
    std::cout << "  PASS: test_empty_features\n";
}

void test_positive_classification() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);
    model.set_weight(1, 1.0f);
    model.set_weight(2, 1.0f);
    auto result = model.classify({0.8f, 0.6f, 0.9f});
    assert(result.prediction == classifier::Prediction::positive);
    assert(result.confidence > 0.5f);
    std::cout << "  PASS: test_positive_classification\n";
}

void test_negative_classification() {
    classifier::Model<3> model;
    model.set_weight(0, -1.0f);
    model.set_weight(1, -1.0f);
    model.set_weight(2, -1.0f);
    auto result = model.classify({0.1f, 0.2f, 0.1f});
    assert(result.prediction == classifier::Prediction::negative);
    assert(result.confidence > 0.5f);
    std::cout << "  PASS: test_negative_classification\n";
}

void test_serialize_deserialize() {
    classifier::Model<3> model;
    model.set_weight(0, 1.5f);
    model.set_weight(1, -0.3f);
    model.set_weight(2, 0.7f);
    model.set_bias(0.25f);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    assert(model.serialize(ss) == classifier::Error::none);

    classifier::Model<3> loaded;
    assert(loaded.deserialize(ss) == classifier::Error::none);

    assert(loaded.weight(0) == 1.5f);
    assert(loaded.weight(1) == -0.3f);
    assert(loaded.weight(2) == 0.7f);
    assert(loaded.bias() == 0.25f);
    std::cout << "  PASS: test_serialize_deserialize\n";
}

void test_serialize_preserves_classification() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);
    model.set_weight(1, 1.0f);
    model.set_weight(2, 1.0f);
    model.set_bias(-1.5f);

    std::array<float, 3> features = {0.8f, 0.6f, 0.9f};
    auto original_result = model.classify(features);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    assert(model.serialize(ss) == classifier::Error::none);

    classifier::Model<3> loaded;
    assert(loaded.deserialize(ss) == classifier::Error::none);
    auto loaded_result = loaded.classify(features);

    assert(original_result.prediction == loaded_result.prediction);
    assert(original_result.confidence == loaded_result.confidence);
    std::cout << "  PASS: test_serialize_preserves_classification\n";
}

void test_deserialize_dimension_mismatch() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    assert(model.serialize(ss) == classifier::Error::none);

    classifier::Model<2> wrong_size;
    assert(wrong_size.deserialize(ss) == classifier::Error::dimension_mismatch);
    std::cout << "  PASS: test_deserialize_dimension_mismatch\n";
}

void test_serialize_empty_model() {
    classifier::Model<0> model;

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    assert(model.serialize(ss) == classifier::Error::none);

    classifier::Model<0> loaded;
    assert(loaded.deserialize(ss) == classifier::Error::none);

    assert(loaded.bias() == 0.0f);
    std::cout << "  PASS: test_serialize_empty_model\n";
}

void test_l2_regularization_reduces_weights() {
    // Train two models on the same data: one without regularization, one with L2.
    // L2 regularization should produce smaller weight magnitudes.
    using TrainingSet = classifier::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {1.0f, 0.5f, 1.0f},
        {0.5f, 1.0f, 1.0f},
        {-1.0f, -0.5f, 0.0f},
        {-0.5f, -1.0f, 0.0f},
    };

    classifier::Model<2> model_unreg;
    classifier::Trainer<2> trainer_unreg(model_unreg);
    assert(trainer_unreg.train(data, 0.5f, 200) == classifier::Error::none);

    classifier::Model<2> model_reg;
    classifier::Trainer<2> trainer_reg(model_reg);
    assert(trainer_reg.train(data, 0.5f, 200, classifier::Regularization::l2, 0.5f) == classifier::Error::none);

    float unreg_norm = model_unreg.weight(0) * model_unreg.weight(0)
                     + model_unreg.weight(1) * model_unreg.weight(1);
    float reg_norm = model_reg.weight(0) * model_reg.weight(0)
                   + model_reg.weight(1) * model_reg.weight(1);

    assert(reg_norm < unreg_norm);
    std::cout << "  PASS: test_l2_regularization_reduces_weights\n";
}

void test_l1_regularization_reduces_weights() {
    // L1 regularization should also produce smaller weight magnitudes.
    using TrainingSet = classifier::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {1.0f, 0.5f, 1.0f},
        {0.5f, 1.0f, 1.0f},
        {-1.0f, -0.5f, 0.0f},
        {-0.5f, -1.0f, 0.0f},
    };

    classifier::Model<2> model_unreg;
    classifier::Trainer<2> trainer_unreg(model_unreg);
    assert(trainer_unreg.train(data, 0.5f, 200) == classifier::Error::none);

    classifier::Model<2> model_reg;
    classifier::Trainer<2> trainer_reg(model_reg);
    assert(trainer_reg.train(data, 0.5f, 200, classifier::Regularization::l1, 0.5f) == classifier::Error::none);

    float unreg_norm = std::abs(model_unreg.weight(0)) + std::abs(model_unreg.weight(1));
    float reg_norm = std::abs(model_reg.weight(0)) + std::abs(model_reg.weight(1));

    assert(reg_norm < unreg_norm);
    std::cout << "  PASS: test_l1_regularization_reduces_weights\n";
}

void test_regularization_none_matches_baseline() {
    // Regularization::none with any strength should behave identically
    // to calling train without regularization parameters.
    using TrainingSet = classifier::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f},
    };

    classifier::Model<2> model_a;
    classifier::Trainer<2> trainer_a(model_a);
    assert(trainer_a.train(data, 0.1f, 50) == classifier::Error::none);

    classifier::Model<2> model_b;
    classifier::Trainer<2> trainer_b(model_b);
    assert(trainer_b.train(data, 0.1f, 50, classifier::Regularization::none, 1.0f) == classifier::Error::none);

    assert(model_a.weight(0) == model_b.weight(0));
    assert(model_a.weight(1) == model_b.weight(1));
    assert(model_a.bias() == model_b.bias());
    std::cout << "  PASS: test_regularization_none_matches_baseline\n";
}

void test_regularization_preserves_correctness() {
    // Model trained with regularization should still classify correctly.
    using TrainingSet = classifier::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {1.0f, 0.5f, 1.0f},
        {0.5f, 1.0f, 1.0f},
        {-1.0f, -0.5f, 0.0f},
        {-0.5f, -1.0f, 0.0f},
    };

    classifier::Model<2> model;
    classifier::Trainer<2> t(model);
    assert(t.train(data, 0.5f, 300, classifier::Regularization::l2, 0.1f) == classifier::Error::none);

    auto pos = model.classify({1.0f, 0.5f});
    assert(pos.prediction == classifier::Prediction::positive);

    auto neg = model.classify({-1.0f, -0.5f});
    assert(neg.prediction == classifier::Prediction::negative);
    std::cout << "  PASS: test_regularization_preserves_correctness\n";
}

void test_negative_regularization_strength() {
    using TrainingSet = classifier::Trainer<2>::TrainingSet;
    TrainingSet data = {{1.0f, 0.0f, 1.0f}};

    classifier::Model<2> model;
    classifier::Trainer<2> t(model);

    assert(t.train(data, 0.1f, 10, classifier::Regularization::l2, -0.1f)
           == classifier::Error::invalid_regularization_strength);
    std::cout << "  PASS: test_negative_regularization_strength\n";
}

void test_deserialize_training_data() {
    using TrainingSet = classifier::Trainer<2>::TrainingSet;
    TrainingSet original = {
        {1.0f, 0.5f, 1.0f},
        {-1.0f, -0.5f, 0.0f},
        {0.3f, 0.7f, 1.0f},
    };

    // Serialize: cols (size_t), rows (size_t), then row-major float data
    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    std::size_t cols = 3;
    std::size_t rows = 3;
    ss.write(reinterpret_cast<char const*>(&cols), sizeof(cols));
    ss.write(reinterpret_cast<char const*>(&rows), sizeof(rows));
    for (auto const& sample : original) {
        ss.write(reinterpret_cast<char const*>(sample.data()),
                 sizeof(float) * 3);
    }

    TrainingSet loaded;
    assert(classifier::Trainer<2>::deserialize_training_data(ss, loaded) == classifier::Error::none);
    assert(loaded.size() == 3);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            assert(loaded[r][c] == original[r][c]);
        }
    }
    std::cout << "  PASS: test_deserialize_training_data\n";
}

void test_deserialize_training_data_column_mismatch() {
    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    std::size_t cols = 4; // Trainer<2> expects 3 columns
    std::size_t rows = 1;
    ss.write(reinterpret_cast<char const*>(&cols), sizeof(cols));
    ss.write(reinterpret_cast<char const*>(&rows), sizeof(rows));

    classifier::Trainer<2>::TrainingSet out;
    assert(classifier::Trainer<2>::deserialize_training_data(ss, out)
           == classifier::Error::dimension_mismatch);
    std::cout << "  PASS: test_deserialize_training_data_column_mismatch\n";
}

void test_deserialize_training_data_round_trip_train() {
    // Deserialize training data, then use it to train a model successfully.
    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    std::size_t cols = 3;
    std::size_t rows = 4;
    ss.write(reinterpret_cast<char const*>(&cols), sizeof(cols));
    ss.write(reinterpret_cast<char const*>(&rows), sizeof(rows));

    std::array<std::array<float, 3>, 4> raw = {{
        {1.0f, 0.5f, 1.0f},
        {0.5f, 1.0f, 1.0f},
        {-1.0f, -0.5f, 0.0f},
        {-0.5f, -1.0f, 0.0f},
    }};
    for (auto const& row : raw) {
        ss.write(reinterpret_cast<char const*>(row.data()), sizeof(float) * 3);
    }

    classifier::Trainer<2>::TrainingSet data;
    assert(classifier::Trainer<2>::deserialize_training_data(ss, data) == classifier::Error::none);

    classifier::Model<2> model;
    classifier::Trainer<2> t(model);
    assert(t.train(data, 0.5f, 300) == classifier::Error::none);

    auto pos = model.classify({1.0f, 0.5f});
    assert(pos.prediction == classifier::Prediction::positive);

    auto neg = model.classify({-1.0f, -0.5f});
    assert(neg.prediction == classifier::Prediction::negative);
    std::cout << "  PASS: test_deserialize_training_data_round_trip_train\n";
}

int main() {
    std::cout << "Running classifier tests...\n";

    test_empty_features();
    test_positive_classification();
    test_negative_classification();
    test_serialize_deserialize();
    test_serialize_preserves_classification();
    test_deserialize_dimension_mismatch();
    test_serialize_empty_model();
    test_l2_regularization_reduces_weights();
    test_l1_regularization_reduces_weights();
    test_regularization_none_matches_baseline();
    test_regularization_preserves_correctness();
    test_negative_regularization_strength();
    test_deserialize_training_data();
    test_deserialize_training_data_column_mismatch();
    test_deserialize_training_data_round_trip_train();

    std::cout << "All tests passed.\n";
    return 0;
}
