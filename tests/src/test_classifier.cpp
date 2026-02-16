#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

#include "classifier/model.h"
#include "trainer/trainer.h"

void test_empty_features() {
    classifier::Model<0> model;
    auto result = model.classify({});
    assert(result.get_prediction() == classifier::Prediction::unknown);
    assert(result.get_confidence() == 0.0f);
    std::cout << "  PASS: test_empty_features\n";
}

void test_positive_classification() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);
    model.set_weight(1, 1.0f);
    model.set_weight(2, 1.0f);
    auto result = model.classify({0.8f, 0.6f, 0.9f});
    assert(result.get_prediction() == classifier::Prediction::positive);
    assert(result.get_confidence() > 0.5f);
    std::cout << "  PASS: test_positive_classification\n";
}

void test_negative_classification() {
    classifier::Model<3> model;
    model.set_weight(0, -1.0f);
    model.set_weight(1, -1.0f);
    model.set_weight(2, -1.0f);
    auto result = model.classify({0.1f, 0.2f, 0.1f});
    assert(result.get_prediction() == classifier::Prediction::negative);
    assert(result.get_confidence() > 0.5f);
    std::cout << "  PASS: test_negative_classification\n";
}

void test_serialize_deserialize() {
    classifier::Model<3> model;
    model.set_weight(0, 1.5f);
    model.set_weight(1, -0.3f);
    model.set_weight(2, 0.7f);
    model.set_bias(0.25f);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    model.serialize(ss);

    classifier::Model<3> loaded;
    loaded.deserialize(ss);

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
    model.serialize(ss);

    classifier::Model<3> loaded;
    loaded.deserialize(ss);
    auto loaded_result = loaded.classify(features);

    assert(original_result.get_prediction() == loaded_result.get_prediction());
    assert(original_result.get_confidence() == loaded_result.get_confidence());
    std::cout << "  PASS: test_serialize_preserves_classification\n";
}

void test_deserialize_dimension_mismatch() {
    classifier::Model<3> model;
    model.set_weight(0, 1.0f);

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    model.serialize(ss);

    classifier::Model<2> wrong_size;
    bool caught = false;
    try {
        wrong_size.deserialize(ss);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  PASS: test_deserialize_dimension_mismatch\n";
}

void test_serialize_empty_model() {
    classifier::Model<0> model;

    std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
    model.serialize(ss);

    classifier::Model<0> loaded;
    loaded.deserialize(ss);

    assert(loaded.bias() == 0.0f);
    std::cout << "  PASS: test_serialize_empty_model\n";
}

void test_l2_regularization_reduces_weights() {
    // Train two models on the same data: one without regularization, one with L2.
    // L2 regularization should produce smaller weight magnitudes.
    using TrainingSet = trainer::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {{1.0f, 0.5f}, 1.0f},
        {{0.5f, 1.0f}, 1.0f},
        {{-1.0f, -0.5f}, 0.0f},
        {{-0.5f, -1.0f}, 0.0f},
    };

    classifier::Model<2> model_unreg;
    trainer::Trainer<2> trainer_unreg(model_unreg);
    trainer_unreg.train(data, 0.5f, 200);

    classifier::Model<2> model_reg;
    trainer::Trainer<2> trainer_reg(model_reg);
    trainer_reg.train(data, 0.5f, 200, trainer::Regularization::l2, 0.5f);

    float unreg_norm = model_unreg.weight(0) * model_unreg.weight(0)
                     + model_unreg.weight(1) * model_unreg.weight(1);
    float reg_norm = model_reg.weight(0) * model_reg.weight(0)
                   + model_reg.weight(1) * model_reg.weight(1);

    assert(reg_norm < unreg_norm);
    std::cout << "  PASS: test_l2_regularization_reduces_weights\n";
}

void test_l1_regularization_reduces_weights() {
    // L1 regularization should also produce smaller weight magnitudes.
    using TrainingSet = trainer::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {{1.0f, 0.5f}, 1.0f},
        {{0.5f, 1.0f}, 1.0f},
        {{-1.0f, -0.5f}, 0.0f},
        {{-0.5f, -1.0f}, 0.0f},
    };

    classifier::Model<2> model_unreg;
    trainer::Trainer<2> trainer_unreg(model_unreg);
    trainer_unreg.train(data, 0.5f, 200);

    classifier::Model<2> model_reg;
    trainer::Trainer<2> trainer_reg(model_reg);
    trainer_reg.train(data, 0.5f, 200, trainer::Regularization::l1, 0.5f);

    float unreg_norm = std::abs(model_unreg.weight(0)) + std::abs(model_unreg.weight(1));
    float reg_norm = std::abs(model_reg.weight(0)) + std::abs(model_reg.weight(1));

    assert(reg_norm < unreg_norm);
    std::cout << "  PASS: test_l1_regularization_reduces_weights\n";
}

void test_regularization_none_matches_baseline() {
    // Regularization::none with any strength should behave identically
    // to calling train without regularization parameters.
    using TrainingSet = trainer::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {{1.0f, 0.0f}, 1.0f},
        {{0.0f, 1.0f}, 0.0f},
    };

    classifier::Model<2> model_a;
    trainer::Trainer<2> trainer_a(model_a);
    trainer_a.train(data, 0.1f, 50);

    classifier::Model<2> model_b;
    trainer::Trainer<2> trainer_b(model_b);
    trainer_b.train(data, 0.1f, 50, trainer::Regularization::none, 1.0f);

    assert(model_a.weight(0) == model_b.weight(0));
    assert(model_a.weight(1) == model_b.weight(1));
    assert(model_a.bias() == model_b.bias());
    std::cout << "  PASS: test_regularization_none_matches_baseline\n";
}

void test_regularization_preserves_correctness() {
    // Model trained with regularization should still classify correctly.
    using TrainingSet = trainer::Trainer<2>::TrainingSet;
    TrainingSet data = {
        {{1.0f, 0.5f}, 1.0f},
        {{0.5f, 1.0f}, 1.0f},
        {{-1.0f, -0.5f}, 0.0f},
        {{-0.5f, -1.0f}, 0.0f},
    };

    classifier::Model<2> model;
    trainer::Trainer<2> t(model);
    t.train(data, 0.5f, 300, trainer::Regularization::l2, 0.1f);

    auto pos = model.classify({1.0f, 0.5f});
    assert(pos.get_prediction() == classifier::Prediction::positive);

    auto neg = model.classify({-1.0f, -0.5f});
    assert(neg.get_prediction() == classifier::Prediction::negative);
    std::cout << "  PASS: test_regularization_preserves_correctness\n";
}

void test_negative_regularization_strength_throws() {
    using TrainingSet = trainer::Trainer<2>::TrainingSet;
    TrainingSet data = {{{1.0f, 0.0f}, 1.0f}};

    classifier::Model<2> model;
    trainer::Trainer<2> t(model);

    bool caught = false;
    try {
        t.train(data, 0.1f, 10, trainer::Regularization::l2, -0.1f);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
    std::cout << "  PASS: test_negative_regularization_strength_throws\n";
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
    test_negative_regularization_strength_throws();

    std::cout << "All tests passed.\n";
    return 0;
}
