#ifndef MACHINE_LEARNING_MOMENTUM_H
#define MACHINE_LEARNING_MOMENTUM_H
#include "Optimiser.h"

struct Momentum : Optimiser {
    double learningRate, mu;
    Model::Parameters v;

    Momentum(double learningRate, double mu = 0.9);

    void step(Model &model, const std::function<Model::Parameters(const Model&)> &computeDerivatives) override;
};

#endif //MACHINE_LEARNING_MOMENTUM_H
