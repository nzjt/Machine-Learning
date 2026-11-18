#ifndef MACHINE_LEARNING_MOMENTUM_H
#define MACHINE_LEARNING_MOMENTUM_H
#include "Optimiser.h"

struct Momentum : Optimiser {
    double learningRate, mu;
    Derivatives v;

    Momentum(double learningRate, double mu);

    void step(Model &model, const std::function<Derivatives(const Model&)> &computeDerivatives) override;
};

#endif //MACHINE_LEARNING_MOMENTUM_H
