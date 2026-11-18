#ifndef MACHINE_LEARNING_NESTEROV_H
#define MACHINE_LEARNING_NESTEROV_H
#include "Optimiser.h"

struct Nesterov : Optimiser {
    double learningRate, mu;
    Derivatives v;

    Nesterov(double learningRate, double mu);

    void step(Model &model, const std::function<Derivatives(const Model&)> &computeDerivatives) override;
};

#endif //MACHINE_LEARNING_NESTEROV_H
