#ifndef MACHINE_LEARNING_NESTEROV_H
#define MACHINE_LEARNING_NESTEROV_H
#include "Optimiser.h"

struct Nesterov : Optimiser {
    double learningRate, mu;
    Model::Parameters v;

    Nesterov(double learningRate, double mu = 0.9);

    void step(Model &model, const std::function<Model::Parameters(const Model&)> &computeDerivatives) override;
};

#endif //MACHINE_LEARNING_NESTEROV_H
