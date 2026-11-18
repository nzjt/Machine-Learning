#ifndef MACHINE_LEARNING_RMSPROP_H
#define MACHINE_LEARNING_RMSPROP_H
#include "Optimiser.h"

struct RMSProp : Optimiser {
    double learningRate, rho, epsilon;
    Model::Parameters v;

    RMSProp(double learningRate, double rho = 0.9, double epsilon = 1e-8);

    void step(Model &model, const std::function<Model::Parameters(const Model&)> &computeDerivatives) override;
};

#endif //MACHINE_LEARNING_RMSPROP_H
