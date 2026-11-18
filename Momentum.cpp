#include "Momentum.h"

Momentum::Momentum(double learningRate, double mu) : learningRate(learningRate), mu(mu) {}

void Momentum::step(Model &model, const std::function<Model::Parameters(const Model&)> &computeDerivatives) {
    if (v.empty()) {
        v = computeDerivatives(model) * -learningRate;
    } else {
        v *= mu;
        v += computeDerivatives(model) * -learningRate;
    }
    model += v;
}