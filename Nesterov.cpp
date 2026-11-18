#include "Nesterov.h"

Nesterov::Nesterov(double learningRate, double mu) : learningRate(learningRate), mu(mu) {}

void Nesterov::step(Model &model, const std::function<Model::Parameters(const Model&)> &computeDerivatives) {
    if (v.empty()) {
        v = computeDerivatives(model) * -learningRate;
    } else {
        v *= mu;
        model += v;
        auto temp = -v;
        v += computeDerivatives(model) * -learningRate;
        model += temp;
    }
    model += v;
}