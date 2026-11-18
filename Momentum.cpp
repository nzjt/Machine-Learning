#include "Momentum.h"

Momentum::Momentum(double learningRate, double mu) : learningRate(learningRate), mu(mu) {}

void Momentum::step(Model &model, const std::function<Derivatives(const Model&)> &computeDerivatives) {
    if (v.dJ_dw.empty()) {
        v = computeDerivatives(model) * -learningRate;
    } else {
        v *= mu;
        v += computeDerivatives(model) * -learningRate;
    }
    auto &layers = model.layers;
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->w += v.dJ_dw[i];
        layers[i]->b += v.dJ_db[i];
    }
}