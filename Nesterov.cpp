#include "Nesterov.h"

Nesterov::Nesterov(double learningRate, double mu) : learningRate(learningRate), mu(mu) {}

void Nesterov::step(Model &model, const std::function<Derivatives(const Model&)> &computeDerivatives) {
    auto &layers = model.layers;
    if (v.dJ_dw.empty()) {
        v = computeDerivatives(model) * -learningRate;
    } else {
        v *= mu;
        for (int i = 0; i < layers.size(); ++i) {
            layers[i]->w += v.dJ_dw[i];
            layers[i]->b += v.dJ_db[i];
        }
        auto temp = v;
        v += computeDerivatives(model) * -learningRate;
        for (int i = 0; i < layers.size(); ++i) {
            layers[i]->w -= temp.dJ_dw[i];
            layers[i]->b -= temp.dJ_db[i];
        }
    }
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->w += v.dJ_dw[i];
        layers[i]->b += v.dJ_db[i];
    }
}