#include "Dense.h"
#include "GradientDescent.h"
#include "linalg.h"
#include "Model.h"
#include <bits/stdc++.h>

void GradientDescent::step(Model &model, const std::function<Derivatives(const Model&)> &computeDerivatives) {
    auto [dJ_dw, dJ_db] = computeDerivatives(model);
    auto &layers = model.layers;
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->w -= dJ_dw[i] * learningRate;
        layers[i]->b -= dJ_db[i] * learningRate;
    }
}