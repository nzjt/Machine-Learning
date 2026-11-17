#include "Dense.h"
#include "GradientDescent.h"
#include "linalg.h"
#include "Model.h"
#include <bits/stdc++.h>

void GradientDescent::optimise(const Derivatives &derivatives, vector<Dense*> &layers) {
    auto [dJ_dw, dJ_db] = derivatives;
    for (int i = 0; i < layers.size(); ++i) {
        layers[i]->w -= dJ_dw[i] * learningRate;
        layers[i]->b -= dJ_db[i] * learningRate;
    }
}