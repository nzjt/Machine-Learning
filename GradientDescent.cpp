#include "Dense.h"
#include "GradientDescent.h"
#include "linalg.h"
#include "Model.h"
#include <bits/stdc++.h>

void GradientDescent::step(Model &model, const std::function<Model::Parameters(const Model&)> &computeDerivatives) {
    model += computeDerivatives(model) * -learningRate;
}