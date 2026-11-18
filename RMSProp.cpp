#include "linalg.h"
#include "RMSProp.h"

RMSProp::RMSProp(double learningRate, double rho, double epsilon) : learningRate(learningRate), rho(rho), epsilon(epsilon) {}

void RMSProp::step(Model &model, const std::function<Model::Parameters(const Model &)> &computeDerivatives) {
    auto gradient = computeDerivatives(model);
    if (v.empty()) {
        v = gradient * gradient;
    } else {
        v *= rho;
        v += gradient * gradient * (1 - rho);
    }
    model += gradient * -learningRate / (sqrt(v) + epsilon);
}
