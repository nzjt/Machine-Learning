#ifndef MACHINE_LEARNING_DENSE_H
#define MACHINE_LEARNING_DENSE_H
#include "ActivationFunction.h"
#include "layers.h"
#include "MyRandom.h"
#include <bits/stdc++.h>

struct Dense : Layer {
    std::vector<std::vector<double>> w;
    std::vector<double> b;
    ActivationFunction *activationFunction;

    Dense(int n, ActivationFunction *activationFunction);

    void randomise(MyRandom &rng);

    void read(ifstream &file);

    void write(ofstream &file) const;

    void resize(int m) override;

    [[nodiscard]] int nParameters() const override;

    [[nodiscard]] std::vector<double> computeZ(const std::vector<double> &activations) const;

    [[nodiscard]] std::vector<double> predict(const std::vector<double> &activations) const override;

    [[nodiscard]] vector<double> derivative(vector<double> &dJ_da, const vector<double> &z, const vector<double> &a) const override;
};

#endif //MACHINE_LEARNING_DENSE_H
