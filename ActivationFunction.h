#ifndef MACHINE_LEARNING_ACTIVATIONFUNCTION_H
#define MACHINE_LEARNING_ACTIVATIONFUNCTION_H
#include "MyRandom.h"
#include <bits/stdc++.h>

struct ActivationFunction {
    [[nodiscard]] virtual std::vector<double> operator()(const std::vector<double> &z) const = 0;

    template <class T>
    [[nodiscard]] std::vector<T> operator()(const std::vector<T> &z) const {
        std::vector<T> ret;
        for (const auto &zi: z) {
            ret.push_back((*this)(zi));
        }
        return ret;
    }

    [[nodiscard]] virtual std::vector<double> gradient(const std::vector<double> &dJ_da, const std::vector<double> &z, const std::vector<double> &a) const = 0;

    template <class T>
    [[nodiscard]] std::vector<T> gradient(const std::vector<T> dJ_da, const std::vector<T> &z, const std::vector<T> &a) const {
        std::vector<T> ret;
        for (int i = 0; i < z.size(); ++i) {
            ret.push_back(derivative(z[i], a[i]));
        }
        return ret;
    }

    virtual void initialise(std::vector<std::vector<double>> &w, std::vector<double> &b, MyRandom &rng) const = 0;

    virtual ~ActivationFunction() = default;
};

#endif //MACHINE_LEARNING_ACTIVATIONFUNCTION_H
