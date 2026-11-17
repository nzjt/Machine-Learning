#ifndef MACHINE_LEARNING_BINARYCROSSENTROPY_H
#define MACHINE_LEARNING_BINARYCROSSENTROPY_H
#include "LossFunction.h"
#include "Sigmoid.h"
#include <bits/stdc++.h>

template <bool fromLogits = false>
struct BinaryCrossentropy : LossFunction {
    [[nodiscard]] double operator()(const vector<double> &y, const vector<double> &yhat) const override {
        int m = y.size();
        double loss = 0;
        if constexpr (fromLogits) {
            vector<double> activation = Sigmoid()(yhat);
            for (int i = 0; i < m; ++i) {
                loss += y[i] * log(activation[i]) + (1 - y[i]) * log(1 - activation[i]);
            }
        } else {
            for (int i = 0; i < m; ++i) {
                loss += y[i] * log(yhat[i]) + (1 - y[i]) * log(1 - yhat[i]);
            }
        }
        return loss / -m;
    }

    [[nodiscard]] vector<double> gradient(const vector<double> &y, const vector<double> &yhat) const override {
        int m = y.size();
        vector<double> ret(m);
        if constexpr (fromLogits) {
            return Sigmoid()(yhat) - y;
        } else {
            for (int i = 0; i < m; ++i) {
                ret[i] = -y[i] / yhat[i] + (1 - y[i]) / (1 - yhat[i]);
            }
        }
        return ret;
    }
};

#endif //MACHINE_LEARNING_BINARYCROSSENTROPY_H
