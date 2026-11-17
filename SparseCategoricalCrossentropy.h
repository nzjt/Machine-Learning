#ifndef MACHINE_LEARNING_SPARSECATEGORICALCROSSENTROPY_H
#define MACHINE_LEARNING_SPARSECATEGORICALCROSSENTROPY_H
#include "LossFunction.h"

template <bool fromLogits = false>
struct SparseCategoricalCrossEntropy : LossFunction {
    [[nodiscard]] double operator()(const vector<double> &y, const vector<double> &yhat) const override {
        if constexpr (fromLogits) {
            return -log(Softmax()(yhat)[(int) y[0]]);
        } else {
            return -log(yhat[(int) y[0]]);
        }
    }

    [[nodiscard]] vector<double> gradient(const vector<double> &y, const vector<double> &yhat) const override {
        int m = yhat.size();
        vector<double> ret;
        int i = y[0];
        if constexpr (fromLogits) {
            ret = Softmax()(yhat);
            --ret[i];
        } else {
            ret.resize(m);
            ret[i] = -1 / yhat[i];
        }
        return ret;
    }
};

#endif //MACHINE_LEARNING_SPARSECATEGORICALCROSSENTROPY_H
