#ifndef MACHINE_LEARNING_MODEL_H
#define MACHINE_LEARNING_MODEL_H
#include "Dense.h"
#include "LossFunction.h"
#include "Optimiser.h"
#include <bits/extc++.h>

using namespace std;

struct Derivatives {
    vector<vector<vector<double>>> dJ_dw;
    vector<vector<double>> dJ_db;
};

struct Model {
    vector<Dense*> layers;
    LossFunction *lossFunction = nullptr;
    double regularisation = 0;

    Model(int inputSize, const vector<Dense*> &layers);

    void setLossFunction(LossFunction *lossFunction);

    void setRegularisation(double regularisation);

    void randomise(MyRandom &rng);

    void read(ifstream &file);

    void write(ofstream &file) const;

    [[nodiscard]] vector<double> predict(const vector<double> &x) const;

    template <class T>
    vector<T> predict(const vector<T> &x) const {
        vector<T> ret;
        for (const auto &xi: x) {
            ret.push_back(predict(xi));
        }
        return ret;
    }

    template <class T>
    [[nodiscard]] double loss(const vector<vector<T>> &x, const vector<vector<T>> &y) const {
        double loss = (*lossFunction)(y, predict(x));
        double L2 = 0;
        for (const auto &layer : layers) {
            for (const auto &row: layer->w) {
                L2 += dot(row, row);
            }
            L2 += dot(layer->b, layer->b);
        }
        return loss + L2 * regularisation / x.size() / 2;
    }

    void computeDerivatives(const vector<double> &x, const vector<double> &y, Derivatives &accumulator);

    Derivatives computeDerivatives(int m, const vector<vector<double>> &x, const vector<vector<double>> &y, const vector<bool> &doSample);

    void fit(const vector<vector<double>> &x, const vector<vector<double>> &y, int epochs, Optimiser &optimiser, MyRandom &rng, double sample = 1);
};

#endif //MACHINE_LEARNING_MODEL_H
