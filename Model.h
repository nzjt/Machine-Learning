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
    LossFunction *lossFunction;
    double regularisation = 0;

    Model(int inputSize, const vector<Dense*> &layers) : layers(layers) {
        this->layers[0]->resize(inputSize);
        for (int i = 1; i < layers.size(); ++i) {
            this->layers[i]->resize(layers[i - 1]->n);
        }
    }

    void setLossFunction(LossFunction *lossFunction) {
        this->lossFunction = lossFunction;
    }

    void setRegularisation(double regularisation) {
        this->regularisation = regularisation;
    }

    void randomise(MyRandom &rng) {
        for (auto &layer: layers) {
            layer->randomise(rng);
        }
    }

    void read(ifstream &file) {
        uint32_t k;
        file.read((char*) &k, sizeof k);
        assert(k == layers.size());
        for (auto layer: layers) {
            layer->read(file);
        }
    }

    void write(ofstream &file) const {
        uint32_t k = layers.size();
        file.write((char*) &k, sizeof k);
        for (auto layer: layers) {
            layer->write(file);
        }
    }

    [[nodiscard]] vector<double> predict(const vector<double> &x) const {
        auto activations = x;
        for (const auto &layer : layers) {
            activations = layer->predict(activations);
        }
        return activations;
    }

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

    Derivatives computeDerivatives(const vector<double> &x, const vector<double> &y) {
        struct Cache {
            vector<double> z, a;
        };
        int K = layers.size();
        vector<Cache> cache(K);
        auto activations = x;
        for (int i = 0; i < K; ++i) {
            cache[i].z = layers[i]->computeZ(activations);
            cache[i].a = (*layers[i]->activationFunction)(cache[i].z);
            activations = cache[i].a;
        }
        vector<double> dJ_da = lossFunction->gradient(y, activations);
        Derivatives ret;
        auto &[dJ_dw, dJ_db] = ret;
        dJ_dw.resize(K);
        dJ_db.resize(K);
        for (int i = K; i--;) {
            int n = layers[i]->n;
            vector<double> dJ_dz = layers[i]->derivative(dJ_da, cache[i].z, cache[i].a);
            const vector<double> &a = i ? cache[i - 1].a : x;
            dJ_dw[i].resize(n);
            for (int j = 0; j < n; ++j) {
                dJ_dw[i][j] = a * dJ_dz[j];
            }
            dJ_db[i] = dJ_dz;
        }
        return ret;
    }

    Derivatives computeDerivatives(const vector<vector<double>> &x, const vector<vector<double>> &y, double regularisation) {
        int m = x.size(), K = layers.size();
        Derivatives ret;
        ret.dJ_dw.resize(K);
        ret.dJ_db.resize(K);
        for (int i = 0; i < K; ++i) {
            ret.dJ_dw[i] = layers[i]->w * regularisation;
            ret.dJ_db[i] = layers[i]->b * regularisation;
        }
        Derivatives empty;
        empty.dJ_dw.resize(K);
        empty.dJ_db.resize(K);
        for (int i = 0; i < K; ++i) {
            empty.dJ_dw[i] = vector<vector<double>>(layers[i]->w.size(), vector<double>(layers[i]->w[0].size()));
            empty.dJ_db[i] = vector<double>(layers[i]->b.size());
        }
#pragma omp parallel
        {
            Derivatives local = empty;
#pragma omp for nowait
            for (int i = 0; i < m; ++i) {
                auto [dJ_dw, dJ_db] = computeDerivatives(x[i], y[i]);
                local.dJ_dw += dJ_dw;
                local.dJ_db += dJ_db;
            }
#pragma omp critical
            ret.dJ_dw += local.dJ_dw;
            ret.dJ_db += local.dJ_db;
        }
        ret.dJ_dw /= m;
        ret.dJ_db /= m;
        return ret;
    }

    void fit(const vector<vector<double>> &x, const vector<vector<double>> &y, int epochs, Optimiser &optimiser, double regularisation, __gnu_cxx::sfmt19937 &rng, int k = -1) {
        if (k == -1) {
            k = x.size();
        }
        for (int t = epochs; t--;) {
            cerr << t << endl;
            vector<vector<double>> xSubset(k), ySubset(k);
            vector<int> order(x.size());
            iota(order.begin(), order.end(), 0);
            shuffle(order.begin(), order.end(), rng);
            for (int i = 0; i < k; ++i) {
                xSubset[i] = x[order[i]];
                ySubset[i] = y[order[i]];
            }
            optimiser.optimise(computeDerivatives(xSubset, ySubset, regularisation), layers);
            if (t % 100 == 0) {
                cerr << loss(x, y) << endl;
            }
        }
    }
};

#endif //MACHINE_LEARNING_MODEL_H
