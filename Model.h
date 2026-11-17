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

    void computeDerivatives(const vector<double> &x, const vector<double> &y, Derivatives &accumulator) {
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
        auto &[dJ_dw, dJ_db] = accumulator;
        for (int i = K; i--;) {
            int n = layers[i]->n, m = layers[i]->w[0].size();
            vector<double> dJ_dz = layers[i]->derivative(dJ_da, cache[i].z, cache[i].a);
            const vector<double> &a = i ? cache[i - 1].a : x;
            for (int j = 0; j < n; ++j) {
                auto &rowW = dJ_dw[i][j];
                double dJ_dz_j = dJ_dz[j];
#pragma omp simd
                for (int k = 0; k < m; ++k) {
                    rowW[k] += a[k] * dJ_dz_j;
                }
            }
            dJ_db[i] += dJ_dz;
        }
    }

    Derivatives computeDerivatives(int m, const vector<vector<double>> &x, const vector<vector<double>> &y, double regularisation, const vector<bool> &doSample) {
        int M = x.size(), K = layers.size();
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
            for (int i = 0; i < M; ++i) {
                if (doSample[i]) {
                    computeDerivatives(x[i], y[i], local);
                }
            }
#pragma omp critical
            ret.dJ_dw += local.dJ_dw;
            ret.dJ_db += local.dJ_db;
        }
        ret.dJ_dw /= m;
        ret.dJ_db /= m;
        return ret;
    }

    void fit(const vector<vector<double>> &x, const vector<vector<double>> &y, int epochs, Optimiser &optimiser, double regularisation, MyRandom &rng, double sample = 1) {
        int M = x.size();
        for (int t = epochs; t--;) {
            cerr << t << endl;
            vector<bool> doSample(M);
            int m = 0;
            for (int i = 0; i < M; ++i) {
                if (rng(0, 1) <= sample) {
                    doSample[i] = true;
                    ++m;
                }
            }
            optimiser.optimise(computeDerivatives(m, x, y, regularisation, doSample), layers);
            if (t % 500 == 0) {
                cerr << loss(x, y) << endl;
            }
        }
    }
};

#endif //MACHINE_LEARNING_MODEL_H
