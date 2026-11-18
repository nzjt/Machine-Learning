#include "Optimiser.h"

void Optimiser::fit(const vector<vector<double>> &x, const vector<vector<double>> &y, int epochs, Model &model, MyRandom &rng, double sample = 1) {
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
        step(model, [&] (const Model &model) { return model.computeDerivatives(m, x, y, doSample); });
        if (t % 500 == 0) {
            cerr << model.loss(x, y) << endl;
        }
    }
}