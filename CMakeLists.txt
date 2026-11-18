cmake_minimum_required(VERSION 4.4)
project(Machine_Learning)

set(CMAKE_CXX_STANDARD 20)

add_executable(Machine_Learning main.cpp
        linalg.cpp
        linalg.h
        MyRandom.h
        Transform.h
        layers.h
        LossFunction.h
        Model.h
        GradientDescent.h
        Optimiser.h
        GradientDescent.cpp
        ActivationFunction.h
        Linear.cpp
        Linear.h
        Sigmoid.cpp
        Sigmoid.h
        Dense.cpp
        Dense.h
        MeanSquaredError.cpp
        MeanSquaredError.h
        BinaryCrossentropy.h
        ReLU.cpp
        ReLU.h
        LeakyReLU.cpp
        LeakyReLU.h
        Softmax.h
        Softmax.cpp
        SparseCategoricalCrossentropy.h
        Model.cpp
        Optimiser.cpp
)
