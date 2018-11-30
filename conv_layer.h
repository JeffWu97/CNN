//
// Created by Jeffery on 2018/11/28.
//

#ifndef SIMPLECNN_CONV_LAYER_H
#define SIMPLECNN_CONV_LAYER_H

#include <Eigen/Dense>
#include "utils.h"

class max_pooling_layer;

using namespace Eigen;

class conv_layer {
public:
    //attributes
    MatrixXd input_data;
    MatrixXd kernels[20];
    VectorXd biases;

    MatrixXd zs[20];
    MatrixXd activations[20];
    MatrixXd deltas[20];

    VectorXd update_biases;
    MatrixXd update_kernels[20];


    //methods
    conv_layer();

    void convolution(MatrixXd input);

    void bp(max_pooling_layer *max_pooling_layer1);

    void update(double eta, int mini_batc);
};


#endif //SIMPLECNN_CONV_LAYER_H
