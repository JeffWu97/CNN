//
// Created by Jeffery on 2018/11/28.
//

#ifndef SIMPLECNN_FULLY_CONNECTED_LAYER_H
#define SIMPLECNN_FULLY_CONNECTED_LAYER_H

#include <Eigen/Dense>
#include "utils.h"
#include "max_pooling_layer.h"

using namespace Eigen;

class fully_connected_layer {
public:
    //attributes
    MatrixXd weights;
    VectorXd bias;
    VectorXd zs;
    VectorXd activations;
    VectorXd deltas;

    MatrixXd update_weights;
    VectorXd update_bias;
    VectorXd activations_last;

    //methods
    fully_connected_layer();

    void feedforward(max_pooling_layer max_pooling_layer1);

    void bp(VectorXd y);

    void update(double eta ,int mini_batch);


};

#endif //SIMPLECNN_FULLY_CONNECTED_LAYER_H
