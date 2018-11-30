//
// Created by Jeffery on 2018/11/28.
//

#ifndef SIMPLECNN_MAX_POOLING_LAYER_H
#define SIMPLECNN_MAX_POOLING_LAYER_H

#include <Eigen/Dense>
#include "conv_layer.h"

using namespace Eigen;

class fully_connected_layer;

class max_pooling_layer {
public:
    //attributes
    MatrixXd activations[20];
    MatrixXd deltas[20];


    //methods
    max_pooling_layer();

    ~max_pooling_layer();

    void max_pooling(conv_layer conv_layer1);

    VectorXd flatten();

    void bp(fully_connected_layer *fully_connected_layer1);

};


#endif //SIMPLECNN_MAX_POOLING_LAYER_H
