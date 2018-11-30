//
// Created by Jeffery on 2018/11/28.
//

#include "conv_layer.h"
#include "max_pooling_layer.h"

//use for test
#include <iostream>


conv_layer::conv_layer() {
    for (int i = 0; i < 20; i++) {
        kernels[i] = MatrixXd::Random(5, 5);
        update_kernels[i] = MatrixXd::Zero(5, 5);

        zs[i] = MatrixXd::Zero(24, 24);
        activations[i] = MatrixXd::Zero(24, 24);
        deltas[i] = MatrixXd::Zero(24, 24);
    }

    biases = VectorXd::Random(20);
    update_biases = VectorXd::Zero(20);
}

void conv_layer::convolution(MatrixXd input) {
    input_data = input;

    for (int i = 0; i < 20; i++)
        for (int j = 0; j < 24; j++)
            for (int k = 0; k < 24; k++) {
                double product = (input.block<5, 5>(j, k).cwiseProduct(kernels[i])).sum();
                double z = product + biases(i);
                zs[i].coeffRef(j, k) = z;
                activations[i].coeffRef(j, k) = Relu(z);
            }
}

void conv_layer::bp(max_pooling_layer *max_pooling_layer1) {
    MatrixXd *deltas_last = max_pooling_layer1->deltas;

    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 12; j++) {
            for (int k = 0; k < 12; k++) {
                MatrixXd tmp = activations[i].block(j * 2, k * 2, 2, 2);
                MatrixXf::Index maxRow, maxCol;

                tmp.maxCoeff(&maxRow, &maxCol);
                deltas[i].coeffRef(maxRow + j * 2, maxCol + k * 2) = deltas_last[i](j, k);
            }
        }
        update_biases[i] = deltas[i].sum();

        //update kernel
        for (int ii = 0; ii < 5; ii++) {
            for (int jj = 0; jj < 5; jj++) {
                update_kernels[i](ii, jj) = (input_data.block(ii, jj, 24, 24).cwiseProduct(deltas[i])).sum();
            }
        }
    }


}

void conv_layer::update(double eta, int mini_batch) {
    for (int i = 0; i < 20; i++)
        kernels[i] -= eta / mini_batch * update_kernels[i];

    biases -= eta / mini_batch * biases;

    //recover
    for (int i = 0; i < 20; i++) {
        update_kernels[i] *= 0;
        deltas[i] *= 0;
    }

    update_biases *= 0;
}



