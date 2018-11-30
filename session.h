//
// Created by Jeffery on 2018/11/29.
//

#ifndef SIMPLECNN_SESSION_H
#define SIMPLECNN_SESSION_H

#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <string>
#include <algorithm>

class session {

public:
    int index_of_train_data;
    int index_of_test_data;

    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> train_data;
    std::vector<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> test_data;

    session();

    void read_csv(std::string file, int number_of_train, int number_of_test);

    Eigen::MatrixXd get_train_data();

    Eigen::VectorXd get_train_label();

    Eigen::MatrixXd get_test_data();

    Eigen::VectorXd get_test_label();
};


#endif //SIMPLECNN_SESSION_H
