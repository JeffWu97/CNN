//
// Created by Jeffery on 2018/11/29.
//

#include "session.h"
#include <iostream>


session::session() {
    index_of_train_data = 0;
    index_of_test_data = 0;
}

Eigen::MatrixXd session::get_train_data() {
    return train_data[index_of_train_data].first;
}

Eigen::VectorXd session::get_train_label() {
    return train_data[index_of_train_data++].second;
}

Eigen::MatrixXd session::get_test_data() {
    return test_data[index_of_test_data].first;
}

Eigen::VectorXd session::get_test_label() {
    return test_data[index_of_test_data++].second;
}

void session::read_csv(std::string file, int number_of_train, int number_of_test) {
    std::ifstream in(file);
    std::string line;

    if (in.is_open()) {
        //read label
        std::getline(in, line);

        //get train data
        for (int i = 0; i < number_of_train; i++) {
            std::getline(in, line);
            int row = 0, col = 0;

            //read a line into a vector and matrix
            std::stringstream ss(line);
            std::string s1;

            std::getline(ss, s1, ',');
            Eigen::VectorXd label = Eigen::VectorXd::Zero(10);
            label(std::stoi(s1)) = 1;

            Eigen::MatrixXd tmp(28, 28);

            while (std::getline(ss, s1, ',')) {
                tmp(row, col++) = stof(s1);
                if (col % 28 == 0) {
                    col = 0;
                    row++;
                }
            }
            train_data.push_back(std::make_pair(tmp, label));
            std::random_shuffle(train_data.begin(), train_data.end());
        }


        //get test data
        for (int i = 0; i < number_of_test; i++) {
            std::getline(in, line);
            int row = 0, col = 0;

            //read a line into a vector and matrix
            std::stringstream ss(line);
            std::string s1;

            std::getline(ss, s1, ',');
            Eigen::VectorXd label = Eigen::VectorXd::Zero(10);
            label(std::stoi(s1)) = 1;

            Eigen::MatrixXd tmp(28, 28);

            while (std::getline(ss, s1, ',')) {
                tmp(row, col++) = stof(s1);
                if (col % 28 == 0) {
                    col = 0;
                    row++;
                }
            }
            test_data.push_back(std::make_pair(tmp, label));
        }
        //close file stream
        in.close();
    } else std::cout << "fail to open file!\n";

}

