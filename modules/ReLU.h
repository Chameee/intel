#ifndef RELU_H
#define RELU_H


#include <vector>
#include <cmath>


class ReLU {
public:
    template<typename T>
    static std::vector<std::vector<T>> myReLU(const std::vector<std::vector<T>> &x);

    template<typename T>
    static std::vector<std::vector<std::vector<T>>>
    myReLU(const std::vector<std::vector<std::vector<T>>> &x);
};

template<typename T>
std::vector<std::vector<T>> ReLU::myReLU(const std::vector<std::vector<T>> &x) {
    std::vector<std::vector<T>> out;

    for (int batch_idx = 0; batch_idx < x.size(); ++batch_idx) {
        std::vector<T> single_out;
        // iterate through the first vector
        for (unsigned long i = 0; i < x[0].size(); i++) {
            T v = 0.0;
            if (x[batch_idx][i] > 0){
                v += x[batch_idx][i];
            }
            single_out.push_back(v);
        }
        out.push_back(single_out);
    }

    return out;
}

template<typename T>
std::vector<std::vector<std::vector<T>>>
ReLU::myReLU(const std::vector<std::vector<std::vector<T>>> &x) {
    std::vector<std::vector<std::vector<T>>> out;

    unsigned long batch_size = x.size();
    unsigned long input_rows = x[0].size();
    unsigned long input_columns = x[0][0].size();
    std::cout << "input_rows:" << input_rows << "，input_cols:" << input_columns << std::endl;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        std::vector<std::vector<T>> single_out;
        // iterate through the first vector
        for (unsigned long row = 0; row < input_rows; row++) {
            std::vector<double> temp_row;

            for (unsigned long column = 0; column < input_columns; column++) {
                T v = 0.0;

                if (x[batch_idx][row][column] > 0){
                    v += x[batch_idx][row][column];
                }

                temp_row.push_back(v);
            }
            single_out.push_back(temp_row);
        }
        out.push_back(single_out);
    }


    return out;
}

#endif //END RELU