#ifndef BATCH_NORM_H
#define BATCH_NORM_H


#include <vector>
#include <cmath>


class BatchNorm {
public:
    template<typename T>
    static std::vector<std::vector<T>> myBatchNorm(const std::vector<std::vector<T>> &x);

    template<typename T>
    static std::vector<std::vector<std::vector<T>>>
    myBatchNorm(const std::vector<std::vector<std::vector<T>>> &x);
};

template<typename T>
std::vector<std::vector<T>> BatchNorm::myBatchNorm(const std::vector<std::vector<T>> &x) {
    std::vector<std::vector<T>> out;
    std::vector<T> mean_values;
    std::vector<T> stand_errors;

    for (unsigned long i = 0; i < x[0].size(); i++) {
        std::vector<T> channel_level_list;
        for (int batch_idx = 0; batch_idx < x.size(); batch_idx++) {
            channel_level_list.push_back(x[batch_idx][i]);
        }
        T sum = std::accumulate(std::begin(channel_level_list), std::end(channel_level_list), 0.001);
        T mean =  sum / channel_level_list.size(); //均值
        T accum  = 0.0;
        std::for_each (std::begin(channel_level_list), std::end(channel_level_list), [&](const double d) {
            accum  += (d-mean)*(d-mean);
        });
        T variance = sqrt(accum/(channel_level_list.size()-1)); //方差

        mean_values.push_back(mean);
        stand_errors.push_back(variance);
    }


    for (int batch_idx = 0; batch_idx < x.size(); batch_idx++) {
        std::vector<T> single_out;
        // iterate through the first vector
        for (unsigned long i = 0; i < x[0].size(); i++) {
            T v = (x[batch_idx][i] - mean_values[i]) / stand_errors[i];
            single_out.push_back(v);
        }
        out.push_back(single_out);
    }

    return out;
}

template<typename T>
std::vector<std::vector<std::vector<T>>>
BatchNorm::myBatchNorm(const std::vector<std::vector<std::vector<T>>> &x) {
    std::vector<std::vector<std::vector<T>>> out;
    std::vector<std::vector<T>> mean_values;
    std::vector<std::vector<T>> stand_errors;

    unsigned long batch_size = x.size();
    unsigned long input_rows = x[0].size();
    unsigned long input_columns = x[0][0].size();



    for (unsigned long i = 0; i < input_rows; i++) {
        std::vector<T> temp_row_means;
        std::vector<T> temp_row_std;
        for (unsigned long j = 0; j < input_columns; j++) {
            std::vector<T> channel_level_list;
            for (int batch_idx = 0; batch_idx < x.size(); batch_idx++) {
                channel_level_list.push_back(x[batch_idx][i][j]);
            }
            T sum = std::accumulate(std::begin(channel_level_list), std::end(channel_level_list), 0.001);
            T mean =  sum / channel_level_list.size(); //均值
            T accum  = 0.0;
            std::for_each (std::begin(channel_level_list), std::end(channel_level_list), [&](const double d) {
                accum  += (d-mean)*(d-mean);
            });
            T variance = sqrt(accum/(channel_level_list.size()-1)); //方差
            temp_row_means.push_back(mean);
            temp_row_std.push_back(variance);
        }
        mean_values.push_back(temp_row_means);
        stand_errors.push_back(temp_row_std);

    }


    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        std::vector<std::vector<T>> single_out;
        // iterate through the first vector
        for (unsigned long row = 0; row < input_rows; row++) {
            std::vector<double> temp_row;
            for (unsigned long column = 0; column < input_columns; column++) {
                T v = (x[batch_idx][row][column] - mean_values[row][column]) / stand_errors[row][column];
                temp_row.push_back(v);
            }

            single_out.push_back(temp_row);
        }
        out.push_back(single_out);
    }


    return out;
}

#endif //BATCHNORM_H