#ifndef CONVOLUTION_CONVOLUTION_H
#define CONVOLUTION_CONVOLUTION_H


#include <vector>
#include <cmath>


class Convolution {
public:
    template<typename T>
    static std::vector<std::vector<T>> myConvolve(const std::vector<std::vector<T>> &x, const std::vector<T> &h);

    template<typename T>
    static std::vector<std::vector<std::vector<T>>>
    myConvolve(const std::vector<std::vector<std::vector<T>>> &x, const std::vector<std::vector<T>> &h);
};

template<typename T>
std::vector<std::vector<T>> Convolution::myConvolve(const std::vector<std::vector<T>> &x, const std::vector<T> &h) {
    std::vector<std::vector<T>> out;

    for (int batch_idx = 0; batch_idx < x.size(); batch_idx++) {
        std::vector<T> single_out;
        // iterate through the first vector
        for (unsigned long i = 0; i < x[0].size(); i++) {
            T v = 0;

            // run the filter on the input vector
            for (unsigned long j = 0, j_end = (i < h.size()) ? i + 1 : h.size(); j < j_end; j++)
                v += x[batch_idx][i - j] * h[j];
            single_out.push_back(v);
        }
        out.push_back(single_out);
    }

    return out;
}

template<typename T>
std::vector<std::vector<std::vector<T>>>
Convolution::myConvolve(const std::vector<std::vector<std::vector<T>>> &x, const std::vector<std::vector<T>> &h) {
    std::vector<std::vector<std::vector<T>>> out;

    unsigned long batch_size = x.size();
    unsigned long input_rows = x[0].size();
    unsigned long input_columns = x[0][0].size();

    unsigned long filter_rows = h.size();
    unsigned long filter_columns = h[0].size();

    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        std::vector<std::vector<T>> single_out;
        // iterate through the first vector
        for (unsigned long row = 0; row < input_rows; row++) {
            // if the filter doesn't fit continue
            if (filter_rows + row > input_rows)
                continue;

            std::vector<double> temp_row;

            for (unsigned long column = 0; column < input_columns; column++) {
                // if the filter doesn't fit continue
                if (filter_columns + column > input_columns)
                    continue;

                T v = 0;

                // run the filter on the input vector
                for (unsigned long i = 0; i < filter_rows; i++)
                    for (unsigned long j = 0; j < filter_columns; j++)
                        v += x[batch_idx][row + i][column + j] * h[i][j];

                temp_row.push_back(v);
            }

            single_out.push_back(temp_row);
        }
        out.push_back(single_out);
    }


    return out;
}

#endif //CONVOLUTION_CONVOLUTION_H