

#include <iostream>
#include <random>
#include <fstream>
#include <chrono>

#include "modules/ConvModule.h"
#include "modules/ReLU.h"
#include "modules/BatchNorm.h"

//! Creates and returns a random vector given the size.
//! Values (-5, 5).
//! \param batch_size the num of input tensor in a batch
//! \param size the desired size of the return vector.
//! \return the random vector
std::vector<std::vector<double>> makeRandomVector(long batch_size, long size) {
    std::vector<std::vector<double>> out;
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    std::default_random_engine re;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        std::vector<double> single_out;

        for (long i = 0; i < size; i++)
            single_out.push_back(unif(re));
        out.push_back(single_out);
    }
    return out;
}

//! Creates and returns a random 2-dimensional vector given the size.
//! \param batch_size the num of input tensor in a batch
//! \param rows how many rows is needed
//! \param columns how many columns is needed
//! \return the random vector
std::vector<std::vector<std::vector<double>>> makeRandom2DVector(long batch_size, long rows, long columns) {

    std::vector<std::vector<std::vector<double>>> out;
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    std::default_random_engine re;

    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        std::vector<std::vector<double>> singel_out;


        for (long i = 0; i < rows; i++) {
            std::vector<double> row;
            for (long j = 0; j < columns; j++)
                row.push_back(unif(re));

            singel_out.push_back(row);
        }
        out.push_back(singel_out);
    }
    return out;
}

//! Creates and returns a single value vector given the size.
//! \param size the desired size of the return vector.
//! \param value the desired value
//! \return the random vector
std::vector<double> makeValueVector(long size, double value) {
    std::vector<double> out;

    for (long i = 0; i < size; i++)
        out.push_back(value);

    return out;
}

//! Creates and returns a 2-dimensional single value vector given the size.
//! \param rows how many rows is needed
//! \param columns how many columns is needed
//! \param value the desired value
//! \return the random vector
std::vector<std::vector<double>> makeValue2DVector(long rows, long columns, double value) {
    std::vector<std::vector<double>> out;

    for (long i = 0; i < rows; i++) {
        std::vector<double> row;
        for (long j = 0; j < columns; j++)
            row.push_back(value);

        out.push_back(row);
    }

    return out;
}

void printFiles(double time, std::vector<std::vector<double>> &input, const std::vector<double> &filter,
                const std::vector<std::vector<double>> &conv) {
    std::ofstream out;
    out.open("results 1d.txt");
    out << "These are the results of the 1d convolution of the input and filter that are shown below." << std::endl
        << std::endl;
    out << "INPUT VECTOR: ";
    for (std::vector<double> i : input) {
        for (double &singe_i : i) {
            out << singe_i << " ";
        }
        out << std::endl << std::endl;
    }
    out << std::endl << std::endl;
    out << "FILTER VECTOR: ";
    for (double i : filter)
        out << i << " ";
    out << std::endl << std::endl;

    out << "CONVOLUTION (time = " << time << " ms) (input*filter) VECTOR: ";
    for (std::vector<double> i : conv) {
        for (double &singe_i : i) {
            out << singe_i << " ";
        }
        out << std::endl << std::endl;
    }
}

void
printFiles(double time, const std::vector<std::vector<std::vector<double>>> &input, const std::vector<std::vector<double>> &filter,
           const std::vector<std::vector<std::vector<double>>> &conv) {
    std::ofstream out;
    out.open("results 2d.txt");
    out << "These are the results of the 1d convolution of the input and filter that are shown below." << std::endl
        << std::endl;
    out << "INPUT ARRAY: " << std::endl;
    for (std::vector<std::vector<double>> r : input) {
        for (std::vector<double> &single_r : r) {
            for (double &i : single_r)
                out << i << " ";
            out << std::endl;
        }
        out << std::endl << std::endl;
    }
    out << std::endl;
    out << "FILTER ARRAY: " << std::endl;
    for (std::vector<double> r : filter) {
        for (double &i : r)
            out << i << " ";

        out << std::endl;
    }
    out << std::endl;

    out << "CONVOLUTION (time = " << time << " ms) (input*filter) ARRAY: " << std::endl;

    for (std::vector<std::vector<double>> r : conv) {
        for (std::vector<double> &single_r : r) {
            for (double &i : single_r)
                out << i << " ";

            out << std::endl;
        }
        out << std::endl << std::endl;
    }
}

void runPredefinedTests() {
    std::vector<std::vector<double>> input_1d;
    std::vector<double> filter_1d;

    unsigned long batch_size;
    unsigned long size = 0;

    std::cout << "please type the batch size : ";
    std::cin >> batch_size;
    std::cout << "please type the length of the 1D Tensor : ";
    std::cin >> size;
    input_1d = makeRandomVector(batch_size, size);
    filter_1d = makeValueVector(5, 1.0 / 5);

    // measure time
    auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> conv_out = Convolution::myConvolve(input_1d, filter_1d);
    std::vector<std::vector<double>> relu_out = ReLU::myReLU(conv_out);
    std::vector<std::vector<double>> bn_out = BatchNorm::myBatchNorm(relu_out);

    auto end = std::chrono::steady_clock::now();

    std::cout << "time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;;

    printFiles(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), input_1d, filter_1d, bn_out);

    /* --------------------------------------------------------------*/
    std::cout << std::endl;

    std::vector<std::vector<std::vector<double>>> input_2d;
    std::vector<std::vector<double>> filter_2d;
    unsigned long rows = 0, columns = 0;

    std::cout << "please type the rows of the input 2D Tensor: ";
    std::cin >> rows;
    std::cout << "please type the columns of the input 2D Tensor: ";
    std::cin >> columns;
    input_2d = makeRandom2DVector(batch_size, rows, columns);
    filter_2d = makeValue2DVector(5, 5, 1.0 / 9);

    // measure time
    start = std::chrono::steady_clock::now();
    std::vector<std::vector<std::vector<double>>> conv_out_2d = Convolution::myConvolve(input_2d, filter_2d);
    std::vector<std::vector<std::vector<double>>> relu_out_2d = ReLU::myReLU(conv_out_2d);
    std::vector<std::vector<std::vector<double>>> bn_out_2d = BatchNorm::myBatchNorm(relu_out_2d);
    end = std::chrono::steady_clock::now();

    std::cout << "time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;

    printFiles(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), input_2d, filter_2d, bn_out_2d);

}

int main() {

    // run the tests
    runPredefinedTests();

    return 0;
}
