#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace py = pybind11;



void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t step = m / batch;
    if(m % batch != 0) step++;
    for(size_t s = 0; s < step; s++){
        // X: batch * n
        // y: batch
        // theta: n * k
        // Z: batch * k
        size_t start = s * batch;
        size_t end = std::min((s + 1) * batch, m);
        size_t batch_size = end - start;


        float *Z = new float[batch_size * k];
        for(size_t i= 0; i < batch_size; i++){
            // 从完一个循环，一行就算完了
            for(size_t j = 0; j < k; j++){
                Z[i * k + j] = 0;
                for(size_t l = 0; l < n; l++){
                    Z[i * k + j] += X[(i + start) * n + l] * theta[l * k + j];
                }
            }
            // 算完一行后可以同时求softmax
            for(size_t j = 0; j < k; j++){
                Z[i * k + j] = exp(Z[i * k + j]);
            }
            float sum = 0;
            for(size_t j = 0; j < k; j++){
                sum += Z[i * k + j];
            }
            for(size_t j = 0; j < k; j++){
                Z[i * k + j] /= sum;
            }
        }

        // 计算Z-I_y
        for(size_t i = 0; i < batch_size; i++){
            for(size_t j = 0; j < k; j++){
                Z[i * k + j] -= (j == y[i + start]);
            }
        }

        // 计算梯度
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < k; j++){
                float grad = 0;
                for(size_t l = 0; l < batch_size; l++){
                    grad += X[(l + start) * n + i] * Z[l * k + j];
                }
                theta[i * k + j] -= lr * grad / batch_size;
            }
        }
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
