/*
 *  MIT License
 *
 *  Copyright (c) 2018 Tom Joy tomjoy@robots.ox.ac.uk
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:

 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.

 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */


#ifndef _FOURD_CONCAT_H_
#define _FOURD_CONCAT_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"



template <typename Device, typename T>
struct FourDConcatFunctor
{
    void operator()(const Device & d,
                    const tensorflow::Tensor & left_tensor_,
                    const tensorflow::Tensor & right_tensor_,
                    tensorflow::Tensor * const concat_tensor_);
};


template <typename Device, typename T>
struct FourDConcatGradFunctor
{
    void operator()(const Device & d,
                    tensorflow::Tensor * const left_tensor_,
                    tensorflow::Tensor * const right_tensor_,
                    const tensorflow::Tensor & grad_tensor_);
};




#if RUN_GPU
//partial specialization 
template <typename T>
struct FourDConcatFunctor<Eigen::GpuDevice, T>
{
    void operator()(const Eigen::GpuDevice & d,
                    const tensorflow::Tensor & left_tensor_,
                    const tensorflow::Tensor & right_tensor_,
                    tensorflow::Tensor * const concat_tensor_);
};


template <typename T>
struct FourDConcatGradFunctor<Eigen::GpuDevice, T>
{
    void operator()(const Eigen::GpuDevice & d,
                    tensorflow::Tensor * const left_tensor_,
                    tensorflow::Tensor * const right_tensor_,
                    const tensorflow::Tensor & grad_tensor_);
};

#endif

#endif //_FEATURE_CORRELATION_H_