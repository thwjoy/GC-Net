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

#if RUN_GPU
 
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "include/four_d_concat.hpp"
#include "include/cuda_helper.cuh"

#include <stdio.h>

using GPUDevice = Eigen::GpuDevice;
using namespace tensorflow;


//how am i going to split these blocks?
// threads 2 * features
//have a block for each disparity and pixel 
template <typename T>
__global__ void concat(const T * const left, const T * const right, T * const out)
{
    int height = gridDim.y;
    int width = gridDim.z;

    int disp = blockIdx.x;
    int row = blockIdx.y;
    int col = blockIdx.z;
    int feat = threadIdx.x;
    int feature_size = blockDim.x / 2;
    T feat_val = 0.0;
    if (feat >= feature_size)
    {
        //this is the right image which needs to be shifted
        if (row < disp) {
            feat_val = 0.0;
        }
        else
        {
            feat_val = right[(col * width * feature_size) + ((row - disp - 1) * feature_size) + feat];
        }
    }
    else
    {
        feat_val = left[(col * width * feature_size) + (row * feature_size) + feat];
    }
      
    out[(disp * height * width * feature_size * 2) + (col * width * feature_size * 2) + row * feature_size * 2 + feat] = feat_val;
}

template <typename T>
__global__ void concatGradLeft(T * const left, const T * const grad)
{
    int width = gridDim.x;
    int height = gridDim.y;
    int feature_size = gridDim.z;

    int row = blockIdx.x;
    int col = blockIdx.y;
    int feat = blockIdx.z;

    int disp = threadIdx.x;

    //put all values for this gradient element into shared memory, then we sum them up
    SharedMemory<T> shared;
    T * shared_data = shared.getPointer();
    T concat_val = grad[(disp * height * width * feature_size * 2) + (row * width * feature_size * 2) + col * feature_size * 2 + feat];
    shared_data[disp] = concat_val;
    __syncthreads();

    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            shared_data[index] += shared_data[index + s];
        }
        __syncthreads();
    }

    T * p = left + (row * width * feature_size) + (col * feature_size) + feat;
    if (threadIdx.x == 0) atomicAdd(p, shared_data[0]);
}

template <typename T>
__global__ void concatGradRight(T * const right, const T * const grad)
{
    int width = gridDim.x;
    int height = gridDim.y;
    int feature_size = gridDim.z;

    int row = blockIdx.x;
    int col = blockIdx.y;
    int feat = blockIdx.z;

    int disp = threadIdx.x;

    //put all values for this gradient element into shared memory, then we sum them up
    SharedMemory<T> shared;
    T * shared_data = shared.getPointer();
    T concat_val = 0;
    if (col + disp < width) concat_val = grad[(disp * height * width * feature_size * 2) + (row * width * feature_size * 2) + (col + disp) * feature_size * 2 + feature_size + feat];
    shared_data[disp] = concat_val;
    __syncthreads();

    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            shared_data[index] += shared_data[index + s];
        }
        __syncthreads();
    }

    T * p = right + (row * width * feature_size) + (col * feature_size) + feat;
    if (threadIdx.x == 0) atomicAdd(p, shared_data[0]);

}




template <typename T>
void FourDConcatFunctor<GPUDevice, T>::operator()(const GPUDevice & d,
    const tensorflow::Tensor & left_tensor_,
    const tensorflow::Tensor & right_tensor_,
    tensorflow::Tensor * const out_tensor_)
{

    size_t batches = left_tensor_.dim_size(0);
    size_t height = left_tensor_.dim_size(1);
    size_t width = left_tensor_.dim_size(2);
    size_t features = left_tensor_.dim_size(3);
    size_t max_disp = out_tensor_->dim_size(1);

    auto left_flat = left_tensor_.flat<T>().data();
    auto right_flat = right_tensor_.flat<T>().data();
    auto concat_flat = out_tensor_->flat<T>().data();


    int threads_per_block = 2 * features;
    dim3 num_blocks(max_disp, height, width);

    for (size_t i = 0; i < batches; ++i)
    { 
        concat<<<num_blocks, threads_per_block>>>(left_flat, right_flat, concat_flat);
        left_flat += height * width * features;
        right_flat += height * width * features;
        concat_flat += max_disp * height * width * features * 2;
    }
    



}

template <typename T>
void FourDConcatGradFunctor<GPUDevice, T>::operator()(const GPUDevice & d,
    tensorflow::Tensor * const left_tensor_,
    tensorflow::Tensor * const right_tensor_,
    const tensorflow::Tensor & grad_tensor_)
{
    size_t batches = left_tensor_->dim_size(0);
    size_t height = left_tensor_->dim_size(1);
    size_t width = left_tensor_->dim_size(2);
    size_t features = left_tensor_->dim_size(3);
    size_t max_disp = grad_tensor_.dim_size(1);

    auto grad_flat = grad_tensor_.flat<T>().data();
    auto left_flat = left_tensor_->flat<T>().data();
    auto right_flat = right_tensor_->flat<T>().data();

    int threads_per_block(max_disp);
    dim3 num_blocks(height, width, features);

    for (size_t batch = 0; batch < batches; ++batch)
    {
        concatGradLeft<<<num_blocks, threads_per_block, max_disp*sizeof(T)>>>(left_flat, grad_flat);
        concatGradRight<<<num_blocks, threads_per_block, max_disp*sizeof(T)>>>(right_flat, grad_flat);
        left_flat += height * width * features;
        right_flat += height * width * features;
        grad_flat += max_disp * height * width * features * 2;
    }
    
}



template struct FourDConcatFunctor<GPUDevice, float>;
template struct FourDConcatFunctor<GPUDevice, int32>;
template struct FourDConcatGradFunctor<GPUDevice, float>;
template struct FourDConcatGradFunctor<GPUDevice, int32>;

#endif