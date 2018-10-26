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


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "include/four_d_concat.hpp"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;




REGISTER_OP("FourDConcat")
    .Attr("T: {float, int32}")
    .Attr("max_disp: int")    
    .Input("left_tensor: T")
    .Input("right_tensor: T")
    .Output("concat_tensor: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims1; 
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &dims1));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({
                                                                c->Dim(dims1, 0),
                                                                c->UnknownDim(),
                                                                c->Dim(dims1, 1),
                                                                c->Dim(dims1, 2), 
                                                                c->Dim(dims1, 3)});
        c->set_output(0, output); 
        return Status::OK();
});




REGISTER_OP("FourDConcatGrad")
    .Attr("T: {float, int32}")
    .Input("gradients: T")
    .Output("grad_left: T")
    .Output("grad_right: T");


template <typename Device, typename T>
class FourDConcatOp : public OpKernel
{
    public:
        explicit FourDConcatOp(OpKernelConstruction* context) 
            : OpKernel(context) 
        { 
            OP_REQUIRES_OK(context, context->GetAttr("max_disp", &m_max_disp));
            OP_REQUIRES(context, m_max_disp >= 0,
                errors::InvalidArgument("Maximum disparity >= 0, got ",
                                        m_max_disp));
        }

        void Compute(OpKernelContext * context) override
        {
            //get the input tensor
            const Tensor left_tensor  = context->input(0);
            const Tensor right_tensor = context->input(1);

            // // Create the output tenso
            Tensor * out_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, 
                                                            TensorShape({left_tensor.dim_size(0),
                                                            m_max_disp,
                                                            left_tensor.dim_size(1),
                                                            left_tensor.dim_size(2),
                                                            left_tensor.dim_size(3) * 2}),
                                                            &out_tensor));

            FourDConcatFunctor<Device, T>()(context->eigen_device<Device>(),
                                                 left_tensor,
                                                 right_tensor,
                                                 out_tensor);
        }


        private:
        int m_max_disp;

};


template <typename Device, typename T>
class FourDConcatGradOp : public OpKernel
{
    public:
        explicit FourDConcatGradOp(OpKernelConstruction* context) 
            : OpKernel(context) 
        {
            
        }

        void Compute(OpKernelContext * context) override
        { 
            const Tensor grad_tensor = context->input(0); 

            Tensor * left_tensor_grad = NULL;
            Tensor * right_tensor_grad = NULL;

            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({grad_tensor.dim_size(0),
                                                                            grad_tensor.dim_size(2),
                                                                            grad_tensor.dim_size(3),
                                                                            grad_tensor.dim_size(4) / 2}),
                                                                            &left_tensor_grad));      

            
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({grad_tensor.dim_size(0),
                                                                            grad_tensor.dim_size(2),
                                                                            grad_tensor.dim_size(3),
                                                                            grad_tensor.dim_size(4) / 2}),
                                                                            &right_tensor_grad));   

            FourDConcatGradFunctor<Device, T>()(context->eigen_device<Device>(),
                                                left_tensor_grad,
                                                right_tensor_grad,
                                                grad_tensor);                   
        }

 
};

template <typename T>
struct FourDConcatFunctor<CPUDevice, T>
{
    void operator()(const CPUDevice & d,
                    const Tensor & left_tensor_,
                    const Tensor & right_tensor_,
                    Tensor * const out_tensor_)
    {
        //loop through all batches

        //then loop through all disparities

        //then for each pixel we concat that feature vector with the reference feature vector

        size_t batches = left_tensor_.dim_size(0);
        size_t height = left_tensor_.dim_size(1);
        size_t width = left_tensor_.dim_size(2);
        size_t features = left_tensor_.dim_size(3);
        size_t max_disp = out_tensor_->dim_size(1);

        auto concat_flat = out_tensor_->flat<T>().data();


        for (size_t batch = 0; batch < batches; ++batch)
        {
            for (size_t disp = 0; disp < max_disp; disp++)
            {  
                auto left_flat = left_tensor_.flat<T>().data();
                auto right_flat = right_tensor_.flat<T>().data();
                left_flat += batch * width * height * features;
                right_flat += batch * width * height * features;
                for (size_t row = 0; row < height; ++row)
                {  
                    for (size_t shifted = 0; shifted < disp; ++shifted)
                    {
                        //map the data entries
                        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> refvec(left_flat, features);
                        Eigen::Matrix<T, Eigen::Dynamic, 1> shiftvec = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(features, 1);
                        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> concatvec(concat_flat, 2 * features);
                        left_flat += features;
                        concat_flat += 2 * features;
                        concatvec << refvec, shiftvec;
                    }
                    
                    for (size_t col = disp; col < width; ++col)
                    {
                        //map the data entries
                        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> refvec(left_flat, features);
                        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> shiftvec(right_flat, features);
                        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> concatvec(concat_flat, 2 * features);
                        left_flat += features;
                        right_flat += features;
                        concat_flat += 2 * features;
                        concatvec << refvec, shiftvec;
                        
                    }
                    right_flat += features * disp;
                    
                }

            }
        }
       

        
    
    }
};


template <typename T>
struct FourDConcatGradFunctor<CPUDevice, T>
{
    void operator()(const CPUDevice & d,
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

        for (size_t batch = 0; batch < batches; ++batch)
        {
            for (size_t disp = 0; disp < max_disp; ++disp)
            {
                //here we need to reset the left and right grad pointers
                auto left_flat = left_tensor_->flat<T>().data();
                auto right_flat = right_tensor_->flat<T>().data();
                left_flat += batch * width * height * features;
                right_flat += batch * width * height * features;
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < disp; ++col)
                    {
                        //here we do not need to pass gradients to the shifted tensors as the gradients will always be zero
                        //create maps of the underlying previous gradient arrays
                        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> ref_grad_vec(grad_flat, features);
                        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> ref_vec(left_flat, features);
                        ref_vec += ref_grad_vec; //pass the gradient back
                        left_flat += features;
                        grad_flat += 2 * features; 
                    }
                    for (size_t col = disp; col < width; ++col)
                    {
                        //create maps of the underlying previous gradient arrays
                        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> ref_grad_vec(grad_flat, features);
                        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> ref_vec(left_flat, features);
                        ref_vec += ref_grad_vec; //pass the gradient back
                        left_flat += features;
                        grad_flat += features; 

                        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> shift_grad_vec(grad_flat, features);
                        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> shift_vec(right_flat, features);
                        shift_vec += shift_grad_vec;
                        right_flat += features;
                        grad_flat += features; 
                    }
                    right_flat += features * disp;
                }

                
            }
        }
        
    }
};



//register kernels
#define REGISTER_CPU(T)                                                         \
    REGISTER_KERNEL_BUILDER(                                              \
        Name("FourDConcat").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
        FourDConcatOp<CPUDevice, T>);                                    \
    REGISTER_KERNEL_BUILDER(                                                 \
        Name("FourDConcatGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
        FourDConcatGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

#if RUN_GPU
#define REGISTER_GPU(T)                                                         \
    extern template struct FourDConcatFunctor<GPUDevice, T>;                    \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("FourDConcat").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
        FourDConcatOp<GPUDevice, T>);                                    \
    extern template struct FourDConcatGradFunctor<GPUDevice, T>;                    \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("FourDConcatGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),   \
        FourDConcatGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif
