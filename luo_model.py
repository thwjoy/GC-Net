"""
MIT License

Copyright (c) 2017 Alessio Tonioni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from utils.layers import conv2d, correlation_tf, correlation_native
import tensorflow as tf

class luo_model():
    """
    Class to implement the matching network from "Efficient Deep Learning for Stereo Matching" in fully convolutional Way.
    """

    def __init__(self, left_batch, right_batch, conv_kernel_size=5, receptive_field_size=37, max_disp=256, training=False, padding=False, reuse=False, correlation_mode='TF', normalize_feature=False, disp_side='left'):
        """
        Constructor for luo model.
        Args:
            left_batch: batch of left images
            right_batch: batch of right images
            conv_kernel_size: dimension fo the convolutional kernel
            receptive_field_size: size of the support used for stereo matching
            max_disp: maximum disparity range
            training: flag to enable training mode
            padding: flag to enable padding of the input 
            reuse: flag to use already existing variables
            correlation_mode: choose which correlation implementation to use, either CUDA or TF
            normalize_feature: perform l2 normalization of features before the correlation computation
            disp_side: choose which disparity to compute ['left', 'right', 'both']
        """
        self.left_batch = left_batch
        self.right_batch = right_batch
        self.kernel_size = conv_kernel_size
        self.receptive_field_size = receptive_field_size
        self.max_disp = max_disp
        self.is_training = training
        self.padding = padding
        self.reuse=reuse
        self.tf_correlation=(correlation_mode=='TF')           
        self.normalize = normalize_feature
        self.disp_side=disp_side
        self._build_model()
    
    @staticmethod
    def soft_argmax(volume):
        """
        Apply the soft argmax operation from GCNET paper to volume to get a differentiable disparity regression starting from a probability distribution.
        Args:
            volume: tensor with the logits corresponding to the unscaled probability distribution of each disparity per pixel, shape [batch_size,height,width,maxDisp]
        Returns:
            disparity_estimation: tensor with the predicted disparity for each pixel as a float value, shape [batch_size, height, width, 1] 
        """
        volume_shape = tf.shape(volume)
        batch_size = volume_shape[0]
        h = volume_shape[1]
        w = volume_shape[2]
        maxDisp = volume_shape[3]

        disparity_probabilities = tf.nn.softmax(volume, axis=-1)

        #build a tensor containing the disparity range
        disparity_range = tf.linspace(0., tf.cast(maxDisp-1,tf.float32),maxDisp)
        #inflate it to be 4 dimensional [1,1,1,maxDisp]
        disparity_range_inflated = tf.expand_dims(tf.expand_dims(tf.expand_dims(disparity_range,axis=0),axis=0),axis=0)
        #tile it to get a disparity volume
        disparity_volume = tf.tile(disparity_range_inflated,[batch_size,h,w,1])
        
        disparity_estimation = tf.reduce_sum(tf.multiply(disparity_probabilities, disparity_volume),-1,keepdims=True) 
        
        return disparity_estimation    

    def pad(self,x, mode='CONSTANT'):
        """
        Pad x with half the receptive field size
        Args:
            x: input tensor
            mode: how to perform padding, see tensorflow docs for available methods
        Returns:
            x: padded input tensor
        """
        pad_size = self.receptive_field_size//2
        x = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]], mode=mode)
        return x

    def _build_feature_extractor(self, x, reuse=False):
        """
        Build a CNN for convolutional feature extraction
        Args:
            x: input tensor
            reuse: boolean, if true reuse already allocated variables, otherway make new ones
        Returns:
            x: activation of last convolutional layer
        """
        if self.padding:
            x = self.pad(x)
        num_convolution = (self.receptive_field_size-1)//(self.kernel_size-1)
        in_channel = x.get_shape()[-1].value
        
        #Feature Extraction
        for i in range(num_convolution-1):
            if i==0:
                kernel = [self.kernel_size,self.kernel_size,in_channel,32]
            elif i<2:
                kernel = [self.kernel_size,self.kernel_size,32,32]
            elif i==2:
                kernel = [self.kernel_size,self.kernel_size,32,64]
            else:
                kernel = [self.kernel_size,self.kernel_size,64,64]    
            
            x = conv2d(x,kernel,activation=lambda x:tf.nn.relu(x),padding='VALID',reuse=reuse,batch_norm=True,training=self.is_training,name='conv{}'.format(i))
        
        #Final Layer
        x = conv2d(x,[self.kernel_size,self.kernel_size,64,64],activation=lambda x:x,padding='VALID',reuse=reuse,batch_norm=True,training=self.is_training,name='conv_final')

        return x
        
    def _build_model(self):
        """
        Build the CNN model for matching cost computation
        Returns:
            Cost volume computed as correlation between convolutional feature extracted from left and right image
        """
        #Extract feature
        with tf.variable_scope('feature_extractor'):
            self.left_features = self._build_feature_extractor(self.left_batch,reuse=self.reuse)
            self.right_features = self._build_feature_extractor(self.right_batch,reuse=True)

        if self.normalize:
            self.left_features = tf.nn.l2_normalize(self.left_features, axis=-1)
            self.right_features = tf.nn.l2_normalize(self.right_features, axis=-1)

        #Compute cost Volume
        #self.cost_volume_tf = correlation_tf(self.left_features,self.right_features,self.max_disp)
        #self.cost_volume_cuda = correlation_native(self.left_features,self.right_features,self.max_disp,'left')
        if self.tf_correlation:
            self.cost_volume = correlation_tf(self.left_features,self.right_features,self.max_disp, disp_side=self.disp_side)
        else:
            self.cost_volume = correlation_native(self.left_features,self.right_features,self.max_disp, disp_side=self.disp_side)
