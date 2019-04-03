from __future__ import absolute_import, division


import tensorflow as tf
from lib.deform_psroi_pooling.ps_roi import ps_roi,tf_repeat,tf_flatten

class PS_roi_offset():

    def __init__(self,features,rois,pool_size,pool,feat_stride,init_normal_stddev = 0.01,**kwargs):
        self.features = features
        self.filters = tf.cast(features.get_shape()[-1],'int32')
        self.rois = rois
        self.pool_size = pool_size   #control nums of relative positions
        self.pool = pool  #control whether ave_pool the ps_score_map
        self.feat_stride = feat_stride
        self.lamda = 0.1


    def call(self, inputs):
        inputs_shape = inputs.get_shape()
        roi_shape = self.rois.get_shape()
        ''''''
        roi_width = self.rois[3] - self.rois[1]    #(1)
        roi_height = self.rois[4] - self.rois[2]    #(1)
        offset_map = tf.keras.layers.Conv2D(self.filters*2,(3,3),padding='same',
                                            use_bias=False,kernel_initializer='zeros')(inputs)
        offset = ps_roi(offset_map,self.rois)  # normalized offset (n*k*k,(c+1)*2)
        offset = tf.reshape(offset,(-1,2)) # normalized offset (n*k*k*(c+1),2)
        repeats = tf.cast(offset.get_shape()[0],'int32')/tf.cast(roi_shape[0],'int32')

        # compute the roi's width and height
        roi_width = tf.reshape(tf_repeat(roi_width,tf.cast(repeats,'int32')), [-1])  #(n*k*k*(c+1),1)
        roi_height = tf.reshape(tf_repeat(roi_height,tf.cast(repeats,'int32')), [-1])  #(n*k*k*(c+1),1)
        roi_width = tf.cast(roi_width,'float32')
        roi_height = tf.cast(roi_height,'float32')

        # transform the normalized offsets to offsets by
        # element-wise product with the roi's width and height
        temp1 = offset[...,0] * roi_width * tf.convert_to_tensor(self.lamda)
        temp2 = offset[...,1] * roi_height * tf.convert_to_tensor(self.lamda)
        offset = tf.stack((temp1,temp2),axis=-1)  #(n*k*k*(c+1),2)
        pooled_response = ps_roi(features=self.features,
                                 boxes=self.rois,pool=self.pool,offsets=offset,
                                 k=self.pool_size,feat_stride=self.feat_stride)   #(n*k*k,depth,n_points)
        return pooled_response
