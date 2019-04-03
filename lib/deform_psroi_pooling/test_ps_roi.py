import numpy as np
import tensorflow as tf
from lib.deform_psroi_pooling.ps_roi import _ps_roi as ps_roi
from lib.deform_psroi_pooling.layer import PS_roi_offset

k=3
feat_stride = 8
boxes = np.array([[0,8,8,65,65],[0,16,16,56,56]])
#boxes = tf.convert_to_tensor(boxes)
offsets1 = range(18*2)
offsets1 = np.reshape(offsets1,(3,3,2*2))
offsets2 = np.zeros((3,3,2*2))
offsets = np.stack((offsets1,offsets2),axis=0)
offsets = np.reshape(offsets,(-1,2))
features = np.ones((1,16,16,18))
ps_roi(features, boxes, False, offsets, k, feat_stride)
#ps_roi_layer = PS_roi_offset(features,boxes,pool_size=3,pool=True,feat_stride=8).call(features)
