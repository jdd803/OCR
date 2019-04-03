import tensorflow as tf
import numpy as np

def tf_flatten(inputs):
    """Flatten tensor"""
    return tf.reshape(inputs, [-1])

def tf_repeat(inputs, repeats, axis=0):
    assert len(inputs.get_shape()) == 1

    a = tf.expand_dims(inputs, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a

def np_flatten(inputs):
    return np.reshape(inputs, [-1])

def np_repeat(inputs, repeats, axis=0):
    assert len(np.shape(inputs)) == 1

    a = np.expand_dims(inputs, -1)
    a = np.tile(a, [1, repeats])
    a = np_flatten(a)
    return a


def ps_roi(features, boxes, pool = True, offsets = None, k = 3, feat_stride = 8):
    pooled_response = tf.py_func(_ps_roi,[features, boxes, pool, offsets, k, feat_stride],tf.float32)
    pooled_fea = tf.convert_to_tensor(pooled_response)
    return pooled_fea

def _ps_roi(features, boxes, pool, offsets, k, feat_stride):
    '''
    Implement the PSROI pooling
    :param features: (1,h,w,2*k^2*(c+1) or (1,h,w,2*K^2*4)
    :param boxes: (n,5)->(0,x1,y1,x2,y2)
    :param pool: control whether ave_pool the features
    :param offsets: (n*k*k*(c+1),2)
    :param k: output size,(x,y)
    :return:(b,k,k,c+1)
    '''
    fea_shape = np.shape(features)
    #num_classes = tf.cast(fea_shape[-1], 'int32') / (k * k)  #channels
    num_classes = fea_shape[-1] / (k * k)  #channels
    depth = num_classes
    boxes_num = np.shape(boxes)[0]
    feature_boxes = np.round(boxes / feat_stride)
    feature_boxes[:,-2:] -= 1  #not include right and bottom edge
    top_left_point = np.hstack((feature_boxes[:,1:3],feature_boxes[:,1:3])).reshape((boxes_num,1,4))
    boxes_part = np.zeros((boxes_num, k * k, 4))  #(n,k^2,4)
    boxes_part[:,] += top_left_point[:,]
    width = (feature_boxes[:,3] - feature_boxes[:,1] + 1) / k   # (n,1)
    height = (feature_boxes[:,4] - feature_boxes[:,2] + 1) / k   # (n,1)

    # split boxes
    for i in range(boxes_num):
        shift_x = np.arange(0, k) * width[i]
        shift_y = np.arange(0, k) * height[i]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        boxes_part[i] += shifts
        boxes_part[i, :, 2] = boxes_part[i, :, 0] + width[i]
        boxes_part[i, :, 3] = boxes_part[i, :, 1] + height[i]
    boxes_part = np.reshape(np.floor(boxes_part),(boxes_num,k*k,-1,4))  #(n,k*k,1,4)

    # add offsets to splitted boxes
    if offsets is not None:
        offsets0 = offsets  #(n*k*k*c,2)
    else:
        offsets0 = np.zeros((boxes_num * k * k * depth, 2))  #(n*k*k*c,2)
    offsets0 = np.reshape(offsets0, (int(boxes_num), int(k * k), int(depth),2))  #(n,k*k,c,2)(x,y,x,y,x,y)
    # offsets1 = tf.stack((offsets0, offsets0),axis=3)
    # offsets1 = tf.reshape(offsets1,(boxes_num, k * k, depth, 4))
    offsets1 = np.tile(offsets0, (1, 1, 1, 2))  #(n,k*k,c,4)
    boxes_part = np.repeat(boxes_part,depth,axis=2)
    boxes_part += offsets1  #(n,k*k,depth,4)
    boxes_part = np.reshape(boxes_part,(int(boxes_num*k*k*depth),4)) #(n*k*k*depth,4)

    # clip split boxes by feature' size
    # temp00 = tf.clip_by_value(boxes_part[..., 0], 0, tf.cast(fea_shape[2],'float32') - 1)
    # temp11 = tf.clip_by_value(boxes_part[..., 1], 0, tf.cast(fea_shape[1],'float32') - 1)
    # temp22 = tf.clip_by_value(boxes_part[..., 2], 0, tf.cast(fea_shape[2],'float32') - 1)
    # temp33 = tf.clip_by_value(boxes_part[..., 3], 0, tf.cast(fea_shape[1],'float32') - 1)
    # boxes_k_offset = tf.stack([temp00, temp11, temp22, temp33], axis=-1)  #(n*k*k*depth,4)
    # boxes_k_offset = tf.reshape(boxes_k_offset,(boxes_num*k*k,depth,4))  #(n*k*k*depth,4)


    # clip split boxes by feature' size
    temp00 = np.clip(boxes_part[..., 0], 0, fea_shape[2] - 1)
    temp11 = np.clip(boxes_part[..., 1], 0, fea_shape[1] - 1)
    temp22 = np.clip(boxes_part[..., 2], 0, fea_shape[2] - 1)
    temp33 = np.clip(boxes_part[..., 3], 0, fea_shape[1] - 1)
    boxes_k_offset = np.stack([temp00,temp11,temp22,temp33],axis=-1)    #(n*k*k*depth,4)
    boxes_k_offset = np.reshape(boxes_k_offset,(int(boxes_num*k*k),int(depth),4))   #(n*k*k,depth,4)


    # num of classes
    all_boxes_num = boxes_num * k * k
    for i in range(all_boxes_num):
        part_k = i % (k * k)
        pooled_fea = map_coordinates(features[0],boxes_k_offset[i],part_k,num_classes,pool)  #(1,depth,1)/(1,depth,n_points)
        try:
            pooled_response = np.concatenate((pooled_response, pooled_fea), 0)
        except UnboundLocalError:
            pooled_response = pooled_fea

    return pooled_response  #(n*k*k,depth,1)/(n*k*k,depth,n_points)


def map_coordinates(inputs,boxes,k,num_classes,pool):
    '''
    Get values in the boxes
    :param inputs: feature map (h,w,2*k^2*(c+1) or (h,w,2*K^2*2)
    :param boxes: (depth,4)(x1,y1,x2,y2) May be fraction
    :param k: relative position
    :param num_classes:
    :param pool: whether ave_pool the features
    :return:
    '''
    # compute box's width and height, both are integer
    width = boxes[0][2] - boxes[0][0] + 1
    height = boxes[0][3] - boxes[0][1] + 1

    depth = np.shape(boxes)[0]
    tp_lf = np.reshape(boxes[:,0:2],(-1,1,2))   #(depth,1,2)
    grid = np.meshgrid(np.array(range(int(height))), np.array(range(int(width))))
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, (1,-1, 2))  #(1,n_points,2)
    coords = grid + tp_lf   #(depth,n_points,2)
    n_coords = np.shape(coords)[1]

    # coords_lt = tf.cast(tf.floor(coords), 'int32')  #(depth,n_points,2)
    # coords_rb = tf.cast(tf.ceil(coords), 'int32')
    # coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    # coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    coords_lt = np.floor(coords)
    coords_rb = np.ceil(coords)
    coords_lb = np.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = np.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)   #(depth,n_points,2)

    idx = np_repeat(range(depth), n_coords)

    def _get_vals_by_coords(input, coords):
        inputs1 = input[:,:,int(k*num_classes):int((k+1)*num_classes)]  #(h,w,depth)
        inputs2 = tf.transpose(inputs1,(2,0,1))  #(depth,h,w)
        indices = np.stack([
            idx,np_flatten(coords[..., 0]), np_flatten(coords[..., 1])
        ], axis=-1)
        # vals1 = tf.gather_nd(inputs2, indices)  #(depth*n_points)
        # vals = tf.reshape(vals1,(depth,n_coords))
        vals = np.take(inputs2,indices)
        vals = np.reshape(vals, (int(depth),int(n_coords)))
        return vals  #(depth,n_points)

    vals_lt = _get_vals_by_coords(inputs, coords_lt)
    vals_rb = _get_vals_by_coords(inputs, coords_rb)
    vals_lb = _get_vals_by_coords(inputs, coords_lb)
    vals_rt = _get_vals_by_coords(inputs, coords_rt)  #(depth,n_points)

    # coords_offset_lt = coords - tf.cast(coords_lt, 'float32')  #(depth,n_points,2)
    coords_offset_lt = coords - coords_lt  # (depth,n_points,2)
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]  # (depth,n_points)
    # def true_fn():
    #     pooled_box = tf.reduce_mean(mapped_vals,axis=1) #(depth,1)
    #     return tf.reshape(pooled_box,(1,depth,-1))
    # def false_fn():
    #     return tf.reshape(mapped_vals, (1, depth, -1))
    #
    #
    # pooled_box = tf.cond(pool, true_fn,false_fn)

    if pool:
        pooled_box = np.mean(mapped_vals, axis=1)   #(depth,1)
        pooled_box = np.reshape(pooled_box, (1, depth, -1))    #(1,depth,1)
    else:
        pooled_box = np.reshape(mapped_vals, (1, depth, -1))    #(1,depth,n)


    return pooled_box  #(1,depth,1)/(1,depth,n_points)