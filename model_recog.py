# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.contrib import learn


#
'''
tf.layers.conv2d(inputs, filters, kernel_size, strides=(1,1),
                 padding='valid', data_format='channels_last',
                 dilation_rate=(1,1), use_bias=True,
                 kernel_initializer=None, bias_initializer=init_ops.zeros_initializer(), 
                　kernel_regularizer=None, bias_regularizer=None, 
                 activation=None, activity_regularizer=None,
                 trainable=True,　name=None, reuse=None)
'''
#
def conv_layer(inputs, params, training):
    '''define a convolutional layer with layer_params'''
    #
    # 输入数据维度为 4-D tensor: [batch_size, width, height, channels]
    #                           [batch_size, height, width, channels]
    #
    #  Layer params:   Filts  K  Strides  Padding   Name   BatchNorm?
    # layer_params = [[  64, 3,  (1,1),  'same',  'conv1',  False], 
    #                 [  64, 3,  (3,3),  'same',  'conv2',  True]]
    #
    batch_norm = params[5] # Boolean
    #
    if batch_norm:
        activation = None
    else:
        activation = tf.nn.relu
    #
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    outputs = tf.layers.conv2d(inputs, 
                               filters = params[0],
                               kernel_size = params[1],
                               strides = params[2],
                               padding = params[3],
                               activation = activation,
                               kernel_initializer = kernel_initializer,
                               bias_initializer = bias_initializer,
                               name = params[4])
    #
    if batch_norm:
        outputs = norm_layer(outputs, training, params[4]+'/batch_norm')
        outputs = tf.nn.relu(outputs, name = params[4]+'/relu')
    #
    return outputs
#
def norm_layer(inputs, training, name):
    '''define a batch-norm layer'''
    return tf.layers.batch_normalization(inputs, axis = 3, # channels last,
                                         training = training,
                                         name = name)
#
'''
tf.layers.max_pooling2d(inputs, pool_size, strides,
                        padding='valid', data_format='channels_last', name=None)
'''
#
def maxpool_layer(inputs, size, stride, padding, name):
    '''define a max-pooling layer'''
    return tf.layers.max_pooling2d(inputs, size, stride,
                                   padding = padding,
                                   name = name)
#
'''
tf.layers.average_pooling2d(inputs, pool_size, strides,
                            padding='valid', data_format='channels_last', name=None)
'''
def averpool_layer(inputs, size, stride, padding, name):
    '''define a average-pooling layer'''
    return tf.layers.average_pooling2d(inputs, size, stride,
                                   padding = padding,
                                   name = name)
#
# tf.pad(tensor, paddings, mode='CONSTANT', name=None)
#
# 't' is [[1, 2, 3], [4, 5, 6]].
# 'paddings' is [[1, 1,], [2, 2]].
# rank of 't' is 2.
#
def pad_layer(tensor, paddings, mode='CONSTANT', name=None):
    ''' define padding layer '''
    return tf.pad(tensor, paddings, mode, name)
#
#
def block_resnet(inputs, layer_params, training, name):
    '''define resnet block'''
    #    
    shape_in = inputs.get_shape().as_list()
    #
    short_cut = inputs   # 1，图像大小不缩小，深度不加深
    #                    # 2，图像大小只能降，1/2, 1/3, 1/4, ...
    #                    # 3，深度，卷积修改
    #
    with tf.variable_scope(name):
        #
        for item in layer_params:
            inputs = conv_layer(inputs, item, training)
        #
        shape_out = inputs.get_shape().as_list()
        #
        # 图片大小，缩小
        if shape_in[1] != shape_out[1] or shape_in[2] != shape_out[2]:
            #
            size = [shape_in[1]//shape_out[1], shape_in[2]//shape_out[2]]
            #
            short_cut = maxpool_layer(short_cut, size, size, 'valid', 'shortcut_pool')
            #
        #
        # 深度
        if shape_in[3] != shape_out[3]:
            #
            item = [shape_out[3], 1, (1,1), 'same', 'shortcut_conv', False]
            #
            short_cut = conv_layer(short_cut, item, training)
            #
        #
        outputs = tf.nn.relu(inputs + short_cut, 'last_relu')                
    #    
    return outputs
#
def bottleneck_block(inputs, depth_arr, name):
    '''define bottleneck block'''
    #
    #shape_in = inputs.get_shape().as_list()
    #
    #short_cut = inputs
    #
    with tf.variable_scope(name):
        #
        out = tf.layers.conv2d(inputs, depth_arr[0], 1, (1,1), 'same',
                               activation = tf.nn.relu, name = 'conv1')
        out = tf.layers.conv2d(out, depth_arr[1], 3, (1,1), 'same',
                               activation = tf.nn.relu, name = 'conv2')
        out = tf.layers.conv2d(out, depth_arr[2], 1, (1,1), 'same',
                               activation = None, name = 'conv3')
        #
        outputs = tf.nn.relu(inputs + out, 'last_relu')            
    #    
    return outputs
#
#
def incept_block(inputs, K, depth_arr, training, name):
    ''' define inception-like block '''
    #
    with tf.variable_scope(name):
        #
        params_1 = [depth_arr[0], [1, K], (1,1), 'same',  'branch1', False]
        params_2 = [depth_arr[1], [K, 1], (1,1), 'same',  'branch2', False]
        params_3_1 = [depth_arr[2], [1, K], (1,1), 'same',  'branch3_1', False]
        params_3_2 = [depth_arr[3], [K, 1], (1,1), 'same',  'branch3_2', False]
        params_4 = [depth_arr[4], [K, K], (1,1), 'same',  'branch4', False]
        #
        branch_1 = conv_layer(inputs, params_1, training)
        branch_2 = conv_layer(inputs, params_2, training)
        branch_3 = conv_layer(inputs, params_3_1, training)
        branch_3 = conv_layer(branch_3, params_3_2, training)
        branch_4 = conv_layer(inputs, params_4, training)
        #
        outputs = tf.concat([branch_1, branch_2, branch_3, branch_4], 3)
    #
    return outputs
#
'''
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                              sequence_length = None, # 输入序列的实际长度（可选，默认为输入序列的最大长度）
                                                      # sequence_length must be a vector of length batch_size
                              initial_state_fw = None,  # 前向的初始化状态（可选）
                              initial_state_bw = None,  # 后向的初始化状态（可选）
                              dtype = None, # 初始化和输出的数据类型（可选）
                              parallel_iterations = None,
                              swap_memory = False,
                              time_major = False, 
                              # 决定了输入输出tensor的格式：如果为true, 向量的形状必须为 `[max_time, batch_size, depth]`. 
                              # 如果为false, tensor的形状必须为`[batch_size, max_time, depth]`. 
                              scope = None)
返回值：一个(outputs, output_states)的元组
其中，
1. outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。
假设time_major = false, tensor的shape为[batch_size, max_time, depth]。
实验中使用tf.concat(outputs, 2)将其拼接。
2. output_states为(output_state_fw, output_state_bw)，包含了前向和后向最后的隐藏状态的组成的元组。
output_state_fw和output_state_bw的类型为LSTMStateTuple。
LSTMStateTuple由（c，h）组成，分别代表memory cell和hidden state。
'''
def rnn_layer(input_sequence, sequence_length, rnn_size, scope):
    '''build bidirectional (concatenated output) lstm layer'''
    #
    weight_initializer = tf.truncated_normal_initializer(stddev = 0.01)
    #
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer = weight_initializer)
    #
    # Include?
    #cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw, 
    #                                         input_keep_prob=dropout_rate )
    #cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw, 
    #                                         input_keep_prob=dropout_rate )
    
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = True,
                                                    dtype = tf.float32,
                                                    scope = scope)
    
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    rnn_output_stack = tf.concat(rnn_output, 2, name = 'output_stack')
    #rnn_output_stack = rnn_output[0] + rnn_output[1]
    
    return rnn_output_stack
#
def gru_layer(input_sequence, sequence_length, rnn_size, scope):
    '''build bidirectional (concatenated output) lstm layer'''
    
    # Default activation is tanh
    cell_fw = tf.contrib.rnn.GRUCell(rnn_size)
    cell_bw = tf.contrib.rnn.GRUCell(rnn_size)
    #
    # tf.nn.rnn_cell.GRUCell(num_units, input_size=None, activation=<function tanh>).
    # tf.contrib.rnn.GRUCell
    #
    # Include?
    #cell_fw = tf.contrib.rnn.DropoutWrapper( cell_fw, 
    #                                         input_keep_prob=dropout_rate )
    #cell_bw = tf.contrib.rnn.DropoutWrapper( cell_bw, 
    #                                         input_keep_prob=dropout_rate )
    
    rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_sequence,
                                                    sequence_length = sequence_length,
                                                    time_major = True,
                                                    dtype = tf.float32,
                                                    scope = scope)
    
    # Concatenation allows a single output op because [A B]*[x;y] = Ax+By
    # [ paddedSeqLen batchSize 2*rnn_size]
    rnn_output_stack = tf.concat(rnn_output, 2, name = 'output_stack')
    #rnn_output_stack = rnn_output[0] + rnn_output[1]
    
    return rnn_output_stack

#
#
# model
#
def conv_feat_layers(inputs, width, mode):
    #
    training = (mode == learn.ModeKeys.TRAIN)
    #
    layer_params = [ [  64, 3, (1,1),  'same',  'conv1', False], # pool
                     [ 128, 3, (1,1),  'same',  'conv2', False], # pool
                     [ 256, 3, (1,1),  'same',  'conv3', False],
                     [ 256, 3, (1,1),  'same',  'conv4', False], # padding
                     [ 512, 3, (1,1),  'same',  'conv5',  True], 
                     [ 512, 3, (1,1),  'same',  'conv6',  True], 
                     [ 512, 2, (1,1),  'same',  'conv7', False]] #
    #
    # inputs should have shape [ batch, 32, width, channel]
    #
    with tf.variable_scope("conv_feat"): # h,w
        #
        conv1 = conv_layer(inputs, layer_params[0], training ) # 32, w
        padd1 = pad_layer(conv1, [[0,0],[0,0],[0,1],[0,0]], name='padd1')
        pool1 = maxpool_layer(padd1, 2, 2, 'valid', 'pool1')   # 16, ceil(w/2)
        #
        conv2 = conv_layer( pool1, layer_params[1], training ) # 16, ceil(w/2)
        padd2 = pad_layer(conv2, [[0,0],[0,0],[0,1],[0,0]], name='padd2')
        pool2 = maxpool_layer(padd2, 2, 2, 'valid', 'pool2' )  # 8, ceil(ceil(w/2)/2)
        #
        conv3 = conv_layer( pool2, layer_params[2], training ) # 8, ceil(ceil(w/2)/2)
        conv4 = conv_layer( conv3, layer_params[3], training ) # 8, ceil(ceil(w/2)/2)
        padd3 = pad_layer(conv4, [[0,0],[0,0],[0,1],[0,0]], name='padd3')
        pool3 = maxpool_layer(padd3, 2, 2, 'valid', 'pool3')   # 4, ceil(ceil(ceil(w/2)/2)/2)
        #
        conv5 = conv_layer( pool3, layer_params[4], training ) # 4, ceil(ceil(ceil(w/2)/2)/2)    
        conv6 = conv_layer( conv5, layer_params[5], training ) # 4, ceil(ceil(ceil(w/2)/2)/2)            
        pool4 = maxpool_layer(conv6, [2,1], [2,1], 'valid', 'pool4')    # 2, ceil(ceil(ceil(w/2)/2)/2)
        #
        conv7 = conv_layer( pool4, layer_params[6], training ) # 2, ceil(ceil(ceil(w/2)/2)/2)
        pool5 = maxpool_layer(conv7, [2,1], [2,1], 'valid', 'pool5')    # 1, ceil(ceil(ceil(w/2)/2)/2)
        #
        features = tf.squeeze(pool5, axis = 1, name = 'features') # squeeze row dim
        # tf.expand_dims()
        #
        #print(features.shape)
        #
        # Calculate resulting sequence length from original image widths
        #
        two = tf.constant(2, dtype=tf.float32, name='two')
        #
        w = tf.cast(width, tf.float32)
        #
        w = tf.div(w, two)
        w = tf.ceil(w)
        #
        w = tf.div(w, two)
        w = tf.ceil(w)
        #
        w = tf.div(w, two)
        w = tf.ceil(w)
        #
        w = tf.cast(w, tf.int32)
        #
        # Vectorize
        sequence_length = tf.reshape(w, [-1], name='seq_len') 
        #
        
        #
        return features, sequence_length
        #
#
def rnn_recog_layers(features, sequence_length, num_classes):
    #
    # Input features is [batchSize paddedSeqLen numFeatures]
    #
    #
    rnn_size = 512
    #
    logit_activation = tf.nn.sigmoid
    #
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    with tf.variable_scope("rnn_recog"):
        #
        # Transpose to time-major order for efficiency
        #  --> [paddedSeqLen batchSize numFeatures]
        #
        rnn_sequence = tf.transpose(features, perm = [1, 0, 2], name = 'time_major')
        #
        rnn1 = rnn_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
        rnn2 = rnn_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
        #
        rnn_logits = tf.layers.dense(rnn2, num_classes,
                                     activation = logit_activation,
                                     kernel_initializer = weight_initializer,
                                     bias_initializer = bias_initializer,
                                     name = 'logits')
        # dense operates on last dim
        #
        
        #
    return rnn_logits
#
#
def ctc_loss_layer(sequence_labels, rnn_logits, sequence_length):
    #
    loss = tf.nn.ctc_loss(sequence_labels, rnn_logits, sequence_length,
                          time_major = True )
    #
    total_loss = tf.reduce_mean(loss)
    #
    return total_loss
    #
#
'''
sequence_length: 1-D int32 vector, size [batch_size].The sequence lengths.
'''

'''
Example: The sparse tensor

SparseTensor(values=[1, 2], indices=[[0, 0], [1, 2]], dense_shape=[3, 4])

represents the dense tensor

  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
'''
#
# labels: An int32 SparseTensor.
#
# labels.indices[i, :] == [b, t] means
# labels.values[i] stores the id for (batch b, time t).
# labels.values[i] must take on values in [0, num_labels).
#
def convert2SparseTensorValue(list_labels):
    #
    # list_labels: batch_major
    #
    
    #
    #print(list_labels)
    #
    num_samples = len(list_labels)
    num_maxlen = max(map(lambda x: len(x), list_labels))
    #
    indices = []
    values = []
    shape = [num_samples, num_maxlen]
    #
    for idx in range(num_samples):
        #
        item = list_labels[idx]
        #
        values.extend(item)
        indices.extend([[idx, posi] for posi in range(len(item))])    
        #
    #
    return tf.SparseTensorValue(indices = indices, values = values, dense_shape = shape)
    #
#

