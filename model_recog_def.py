# -*- coding: utf-8 -*-


import tensorflow as tf

import zoo_layers as layers


#
# model
#
def conv_feat_layers(inputs, width, training):
    #
    # convolutional features maps for recognition
    #
    
    #
    # recog-inputs should have shape [ batch, 36, width, channel]
    #
    # height_norm = 36
    #

    #
    # [3,1; 1,1],
    # [9,2; 3,2], [9,2; 3,2], [9,2; 3,2]
    # [18,4; 6,4], [18,4; 6,4], [18,4; 6,4]
    # [36,8; 12,8], [36,8; 12,8], [36,8; 12,8], 
    #
  

    #
    layer_params = [ [  64, (3,3), (1,1),  'same', True, True, 'conv1'], 
                     [  64, (3,3), (1,1),  'same', True, True, 'conv2'],
                     [  64, (2,2), (2,2), 'valid', True, True, 'pool1'], # for pool
                     [ 128, (3,3), (1,1),  'same', True, True, 'conv3'], 
                     [ 128, (3,3), (1,1),  'same', True, True, 'conv4'],
                     [ 128, (2,2), (2,2), 'valid', True, True, 'pool2'], # for pool
                     [ 256, (3,3), (1,1),  'same', True, True, 'conv5'],
                     [ 256, (3,3), (1,1),  'same', True, True, 'conv6'],
                     [ 256, (3,2), (3,2), 'valid', True, True, 'pool3'], # for pool
                     [ 512, (3,1), (1,1), 'valid', True, True, 'conv_feat'] ] # for feat
    
    #
    with tf.variable_scope("conv_comm"):
        #        
        inputs = layers.conv_layer(inputs, layer_params[0], training)
        inputs = layers.conv_layer(inputs, layer_params[1], training)
        inputs = layers.padd_layer(inputs, [[0,0],[0,0],[0,1],[0,0]], name='padd1')
        inputs = layers.conv_layer(inputs, layer_params[2], training)
        #inputs = layers.maxpool_layer(inputs, (2,2), (2,2), 'valid', 'pool1')
        #        
        params = [[ 64, 3, (1,1), 'same', True,  True, 'conv1'], 
                  [ 64, 3, (1,1), 'same', True, False, 'conv2']] 
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res1')
        #
        inputs = layers.conv_layer(inputs, layer_params[3], training)
        inputs = layers.conv_layer(inputs, layer_params[4], training)
        inputs = layers.padd_layer(inputs, [[0,0],[0,0],[0,1],[0,0]], name='padd2')
        inputs = layers.conv_layer(inputs, layer_params[5], training)
        #inputs = layers.maxpool_layer(inputs, (2,2), (2,2), 'valid', 'pool2')
        #
        params = [[ 128, 3, (1,1), 'same', True,  True, 'conv1'], 
                  [ 128, 3, (1,1), 'same', True, False, 'conv2']] 
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res2')
        #
        inputs = layers.conv_layer(inputs, layer_params[6], training)
        inputs = layers.conv_layer(inputs, layer_params[7], training)
        inputs = layers.padd_layer(inputs, [[0,0],[0,0],[0,1],[0,0]], name='padd3')
        inputs = layers.conv_layer(inputs, layer_params[8], training)
        #inputs = layers.maxpool_layer(inputs, (3,2), (3,2), 'valid', 'pool3')
        #
        params = [[ 256, 3, (1,1), 'same', True,  True, 'conv1'], 
                  [ 256, 3, (1,1), 'same', True, False, 'conv2']] 
        inputs = layers.block_resnet_others(inputs, params, True, training, 'res3')
        #
        conv_feat = layers.conv_layer(inputs, layer_params[9], training)
        # 
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
    return conv_feat, sequence_length
    #
#
def rnn_recog_layers(features, sequence_length, num_classes):
    #
    # batch-picture features
    features = tf.squeeze(features, axis = 1) # squeeze
    #
    # [batchSize paddedSeqLen numFeatures]
    #
    #
    rnn_size = 256  # 256, 512
    #fc_size = 512  # 256, 384, 512
    #
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    #
    #
    # Transpose to time-major order for efficiency
    #  --> [paddedSeqLen batchSize numFeatures]
    #
    rnn_sequence = tf.transpose(features, perm = [1, 0, 2], name = 'time_major')
    #
    rnn1 = layers.gru_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
    rnn2 = layers.gru_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
    #
    # out
    #
    rnn_logits = tf.layers.dense(rnn2, num_classes,
                                 activation = None, #tf.nn.sigmoid,
                                 kernel_initializer = weight_initializer,
                                 bias_initializer = bias_initializer,
                                 name = 'rnn_logits')
    #
    # dense operates on last dim
    #
        
    #
    return rnn_logits
    #

def ctc_loss_layer(sequence_labels, rnn_logits, sequence_length):
    #
    loss = tf.nn.ctc_loss(inputs = rnn_logits, 
                          labels = sequence_labels,
                          sequence_length = sequence_length,
                          ignore_longer_outputs_than_inputs = True,
                          time_major = True )
    #
    total_loss = tf.reduce_mean(loss, name = 'loss')
    #
    return total_loss
    #

def decode_rnn_results_ctc_beam(results, seq_len):
    #
    # tf.nn.ctc_beam_search_decoder
    #
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(results, seq_len, merge_repeated=False)
    #
    return decoded
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
# sparse_targets = sparse_tuple_from(targets)
#
def convert2SparseTensorValue(list_labels):
    #
    # list_labels: batch_major
    #
   
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
def convert2ListLabels(sparse_tensor_value):
    #
    # list_labels: batch_major
    #
    
    shape = sparse_tensor_value.dense_shape
    indices = sparse_tensor_value.indices
    values = sparse_tensor_value.values

    
    list_labels = []
    #
    item = [0]*shape[1]
    for i in range(shape[0]): list_labels.append(item)
    #
    
    for idx, value in enumerate(values):
        #
        posi = indices[idx]
        #
        list_labels[posi[0]][posi[1]] = value
        #
    
    return list_labels
    #

