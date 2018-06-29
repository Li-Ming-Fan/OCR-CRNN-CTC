# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import os
import time
import random

from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_util

import model_recog_def as model_def
import model_recog_meta as meta


#
TRAINING_STEPS = 16000
BATCH_SIZE = 128
#
LEARNING_RATE_BASE = 0.001
MOMENTUM = 0.9
REG_LAMBDA = 0.0001
GRAD_CLIP = 5.0
#
BATCH_SIZE_VALID = 1
VALID_FREQ = 100
LOSS_FREQ = 1
#



class ModelRecog():
    #
    HEIGHT_NORM = meta.height_norm
    #
    def __init__(self):
        #
        # default pb path 
        self.pb_file = os.path.join(meta.model_recog_dir, meta.model_recog_pb_file)
        #        
        self.sess_config = tf.ConfigProto()
        # self.sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #
        self.is_train = False 
        #
        self.graph = None
        self.sess = None               
        #
        self.train_steps = TRAINING_STEPS
        self.batch_size = BATCH_SIZE
        #
        self.learning_rate_base = LEARNING_RATE_BASE
        self.momentum = MOMENTUM
        self.reg_lambda = REG_LAMBDA
        self.grad_clip = GRAD_CLIP
        #
        self.batch_size_valid = BATCH_SIZE_VALID
        self.valid_freq = VALID_FREQ
        self.loss_freq = LOSS_FREQ
        #
        
    def prepare_for_prediction(self, pb_file_path = None):
        #
        if pb_file_path == None: pb_file_path = self.pb_file 
        #
        if not os.path.exists(pb_file_path):
            print('ERROR: %s NOT exists, when load_pb_for_predict()' % pb_file_path)
            return -1
        #
        self.graph = tf.Graph()
        #
        with self.graph.as_default():
            #
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                #
                tf.import_graph_def(graph_def, name="")
                #
            #
            # input/output variables
            #
            self.x = self.graph.get_tensor_by_name('x-input:0')
            self.w = self.graph.get_tensor_by_name('w-input:0')
            #
            self.seq_len = self.graph.get_tensor_by_name('seq_len:0')
            self.result_logits = self.graph.get_tensor_by_name('rnn_logits/BiasAdd:0')
            #
            self.result_i = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
            self.result_v = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
            self.result_s = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
            #            
        #
        print('graph loaded for prediction')
        #
        self.sess = tf.Session(graph = self.graph, config = self.sess_config)
        #

    def predict(self, image_in):
        #
        # input data
        if  isinstance(image_in, str):
            img = Image.open(image_in)
            img = img.convert('RGB')
            img_size = img.size
            if img_size[1] != meta.height_norm:
                w = int(img_size[0] * meta.height_norm *1.0/img_size[1])
                img = img.resize((w, meta.height_norm))
            img_data = np.array(img, dtype = np.float32)/255  # (height, width, channel)
            img_data = [ img_data[:,:,0:3] ]
        else:
            # np array
            img_data = image_in
        #
        w_arr = [ img_data[0].shape[1] ]  # batch, height, width, channel
        #
        with self.graph.as_default():              
            #
            feed_dict = {self.x: img_data, self.w: w_arr}
            #
            results, seq_length, d_i, d_v, d_s = \
            self.sess.run([self.result_logits, self.seq_len, 
                      self.result_i, self.result_v, self.result_s], feed_dict)
            #
            decoded = tf.SparseTensorValue(indices = d_i, values = d_v, dense_shape = d_s)
            trans = model_def.convert2ListLabels(decoded)
            #print(trans)
            #
            for item in trans:
                seq = list(map(meta.mapOrder2Char, item))
                str_result = ''.join(seq)
                #
        #
        return str_result
        #
    
    def create_graph_all(self, training):
        #
        self.is_train = training
        self.graph = tf.Graph()
        #
        with self.graph.as_default():
            #
            self.x = tf.placeholder(tf.float32, (None, None, None, 3), name = 'x-input')
            self.w = tf.placeholder(tf.int32, (None,), name = 'w-input') # width
            #
            self.conv_feat, self.seq_len = model_def.conv_feat_layers(self.x, self.w, self.is_train)   # train
            self.result_logits = model_def.rnn_recog_layers(self.conv_feat, self.seq_len, len(meta.alphabet) + 1)
            self.result_decoded_list = model_def.decode_rnn_results_ctc_beam(self.result_logits, self.seq_len)
            #
            print()
            print('self.result_logits.op.name: ' + self.result_logits.op.name)    # recog_logits/Sigmoid   /BiasAdd
            #print(self.seq_len)          # seq_len
            print('self.result_decoded_list[0]: ')
            print(self.result_decoded_list[0])   # CTCBeamSearchDecoder
            print()
            #
            # SparseTensor(indices=Tensor("CTCBeamSearchDecoder:0", shape=(?, 2), dtype=int64),
            #              values=Tensor("CTCBeamSearchDecoder:1", shape=(?,), dtype=int64),
            #              dense_shape=Tensor("CTCBeamSearchDecoder:2", shape=(2,), dtype=int64))
            #
            self.y = tf.sparse_placeholder(tf.int32, name = 'y-input') 
            #
            print('self.y: ')
            print(self.y)
            #
            # <tf.Operation 'y-input/shape' type=Placeholder>,
            # <tf.Operation 'y-input/values' type=Placeholder>,
            # <tf.Operation 'y-input/indices' type=Placeholder>]
            #            
            #print(graph.get_operations())
            #
            self.loss = model_def.ctc_loss_layer(self.y, self.result_logits, self.seq_len)
            #
            #print(self.loss.op.name)  # loss
            #
            # train
            self.global_step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
            
            #optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
            #optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)              
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = MOMENTUM)
            #
            '''
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            grads = optimizer.compute_gradients(self.loss + l2_loss * REG_LAMBDA)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, GRAD_CLIP)
            grads_applying = zip(capped_grads, variables)
            '''
            grads_applying = optimizer.compute_gradients(self.loss)
            
            self.train_op = optimizer.apply_gradients(grads_applying, global_step=self.global_step)                
            #
            
            '''
            #
            # Update batch norm stats [http://stackoverflow.com/questions/43234667]
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # 
            with tf.control_dependencies(extra_update_ops):
                #
                learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                           tf.train.get_global_step(),
                                                           DECAY_STEPS,
                                                           DECAY_RATE,
                                                           staircase = DECAY_STAIRCASE,
                                                           name = 'learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                                   beta1 = MOMENTUM)
                # train_op =                     
                train_op = tf.contrib.layers.optimize_loss(loss = loss,
                                                           global_step = tf.train.get_global_step(),
                                                           learning_rate = learning_rate,
                                                           optimizer = optimizer,
                                                           name = 'train_op')  #variables = rnn_vars)
                
            #
            '''
            
            
            #
            print('global_step.op.name: ' + self.global_step.op.name)
            print('train_op.name: ' + self.train_op.name)  
            print()
            #
            print('graph defined for training') if self.is_train else print('graph defined for validation')              
            #
            #

    def train_and_valid(self, data_train, data_valid):
        #
        # model save-path
        if not os.path.exists(meta.model_recog_dir): os.mkdir(meta.model_recog_dir)
        #
        # graph
        self.create_graph_all(training = True)
        #
        # restore and train
        with self.graph.as_default():
            #
            saver = tf.train.Saver()
            with tf.Session(config = self.sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                sess.run(tf.assign(self.learning_rate, tf.constant(self.learning_rate_base, dtype=tf.float32)))
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                #
                print('begin to train ...')
                #
                # variables
                #
                # y_s = self.graph.get_tensor_by_name('y-input/shape:0')
                # y_i = self.graph.get_tensor_by_name('y-input/indices:0')
                # y_v = self.graph.get_tensor_by_name('y-input/values:0')
                #
                # <tf.Operation 'y-input/shape' type=Placeholder>,
                # <tf.Operation 'y-input/values' type=Placeholder>,
                # <tf.Operation 'y-input/indices' type=Placeholder>]
                #
                #
                num_samples = len(data_train['x'])
                #
                # start training
                start_time = time.time()
                begin_time = start_time 
                #
                step = sess.run(self.global_step)
                #
                train_step_half = int(self.train_steps * 0.5)
                train_step_quar = int(self.train_steps * 0.75)
                #
                while step < self.train_steps:
                    #
                    if step == train_step_half:
                        sess.run(tf.assign(self.learning_rate, tf.constant(self.learning_rate_base/10, dtype=tf.float32)))
                    if step == train_step_quar:
                        sess.run(tf.assign(self.learning_rate, tf.constant(self.learning_rate_base/100, dtype=tf.float32)))
                    #
                    # save and validate
                    if step % self.valid_freq == 0:
                        #
                        print('save model to ckpt ...')
                        saver.save(sess, os.path.join(meta.model_recog_dir, meta.model_recog_name), \
                                   global_step = step)
                        #
                        print('validating ...')
                        model_v = ModelRecog()
                        model_v.validate(data_valid, step)
                        #
                    #
                    # train
                    index_batch = random.sample(range(num_samples), self.batch_size)
                    #
                    images= [data_train['x'][i] for i in index_batch] 
                    targets = [data_train['y'][i] for i in index_batch]
                    #
                    w_arr = [item.shape[1] for item in images]
                    max_w = max(w_arr)
                    img_padd = []
                    for item in images:
                        if item.shape[1] != max_w:
                            img_zeros = np.zeros(shape=[meta.height_norm, max_w - item.shape[1], 3], dtype=np.float32)            
                            item = np.concatenate([item, img_zeros], axis = 1)
                        img_padd.append(item)
                    images = img_padd
                    #
                    # targets_sparse_value
                    tsv = model_def.convert2SparseTensorValue(targets)
                    #
                    #
                    #feed_dict = {self.x: images, self.w: w_arr, y_s: tsv.dense_shape, y_i: tsv.indices, y_v: tsv.values}
                    #
                    feed_dict = {self.x: images, self.w: w_arr, self.y: tsv}
                    #
                    
                    #print(width)
                    #conv_v = sess.run(conv_feat, feed_dict)
                    #print(len(conv_v))
                    
                    # sess.run
                    _, loss_value, step, lr = sess.run([self.train_op, self.loss, self.global_step, self.learning_rate],\
                                                       feed_dict)
                    #
                    if step % self.loss_freq == 0:                        
                        #
                        curr_time = time.time()            
                        #
                        print('step: %d, loss: %g, lr: %g, sect_time: %.1f, total_time: %.1f' %
                              (step, loss_value, lr, curr_time - begin_time, curr_time - start_time) )
                        #
                        begin_time = curr_time
                        #
                    #
        #
    
    def validate(self, data_valid, step):
        #
        # valid_result save-path
        if not os.path.exists(meta.dir_results_valid): os.mkdir(meta.dir_results_valid)
        #
        self.create_graph_all(training = False)
        #
        with self.graph.as_default():
            #
            saver = tf.train.Saver()
            with tf.Session(config = self.sess_config) as sess:                
                #
                tf.global_variables_initializer().run()
                #sess.run(tf.assign(self.is_train, tf.constant(False, dtype=tf.bool)))
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # pb
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names = \
                                                                           ['rnn_logits/BiasAdd','seq_len',\
                                                                            'CTCBeamSearchDecoder'])
                with tf.gfile.FastGFile(self.pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                #
                # variables
                #
                result_i = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
                result_v = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
                result_s = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
                #
                #y_s = self.graph.get_tensor_by_name('y-input/shape:0')
                #y_i = self.graph.get_tensor_by_name('y-input/indices:0')
                #y_v = self.graph.get_tensor_by_name('y-input/values:0')
                #
                # <tf.Operation 'y-input/shape' type=Placeholder>,
                # <tf.Operation 'y-input/values' type=Placeholder>,
                # <tf.Operation 'y-input/indices' type=Placeholder>]
                #
                # validate
                num_samples = len(data_valid['x'])
                num_right = 0
                num_total = 0
                num_batches = np.ceil(num_samples * 1.0 / self.batch_size_valid)
                #
                curr = 0
                batch_start = 0
                batch_end = self.batch_size_valid
                while batch_start < num_samples:
                    #
                    images = data_valid['x'][batch_start:batch_end]
                    targets = data_valid['y'][batch_start:batch_end]
                    #
                    w_arr = [item.shape[1] for item in images]
                    max_w = max(w_arr)
                    img_padd = []
                    for item in images:
                        if item.shape[1] != max_w:
                            img_zeros = np.zeros(shape=[meta.height_norm, max_w - item.shape[1], 3], dtype=np.float32)            
                            item = np.concatenate([item, img_zeros], axis = 1)
                        img_padd.append(item)
                    images = img_padd
                    #
                    batch_start = batch_start + self.batch_size_valid
                    batch_end = min(batch_end + self.batch_size_valid, num_samples)
                    #
                    # targets_sparse_value
                    tsv = model_def.convert2SparseTensorValue(targets)
                    #
                    feed_dict = {self.x: images, self.w: w_arr, self.y: tsv}
                    #
                    results, loss, seq_length, d_i, d_v, d_s = \
                    sess.run([self.result_logits, self.loss, self.seq_len, \
                                                                        result_i, result_v, result_s], feed_dict)
                    #
                    decoded = tf.SparseTensorValue(indices = d_i, values = d_v, dense_shape = d_s)
                    #
                    curr += 1
                    print('curr: %d / %d, loss: %f' % (curr, num_batches, loss))
                    #
                    #print(targets)               
                    #print(results)
                    #print(decoded)
                    #
                    trans = model_def.convert2ListLabels(decoded)
                    #
                    print(targets)
                    print(trans)
                    #
                    diff = [item[0]-item[1] for item in zip(targets[0], trans[0])]
                    num_right += diff.count(0)
                    num_total += sum([ len(word) for word in targets ])
                    #                
                #
                acc = num_right/num_total
                #
                print('validation finished, char-accuracy: %f' %  acc )
                print()
                #
    
    
  
