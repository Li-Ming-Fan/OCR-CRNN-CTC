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

import model_comm_def as model_def
import model_comm_meta as meta

import model_recog_data



#
TRAINING_STEPS = 30000*5
BATCH_SIZE = 128
REG_LAMBDA = 0.0001
#
LEARNING_RATE_BASE = 0.001
DECAY_RATE = 0.1
DECAY_STAIRCASE = True
DECAY_STEPS = 30000
#
MOMENTUM = 0.9
#



class ModelRecog():
    #
    def __init__(self):
        #
        self.z_pb_file = os.path.join(meta.model_recog_dir, meta.model_recog_pb_file)
        #        
        self.z_sess_config = tf.ConfigProto()
        # self.z_sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95
        #
        self.z_valid_freq = 100
        self.z_valid_option = False
        #
        self.z_batch_size = BATCH_SIZE
        #
        
    def load_pb_for_prediction(self, pb_file_path = None):
        #
        if pb_file_path == None: pb_file_path = self.z_pb_file 
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
            # change the input/output variables
            #
            self.x = self.graph.get_tensor_by_name('x-input:0')
            self.w = self.graph.get_tensor_by_name('w-input:0')
            #
            self.seq_len = self.graph.get_tensor_by_name('seq_len:0')
            self.result_logits = self.graph.get_tensor_by_name('rnn_logits/Sigmoid:0')
            #
            self.result_i = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
            self.result_v = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
            self.result_s = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
            #
            
        #
        print('graph loaded for prediction')
        #
        return 0
        #
        
    def create_session_for_prediction(self):
        #
        with self.graph.as_default():
            sess = tf.Session(config = self.z_sess_config)
            
            return sess
        #

    def predict(self, sess, img_file, out_dir = './results_prediction'):
        #
        # input data
        img = Image.open(img_file)
        img = np.array(img, dtype = np.float32)/255
        img_data = img[:,:,0:3]
        #
        img_size = img.size  # (width, height)
        w_arr = np.ones((1,), dtype = np.int32) * img_size[0]
        #
        # predication_result save-path
        if not os.path.exists(out_dir): os.mkdir(out_dir)
        #
        with self.graph.as_default():              
            #
            feed_dict = {self.x: img_data, self.w: w_arr}
            #
            results, seq_length, d_i, d_v, d_s = sess.run([self.result_logits, self.seq_len, \
                                                           self.result_i, self.result_v, self.result_s], feed_dict)
            #
            #
            filename = os.path.basename(img_file)
            arr_str = os.path.splitext(filename)
            #
            # image
            r = Image.fromarray(img_data[0][:,:,0] *255).convert('L')
            g = Image.fromarray(img_data[0][:,:,1] *255).convert('L')
            b = Image.fromarray(img_data[0][:,:,2] *255).convert('L')
            #
            file_target = os.path.join(out_dir, arr_str[0] + '_predict.png')
            img_target = Image.merge("RGB", (r, g, b))
            img_target.save(file_target)
            #
            #
            decoded = tf.SparseTensorValue(indices = d_i, values = d_v, dense_shape = d_s)
            #
            trans = model_def.convert2ListLabels(decoded)
            # 
            #print(trans)
            #
            filename = os.path.basename(img_file)
            arr_str = os.path.splitext(filename)
            result_file = os.path.join(out_dir, arr_str[0] + '_predict.txt')
            #
            with open(result_file, 'w') as fp:
                for item in trans:
                    seq = list(map(meta.mapOrder2Char, item))
                    str_result = ''.join(seq)
                    fp.write(str_result + '\n')
                    #
                    print(str_result)
                    #
            #
        #
        return str_result
        #
    
    @staticmethod
    def z_define_graph_all(graph, train = True): # learn.ModeKeys.TRAIN  INFER
        #
        with graph.as_default():
            #
            x = tf.placeholder(tf.float32, (None, None, None, 3), name = 'x-input')
            w = tf.placeholder(tf.int32, (None,), name = 'w-input') # width
            #
            conv_feat, sequence_length = model_def.conv_feat_layers(x, w, train)   # train
            #
            result_logits = model_def.rnn_recog_layers(conv_feat, sequence_length, len(meta.alphabet) + 1)
            #
            result_decoded_list = model_def.decode_rnn_results_ctc_beam(result_logits, sequence_length)
            #
            print(' ')
            #print(result_logits.op.name)    # recog_logits/Sigmoid
            #print(sequence_length)          # seq_len
            print(result_decoded_list[0])   # CTCBeamSearchDecoder
            print(' ')
            #
            # SparseTensor(indices=Tensor("CTCBeamSearchDecoder:0", shape=(?, 2), dtype=int64),
            #              values=Tensor("CTCBeamSearchDecoder:1", shape=(?,), dtype=int64),
            #              dense_shape=Tensor("CTCBeamSearchDecoder:2", shape=(2,), dtype=int64))
            #
            
            #
            print('forward graph defined, training = %s' % train)
            #
            #
            y = tf.sparse_placeholder(tf.int32, shape = (None, None), name = 'y-input') 
            #
            # <tf.Operation 'y-input/shape' type=Placeholder>,
            # <tf.Operation 'y-input/values' type=Placeholder>,
            # <tf.Operation 'y-input/indices' type=Placeholder>]
            #            
            #print(graph.get_operations())
            #
            loss = model_def.ctc_loss_layer(y, result_logits, sequence_length)
            #
            #print(loss.op.name)  # loss
            #
            # train
            global_step = tf.train.get_or_create_global_step()
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
            print('train graph defined, training = %s' % train)
            #
            print('global_step.op.name: ' + global_step.op.name)
            print('train_op.op.name: ' + train_op.op.name)                
            #
            #
                
    def validate(self, data_valid, step):
        #
        # valid_result save-path
        if not os.path.exists(meta.dir_results_valid): os.mkdir(meta.dir_results_valid)
        #
        # if os.path.exists(dir_results): shutil.rmtree(dir_results)
        # time.sleep(0.1)
        # os.mkdir(dir_results)
        #
        # validation graph
        self.graph = tf.Graph()
        #
        self.z_define_graph_all(self.graph, self.z_valid_option)
        #
        with self.graph.as_default():
            #
            saver = tf.train.Saver()
            #
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            #
            with tf.Session(config = self.z_sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # pb
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names = \
                                                                           ['rnn_logits/Sigmoid','seq_len',\
                                                                            'CTCBeamSearchDecoder'])
                with tf.gfile.FastGFile(self.z_pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())
                #
                # variables
                #
                x = self.graph.get_tensor_by_name('x-input:0')
                w = self.graph.get_tensor_by_name('w-input:0')
                #
                seq_len = self.graph.get_tensor_by_name('seq_len:0')
                result_logits = self.graph.get_tensor_by_name('rnn_logits/Sigmoid:0')
                #
                result_i = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:0')
                result_v = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:1')
                result_s = self.graph.get_tensor_by_name('CTCBeamSearchDecoder:2')
                #
                y_s = self.graph.get_tensor_by_name('y-input/shape:0')
                y_i = self.graph.get_tensor_by_name('y-input/indices:0')
                y_v = self.graph.get_tensor_by_name('y-input/values:0')
                #
                # <tf.Operation 'y-input/shape' type=Placeholder>,
                # <tf.Operation 'y-input/values' type=Placeholder>,
                # <tf.Operation 'y-input/indices' type=Placeholder>]
                #
                #
                loss_ts = self.graph.get_tensor_by_name('loss:0')
                #
                # validate
                num_samples = len(data_valid['x'])
                num_right = 0
                num_batches = np.ceil(num_samples * 1.0 / self.z_batch_size)
                # loss_sum = 0.0
                #
                curr = 0
                batch_start = 0
                batch_end = self.z_batch_size
                while batch_start < num_samples:
                    #
                    images = data_valid['x'][batch_start:batch_end]
                    targets = data_valid['y'][batch_start:batch_end]
                    w_arr = np.ones((batch_end - batch_start,), dtype = np.float32) * meta.width_norm
                    #
                    batch_start = batch_start + self.z_batch_size
                    batch_end = min(batch_end + self.z_batch_size, num_samples)
                    #
                    # targets_sparse_value
                    tsv = model_def.convert2SparseTensorValue(targets)
                    #
                    feed_dict = {x: images, w: w_arr, y_s: tsv.dense_shape, y_i: tsv.indices, y_v: tsv.values}
                    results, loss, seq_length, d_i, d_v, d_s = sess.run([result_logits, loss_ts, seq_len, \
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
                
                #
                print('validation finished')
                print(' ')
                #
    
    def train_and_valid(self, data_train, data_valid, load_from_pretrained = True):
        #
        # model save-path
        if not os.path.exists(meta.model_recog_dir): os.mkdir(meta.model_recog_dir)
        #
        # training graph
        self.z_graph = tf.Graph()
        self.z_define_graph_all(self.z_graph, True)
        #
        # load from pretained
        list_ckpt = model_recog_data.getFilesInDirect(meta.model_recog_dir, '.meta')
        #
        print(' ')
        #
        if len(list_ckpt) > 0:
            print('model_recog ckpt already exists, no need to load common tensors.')            
        elif load_from_pretrained == False:
            print('not to load common tensors, by manual setting.')
        else:
            print('load common tensors from pretrained detection model.')
            self.z_load_from_pretrained_detection_model()
        print(' ')
        #
        # restore and train
        with self.z_graph.as_default():
            #
            saver = tf.train.Saver()
            with tf.Session(config = self.z_sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(meta.model_recog_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                #
                # variables
                #
                x = self.z_graph.get_tensor_by_name('x-input:0')
                w = self.z_graph.get_tensor_by_name('w-input:0')
                #
                y_s = self.z_graph.get_tensor_by_name('y-input/shape:0')
                y_i = self.z_graph.get_tensor_by_name('y-input/indices:0')
                y_v = self.z_graph.get_tensor_by_name('y-input/values:0')
                #
                # <tf.Operation 'y-input/shape' type=Placeholder>,
                # <tf.Operation 'y-input/values' type=Placeholder>,
                # <tf.Operation 'y-input/indices' type=Placeholder>]
                #
                #conv_feat = self.z_graph.get_tensor_by_name('conv_comm/conv_feat:0')
                #
                loss = self.z_graph.get_tensor_by_name('loss:0')
                #
                global_step = self.z_graph.get_tensor_by_name('global_step:0')
                learning_rate = self.z_graph.get_tensor_by_name('learning_rate:0')
                train_op = self.z_graph.get_tensor_by_name('train_op/control_dependency:0')
                #
                print('begin to train ...')
                #
                num_samples = len(data_train['x'])
                #
                # start training
                start_time = time.time()
                begin_time = start_time 
                step = 0
                #
                for curr_iter in range(TRAINING_STEPS):
                    #
                    # save and validate
                    if step % self.z_valid_freq == 0:
                        #
                        # ckpt
                        print('save model to ckpt ...')
                        saver.save(sess, os.path.join(meta.model_recog_dir, meta.model_recog_name), \
                                   global_step = step)
                        #
                        # validate
                        print('validating ...')
                        self.validate(data_valid, step)
                        #
                    #
                    # train
                    index_batch = random.sample(range(num_samples), self.z_batch_size)
                    #
                    images= [data_train['x'][i] for i in index_batch] 
                    targets = [data_train['y'][i] for i in index_batch]
                    w_arr = np.ones((self.z_batch_size,), dtype = np.float32) * meta.width_norm
                    #
                    # targets_sparse_value
                    tsv = model_def.convert2SparseTensorValue(targets)
                    #
                    #
                    feed_dict = {x: images, w: w_arr, y_s: tsv.dense_shape, y_i: tsv.indices, y_v: tsv.values}
                    #
                    
                    #print(width)
                    #conv_v = sess.run(conv_feat, feed_dict)
                    #print(len(conv_v))
                    
                    # sess.run
                    _, loss_value, step, lr = sess.run([train_op, loss, global_step, learning_rate], \
                                                        feed_dict)
                    #
                    if step % 1 == 0:                        
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
    
    def z_load_from_pretrained_detection_model(self):
        #
        # model save-path
        # if not os.path.exists(meta.model_recog_dir): os.mkdir(meta.model_recog_dir)
        #
        # training graph
        # self.z_graph = tf.Graph()
        # self.z_define_graph_all(self.z_graph, True)
        #
        print('check common tensors to load ...')
        #
        comm_op_list = []
        #
        op_list = self.z_graph.get_operations()
        for op in op_list:
            if 'comm' in op.name and 'train' not in op.name and 'Variable' in op.type:
                #
                #print(op)
                #print(op.name)
                #print(op.type)
                #
                op_tensor = self.z_graph.get_tensor_by_name(op.name + ':0')
                #
                comm_op_list.append(op_tensor)
                #
        #
        print('checked.')
        #
        with self.z_graph.as_default():
            #
            saver = tf.train.Saver(comm_op_list)
            #
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.95)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            #
            with tf.Session(config = self.z_sess_config) as sess:
                #
                tf.global_variables_initializer().run()
                #
                # restore with saved data
                ckpt = tf.train.get_checkpoint_state(meta.model_detect_dir)
                #
                if ckpt and ckpt.model_checkpoint_path:
                    print('loading comm_tensors of detection and recognition ...')
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('loaded.')
                else:
                    print('NO pretrained detection model.')
                #
        #

#
