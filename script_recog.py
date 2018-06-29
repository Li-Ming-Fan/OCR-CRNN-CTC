# -*- coding: utf-8 -*-
"""
@author: limingfan

"""

import model_recog_meta as meta
from model_recog_wrap import ModelRecog

import model_recog_data as model_data



import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = '2' #使用 GPU 0
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
#




#
model = ModelRecog()
#


# data
print('loading data ...')
data_train = model_data.load_data(meta.dir_data_train)
data_valid = model_data.load_data(meta.dir_data_valid)
print('load finished.')

#
# train
model.train_and_valid(data_train, data_valid)
#


#
# predict
model.prepare_for_prediction()
#
list_images_valid = model_data.getFilesInDirect(meta.dir_images_valid, meta.str_dot_img_ext)
for img_file in list_images_valid:
    #
    # img_file = './data_test/images/bkgd_1_0_generated_0.png'
    #
    print(img_file)
    #
    model.predict(img_file)
    #

