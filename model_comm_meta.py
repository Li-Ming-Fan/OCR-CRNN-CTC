# -*- coding: utf-8 -*-
"""
@author: limingfan

"""


# data for train
dir_data_train = './data_generated'
#
dir_images_train = dir_data_train + '/images'
dir_contents_train = dir_data_train + '/contents'

# data for validation
dir_data_valid = './data_test'
#
dir_images_valid = dir_data_valid + '/images'
dir_contents_valid = dir_data_valid + '/contents'
#
dir_results_valid = dir_data_valid + '/results'
#
str_dot_img_ext = '.png'
#



#
model_detect_dir = './model_detect'
model_detect_name = 'model_detect'
model_detect_pb_file = model_detect_name + '.pb'
#
anchor_heights = [12, 24, 36]
#
threshold = 0.5  #
#




#
model_recog_dir = './model_recog'
model_recog_name = 'model_recog'
model_recog_pb_file = model_recog_name + '.pb'
#

#
num_chars = 6
#
height_norm = 36
width_norm = 100
#
xs = 3
ys = 3
#

#
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet += ''',./<>?;':"[]\{}-=_+~!@#$%^&*()
            '''.replace('space','').strip()
#
alphabet_blank = '`'
#


def define_alphabet():
    #
    pass

def mapChar2Order(char): return alphabet.index(char)
def mapOrder2Char(order):
    if order == len(alphabet):
        return alphabet_blank
    else:
        return alphabet[order]
    #
    



