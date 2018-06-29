# -*- coding: utf-8 -*-
"""
@author: limingfan

"""


# data for train
dir_data_train = './data_rects_train'
#
dir_images_train = dir_data_train + '/images'
dir_contents_train = dir_data_train + '/contents'

# data for validation
dir_data_valid = './data_rects_valid'
#
dir_images_valid = dir_data_valid + '/images'
dir_contents_valid = dir_data_valid + '/contents'
#
dir_results_valid = dir_data_valid + '/results'
#
str_dot_img_ext = '.jpg'  # png
#



#
model_recog_dir = './model_recog'
model_recog_name = 'model_recog'
model_recog_pb_file = model_recog_name + '.pb'
#

#
height_norm = 36
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
    



