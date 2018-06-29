# -*- coding: utf-8 -*-



import os
import random
from PIL import Image

import model_recog_meta as meta

height_norm = meta.height_norm


'''
extract rects for recognition

'''

import sys


#
# change this value to change the default purpose of data-generating
#
data_for_training = 1              # 1 for training, 0 for test  
# 

#
str_dot_img_ext = '.jpg'
#
ratio_extract = 0.2
#



#
if len(sys.argv) >= 2:
    #
    data_for_training = int(sys.argv[1])
    #
    if data_for_training != 0 and data_for_training != 1:
        print('The parameter should be 0 or 1.')
        print('Set to 0 by default.')
        data_for_training = 0
        #

#
if data_for_training > 0:
    dir_walk = './data_train'
else:
    dir_walk = './data_valid'
#
    
#
dir_data_rects = './data_rects_' + dir_walk.split('_')[-1]
#
dir_rects_images = dir_data_rects + '/images'
dir_rects_contents = dir_data_rects + '/contents'
if not os.path.exists(dir_data_rects): os.mkdir(dir_data_rects)
if not os.path.exists(dir_rects_images): os.mkdir(dir_rects_images)
if not os.path.exists(dir_rects_contents): os.mkdir(dir_rects_contents)
#

#
def get_target_txt_file(img_file):
    #
    pre_dir = os.path.abspath(os.path.dirname(img_file)+os.path.sep+"..")
    txt_dir = os.path.join(pre_dir, 'contents')
    #
    filename = os.path.basename(img_file)
    arr_split = os.path.splitext(filename)
    filename = arr_split[0] + '.txt'
    #
    txt_file = os.path.join(txt_dir, filename)
    #
    return txt_file
    #
#
def get_list_contents(content_file):
    #
    contents = []
    #
    if not os.path.exists(content_file): return contents
    #
    with open(content_file, 'r') as fp:
        lines = fp.readlines()
    #
    for line in lines:
        arr_str = line.split('|')
        item = list(map(lambda x: int(x), arr_str[0].split('-')))
        #
        contents.append([ item, arr_str[1] ] )
        #
    return contents
#

#
def extract_rects_walk(path, str_ext):
    curr = 0
    list_files = []
    for (root, dirs, files) in os.walk(path):
        #  列出目录下的所有文件和文件名        
        # for filename in files: print(os.path.join(root,filename) )            
        # for dirc in dirs: print(os.path.join(root,dirc) )
        for filename in files:            
            if not filename.endswith(str_ext): continue
            #
            img_file = os.path.join(root,filename)
            if not os.path.exists(img_file):
                print('image_file: %s NOT exist' % img_file)
                continue
            #
            txt_file = get_target_txt_file(img_file)
            if not os.path.exists(txt_file):
                print('label_file: %s NOT exist' % txt_file)
                continue
            #
            try:
                img = Image.open(img_file)
            except BaseException:
                print('Error file: %s, removed' % img_file)
                os.remove(img_file)
            #
            print(img_file)
            #
            list_contents = get_list_contents(txt_file)
            for item in list_contents:
                if random.random() > ratio_extract: continue
                #
                rect = img.crop(item[0])
                word = item[1].replace('\n', '')
                #
                #print(item[0])
                #print(word)
                #
                rect_size = rect.size  # (width, height)
                w = int(rect_size[0] * height_norm *1.0/rect_size[1])
                rect = rect.resize((w, height_norm))
                #
                # save
                filepath = os.path.join(dir_rects_images, str(curr) + '_' + word + '_0' + str_dot_img_ext)
                rect = rect.convert('RGB')
                rect.save(filepath)
                #
                list_files.append(filepath)
                #
                filepath = os.path.join(dir_rects_contents, str(curr) + '_' + word + '_0.txt')
                with open(filepath, 'w') as fp:
                    fp.write(word)
                #                
            #
            curr = curr + 1
            #
    #
    return list_files
    #

#
extract_rects_walk(dir_walk, str_dot_img_ext)
#


    
    
