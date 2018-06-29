 # -*- coding: utf-8 -*-

import os

from PIL import Image
import numpy as np

import model_recog_meta as meta


'''
#
dir_data = './data_generated'
dir_images = dir_data + '/images'
dir_contents = dir_data + '/contents'
#
'''


#
def getFilesInDirect(path, str_dot_ext):
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)  
        if os.path.splitext(file_path)[1] == str_dot_ext:  
            file_list.append(file_path)
            #print(file_path)
        #
    return file_list;
    #
    
def getTargetTxtFile(img_file):
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

def getImageSize(img_file):
    #
    img = Image.open(img_file)
    return img.size  # (width, height)
    #

def getListContents(content_file):
    #
    contents = []
    #
    with open(content_file, 'r') as fp:
        lines = fp.readlines()
    #
    '''
    for line in lines:
        arr_str = line.split('|')
        item = list(map(lambda x: int(x), arr_str[0].split('-')))
        #
        contents.append([item, arr_str[1][:-1]])
        #
    '''
    word_str = lines[0].replace('\n', '').strip()
    contents = [[None, word_str] ]

    return contents

#
# util function
#
def load_data(dir_data):
    #
    dir_images = dir_data + '/images'
    #dir_contents = dir_data + '/contents'
    #
    list_imgs = getFilesInDirect(dir_images, meta.str_dot_img_ext)
    #
    data = []
    labels = []
    #
    for img_file in list_imgs:
        #
        # label
        txt_file = getTargetTxtFile(img_file)
        content_list = getListContents(txt_file)
        #
        try:
            list_chars = list(map(meta.mapChar2Order, content_list[0][1])) 
        except BaseException:
            print('Unknown char in file: %s' % img_file)
            continue
        #
        #if len(list_chars) != meta.num_chars: continue
        #
        print(img_file)
        print(content_list[0][1])
        print(list_chars)
        #
        labels.append(list_chars)  #
        #
        # img_data
        img = Image.open(img_file)
        img = np.array(img, dtype = np.float32)/255
        img = img[:,:,0:3]
        #
        data.append(img)  #
        #
        '''
        try:
            chosen = random.choice(range(len(content_list)))
            list_chars = list(map(mapChar2Order, content_list[chosen][1]))  
        except BaseException:
            print('character OUT OF ALPHABET')
            print(content_list[chosen][1])
            chosen = -1
        '''
    
    #
    return {'x': data, 'y': labels}
    #

