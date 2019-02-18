#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : merge_all_images.py
## Authors    : ydwu@ydwu-OptiPlex-5050
## Create Time: 2018-10-18:14:18:33
## Description:
## 
##
import os
import sys
import shutil

#########################
# path = '/home/ydwu/datasets/face_near_infrared_test/black'
# tag_path='/home/ydwu/datasets/face_near_infrared_test/merge_black'

path = '/home/ydwu/datasets/cfp-dataset/Data/Images'
tag_path='/home/ydwu/datasets/white-cfp-dataset-merge'

if not os.path.exists(tag_path):
    os.makedirs(tag_path)


count = 1
for person in os.listdir(path):
    for each_img in os.listdir(path+'/'+person):
        old_fold=path+'/'+person+'/'+each_img
        for xxx in os.listdir(old_fold):
            oldname=old_fold+'/'+xxx
            new_fold=tag_path+'/'+person
            if not os.path.exists(new_fold):
                os.makedirs(new_fold)
            newname=new_fold+'/'+each_img+'_'+xxx
            shutil.copyfile(oldname,newname)

            
            print(oldname)
            print(newname)
        
        count+=1
    count = 1
