#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np

filename = '/home/ydwu/project3/pose_and_emb.txt'

aaa = []
bbb = []
ccc = []
ddd = []

with open(filename, 'r') as file_to_read:
  while True:
    lines = file_to_read.readline() # 整行读取数据
    print(lines)
    if not lines:
      break
      pass

    l = lines.split()
    aaa.append(l[0])
    bbb.append(l[1])
    ccc.append(l[2:5])
    ddd.append(l[5:])
