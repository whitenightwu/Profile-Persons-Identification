#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np


class BaseDataset():
    def __init__(self, file, batch_size=8, shuffle=True, validation_split=0.0001, embedding_size=128):
        self.file = file
        self.embedding_size = embedding_size
        self.batch_size = batch_size

    def trainingset_iterator(self):
        pass

    def get_validation_set(self):
        pass

    @staticmethod
    def norm_angle(angle):
        if len(np.shape(angle)) > 1:
            angle = np.squeeze(angle)
        return BaseDataset.__sigmoid(10 * (np.abs(angle) / 45.0 - 1))

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))
