from __future__ import division

import math
import warnings

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import scipy.ndimage
import six
import skimage
import skimage.color
from skimage import img_as_ubyte
import os
import os.path as osp
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import scipy.signal
# -----------------------------------------------------------------------------
#   IJB-A helper code
# -----------------------------------------------------------------------------
def get_ijba_1_1_metadata(protocol_file):
    metadata = {}
    template_id = []
    subject_id = []
    img_filename = []
    media_id = []
    sighting_id = []

    with open(protocol_file, 'r') as f:
        for line in f.readlines()[1:]:
            line_fields = line.strip().split(',')
            template_id.append(int(line_fields[0]))
            subject_id.append(int(line_fields[1]))
            img_filename.append(line_fields[2])
            media_id.append(int(line_fields[3]))
            sighting_id.append(int(line_fields[4]))

    metadata['template_id'] = np.array(template_id)
    metadata['subject_id'] = np.array(subject_id)
    metadata['img_filename'] = np.array(img_filename)
    metadata['media_id'] = np.array(media_id)
    metadata['sighting_id'] = np.array(sighting_id)
    return metadata


def read_ijba_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split(',')
            pairs.append(pair)
    return np.array(pairs).astype(np.int)