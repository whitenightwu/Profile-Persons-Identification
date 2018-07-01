from __future__ import division

import math
import warnings
import cv2
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


def plot_log_csv(log_path):
    log_dir, _ = osp.split(log_path)
    dat = np.genfromtxt(log_path, names=True, delimiter=',', autostrip=True)

    train_loss = dat['trainloss']
    train_loss_sel = ~np.isnan(train_loss)
    train_loss = train_loss[train_loss_sel]
    iter_train_loss = dat['iteration'][train_loss_sel]

    train_acc = dat['trainacc']
    train_acc_sel = ~np.isnan(train_acc)
    train_acc = train_acc[train_acc_sel]
    iter_train_acc = dat['iteration'][train_acc_sel]

    val_loss = dat['validloss']
    val_loss_sel = ~np.isnan(val_loss)
    val_loss = val_loss[val_loss_sel]
    iter_val_loss = dat['iteration'][val_loss_sel]

    mean_iu = dat['validacc']
    mean_iu_sel = ~np.isnan(mean_iu)
    mean_iu = mean_iu[mean_iu_sel]
    iter_mean_iu = dat['iteration'][mean_iu_sel]

    fig, ax = plt.subplots(nrows=2, ncols=2)

    plt.subplot(2, 2, 1)
    plt.plot(iter_train_acc, train_acc, label='train')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.plot(iter_mean_iu, mean_iu, label='val')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.plot(iter_train_loss, train_loss, label='train')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(iter_val_loss, val_loss, label='val')
    plt.xlabel('iteration')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(osp.join(log_dir, 'log_plots.png'), bbox_inches='tight')


def plot_log(log_path):
    log_dir, _ = osp.split(log_path)
    epoch = []
    iteration = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    g = lambda x: x if x != '' else float('nan')
    reader = csv.reader(open(log_path, 'rb'))
    next(reader)  # Skip header row.
    for line in reader:
        line_fields = [g(x) for x in line]
        epoch.append(float(line_fields[0]))
        iteration.append(float(line_fields[1]))
        train_loss.append(float(line_fields[2]))
        train_acc.append(float(line_fields[3]))
        val_loss.append(float(line_fields[4]))
        val_acc.append(float(line_fields[5]))

    epoch = np.array(epoch)
    iteration = np.array(iteration)
    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    val_loss = np.array(val_loss)
    val_acc = np.array(val_acc)

    train_loss_sel = ~np.isnan(train_loss)
    train_loss = train_loss[train_loss_sel]
    iter_train_loss = iteration[train_loss_sel]

    train_acc_sel = ~np.isnan(train_acc)
    train_acc = train_acc[train_acc_sel]
    iter_train_acc = iteration[train_acc_sel]

    val_loss_sel = ~np.isnan(val_loss)
    val_loss = val_loss[val_loss_sel]
    iter_val_loss = iteration[val_loss_sel]

    val_acc_sel = ~np.isnan(val_acc)
    val_acc = val_acc[val_acc_sel]
    iter_val_acc = iteration[val_acc_sel]

    fig, ax = plt.subplots(nrows=2, ncols=2)

    plt.subplot(2, 2, 1)
    plt.plot(iter_train_acc, train_acc, label='train', alpha=0.5, color='C0')
    box_pts = np.rint(np.sqrt(len(train_acc))).astype(np.int)
    plt.plot(iter_train_acc, savgol_smooth(train_acc, box_pts), color='C0')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.title('Training')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    plt.plot(iter_val_acc, val_acc, label='val', alpha=0.5, color='C1')
    box_pts = np.rint(np.sqrt(len(val_acc))).astype(np.int)
    plt.plot(iter_val_acc, savgol_smooth(val_acc, box_pts), color='C1')
    plt.grid()
    plt.legend()
    plt.title('Validation')
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    plt.plot(iter_train_loss, train_loss, label='train', alpha=0.5, color='C0')
    box_pts = np.rint(np.sqrt(len(train_loss))).astype(np.int)
    plt.plot(iter_train_loss, savgol_smooth(train_loss, box_pts), color='C0')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.plot(iter_val_loss, val_loss, label='val', alpha=0.5, color='C1')
    box_pts = np.rint(np.sqrt(len(val_loss))).astype(np.int)
    plt.plot(iter_val_loss, savgol_smooth(val_loss, box_pts), color='C1')
    plt.xlabel('iteration')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(osp.join(log_dir, 'log_plots.png'), bbox_inches='tight')


def savgol_smooth(y, box_pts):
    # use the Savitzky-Golay filter for 1-D smoothing
    if box_pts % 2 == 0:
        box_pts += 1
    y_smooth = scipy.signal.savgol_filter(y, box_pts, 2)
    return y_smooth



#python -c "from utils import plot_log_csv; plot_log_csv('$LOG_FILE')"