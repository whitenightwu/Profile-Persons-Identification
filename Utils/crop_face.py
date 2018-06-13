import csv
import io
import math
import os

import cv2


def extract(saveFile):
    with io.open(saveFile, 'w', encoding='utf-8') as saveF:
        with io.open("./umdfaces_batch3/" + "umdfaces_batch3_ultraface.csv", 'r', encoding='utf-8') as F:
            reader = csv.reader(F, delimiter=',')
            for row in reader:
                ID, path, x, y, w, h, yaw = row[0], row[1], row[4], row[5], row[6], row[7], row[8]
                if os.path.exists(path):
                    saveF.write('\t'.join([row[0], row[1], row[4], row[5], row[6], row[7], row[8]]) + '\n')


def cropped(dataset_path):
    target_path = "./cropped/"
    dataset_path = "./umdfaces_batch2/"  # "./umdfaces_batch3/"
    with io.open(saveFile, 'r', encoding='utf-8') as saveF:
        reader = csv.reader(saveF, delimiter='\t')
        for row in reader:
            ID, path_img, x, y, w, h, yaw = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            h = round(float(h))
            w = round(float(w))
            x = math.floor(float(x))
            y = math.floor(float(y))
            if (w > 180 and h > 180):
                target_path = dataset_path + path_img
                image = cv2.imread(target_path)
                frame = image[y:y + h, x:x + w]
                img = target_path + path_img
                directory = target_path + path_img.split('/')[0]
                if not os.path.exists(directory):
                    os.makedirs(directory)
                if (w > 300 or h > 300):
                    frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(img, frame)


def file_create():
    list1 = "img_list.csv"
    list2 = "img_list2.csv"
    saveTrain = "train.txt"
    saveVal = "val.txt"
    saveFile = "list.txt"

    with io.open(saveFile, 'w', encoding='utf-8') as saveF:
        prev_ID = 0
        ID = 1
        count = 1
        with io.open(list1, 'r', encoding='utf-8') as list1:
            reader = csv.reader(list1, delimiter='\t')
            for row in reader:
                path_img, yaw = row[1], row[6]
                directory = "./cropped/" + path_img
                if os.path.exists(directory):
                    if (prev_ID != int(row[0])):
                        ID += 1
                    saveF.write('\t'.join([str(ID), path_img, yaw]) + '\n')
                    count += 1
                    prev_ID = int(row[0])

        prev_ID = 2
        with io.open(list2, 'r', encoding='utf-8') as list2:
            reader = csv.reader(list2, delimiter='\t')
            for row in reader:
                path_img, yaw = row[1], row[6]
                directory = "./cropped/" + path_img
                if os.path.exists(directory):
                    if (prev_ID != int(row[0])):
                        ID += 1
                    saveF.write('\t'.join([str(ID), path_img, yaw]) + '\n')
                    count += 1
                    prev_ID = int(row[0])

        print('Total ' + str(count))
        print('ID ' + str(ID))


def create_samples():
    list = "list.txt"
    train = "train.txt"
    val = "val.txt"
    hash = dict()

    with io.open(list, 'r', encoding='utf-8') as list:
        reader = csv.reader(list, delimiter='\t')
        for row in reader:
            ID, path_img, yaw = row[0], row[1], row[2]
            if ID not in hash:
                hash[ID] = []
            hash[ID].append([path_img, yaw])

    keys = []
    for key in hash.keys():
        if len(hash[key]) < 2:
            keys.append(key)

    for key in keys:
        if key in hash and len(hash[key]) < 2:
            del hash[key]

    train_dict = dict()
    val_dict = dict()
    for key in hash.keys():
        train_dict[key] = []
        val_dict[key] = []

    for key in hash.keys():
        value = hash[key]
        size = len(value)
        if size > 40:
            size = 40
        col1 = round(size * 0.6)
        i = 0
        for v in value:
            if i < col1:
                train_dict[key].append(v)
            if col1 <= i < size:
                val_dict[key].append(v)
            i += 1

    with io.open(train, 'w', encoding='utf-8') as train:
        with io.open(val, 'w', encoding='utf-8') as val:
            for key in hash.keys():
                train_v = train_dict[key]
                val_v = val_dict[key]
                for v in train_v:
                    train.write('\t'.join([key, v[0], v[1]]) + '\n')
                for v in val_v:
                    val.write('\t'.join([key, v[0], v[1]]) + '\n')

