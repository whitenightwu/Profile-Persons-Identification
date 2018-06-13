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
    global saveFile
    list1 = "img_list.csv"
    list2 = "img_list2.csv"
    saveTrain = "train.txt"
    saveVal = "val.txt"
    saveFile = "list.csv"

    with io.open(saveFile, 'w', encoding='utf-8') as saveF:
        ID = 0
        count = 0
        with io.open(list1, 'r', encoding='utf-8') as list1:
            reader = csv.reader(list1, delimiter='\t')
            for row in reader:
                path_img, yaw = row[1], row[6]
                directory = "./cropped/" + path_img.split
                if os.path.exists(directory):
                    count += 1
                    if (row[0] != ID):
                        ID += 1
                    saveF.write('\t'.join(ID, path_img, yaw) + '\n')
                prev_ID = row[0]
        print('Total' + count)
        print('ID ' + ID)


extract("img_list.csv")
