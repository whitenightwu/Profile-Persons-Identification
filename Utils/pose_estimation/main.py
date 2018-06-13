import argparse
import os

import cv2
from face_align import *
from pose_estimation import get_angle
from facial_landmarks import detect_landmarks

parser = argparse.ArgumentParser(description='Pytorch Branch Finetuning')
parser.add_argument('--path', metavar='DIR', default='', help='path to folder')


# Read all jpg images in folder.
def readImages(path):
    # Create array of array of images.
    imagesArray = dict();

    # List all files in the directory and read points from text files one by one
    for filePath in os.listdir(path):
        if filePath.endswith(".jpg"):
            img_path = os.path.join(path, filePath)
            img = cv2.imread(img_path);
            imagesArray[filePath] = img;

    return imagesArray;


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path

    images_dict = readImages(path);
    w_out = 300;
    h_out = 300;

    for file, img in images_dict.items():
        angle = get_angle(img, file)
        print(file + '   ' + '  '.join(angle))
