import dlib
import numpy as np
import argparse
import os
from imutils import face_utils
from facial_landmarks import detect_landmarks


parser = argparse.ArgumentParser(description='Pytorch Branch Finetuning')
parser.add_argument('--ilf', '--image-list-file', default='img.txt', type=str, help='image list file')
parser.add_argument('--path', metavar='DIR', default='', help='path to dataset')

if __name__ == '__main__':
    args = parser.parse_args()
    path = os.getcwd()

    file_image_names = open(args.ilf, 'r')
    for image_name in iter(file_image_names):
        img_path = str(os.path.join(path, image_name))
        print(img_path)
        img_path = "/home/u0060/Profile-Persons-Identification/utils/37_0.jpg"
        frame = cv2.imread(img_path)
        landmarks = detect_landmarks(frame, fileName = image_name)
        imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)

        cv2.line(frame, tuple(nose), tuple(imgpts[1].ravel()), (0, 0, 255), 1)  # GREEN
        cv2.line(frame, tuple(nose), tuple(imgpts[0].ravel()), (0, 0, 255), 1)  # BLUE
        cv2.line(frame, tuple(nose), tuple(imgpts[2].ravel()), (0, 0, 255), 1)  # RED

        for index in range(len(landmarks)):
            cv2.circle(frame, tuple(modelpts[index].ravel().astype(int)), 2, (0, 255, 0), -1)

        cv2.imwrite("new01.jpg", frame)

    file_image_names.close()