import cv2
import math
import dlib
import numpy as np
import argparse
from imutils import face_utils

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

parser = argparse.ArgumentParser(description='Pytorch Branch Finetuning')
parser.add_argument('-ilf', '--image-list-file', default='img.txt', type=str, metavar='N', help='image list file')
parser.add_argument('--path', metavar='DIR', default='', help='path to dataset')


def main():
    args = parser.parse_args()
    path = args.path

    file_image_names = open(args.ilf, 'r')
    for image_name in iter(file_image_names):
        img_path = path + image_name
        frame = cv2.imread(img_path)
        landmarks = detect_landmarks(img_path)
        imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)

        cv2.line(frame, nose, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
        cv2.line(frame, nose, tuple(imgpts[0].ravel()), (255, 0,), 3)  # BLUE
        cv2.line(frame, nose, tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED

        remapping = [2, 3, 0, 4, 5, 1]
        for index in range(len(landmarks) / 2):
            random_color = tuple(np.random.random_integers(0, 255, size=3))
            cv2.circle(frame, (landmarks[index * 2], landmarks[index * 2 + 1]), 5, random_color, -1)
            cv2.circle(frame, tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)

        for j in xrange(len(rotate_degree)):
            cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

        cv2.imwrite('poseEstimation/' + img_path, frame)

    file_image_names.close()


def face_orientation(frame, landmarks):
    size = frame.shape  # (height, width, color_channel)

    image_points = np.array([
        (landmarks[4], landmarks[5]),  # Nose tip
        (landmarks[10], landmarks[11]),  # Chin
        (landmarks[0], landmarks[1]),  # Left eye left corner
        (landmarks[2], landmarks[3]),  # Right eye right corne
        (landmarks[6], landmarks[7]),  # Left Mouth corner
        (landmarks[8], landmarks[9])  # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals

    center = (size[1] / 2, size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.CV_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    print(yaw)

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[4], landmarks[5])


def detect_landmarks(file):
    # load the input image and convert it to grayscale
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        needed = [shape[33], shape[8], shape[36], shape[45], shape[48], shape[54]]

        return needed
        # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
    # for (x, y) in needed:
    #    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # show the output image with the face detections + facial landmarks
    # cv2.imwrite('lands' + file, image)


if __name__ == '__main__':
    main()
