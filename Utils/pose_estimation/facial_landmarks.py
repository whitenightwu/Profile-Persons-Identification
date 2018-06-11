import os
import cv2
import dlib
from imutils import face_utils

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def detect_landmarks(image, fileName="noname", write=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)  # detect faces in the grayscale image
    needed = []

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        needed = [shape[33], shape[8], shape[36], shape[45], shape[48], shape[54]]

        # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
        for (x, y) in needed:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # show the output image with the face detections + facial landmarks
    if (write):
        img_path = str(os.path.join(os.getcwd() + '\landmarks', fileName))
        cv2.imwrite(img_path, image)
    return needed
