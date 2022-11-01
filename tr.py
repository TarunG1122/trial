#Computer Vision - display camera output, read image and video formats
import cv2

#system 
import os

#to open the trained image dataset
from keras.models import load_model

#to perform mathematical operations on arrays
import numpy as np



# to perform time related operations
import time

from google.colab.patches import cv2_imshow

def model():
    import cv2
    face = cv2.CascadeClassifier(r"/content/Driver-Drowsiness-detection-using-CNN-and-open-cv-with-warning-alarm/haar cascade files/haarcascade_frontalface_alt.xml")

# loading haarcascade file for left eye detection
    leye = cv2.CascadeClassifier(r"/content/Driver-Drowsiness-detection-using-CNN-and-open-cv-with-warning-alarm/haar cascade files/haarcascade_lefteye_2splits.xml")

# loading haarcascade file for right eye detection
    reye = cv2.CascadeClassifier(r"/content/Driver-Drowsiness-detection-using-CNN-and-open-cv-with-warning-alarm/haar cascade files/haarcascade_righteye_2splits.xml")
    
    return face, leye , reye


def live_detection(face,leye,reye):
    # Computer Vision - display camera output, read image and video formats
    import cv2

    # system
    import os

    # to open the trained image dataset
    from keras.models import load_model

    # to perform mathematical operations on arrays
    import numpy as np

    # to perform time related operations
    import time

    from google.colab.patches import cv2_imshow

    cnt = 0
    cap = cv2.VideoCapture("/content/oc.mp4")
    
    face = cv2.CascadeClassifier(r"/content/Driver-Drowsiness-detection-using-CNN-and-open-cv-with-warning-alarm/haar cascade files/haarcascade_frontalface_alt.xml")

# loading haarcascade file for left eye detection
    leye = cv2.CascadeClassifier(r"/content/Driver-Drowsiness-detection-using-CNN-and-open-cv-with-warning-alarm/haar cascade files/haarcascade_lefteye_2splits.xml")

# loading haarcascade file for right eye detection
    reye = cv2.CascadeClassifier(r"/content/Driver-Drowsine
    while True:
        # will read each frame and we store the image in a frame variable.
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        # OpenCV algorithm for object detection takes gray images in the input.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Face detection
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))  # Face detection
        # Left eye detection
        left_eye = leye.detectMultiScale(gray)
        # Right eye detection
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            # Draws rectangle for detected face.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
            count = count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            # My model is trained on 24*24 images
            r_eye = cv2.resize(r_eye, (24, 24))
            # Normalization so the model can works efficiently
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = (model.predict(r_eye) > 0.5).astype("int32")
            if (rpred[0][1] == 1):
                lbl = 'Open'
            if (rpred[0][1] == 0):
                lbl = 'Closed'
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]
            count = count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = (model.predict(l_eye) > 0.5).astype("int32")
            if (lpred[0][1] == 1):
                lbl = 'Open'
            if (lpred[0][1] == 0):
                lbl = 'Closed'
            break

        if (rpred[0][1] == 0 and lpred[0][1] == 0):
            score = score + 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            score = score - 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if (score < 0):
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if (score > 15):
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)

            if (thicc < 16):
                thicc = thicc + 2
            else:
                thicc = thicc - 2
                if (thicc < 2):
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        cv2_imshow(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
