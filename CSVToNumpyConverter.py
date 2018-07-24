import Constants
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join

cascade_classifier = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_default.xml')

'''
Wrapping CSV strings into numpy arrays so we can dump them to disk.
Handling data with numpy is much easier and several times faster
than using standard library data structures.
'''


def data_to_image(data):
    new_image = np.fromstring(str(data),
                              dtype=np.uint8,
                              sep=' ')\
        .reshape((Constants.FACE_SIZE, Constants.FACE_SIZE))

    new_image = Image.fromarray(new_image).convert('RGB')
    # Need to flip
    new_image = np.array(new_image)[:, :, ::-1].copy()
    new_image = format_image(new_image)
    return new_image


'''
Basic one hot encoding vector. 
Specific emotion index is set to 1, everything else is 0.
'''


def encode_one_hot_emotion(x):
    d = np.zeros(len(Constants.EMOTIONS))
    d[x] = 1.0
    return d

'''
We need to isolate all faces in the image and retrieve
the one with the largest "area". 
Crop/transform it to network specs and return it.
'''


def format_image(image_to_format):
    image_to_format = cv2.cvtColor(image_to_format, cv2.COLOR_BGR2GRAY)

    image_border = np.zeros((150, 150), np.uint8)
    image_border[:, :] = 200
    image_border[
        int((150 / 2) - (Constants.FACE_SIZE / 2)): int((150 / 2) + (Constants.FACE_SIZE / 2)),
        int((150 / 2) - (Constants.FACE_SIZE / 2)): int((150 / 2) + (Constants.FACE_SIZE / 2))
    ] = image_to_format

    image_to_format = image_border
    detected_faces = cascade_classifier.detectMultiScale(
        image_to_format,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If no faces are found, return Null
    if not detected_faces:
        return None

    max_face = detected_faces[0]
    for face in detected_faces:
        if face[2] * face[3] > max_face[2] * max_face[3]:
            max_face = face

    # Chop image to face
    face = max_face
    image_to_format = image_to_format[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    # Resize image to fit network specs
    try:
        image_to_format = cv2.resize(image_to_format, (Constants.FACE_SIZE, Constants.FACE_SIZE),
                                     interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        # This happened once and now I'm scared to remove it.
        print("Image resize exception. Check input resolution inconsistency.")
        return None
    return image_to_format


data = pd.read_csv(join(Constants.DATA_DIR, Constants.DATASET_CSV_FILENAME))
# This data wrangling took me longer than I care to admit.
# Pandas + Numpy ftw
labels = []
images = []
total = data.shape[0]
for index, row in data.iterrows():
    emotion = encode_one_hot_emotion(row['emotion'])
    image = data_to_image(row['pixels'])

    if image is not None:
        labels.append(emotion)
        images.append(image)

    print("Conversion Progress: {}/{}".format(index+1, total))

print("Total: " + str(len(images)))
np.save(join(Constants.DATA_DIR, Constants.DATA_IMAGE_FILE), images)
np.save(join(Constants.DATA_DIR, Constants.DATA_LABEL_FILE), labels)

