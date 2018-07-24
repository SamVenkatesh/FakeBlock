from os.path import join
import numpy as np
from Constants import *
from sklearn.model_selection import train_test_split

class DataLoader:
    def load_from_save(self):
        images = np.load(join(DATA_DIR, DATA_IMAGE_FILE))
        images = images.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        labels = np.load(join(DATA_DIR, DATA_LABEL_FILE)).reshape([-1, len(EMOTIONS)])
        return train_test_split(images, labels, test_size=0.20, random_state=42)
