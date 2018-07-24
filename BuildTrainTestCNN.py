from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from DataLoader import DataLoader
from Constants import *

'''
CNN described in https://github.com/SamVenkatesh/FakeBlock/blob/master/README.md
Training for 200 epochs took ~ 3 hours on my 2017 Macbook Pro. 
If you have a computer with a beefy GPU I recommend you try 500+, Keras and TensorFlow have crazy GPU optimizations.
'''


class NNModel:
    def __init__(self):
        self.dataLoader = DataLoader()
        self.model = None

    def build_model(self, learning_rate=0.03, learning_decay=1e-5, learning_momentum=0.4):
        # Inspired by AlexNet:
        # https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        inputs = Input(shape=(FACE_SIZE, FACE_SIZE, 1))
        x = Conv2D(filters=64, kernel_size=5, activation='relu', input_shape=(FACE_SIZE, FACE_SIZE, 1))(inputs)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Conv2D(filters=64, kernel_size=5, activation='relu')(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Conv2D(filters=128, kernel_size=4, activation='relu')(x)
        # Trying to avoid overfitting.
        # Possibly  unnecessary if dataset is sufficiently diversified
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        outputs = Dense(units=len(EMOTIONS), activation='softmax')(x)

        self.model = Model(inputs, outputs)
        sgd = SGD(lr=learning_rate, decay=learning_decay, momentum=learning_momentum)
        self.model.compile(loss='mse', optimizer=sgd)

    def train_model(self, training_epochs=200, training_batch_size=50):
        x_train, x_test, y_train, y_test = self.dataLoader.load_from_save()
        print('->Training Model')
        # TODO: If deploying to AWS, dump to log
        self.model.fit(x=x_train, y=y_train, epochs=training_epochs, batch_size=training_batch_size, verbose=1, shuffle=True)

    def eval_model(self, eval_batch_size=50):
        x_train, x_test, y_train, y_test = self.dataLoader.load_from_save()
        print('->Evaluating Model')
        # TODO: If deploying to AWS, dump to log
        eval = self.model.evaluate(x_test, y_test, batch_size=eval_batch_size, verbose=1)
        return eval

    def make_prediction(self, image):
        if image is None:
            return None
        image = image.reshape([-1, FACE_SIZE, FACE_SIZE, 1])
        return self.model.predict(image)
