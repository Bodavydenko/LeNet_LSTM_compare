import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Flatten, Dense, Dropout,SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.datasets import mnist
from keras.utils import np_utils
from keras import utils
from keras.optimizer_v1 import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import random

# Класс для создания нейронной сети LeNet
class LeNet:
    @staticmethod
    def build():
        model = Sequential()

        model.add(Conv2D(20, kernel_size=5, padding="same", input_shape=(28, 28, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2), strides=2))

        model.add(Conv2D(50, kernel_size=5, padding="same", input_shape=(28, 28, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D((2, 2), strides=2))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(10))
        model.add(Activation("softmax"))

        return model

# Класс для создания нейронной сети LSTM
class Lstm:
    @staticmethod
    def build():
        model = Sequential()

        model.add(LSTM(128, input_shape=(28, 28)))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        return model


# Собираем модели
opt = 'adam'

model_LeNet = LeNet.build()
model_LeNet.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print(model_LeNet.summary())

model_LSTM = Lstm.build()
model_LSTM.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print(model_LSTM.summary())

# Форматируем датасет
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Обучаем модели

epochs_num = 30

LeNet_log = model_LeNet.fit(x_train, y_train, batch_size=32, epochs=epochs_num, validation_split=0.2)
model_LeNet.evaluate(x_test, y_test)

Lstm_log = model_LSTM.fit(x_train, y_train, batch_size=32, epochs=epochs_num, validation_split=0.2)
model_LSTM.evaluate(x_test, y_test)

#Строим график ошибки

plt.plot(LeNet_log.history['loss'], label="Тренировочная_LeNet")
plt.plot(LeNet_log.history['val_loss'], label="Валидационная_LeNet")
plt.plot(Lstm_log.history['loss'], label="Тренировочная_Lstm")
plt.plot(Lstm_log.history['val_loss'], label="Валидационная_Lstm")
plt.title(label='Функция потери')
plt.legend()
plt.xlabel('Эпоха')
plt.ylabel('Ошибка')
plt.show()

#Строим график ошибки

plt.plot(LeNet_log.history['accuracy'], label="Тренировочная_LeNet")
plt.plot(LeNet_log.history['val_accuracy'], label="Валидационная_LeNet")
plt.plot(Lstm_log.history['accuracy'], label="Тренировочная_Lstm")
plt.plot(Lstm_log.history['val_accuracy'], label="Валидационная_Lstm")
plt.title(label='Точность')
plt.legend()
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.show()

# Проверяем случайные варианты

while True:
    #n = int(input())
    n = random.randrange(1,1000);
    x = np.expand_dims(x_test[n], axis=0)
    print("LeNet: "+str(np.argmax(model_LeNet.predict(x))))
    print("LSTM: " + str(np.argmax(model_LSTM.predict(x))))
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()
