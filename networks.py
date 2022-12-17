import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from distiller import Distiller
from keras.utils.vis_utils import plot_model
from tensorflow.keras import layers

IMAGE_DIR = "images"


# 63% accuracy
def get_teacher_cifar100(summarize = False):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation='softmax'))
    teacher = model
    
    teacher = model 
    if summarize:
        plot_model(teacher, to_file=f'{IMAGE_DIR}/teacher_network_cifar.png', show_layer_names=False, show_shapes=True)
        teacher.summary()

    return teacher

# TODO: zelfde model als hierboven gebruiken maar kleiner? Of eventueel get_student_smaller_cifar10 gebruiken (heeft minder lagen)
# vooral letten op dat er geen overfitting is na nog maar 5 epochs ofzo
def get_student_smaller_cifar100(summarize = False):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters = 8, kernel_size=(2, 2), input_shape=(32, 32, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 16, kernel_size=(2, 2), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters = 32, kernel_size=(2, 2),activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(100, activation="softmax"))
    

    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

# 85.2% accuracy teacher network after 20 epochs
def get_teacher_cifar10(summarize = False):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10))
    teacher = model
   
    if summarize:
        plot_model(teacher, to_file=f'{IMAGE_DIR}/teacher_network_cifar.png', show_layer_names=False, show_shapes=True)
        teacher.summary()
    return teacher

def get_student_smaller_2_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=8,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=10,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_student_smaller_3_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=8,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=8,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_student_smaller_4_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=4,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=4,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_student_smaller_5_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=2,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=2,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_student_smaller_6_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=2,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=1,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_student_smaller_7_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=1,kernel_size=(8,8),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=1,kernel_size=(8,8),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_student_smaller_8_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=1,kernel_size=(8,8),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=1,kernel_size=(8,8),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(4,4)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

# teacher = 85, scratch = 67, student = 67 (beste optie)
def get_student_smaller_cifar10(summarize = False):
    model= tf.keras.Sequential()
    model.add(layers.Conv2D(filters=8,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # batch normalisatie verbert het (zonder netwerk groter te maken), maar maakt het precies minder stabiel?
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25)) # Drop 25% of the units from the layer.
    model.add(layers.Conv2D(filters=16,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
    model.add(layers.BatchNormalization()) # zie ^^ 
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(10))
    
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

# teacher= 85, scratch = 73, student = 73 (maar groter dan bovenste)
def get_student_small_cifar10(summarize = False):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    student = model
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

# teacher= 85, scratch = 64.7, student = 63.4 (redelijk slecht, gaat snel overfitten)
def get_student_cifar10(summarize = False):
    student = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ],
        name="student",
    )
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_teacher_mnist(summarize = False):
    # Create the teacher
    teacher = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
            layers.Flatten(),
            layers.Dense(10),
        ],
        name="teacher",
    )
    if summarize:
        plot_model(teacher, to_file=f'{IMAGE_DIR}/teacher_mnist.png', show_layer_names=False, show_shapes=True)
        teacher.summary()
    return teacher

def get_student_mnist(summarize = False):
    # Create the student
    student = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
            layers.Flatten(),
            layers.Dense(10),
        ],
        name="student",
    )
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/student_mnist.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student
