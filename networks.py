import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from distiller import Distiller
from keras.utils.vis_utils import plot_model
from tensorflow.keras import layers

IMAGE_DIR = "images"

def get_teacher_cifar10(summarize = False):
    teacher = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            
            tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ],
        name="teacher",
    )
    if summarize:
        plot_model(teacher, to_file=f'{IMAGE_DIR}/teacher_network_cifar.png', show_layer_names=False, show_shapes=True)
        teacher.summary()
    return teacher

def get_student_smaller_cifar10(summarize = False):
    student = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ],
        name="smaller student",
    )
    if summarize:
        plot_model(student, to_file=f'{IMAGE_DIR}/small_network_cifar.png', show_layer_names=False, show_shapes=True)
        student.summary()
    return student

def get_student_cifar10(summarize = False):
    student = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
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
