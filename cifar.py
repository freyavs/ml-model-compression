import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from distiller import Distiller
from keras.utils.vis_utils import plot_model

def writetofile(f, text):
    with open(f, 'a') as f:
        f.write(text)
        f.write('\n')

def get_teacher():
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
    plot_model(teacher, to_file='teacher_network_cifar.png', show_layer_names=False, show_shapes=True)
    teacher.summary()
    return teacher

def get_student_smaller():
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
        name="student",
    )
    plot_model(student, to_file='small_network_cifar.png', show_layer_names=False, show_shapes=True)
    student.summary()
    return student

def get_student():
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
    plot_model(student, to_file='small_network_cifar.png', show_layer_names=False, show_shapes=True)
    student.summary()
    return student

def main():
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    student = get_student()
    scratch = keras.models.clone_model(student)
    

    # Prepare the train and test dataset.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 32, 32, 3))
    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 32, 32,3))

    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        teacher = keras.models.load_model('models/teacher_cifar')
        teacher.evaluate(x_test, y_test)
    else:
        teacher = get_teacher()
        teacher.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        teacher.fit(x_train, y_train, epochs=4)
        teacher.evaluate(x_test, y_test)
        teacher.save('models/teacher_cifar')
        exit()

    # distiller= Distiller(student=student, teacher=teacher)
    # distiller.compile(optimizer=keras.optimizers.Adam(),
    #                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
    #                  student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                  distillation_loss_fn=keras.losses.KLDivergence(),
    #                  alpha=0.3,
    #                  temperature=7)
    # distiller.fit(x_train, y_train, epochs=4)
    # distiller.evaluate(x_test, y_test)
    # distiller.student.save(f'models/student_cifar_smaller')
    # exit()

    scratch.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    scratch.fit(x_train, y_train, epochs=4)
    scratch.evaluate(x_test, y_test)
    scratch.save('models/scratch_cifar')

if __name__ == '__main__':
    main()
