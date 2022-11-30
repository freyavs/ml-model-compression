import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from cifar import Distiller
from keras.utils.vis_utils import plot_model

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_teacher():
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
    plot_model(teacher, to_file='teacher_network.png', show_layer_names=False, show_shapes=True)
    teacher.summary()
    return teacher

def get_student():
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
    plot_model(student, to_file='small_network.png', show_layer_names=False, show_shapes=True)
    student.summary()
    return student


def main():
    teacher = get_teacher()
    student = get_student()

    # Clone student for later comparison
    scratch = keras.models.clone_model(student)

    # Prepare the train and test dataset.
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    # Train teacher as usual
    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train and evaluate teacher on data.
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        teacher = keras.models.load_model('teacher')
        teacher.evaluate(x_test, y_test)
    else:
        teacher.fit(x_train, y_train, epochs=5)
        teacher.evaluate(x_test, y_test)
        teacher.save('teacher')

    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.3,
        temperature=7,
    )

    # Distill teacher to student
    distiller.fit(x_train, y_train, epochs=5)

    # Evaluate student on test dataset
    distiller.evaluate(x_test, y_test)
    distiller.student.save('student')

    # Train student as doen usually
    scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train and evaluate student trained from scratch.
    scratch.fit(x_train, y_train, epochs=5)
    scratch.evaluate(x_test, y_test)
    scratch.save('scratch')

if __name__ == '__main__':
    main()
