import sys
from tensorflow import keras
import numpy as np
from cifar import Distiller
from distiller import Distiller
from result_metrics import compression_result
from networks import get_student_mnist, get_teacher_mnist
from networks import get_student_cifar10, get_student_smaller_cifar10, get_teacher_cifar10

def mnist():
    teacher = get_teacher_mnist()
    student = get_student_mnist()

    # Prepare the train and test dataset.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    return teacher, student, x_train, x_test, y_train, y_test

def cifar10():
    teacher = get_teacher_cifar10()
    student = get_student_cifar10()

    # Prepare the train and test dataset.
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 32, 32, 3))
    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 32, 32,3))

    return teacher, student, x_train, x_test, y_train, y_test


def kd_loop(data = "mnist", epochs = 1):
    if data == "mnist":
        teacher, student, x_train, x_test, y_train, y_test = mnist() 
    elif data == "cifar10":
        teacher, student, x_train, x_test, y_train, y_test = cifar10() 
    else:
        return

    # Clone student for later comparison
    scratch = keras.models.clone_model(student)

    # Train teacher as usual
    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    # Train and evaluate teacher on data.
    print("\n--- EVALUATING TEACHER ---\n")
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        teacher = keras.models.load_model(f'output/teacher-{data}')
        teacher.evaluate(x_test, y_test)
    else:
        teacher.fit(x_train, y_train, epochs=epochs)
        teacher.evaluate(x_test, y_test)
        teacher.save(f'output/teacher-{data}')

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
    distiller.fit(x_train, y_train, epochs=epochs)

    print("\n--- EVALUATING STUDENT ---\n")
    # Evaluate student on test dataset
    distiller.evaluate(x_test, y_test)
    distiller.student.save(f'output/student-{data}')

    # Train student as done usually
    scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("\n--- EVALUATING SCRATCH ---\n")
    # Train and evaluate student trained from scratch.
    scratch.fit(x_train, y_train, epochs=epochs)
    scratch.evaluate(x_test, y_test)
    scratch.save(f'output/scratch-{data}')

    return teacher, student


if __name__ == '__main__':
    #physical_devices = tf.config.list_physical_devices('GPU') 
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    teacher, student = kd_loop("cifar10")
    compression_result(teacher, student)
