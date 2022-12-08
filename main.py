import sys
from tensorflow import keras
import numpy as np
from distiller import Distiller
from result_metrics import compression_result
from networks import get_student_mnist, get_teacher_mnist
from networks import get_student_cifar10, get_student_smaller_cifar10, get_teacher_cifar10
from prune import prune

def mnist():
    teacher = get_teacher_mnist()
    student = get_student_mnist()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    return teacher, student, x_train, x_test, y_train, y_test

def cifar10():
    teacher = get_teacher_cifar10()
    student = get_student_smaller_cifar10()

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 32, 32, 3))
    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 32, 32,3))

    return teacher, student, x_train, x_test, y_train, y_test


def kd_loop(data = "mnist", epochs = 1, prune_before = False, prune_after = False):
    if data == "mnist":
        teacher, student, x_train, x_test, y_train, y_test = mnist() 
    elif data == "cifar10":
        teacher, student, x_train, x_test, y_train, y_test = cifar10() 
    else:
        raise ValueError(f"{data} is not a valid dataset")

    scratch = keras.models.clone_model(student)

    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("\n--- EVALUATING TEACHER ---\n")
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        teacher = keras.models.load_model(f'output/teacher-{data}')
        _, teacher_accuracy = teacher.evaluate(x_test, y_test, verbose=0)
    else:
        teacher.fit(x_train, y_train, epochs=epochs)
        _, teacher_accuracy = teacher.evaluate(x_test, y_test, verbose=0)
        teacher.save(f'output/teacher-{data}')

    print('Teacher test accuracy:', teacher_accuracy)

    if prune_before:
        print("\n--- PRUNING & RE-EVALUATING TEACHER ---\n")
        new_teacher = keras.models.clone_model(teacher)
        new_teacher = prune(new_teacher, x_train, y_train, x_test, y_test, epochs=epochs)
        compression_result(teacher, new_teacher)
        teacher = new_teacher

    print("\n--- EVALUATING STUDENT ---\n")
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.3,
        temperature=7,
    )

    distiller.fit(x_train, y_train, epochs=epochs)

    result = distiller.evaluate(x_test, y_test, verbose=0)
    print('Student test accuracy:', result)
    distiller.student.save(f'output/student-{data}')

    if prune_after:
        print("\n--- PRUNING & RE-EVALUATING STUDENT ---\n")
        new_student = keras.models.clone_model(student)
        new_student = prune(new_student, x_train, y_train, x_test, y_test, epochs=epochs)
        compression_result(student, new_student)
        student = new_student

    scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("\n--- EVALUATING SCRATCH ---\n")
    scratch.fit(x_train, y_train, epochs=epochs)
    _, scratch_accuracy = scratch.evaluate(x_test, y_test, verbose=0)
    print('Scratch test accuracy:', scratch_accuracy)
    scratch.save(f'output/scratch-{data}')

    return teacher, student, scratch

if __name__ == '__main__':
    #physical_devices = tf.config.list_physical_devices('GPU') 
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    teacher, student, scratch = kd_loop("mnist", epochs=3, prune_before=True)
    compression_result(teacher, student)
    compression_result(student, scratch)
