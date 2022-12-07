import sys
from tensorflow import keras
import numpy as np
from distiller import Distiller
from result_metrics import compression_result
from networks import get_student_mnist, get_teacher_mnist
from networks import get_student_cifar10, get_student_smaller_cifar10, get_teacher_cifar10
from prune import prune

def mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype('float32') / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    return x_train, x_test, y_train, y_test

def cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_train = np.reshape(x_train, (-1, 32, 32, 3))
    x_test = x_test.astype('float32') / 255.0
    x_test = np.reshape(x_test, (-1, 32, 32,3))

    return x_train, x_test, y_train, y_test

def get_data(data):
    match data:
        case 'mnist':
            x_train, x_test, y_train, y_test = mnist() 
        case 'cifar10':
            x_train, x_test, y_train, y_test = cifar10() 
        case _:
            raise ValueError(f'{data} is not a valid dataset')

    return x_train, x_test, y_train, y_test

def get_model_and_data(model, data):
    match (data, model):
        case ('mnist', 'teacher'):
            model = get_teacher_mnist()
        case ('mnist', 'student' | 'scratch'):
            model = get_student_mnist()
        case ('cifar10', 'teacher'):
            model = get_teacher_cifar10()
        case ('cifar10', 'student' | 'scratch'):
            model = get_student_smaller_cifar10()
        case _:
            raise ValueError(f'{data}, {model} is no valid model configuration')

    return model, *get_data(data)

def kd_loop(data="mnist", teacher=None, epochs=1, prune_teacher=False, prune_student=False, temperature=7, alpha=0.3, save_accuracy=lambda f,a: (f,a)):
    teacher = kd_loop_teacher(data, epochs, prune_teacher, save_accuracy)
    student = kd_loop_student(data, epochs, teacher, prune_student, temperature, alpha, save_accuracy)
    scratch = kd_loop_scratch(data, epochs, save_accuracy)

    return teacher, student, scratch


def kd_loop_teacher(data="mnist", epochs=1, apply_pruning=False, save_accuracy=lambda f,a: (f,a)):
    teacher, x_train, x_test, y_train, y_test = get_model_and_data('teacher', data)

    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("\n--- EVALUATING TEACHER ---\n")
    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        teacher = keras.models.load_model(f'output/teacher-{data}')
    else:
        teacher.fit(x_train, y_train, epochs=epochs)
        teacher.save(f'output/teacher-{data}')
    _, teacher_accuracy = teacher.evaluate(x_test, y_test, verbose=0)

    save_accuracy('teacher', teacher_accuracy)
    print('Teacher test accuracy:', teacher_accuracy)

    if apply_pruning:
        print("\n--- PRUNING & RE-EVALUATING TEACHER ---\n")
        new_teacher = teacher.clone_model()
        new_teacher = prune(new_teacher, x_train, y_train, x_test, y_test, epochs=epochs)
        compression_result(teacher, new_teacher)
        teacher = new_teacher
    
    return teacher

def kd_loop_student(data="mnist", teacher=None, epochs=1, apply_pruning=False, temperature=7, alpha=0.3, save_accuracy=lambda f,a: (f,a)):
    if not teacher:
        raise ValueError("No teacher was provided")

    student, x_train, x_test, y_train, y_test = get_model_and_data('student', data)

    print("\n--- EVALUATING STUDENT ---\n")
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=alpha,
        temperature=temperature,
    )

    distiller.fit(x_train, y_train, epochs=epochs)

    student_accuracy = distiller.evaluate(x_test, y_test, verbose=0)
    save_accuracy('student', student_accuracy)
    print('Student test accuracy:', student_accuracy)

    distiller.student.save(f'output/student-{data}')

    if apply_pruning:
        print("\n--- PRUNING & RE-EVALUATING STUDENT ---\n")
        new_student = student.clone_model()
        new_student = prune(new_student, x_train, y_train, x_test, y_test, epochs=epochs)
        compression_result(student, new_student)
        student = new_student

    return student

def kd_loop_scratch(data="mnist", epochs=1, save_accuracy=lambda f,a: (f,a)):
    scratch, x_train, x_test, y_train, y_test = get_model_and_data('scratch', data)

    print("\n--- EVALUATING SCRATCH ---\n")
    scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    scratch.fit(x_train, y_train, epochs=epochs)

    _, scratch_accuracy = scratch.evaluate(x_test, y_test, verbose=0)
    save_accuracy('scratch', scratch_accuracy)
    print('Scratch test accuracy:', scratch_accuracy)

    scratch.save(f'output/scratch-{data}')

    return scratch