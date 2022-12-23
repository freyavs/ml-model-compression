import sys
from tensorflow import keras
import numpy as np
from distiller import Distiller
from result_metrics import compression_result
from networks import *
from prune import prune
from util import mean_accuracy

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

# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-convnet-for-cifar-10-and-cifar-100-classification-with-keras.md
def cifar100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_train = np.reshape(x_train, (-1, 32, 32, 3))
    x_test = x_test.astype('float32') / 255.0
    x_test = np.reshape(x_test, (-1, 32, 32,3))

    return x_train, x_test, y_train, y_test

def get_data(data):
    if data == 'mnist':
        x_train, x_test, y_train, y_test = mnist() 
    elif data == 'cifar10':
        x_train, x_test, y_train, y_test = cifar10() 
    elif data == 'cifar100':
        x_train, x_test, y_train, y_test = cifar100() 
    else:
        raise ValueError(f'{data} is not a valid dataset')

    return x_train, x_test, y_train, y_test

def get_model_and_data(model, data):
    x = (data, model)
    if x == ('mnist', 'teacher'):
         model = get_teacher_mnist()
    elif x == ('mnist','scratch') or x == ('mnist','student'):
        model = get_student_mnist()
    elif x == ('cifar10', 'teacher'):
        model = get_teacher_cifar10()
    elif x == ('cifar100', 'teacher'):
        model = get_teacher_cifar100()
    elif x == ('cifar10', 'student') or x == ('cifar10', 'scratch'):
        model = get_student_small_0_cifar10()
    elif x == ('cifar100', 'student') or x == ('cifar100', 'scratch'):
        model = get_student_small_1_cifar10()
    else:
        raise ValueError(f'{data}, {model} is no valid model configuration')

    return model, *get_data(data)

def kd_loop(data="mnist", teacher=None, epochs=1, prune_teacher=False, prune_student=False, temperature=7, alpha=0.3, save=lambda f,a: (f,a)):
    teacher = kd_loop_teacher(data, epochs, prune_teacher, save)
    student = kd_loop_student(data, epochs, teacher, prune_student, temperature, alpha, save)
    scratch = kd_loop_scratch(data, epochs, save)

    return teacher, student, scratch


def kd_loop_teacher(data="mnist", epochs=1, apply_pruning=False, save=lambda f,a: (f,a), load_teacher=False):
    teacher, x_train, x_test, y_train, y_test = get_model_and_data('teacher', data)
    
    # optimizer = 'SGD' #if resnet!!
    optimizer = keras.optimizers.Adam()

    teacher.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("\n--- EVALUATING TEACHER ---\n")
    if (len(sys.argv) > 1 and sys.argv[1] == 'load') or load_teacher:
        teacher = keras.models.load_model(f'output/teacher-{data}')
        history = None
    else:
        history = teacher.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        teacher.save(f'output/teacher-{data}')
    _, teacher_accuracy = teacher.evaluate(x_test, y_test, verbose=0)

    save('teacher', teacher_accuracy)
    print('Teacher test accuracy:', teacher_accuracy)

    if apply_pruning:
        print("\n--- PRUNING & RE-EVALUATING TEACHER ---\n")
        new_teacher = keras.models.clone_model(teacher)
        new_teacher = prune(new_teacher, x_train, y_train, x_test, y_test, epochs=20)
        compression_result(teacher, 'teacher', True, save)
        compression_result(new_teacher, 'teacher', True, save)
        teacher = new_teacher
    
    return teacher, history

def kd_loop_student(data="mnist", teacher=None, student=None, epochs=1, apply_pruning=False, temperature=4, alpha=0.1, save=lambda f,a: (f,a)):
    if not teacher:
        raise ValueError("No teacher was provided")

    if not student:
        student, x_train, x_test, y_train, y_test = get_model_and_data('student', data)
    else:
        _, x_train, x_test, y_train, y_test = get_model_and_data('student', data)

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

    history = distiller.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    student_accuracy, _ = distiller.evaluate(x_test, y_test, verbose=0)
    student_accuracy = mean_accuracy(history)
    save('student', student_accuracy)
    print('Student test accuracy:', student_accuracy)

    distiller.student.save(f'output/student-{data}')

    compression_result(student, 'student', True, save)
    if apply_pruning:
        print("\n--- PRUNING & RE-EVALUATING STUDENT ---\n")
        new_student = keras.models.clone_model(student)
        new_student = prune(new_student, x_train, y_train, x_test, y_test, epochs=15)
        compression_result(new_student, 'student', True, save)
        student = new_student

    return student, history

def kd_loop_scratch(data="mnist", scratch=None, epochs=1, apply_pruning=True, save=lambda f,a: (f,a)):
    if not scratch:
        scratch, x_train, x_test, y_train, y_test = get_model_and_data('student', data)
    else:
        _, x_train, x_test, y_train, y_test = get_model_and_data('student', data)

    print("\n--- EVALUATING SCRATCH ---\n")
    scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    history = scratch.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    _, scratch_accuracy = scratch.evaluate(x_test, y_test, verbose=0)
    scratch_accuracy = mean_accuracy(history)
    save('scratch', scratch_accuracy)
    print('Scratch test accuracy:', scratch_accuracy)

    scratch.save(f'output/scratch-{data}')

    compression_result(scratch, 'scratch', True, save)
    if apply_pruning:
        print("\n--- PRUNING & RE-EVALUATING SCRATCH ---\n")
        new_scratch = keras.models.clone_model(scratch)
        new_scratch = prune(new_scratch, x_train, y_train, x_test, y_test, epochs=15)
        compression_result(new_scratch, 'scratch', True, save)
        scratch = new_scratch

    return scratch
