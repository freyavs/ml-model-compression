from kd_loop import *
from networks import *
from util import get_accuracy_saver
from result_metrics import compression_result
import tensorflow as tf
from pathlib import Path
import pandas as pd

def teacher_loss_graphic(data:str):
    _, history = kd_loop_teacher(data, epochs=50, load_teacher=False)
    df = pd.DataFrame(history.history)
    df = df.rename(columns={"loss": "train loss", "sparse_categorical_accuracy": "train accuracy", "val_loss": "test loss", "val_sparse_categorical_accuracy": "test accuracy"})
    ax = df.plot()
    ax.set_xlabel("epochs")
    ax.get_figure().savefig('teacher_loss.png')

def student_loss_graphic(data:str):
    teacher, _ = kd_loop_teacher(data, load_teacher=True)
    _, history = kd_loop_student(data, epochs=30, teacher=teacher)
    df = pd.DataFrame(history.history)
    df = df.rename(columns={"loss": "train loss", "sparse_categorical_accuracy": "train accuracy", "val_loss": "test loss", "val_sparse_categorical_accuracy": "test accuracy"})
    df.plot().get_figure().savefig('student_loss.png')

def big_to_small(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)
    teacher, history = kd_loop_teacher(data, epochs=20, save=save_accuracy, load_teacher=False)

    s = get_student_smaller_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()

    s = get_student_smaller_2_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()

    s = get_student_smaller_3_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()

    s = get_student_smaller_4_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()

    s = get_student_smaller_5_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()

    s = get_student_smaller_6_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()

    s = get_student_smaller_7_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()

    s = get_student_smaller_8_cifar10()
    _, history = kd_loop_student(data, student=s, epochs=13, teacher=teacher, save=save_accuracy)
    kd_loop_scratch(data, scratch=s, epochs=13, save=save_accuracy)
    # pd.DataFrame(history.history).plot()


def normal(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    for _ in range(1):
        # TODO: voor grafiekjes is het interessanter om epochs hoger te zetten
        teacher = kd_loop_teacher(data, epochs=1, save=save_accuracy, load_teacher=True)
        student = kd_loop_student(data, epochs=25, teacher=teacher, save=save_accuracy)
        scratch = kd_loop_scratch(data, epochs=25, save=save_accuracy)
        compression_result(teacher,student, "teacher_file")

def teacher_pruned(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    for _ in range(1):
        teacher = kd_loop_teacher(data, epochs=5, apply_pruning=True, save=save_accuracy)
        kd_loop_student(data, epochs=3, teacher=teacher, save=save_accuracy)

def student_pruned(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    teacher = kd_loop_teacher(data, epochs=5, save=save_accuracy)
    for _ in range(1):
        kd_loop_student(data, epochs=3, teacher=teacher, apply_pruning=True, save=save_accuracy)

def teacher_and_student_pruned(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    for _ in range(1):
        teacher = kd_loop_teacher(data, epochs=5, apply_pruning=True, save=save_accuracy)
        kd_loop_student(data, epochs=3, teacher=teacher, apply_pruning=True, save=save_accuracy)

def temperature_influence(teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    dataset = "cifar10"
    teacher = kd_loop_teacher(dataset, epochs=5)
    temperatures = range(21)
    alpha = 0.3

    for T in temperatures:
        kd_loop_student(dataset, teacher, epochs=3, temperature=T, alpha=alpha, save=save_accuracy)

def alpha_influence(teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    dataset = "cifar10"
    teacher = kd_loop_teacher(dataset, epochs=5)
    temperature = 7
    alphas = [i/10 for i in range(11)]

    for alpha in alphas:
        kd_loop_student(dataset, teacher, epochs=3, temperature=temperature, alpha=alpha, save=save_accuracy)
    
def main():
    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    OUTPUT_DIR = './experiment_results'
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # file_names(name)
    def filenames(name, dataset=None):
        teacher = f'{OUTPUT_DIR}/{dataset}_{name}_teacher' if dataset else f'{OUTPUT_DIR}/{name}_teacher'
        student = f'{OUTPUT_DIR}/{dataset}_{name}_student' if dataset else f'{OUTPUT_DIR}/{name}_student'
        scratch = f'{OUTPUT_DIR}/{dataset}_{name}_scratch' if dataset else f'{OUTPUT_DIR}/{name}_scratch'
        return (teacher, student, scratch) 

    # teacher_loss_graphic('cifar10')
    student_loss_graphic('cifar10')
    # big_to_small('cifar10', *filenames('big_to_small', 'cifar10')) 
    return

    # normal('cifar100', *filenames('normal', 'cifar100')) 
    normal('cifar10', *filenames('normal', 'cifar10')) 
    return
    normal('mnist', *filenames('normal', 'mnist')) 
    normal('cifar100', *filenames('normal', 'cifar100')) 

    big_to_small('cifar10', *filenames('big_to_small', 'cifar10')) 

    teacher_pruned('mnist', *filenames('teacher_pruned', 'mnist')) 
    teacher_pruned('cifar10', *filenames('teacher_pruned', 'cifar10')) 

    student_pruned('mnist', *filenames('student_pruned', 'mnist')) 
    student_pruned('cifar10', *filenames('student_pruned', 'cifar10')) 

    teacher_and_student_pruned('mnist', *filenames('normal', 'mnist')) 
    teacher_and_student_pruned('cifar10', *filenames('normal', 'cifar10')) 

    temperature_influence(*filenames('temperature_influence'))
    alpha_influence(*filenames('alpha_influence'))

if __name__ == '__main__':
    main()
    
    # experiment_1('teacher_file1.txt', 'student_file1.txt', 'scratch_file1.txt')
    # experiment_2('teacher_file2.txt', 'student_file2.txt', 'scratch_file2.txt')
