from kd_loop import *
from util import get_accuracy_saver
from result_metrics import compression_result
import tensorflow as tf
from pathlib import Path

def normal(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    for _ in range(5):
        teacher = kd_loop_teacher(data, epochs=5, save_accuracy=save_accuracy)
        kd_loop_student(data, epochs=3, teacher=teacher, save_accuracy=save_accuracy)
        kd_loop_scratch(data, epochs=3, save_accuracy=save_accuracy)

def teacher_pruned(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    for _ in range(5):
        teacher = kd_loop_teacher(data, epochs=5, apply_pruning=True, save_accuracy=save_accuracy)
        kd_loop_student(data, epochs=3, teacher=teacher, save_accuracy=save_accuracy)
        kd_loop_scratch(data, epochs=3, save_accuracy=save_accuracy)

def student_pruned(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    teacher = kd_loop_teacher(data, epochs=5, save_accuracy=save_accuracy)
    for _ in range(5):
        kd_loop_student(data, epochs=3, teacher=teacher, apply_pruning=True, save_accuracy=save_accuracy)

def teacher_and_student_pruned(data:str , teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    for _ in range(5):
        teacher = kd_loop_teacher(data, epochs=5, apply_pruning=True, save_accuracy=save_accuracy)
        kd_loop_student(data, epochs=3, teacher=teacher, apply_pruning=True, save_accuracy=save_accuracy)
        kd_loop_scratch(data, epochs=3, save_accuracy=save_accuracy)

def temperature_influence(teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    dataset = "cifar10"
    teacher = kd_loop_teacher(dataset, epochs=5)
    temperatures = range(21)
    alpha = 0.3

    for T in temperatures:
        kd_loop_student(dataset, teacher, epochs=3, temperature=T, alpha=alpha, save_accuracy=save_accuracy)

def alpha_influence(teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    dataset = "cifar10"
    teacher = kd_loop_teacher(dataset, epochs=5)
    temperature = 7
    alphas = [i/10 for i in range(11)]

    for alpha in alphas:
        kd_loop_student(dataset, teacher, epochs=3, temperature=temperature, alpha=alpha, save_accuracy=save_accuracy)

if __name__ == '__main__':
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

    normal('mnist', *filenames('normal', 'mnist')) 
    normal('cifar10', *filenames('normal', 'cifar10')) 

    teacher_pruned('mnist', *filenames('teacher_pruned', 'mnist')) 
    teacher_pruned('cifar10', *filenames('teacher_pruned', 'cifar10')) 

    student_pruned('mnist', *filenames('student_pruned', 'mnist')) 
    student_pruned('cifar10', *filenames('student_pruned', 'cifar10')) 

    teacher_and_student_pruned('mnist', *filenames('normal', 'mnist')) 
    teacher_and_student_pruned('cifar10', *filenames('normal', 'cifar10')) 

    temperature_influence(*filenames('temperature_influence'))
    alpha_influence(*filenames('alpha_influence'))

    


    # experiment_1('teacher_file1.txt', 'student_file1.txt', 'scratch_file1.txt')
    # experiment_2('teacher_file2.txt', 'student_file2.txt', 'scratch_file2.txt')
