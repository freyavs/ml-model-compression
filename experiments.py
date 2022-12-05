from kd_loop import kd_loop
from util import get_accuracy_saver
from result_metrics import compression_result

def experiment_1(teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    teacher, student, scratch = kd_loop("mnist", epochs=3, prune_before=True, save_accuracy=save_accuracy)
    compression_result(teacher, student)
    compression_result(student, scratch)

def experiment_2(teacher_file: str, student_file: str, scratch_file: str):
    save_accuracy = get_accuracy_saver(teacher_file, student_file, scratch_file)

    teacher, student, scratch = kd_loop("mnist", epochs=3, prune_before=True, save_accuracy=save_accuracy)
    compression_result(teacher, student)
    compression_result(student, scratch)

if __name__ == '__main__':
    #physical_devices = tf.config.list_physical_devices('GPU') 
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    experiment_1('teacher_file1.txt', 'student_file1.txt', 'scratch_file1.txt')
    experiment_2('teacher_file2.txt', 'student_file2.txt', 'scratch_file2.txt')
