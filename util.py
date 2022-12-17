from collections.abc import Callable
import pandas as pd


def write_to_file(f: str, text: str):
    with open(f, 'a') as file:
        file.write(text)

def write_accuracy_to_file(file: str, accuracy: float):
    write_to_file(file, f"{accuracy}, ")

def get_accuracy_saver(teacher_file: str, student_file: str, scratch_file: str) -> Callable:
    def save_accuracy(who: str, accuracy: float) -> None:
        if who == 'teacher':
            write_accuracy_to_file(teacher_file, accuracy)
        elif who == 'teacher_size':
            write_accuracy_to_file(f"{teacher_file}_size", accuracy)
        elif who == 'teacher_parameters':
            write_accuracy_to_file(f"{teacher_file}_parameters", accuracy)

        elif who == 'student':
            write_accuracy_to_file(student_file, accuracy)
        elif who == 'student_size':
            write_accuracy_to_file(f"{student_file}_size", accuracy)
        elif who == 'student_parameters':
            write_accuracy_to_file(f"{student_file}_parameters", accuracy)

        elif who == 'scratch':
            write_accuracy_to_file(scratch_file, accuracy)
        else:
            raise ValueError(f"Can't save accuracy for {who}")

    return save_accuracy

def mean_accuracy(history):
    df = pd.DataFrame(history.history)
    mean_accuracy = df.tail(5)['val_sparse_categorical_accuracy'].mean()
    return mean_accuracy
