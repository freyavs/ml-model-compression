from collections.abc import Callable


def write_to_file(f: str, text: str):
    with open(f, 'a') as file:
        file.write(text)

def write_accuracy_to_file(file: str, accuracy: float):
    write_to_file(file, f"{accuracy}, ")

def get_accuracy_saver(teacher_file: str, student_file: str, scratch_file: str) -> Callable:
    def save_accuracy(who: str, accuracy: float) -> None:
        if who == 'teacher':
            write_accuracy_to_file(teacher_file, accuracy)
        if who == 'student':
            write_accuracy_to_file(student_file, accuracy)
        if who == 'scratch':
            write_accuracy_to_file(scratch_file, accuracy)
        else:
            raise ValueError(f"Can't save accuracy for {who}")

    return save_accuracy

