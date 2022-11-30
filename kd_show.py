from matplotlib import pyplot as plt

teacher = [0.6579, 0.6922, 0.6866, 0.6849, 0.6775, 0.6936]
student = [0.7098, 0.6817, 0.7011, 0.6979, 0.7133, 0.724]
scratch = [0.6977, 0.6963, 0.6950, 0.7034, 0.697, 0.7094]

# fig = plt.gcf()
# fig.set_size_inches(5, 4)
plt.scatter([ 0.25 for _ in range(len(teacher))], teacher, color='orange', label='teacher')
plt.scatter([ 0.50 for _ in range(len(student))], student, color='red', label='student')
plt.scatter([ 0.75 for _ in range(len(scratch))], scratch, color='blue', label='scratch')
# plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.xticks([0.25, 0.50, 0.75], ['teacher', 'student', 'scratch'])
plt.xlim([0, 1])
# plt.ylim([0, max(accs)+0.5])
plt.ylim([0.5, 1])
plt.legend()
plt.savefig('teacher_student.png', dpi=200)
plt.show()

teacher = []
student = []

