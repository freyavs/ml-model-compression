from matplotlib import pyplot as plt


accs_student_2 = [0.6371999979019165, 0.5964000225067139, 0.6265000104904175, 0.5170000195503235, 0.4260999858379364, 0.2770000100135803, 0.22840000689029694, 0.19359999895095825, ]
accs_scratch_2 = [0.6777999997138977, 0.6111000180244446, 0.6560999751091003, 0.5562999844551086, 0.43970000743865967, 0.2770000100135803, 0.23849999904632568, 0.17980000376701355, ]

sizes = [28866, 18468, 15002, 7602, 4094, 2457, 1236, 1044]
accs_student_2.reverse()
accs_scratch_2.reverse()
sizes.reverse()

plt.plot(sizes, accs_student_2, label="student2")
# plt.plot(sizes, accs_scratch_2, label="scratch2")
#
# plt.legend()
# plt.show()

accuracy_student = [0.684660005569458, 0.6176999926567077, 0.5299799859523773, 0.5642399847507477, 0.45663999319076537, 0.41858, 0.3125999987125397, 0.2224399983882904, 0.20160000026226044]
kbs = [138.296, 111.11, 72.497, 59.492000000000004, 31.921, 18.735, 12.5, 7.853, 7.125]
params = [35506.0, 28866.0, 18468.0, 15002.0, 7602.0, 4094.0, 2457.0, 1236.0, 1044.0]
accuracy_scratch = [0.7289000034332276, 0.6559399962425232, 0.5709400057792664, 0.6042800068855285, 0.4859600067138672, 0.43442, 0.344, 0.2213799983263, 0.20116000175476073]

accuracy_student.reverse()
kbs.reverse()
params.reverse()
accuracy_scratch.reverse()

plt.plot(params, accuracy_student, label="student")
plt.plot(params, accuracy_scratch, label="scratch")
plt.legend()
plt.show()

exit()
alphas = []
accs   = []
losses = []
with open('alpha.txt', 'r') as f:
    for line in f.readlines():
        results = line.split(':')[1]
        results = results.split(',')
        
        alphas.append(float(results[0]))
        accs.append(round(float(results[1]), 2))
        losses.append(round(float(results[2]), 2))

# fig = plt.gcf()
# fig.set_size_inches(5, 4)
plt.scatter(alphas, accs, color='orange')
plt.plot(alphas, accs, color='orange', linestyle='dotted')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.xticks([i/10 for i in range(11)])
plt.xlim([-0.05, max(alphas)+0.05])
# plt.ylim([0, max(accs)+0.5])
plt.ylim([0, 1])
plt.savefig('alpha_acc.png', dpi=200)
plt.show()
plt.scatter(alphas, losses, color='orange')
plt.plot(alphas, losses, color='orange', linestyle='dotted')
plt.xlabel('Alpha')
plt.ylabel('Loss')
plt.xticks([i/10 for i in range(11)])
plt.xlim([-0.05, max(alphas)+0.05])
plt.ylim([0, max(losses)+0.25])
plt.savefig('alpha_loss.png', dpi=200)
plt.show()
