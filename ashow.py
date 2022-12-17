from matplotlib import pyplot as plt

accs_student = [0.6638000011444092, 0.6355000138282776, 0.5612999796867371, 0.5196999907493591, 0.4307999908924103, 0.3244999945163727, 0.2313999980688095, 0.1873999983072281]
accs_scratch = [0.6700999736785889, 0.6211000084877014, 0.6251999735832214, 0.5266000032424927, 0.4327000081539154, 0.30559998750686646, 0.21889999508857727, 0.20640000700950623]

accs_student_2 = [0.6371999979019165, 0.5964000225067139, 0.6265000104904175, 0.5170000195503235, 0.4260999858379364, 0.2770000100135803, 0.22840000689029694, 0.19359999895095825, ]
accs_scratch_2 = [0.6777999997138977, 0.6111000180244446, 0.6560999751091003, 0.5562999844551086, 0.43970000743865967, 0.2770000100135803, 0.23849999904632568, 0.17980000376701355, ]

sizes = [28866, 18468, 15002, 7602, 4094, 2457, 1236, 1044]
accs_student.reverse()
accs_scratch.reverse()
accs_student_2.reverse()
accs_scratch_2.reverse()
sizes.reverse()
plt.plot(sizes, accs_student, label="student")
plt.plot(sizes, accs_scratch, label="scratch")

plt.plot(sizes, accs_student_2, label="student2")
plt.plot(sizes, accs_scratch_2, label="scratch2")

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
