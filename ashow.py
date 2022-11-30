from matplotlib import pyplot as plt

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
