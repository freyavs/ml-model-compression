from matplotlib import pyplot as plt

alphas = []
accs   = []
losses = []
with open('out.txt', 'r') as f:
    for line in f.readlines():
        # results = line.split(':')[1]
        results = line
        results = results.split(',')
        
        alphas.append(float(results[0]))
        accs.append(round(float(results[1]), 2))
        losses.append(round(float(results[2]), 2))

# fig = plt.gcf()
# fig.set_size_inches(5, 4)
plt.scatter(alphas, accs, color='orange')
plt.plot(alphas, accs, color='orange', linestyle='dotted')
plt.xlabel('Temperature')
plt.ylabel('Accuracy')
plt.xticks([1, 5, 10, 15])
plt.xlim([0, max(alphas)+0.5])
# plt.ylim([0, max(accs)+0.5])
plt.ylim([0, 1])
plt.savefig('temp_acc.png', dpi=200)
plt.show()
plt.scatter(alphas, losses, color='orange')
plt.plot(alphas, losses, color='orange', linestyle='dotted')
plt.xlabel('Temperature')
plt.ylabel('Loss')
plt.xticks([1, 5, 10, 15])
plt.xlim([0, max(alphas)+0.5])
plt.ylim([0, max(losses)+0.25])
plt.savefig('temp_loss.png', dpi=200)
plt.show()
