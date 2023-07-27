
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


average_rewards = []
with open('rewards.txt') as f:
    for line in f:
        score = f.readline()
        try:
            average_rewards.append(float(score))
        except:
            print("no float")

print(average_rewards)

x = [i+1 for i in range(len(average_rewards))]

plt.plot(x, average_rewards)
plt.title('Running average of previous 250 scores')
plt.savefig("graph")