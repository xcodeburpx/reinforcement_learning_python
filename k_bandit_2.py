import matplotlib.pyplot as plt
import numpy as np


def k_bandit():
    k_number = 1000
    stepSize = 0.1
    indices = np.arange(k_number)
    sampleAverage = True
    epsilons = [0.3, 0.1, 0.01, 0.0]

    bandit_q = []
    figureNumber = 0

    for i in range(k_number):
        bandit_q.append(np.random.randn())
    print("Bandit Q:",bandit_q)

    plt.figure(figureNumber)

    for epsilon in epsilons:

        timer = 0
        averageReward = 0
        averageLog = []
        estimate_q = []
        actionCount = []

        for i in range(k_number):
            estimate_q.append(0)
            actionCount.append(0)

        for i in range(1000000):
            timer += 1
            if epsilon > 0:
                if np.random.binomial(1, epsilon) == 1:
                    np.random.shuffle(indices)
                    action = indices[0]

                else:
                    action = np.argmax(estimate_q)



            reward = bandit_q[action]
            averageReward = (timer-1.0)/timer * averageReward + reward/timer
            averageLog.append(averageReward)
            actionCount[action] += 1
            if sampleAverage:
                estimate_q[action] += 1/actionCount[action] *(reward - estimate_q[action])
            else:
                estimate_q[action] += stepSize *(reward - estimate_q[action])
        print("Estimated Q:",estimate_q)
        plt.plot(averageLog, label="epsilon=%.5f"%epsilon)
    plt.title("Epsilons")
    plt.legend()
    plt.show()

k_bandit()