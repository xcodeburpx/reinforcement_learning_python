import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import sys

style.use('ggplot')

K_NUMBER = 1000

def k_bandit():



    mu, sigma = 0, 1
    rewards = np.random.normal(mu,sigma,K_NUMBER)
    epsilons = [0.01, 0.1]

    decisions = [0,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(rewards)
    for decision in decisions:

        if decision == 0:
            ax1 = fig.add_subplot(211)
            alpha = 0.6
            for epsilon in epsilons:
                timer = 0
                q_values = [{"qval": 0, "occur": 0, "value": 0} for i in range(K_NUMBER)]
                data_time = []
                data_log = []
                values = [q_values[i]['value'] for i in range(K_NUMBER)]

                for i in range(10000):
                    prob = np.random.random()
                    #print(prob)
                    if prob >= 1-epsilon:
                        action = np.argmax(values)
                    else:
                        action = np.random.randint(0,K_NUMBER)

                    reward = rewards[action]
                    q_val = q_values[action]
                    q_val['occur'] += 1
                    #q_val['value'] = q_val['value'] + (1/q_val['occur'])*(reward - q_val['value'])
                    q_val['value'] = q_val['value'] + alpha * (reward - q_val['value'])
                    q_values[action] = q_val

                    values[action] = q_values[action]['value']

                    timer += 1

                    runnMean = np.mean(values)

                    data_time.append(timer)
                    data_log.append(runnMean)


                ax1.plot(data_time, data_log, label='Epsilon %.4f'%epsilon)
            ax1.set_xlabel('Time')
            ax1.set_ylabel("Average reward")
            ax1.set_title("K-bandit problem\nAlpha = %f"%alpha)
            ax1.legend()

        else:
            ax2 = fig.add_subplot(212)
            for epsilon in epsilons:
                    timer = 0
                    q_values = [{"qval": 0, "occur": 0, "value": 0} for i in range(K_NUMBER)]
                    data_time = []
                    data_log = []
                    values = [q_values[i]['value'] for i in range(K_NUMBER)]

                    for i in range(10000):
                        prob = np.random.random()
                        # print(prob)
                        if prob >= 1 - epsilon:
                            action = np.argmax(values)
                        else:
                            action = np.random.randint(0, K_NUMBER)

                        reward = rewards[action]
                        q_val = q_values[action]
                        q_val['occur'] += 1
                        q_val['value'] = q_val['value'] + (1/q_val['occur'])*(reward - q_val['value'])
                        #q_val['value'] = q_val['value'] + alpha * (reward - q_val['value'])
                        q_values[action] = q_val

                        values[action] = q_values[action]['value']

                        timer += 1

                        runnMean = np.mean(values)

                        data_time.append(timer)
                        data_log.append(runnMean)

                    ax2.plot(data_time, data_log, label='Epsilon %.4f' % epsilon)
            ax2.set_xlabel('Time')
            ax2.set_ylabel("Average reward")
            ax2.set_title("K-bandit problem\nAlpha = 1/n")
            ax2.legend()
            plt.show()

plt.show()

if __name__ == '__main__':
    k_bandit()


