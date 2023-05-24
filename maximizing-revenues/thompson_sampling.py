import numpy as np
import matplotlib.pyplot as plt
import random

N = 1000
d = 9

conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.2, 0.08, 0.01]
x = np.array(np.zeros([N, d]))

for i in range(0, N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            x[i,j] = 1

strategies_selected_rs = []
strategies_selected_ts = []
total_reward_rs = 0
total_reward_ts = 0
numbers_of_reward_1 = [0] * d
numbers_of_reward_0 = [0] * d

for n in range(0, N):
    # Random Sampling
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = x[n, strategy_rs]
    total_reward_rs += reward_rs

    # Thompson Sampling
    strategy_ts = 0
    max_random = 0
    for strategy in range(0, d):
        random_beta = random.betavariate(numbers_of_reward_1[strategy] + 1, numbers_of_reward_0[strategy] + 1)
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = strategy
    reward_ts = x[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_reward_1[strategy_ts] += 1
    else:
        numbers_of_reward_0[strategy_ts] += 1

    strategies_selected_ts.append(strategy_ts)
    total_reward_ts = total_reward_ts + reward_ts

absolute_return = total_reward_ts - total_reward_rs
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100

print("Absolute return: {:.0f}".format(absolute_return))
print("Relative return: {:.0f}%".format(relative_return))

plt.hist(strategies_selected_ts)
plt.title("Strategies selected using TS")
plt.xlabel("Strategies")
plt.ylabel("Number of selections")
plt.show()
