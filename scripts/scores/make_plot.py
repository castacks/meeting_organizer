import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

if __name__ == '__main__':
    fps = [x for x in os.listdir() if '.npy' in x]
    res = [np.load(x) for x in fps]
    res = np.stack(res, axis=-1)

    x = np.arange(len(res))
    resm = -res.mean(axis=-1)
    ress = res.std(axis=-1)

    sns.set_style('darkgrid')
    plt.plot(x, resm)
    plt.fill_between(x, resm - ress, resm + ress, alpha=0.3)
    plt.title('ScheduleWorld-v0')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.show()
