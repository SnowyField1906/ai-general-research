import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression

np.random.seed(seed=42)


def test(dataset, plot=True):
    if dataset == 'uscrime':
        df = pd.read_csv('data/uscrime.txt', sep='\t')
    elif dataset == 'BostonHousing':
        df = pd.read_csv('data/BostonHousing.txt', sep=',')
    elif dataset == 'diamonds':
        df = pd.read_csv('data/diamonds.csv', sep=' ')
    else:
        print('No data set under the input name.')
        return

    print(df.head(10))

    df = df.to_numpy()
    x = df[:, 0:-1]
    y = df[:, -1]

    learner = LinearRegression(x, y, plot=plot)
    learner.fit()
    y_pred = learner.predict(x)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    ax.set_title(f'{dataset}')
    ax.plot(y, y_pred, marker='o', markersize=4, linestyle='None', color='#1f77b4')
    ax.axline((np.mean(y), np.mean(y)), slope=1., color='red')
    ax.set_ylabel('Predicted value')
    ax.set_xlabel('True value')

    plt.tight_layout()
    plt.savefig(f'document/figures/our-result-{dataset}.png')


def test_learning_rate(dataset, plot=True):
    if dataset == 'uscrime':
        df = pd.read_csv('data/uscrime.txt', sep='\t')
    elif dataset == 'BostonHousing':
        df = pd.read_csv('data/BostonHousing.txt', sep=',')
    elif dataset == 'diamonds':
        df = pd.read_csv('data/diamonds.csv', sep=' ')
    else:
        print('No data set under the input name.')
        return

    df = df.to_numpy()
    x = df[:, 0:-1]
    y = df[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    lr_list = [0.2, 0.1, 0.001]
    mse_log_list = []
    for i in range(len(lr_list)):
        learner = LinearRegression(x_train, y_train, plot=plot, verbose=True)
        w, mse_log = learner.gradient_descend(learner.x_train, learner.y_train, lr=lr_list[i])
        mse_log_list.append(mse_log)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    ax.set_title('Learning curves of different learning rates')
    ax.plot(mse_log_list[0][:10], marker='.', label=f'lr = {lr_list[0]}')
    ax.plot(mse_log_list[1][:200], label=f'lr = {lr_list[1]}')
    ax.plot(mse_log_list[2][:200], linestyle='dashed', label=f'lr = {lr_list[2]}')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Iteration')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig('document/figures/diff-learning-rates.png')



# test('uscrime')
# test('BostonHousing')
test('diamonds')

# test_learning_rate('BostonHousing')

