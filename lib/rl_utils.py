import numpy as np


def compute_returns(rewards, gamma=1.0):
    assert 0 <= gamma <= 1.0
    if gamma == 1.0:
        returns = np.cumsum(rewards)[::-1]
    elif gamma == 0:
        returns = rewards
    else:
        returns = np.zeros_like(rewards)
        g = 0  # G_T
        for t in reversed(range(returns.shape[0])):  # T-1, T-2, ..., 0
            r = rewards[t]  # r_t+1
            g = r + gamma * g  # G_t = r_t+1 + gamma * G_t+1
            returns[t] = g

    total_return = returns[0]
    return total_return, returns  # G_0, G_1, ..., G_T-1


def one_hot_test(tensor, axis=-1):
    test = np.sum(tensor, axis=axis, dtype=np.int)
    assert (
        np.equal(test, np.ones_like(test, dtype=np.int)).all()
    )
