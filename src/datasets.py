from pathlib import Path
from statistics import NormalDist
import numpy as np
from sklearn.model_selection import train_test_split

W_STAR = np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)]) 

def make_separable(n, gamma, rng):
    X = np.empty((0, 2))
    while True:
        Z = rng.uniform(-1, 1, size=(n, 2))
        X = np.vstack([X, Z[np.abs(Z @ W_STAR) >= gamma]])
        pos, neg = X[X @ W_STAR > 0], X[X @ W_STAR < 0]
        if min(len(pos), len(neg)) >= n // 2: break
    X = np.vstack([pos[: n // 2], neg[: n // 2]])
    y = np.repeat([1, -1], n // 2)
    perm = rng.permutation(n)
    return X[perm], y[perm]


def flip_labels(y, rho, rng):
    return np.where(rng.uniform(size=len(y)) < rho, -y, y)


def gaussian_mean(bayes_risk):
    return NormalDist().inv_cdf(1 - bayes_risk) * W_STAR


def make_gaussian(n, mu, rng):
    y = np.repeat([1, -1], n // 2)
    X = y[:, None] * mu + rng.standard_normal((n, 2))
    perm = rng.permutation(n)
    return X[perm], y[perm]

def eta_separable(X):
    return (X @ W_STAR > 0).astype(float)


def eta_label_flip(X, rho):
    return np.where(X @ W_STAR > 0, 1 - rho, rho)


def eta_gaussian(X, mu):
    return 1 / (1 + np.exp(-2 * X @ mu))


def stratified_splits(y, seeds, test_size):
    splits = [train_test_split(np.arange(len(y)), test_size=test_size,
                               stratify=y, random_state=s) for s in seeds]
    train_idx, test_idx = zip(*splits)
    return np.array(train_idx), np.array(test_idx)

def add_bias(X):
    return np.column_stack([X, np.ones(len(X))])


def load_dataset(name, run, data_dir):
    data = np.load(Path(data_dir) / f"{name}.npz")
    train, test = data["train_idx"][run], data["test_idx"][run]
    eta_test = data["eta"][test] if "eta" in data else None
    return add_bias(data["X"][train]), data["y"][train], add_bias(data["X"][test]), data["y"][test], eta_test
