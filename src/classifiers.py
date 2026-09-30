import numpy as np

class ClassifierLin:
    def __init__(self, loss, lam=0.05, EPOCHS=2000, tol=1e-10):
        self.loss = loss
        self.lam = lam
        self.EPOCHS = EPOCHS
        self.tol = tol
        self.w = None
        self.mask = None
        self.n_iter = 0
        self.history = {}

    def decision_function(self, X):
        return X @ self.w

    def predict(self, X):
        return np.where(self.decision_function(X) >0, 1,-1)

    def error(self, X, y):
        return np.mean(self.predict(X) != y)

    def phi_risk(self, X, y):
        return np.mean(self.loss.phi(y*self.decision_function(X)))

    def fit(self, X, y, X_test=None, y_test=None):
        samples, features = X.shape
        self.mask = np.r_[np.ones(features-1), 0.0]
        w = np.zeros(features)
        self.w = w.copy()
        L = None
        if self.loss.c_phi is not None:
            L = self.loss.c_phi * np.linalg.norm(X,2)**2 / samples + self.lam
        self.history = {k: [] for k in ("objective", "train_phi", "train_error", "test_phi", "test_error")}
        self._log(X,y, X_test, y_test)

        for epoch in range(1, self.EPOCHS+1):
            grad = X.T @(self.loss.dphi(y*(X @ w)) * y) / samples + self.lam * self.mask * w
            w -= self.loss.step_size(epoch, self.lam, L)*grad
            self.w = self.w + (w - self.w) / (epoch + 1) if self.loss.average else w
            self._log(X,y,X_test, y_test)
            F = self.history["objective"]
            if L is not None and abs(F[-1]-F[-2]) < self.tol * max(1.0, abs (F[-2])): break

        self.n_iter = epoch
        return self

    def _log(self, X,y,X_test,y_test):
        h, phi = self.history, self.phi_risk(X,y)
        h["objective"].append(phi+self.lam/2*np.sum(self.mask * self.w **2))
        h["train_phi"].append(phi)
        h["train_error"].append(self.error(X,y))
        if X_test is not None:
            h["test_phi"].append(self.phi_risk(X_test, y_test))
            h["test_error"].append(self.error(X_test, y_test))
