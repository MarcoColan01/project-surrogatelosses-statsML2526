from abc import ABC, abstractmethod
import numpy as np 

class Loss(ABC):
    c_phi = None
    average = False

    @abstractmethod
    def phi(self, z):
        pass

    @abstractmethod
    def dphi(self, z):
        pass

    @abstractmethod
    def step_size(self, t, lamb, L):
        pass


class HingeLoss(Loss):
    average = True

    def phi(self, z):
        return np.maximum(0,1-z)   
    def dphi(self, z):
        return -(z<1).astype(float)
    def step_size(self, t, lamb, L):
        return 1/(lamb*t)


class LogisticLoss(Loss):
    c_phi = 1/(4*np.log(2))

    def phi(self, z):
        return np.logaddexp(0, -z) / np.log(2)
    def dphi(self,z):
        return -np.exp(-np.logaddexp(0,z)) / np.log(2)
    def step_size(self, t, lamb, L):
        return 1/L

class SquaredLoss(Loss):
    c_phi = 2.0

    def phi(self, z):
        return (1-z)**2
    def dphi(self,z):
        return -2 * (1-z)
    def step_size(self, t, lamb, L):
        return 1/L
    