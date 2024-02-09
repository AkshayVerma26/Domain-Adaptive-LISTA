import torch
import numpy as np
from LISTA_essential import LISTA_train, LISTA_test
import matplotlib.pyplot as plt
import time
import torch.nn as nn

# Dataset class
class datagen():
    def __init__(self, n, m, k, seed):
        self.n = n
        self.m = m
        self.k = k
        self.seed = seed

    # Generate sparse signal X with amplitude 1 and equally spaced
    def generate_sparse_signal(self, p):
        x_lst = np.zeros((self.n, p))
        for i in range(x_lst.shape[1]):
            index = self.seed.choice(self.n//self.k, 1, replace=False)
            indices = np.array([index[0], index[0]+self.n//self.k, index[0]+2*self.n//self.k])
            x = np.ones(self.k)
            x_lst[indices,i] = x
        return x_lst

    # Generate measurement matrix A
    def generate_measurement_matrix(self):
        mes_mat = self.seed.randn(self.m, self.n)
        return mes_mat

    # Generate measurements Y
    def generate_measurement(self, A, x_lst):
        return np.matmul(A, x_lst)

    # Add noise to measurements Y
    def add_noise(self, y_lst,sigma):
        noise = self.seed.randn(y_lst.shape[0])*sigma
        for i in range(y_lst.shape[1]):
            y_lst[:,i] += noise
        return y_lst

    def data_gen(self, A, p, sigma_lst):
        Y = []
        X = self.generate_sparse_signal(p)
        SIGMA = []
        for sigma in sigma_lst:
            y = self.generate_measurement(A, X)
            y_noisy = self.add_noise(y,sigma)
            Y.append(y_noisy)
            SIGMA.append(sigma)
        return X, np.array(Y), np.array(SIGMA)

def data_mix(seed, x1, y1, sigma1, x2, y2, sigma2, x3, y3, sigma3):
    l = x1.shape[1]
    sel = l//3
    
    idx1 = seed.choice(l, sel, replace=False)
    idx2 = seed.choice(l, sel, replace=False)
    idx3 = seed.choice(l, sel, replace=False)
    
    x_mixed = np.concatenate((x1[:,idx1], x2[:,idx2], x3[:,idx3]), axis=1)
    y_mixed = np.concatenate((y1[:,idx1], y2[:,idx2], y3[:,idx3]), axis=1)
    sigma_mixed = np.concatenate((sigma1[idx1], sigma2[idx2], sigma3[idx3]), axis=1)
    
    indices = np.arange(x_mixed.shape[1])
    np.random.shuffle(indices)
    X_mixed = x_mixed[:, indices]
    Y_mixed = y_mixed[:, indices]
    Sigma_mixed = sigma_mixed[:, indices]
    
    return X_mixed, Y_mixed, Sigma_mixed

# Plotter class
class plotter():
    # line plot for HR, MSE variation with SNR
    def hr_mse_vs_sigma(self, sigma_list, MSE_list, HR_list, sigma_train):
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax1.set_xlabel(r'$\sigma$')
        ax1.set_ylabel('MSE(dB)', color="green")
        ax1.plot(sigma_list, MSE_list, color="green")
        ax1.tick_params(axis='y', labelcolor="green")
        plt.grid()
        ax2 = ax1.twinx()
        ax2.set_ylabel('Hit Rate(%)', color="red")
        ax2.plot(sigma_list, HR_list, color="red")
        ax2.tick_params(axis='y', labelcolor="red")
        plt.title(r"MSE, HR vs $\sigma$"+ f"\nLISTA trained at "+r"$\sigma$"+f"={sigma_train}")
        # plt.xticks(sigma_list)
        plt.show()

    # stem plot to analyze recovered and original sparse signal
    def stem_plot(self, X_test, X_est, sigma_i, sigma_t, samp=0):
        plt.figure(figsize=(12,7))
        plt.stem(X_test[:,samp], basefmt=" ", linefmt="y-", markerfmt="yo", label="Original X")
        plt.stem(X_est[:,samp], basefmt=" ", linefmt="g--", markerfmt="gx", label="Recovered X")
        plt.legend(loc='best')
        plt.grid()
        plt.title("Signal "+r"$\sigma$"+f"={sigma_i}\nLISTA trained at "+r"$\sigma$"+f"={sigma_t}dB")
        plt.show()