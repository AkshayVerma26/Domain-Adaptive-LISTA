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
    def add_noise(self, y_lst, snr):
        std_arr = np.array([])
        for i in range(y_lst.shape[1]):
            noise_std = np.linalg.norm(y_lst[:,i]) / (10**(snr / 20.0) * np.sqrt(y_lst.shape[0]))
            std_arr = np.append(std_arr, noise_std)
        std = np.mean(std_arr)
        noise = self.seed.randn(y_lst.shape[0])*std
        for i in range(y_lst.shape[1]):
            y_lst[:,i] += noise
        return y_lst

    def data_gen(self, A, p, snr_lst):
        Y = []
        X = self.generate_sparse_signal(p)
        for snr in snr_lst:
            y = self.generate_measurement(A, X)
            y_noisy = self.add_noise(y, snr)
            Y.append(y_noisy)
        return X, Y

def data_mix(seed, x5, y5, x15, y15, x30, y30):
    l = x5.shape[1]
    sel = l//3
    
    idx5 = seed.choice(l, sel, replace=False)
    idx15 = seed.choice(l, sel, replace=False)
    idx30 = seed.choice(l, sel, replace=False)
    
    x_mixed = np.concatenate((x5[:,idx5], x15[:,idx15], x30[:,idx30]), axis=1)
    y_mixed = np.concatenate((y5[:,idx5], y15[:,idx15], y30[:,idx30]), axis=1)
    
    indices = np.arange(x_mixed.shape[1])
    np.random.shuffle(indices)
    X_mixed = x_mixed[:, indices]
    Y_mixed = y_mixed[:, indices]
    
    return X_mixed, Y_mixed

# Plotter class
class plotter():
    # line plot for HR, MSE variation with SNR
    def hr_mse_vs_sigma(self, SNR_list, MSE_list, HR_list, SNR_train):
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax1.set_xlabel('X Axis')
        ax1.set_ylabel('MSE(dB)', color="green")
        ax1.plot(SNR_list, MSE_list, color="green")
        ax1.tick_params(axis='y', labelcolor="green")
        plt.grid()
        ax2 = ax1.twinx()
        ax2.set_ylabel('Hit Rate(%)', color="red")
        ax2.plot(SNR_list, HR_list, color="red")
        ax2.tick_params(axis='y', labelcolor="red")
        plt.title(f"MSE, HR vs SNR\nLISTA trained at SNR={SNR_train}dB")
        plt.xticks(range(5, 55, 5))
        plt.show()

    # stem plot to analyze recovered and original sparse signal
    def stem_plot(self, X_test, X_est, SNR_list, SNR_input, SNR_train, samp=0):
        plt.figure(figsize=(10,8))
        plt.stem(X_test[:,samp], basefmt=" ", linefmt="y-", markerfmt="yo", label="Original X")
        plt.stem(X_est[SNR_list.index(SNR_input)][:,samp], basefmt=" ", linefmt="g--", markerfmt="gx", label="Recovered X")
        plt.legend(loc='best')
        plt.grid()
        plt.title(f"Signal SNR={SNR_input}dB\nLISTA trained at SNR={SNR_train}dB")
        plt.show()