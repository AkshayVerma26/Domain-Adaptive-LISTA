import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

def soft_thr(input_, theta_):
    return F.relu(input_-theta_)-F.relu(-input_-theta_)

def visu_shrink(sigma, M, numLayers):
    thr_visu = sigma*math.sqrt(2*math.log(M))
    thr_lst = [thr_visu*0.05*(0.1325**i) for i in range(numLayers)]
    threshold = torch.tensor(thr_lst)
    return threshold

class LISTA(nn.Module):
    def __init__(self, m, n, A, L, numIter, device):
        super(LISTA, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n, bias=False)
        
        self.numIter = numIter
        self.A = A
        self.L = L
        # self.alpha = alpha
        self.device = device

        self.thr = visu_shrink(0.057, 30, 15)
        # self.thr = nn.Parameter(thr.to(self.device), requires_grad=False)
        
    def __setter__(self, threshold):
        self.thr = threshold

    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        L = self.L
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/L)*np.matmul(A.T, A))
        S = S.float().to(self.device)
        B = torch.from_numpy((1/L)*A.T)
        B = B.float().to(self.device)
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)

    def forward(self, y):
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
            
        for iter in range(self.numIter):
            d = soft_thr(self._W(y) + self._S(d), self.thr[iter])
            x.append(d)
        return x    

# defining custom dataset
class dataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :] # since the input to the datset is tensor

def LISTA_train(X , Y, D, numEpochs, numLayers, device, learning_rate, sigma):

    m, n = D.shape
    
    Train_size = Y.shape[1]
    batch_size = 250
    print('Total dataset size is ', Train_size)
    if Train_size % batch_size != 0:
        print('Bad Training dataset size')

    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    Y_t = Y_t.float().to(device)

    D_t = torch.from_numpy(D.T)
    D_t = D_t.float().to(device)

    # we need to use ISTA to get X
    X_t = torch.from_numpy(X.T)
    X_t = X_t.float().to(device)

    # dataset_train = dataset(X_t, Y_t)
    valid_size = int(0.2 * Train_size)
    dataset_train = dataset(X_t[:-valid_size, :], Y_t[:-valid_size, :])
    dataset_valid = dataset(X_t[-valid_size:, :], Y_t[-valid_size:, :])
    print('DataSet size is: ', dataset_train.__len__())
    dataLoader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle = True)
    dataLoader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle = False)
    
    # compute the max eigen value of the D'*D
    T = np.matmul(D.T, D)
    eg, _ = np.linalg.eig(T)
    eg = np.abs(eg)
    L = np.max(eg)*1.001

    # calculating threshold
    # threshold = visu_shrink(sigma, Y.shape[0], numLayers)

    # Numpy Random State if rng (passed through arguments)
    net = LISTA(m, n, D, L, numLayers, device = device)
    net = net.float().to(device)
    #-------------------*****************-_-----------------
    net.weights_init()

    # build the optimizer and criterion
    criterion = nn.MSELoss()            # nn.SmoothL1Loss() -> original
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, betas = (0.9, 0.999))

    # list of losses at every epoch
    train_loss_list = []
    valid_loss_list = []
    time_list = []
    weight_list_S = []; weight_list_S_B = []
    weight_list_W = []; weight_list_W_B = []
    thr_list = []
    import time
    start = time.time()
    tol = 1e-5
    
    best_model = net; best_loss = 1e6
    lr = learning_rate
    # ------- Training phase --------------
    for epoch in range(numEpochs):
        
        if epoch == round(numEpochs*0.5):
            lr = learning_rate * 0.2
            optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas = (0.9, 0.999))
        elif epoch == round(numEpochs*0.75):
            lr = learning_rate * 0.02
            optimizer = torch.optim.Adam(net.parameters(), lr = lr, betas = (0.9, 0.999))
        else:
            pass

        tot_loss = 0
        net.train()
        
        for iter, data in enumerate(dataLoader_train):
            X_GT_batch, Y_batch = data
            net.__setter__(visu_shrink(sigma, 30, 15))
            X_batch_hat = net(Y_batch.float())  # get the outputs
            loss = 0
            for i in range(numLayers):
                loss += criterion(X_batch_hat[i].float(), X_GT_batch.float()) 
            # compute the losss
            loss /= numLayers
            tot_loss += loss.detach().cpu().data
            optimizer.zero_grad()   #clear the gradients
            loss.backward()     # compute the gradiettns

            optimizer.step()    # Update the weights
            net.zero_grad()

                
        if epoch % 1 == 0 :
            print('Training - Epoch: {}, Loss: {}'.format( epoch+1, tot_loss))

        # Validation stage
        with torch.no_grad():
            train_loss_list.append(tot_loss.detach().data/4) 
            tot_loss = 0
            for iter, data in enumerate(dataLoader_valid):
    
                X_GT_batch, Y_batch = data
                net.__setter__(visu_shrink(sigma, 30, 15))
                X_batch_hat = net(Y_batch.float())  # get the outputs
                loss = 0
                for i in range(numLayers):
                    loss += criterion(X_batch_hat[i].float(), X_GT_batch.float()) 
                # compute the losss
                loss /= numLayers
                tot_loss += loss.detach().cpu().data
            valid_loss_list.append(tot_loss)
            
            if best_loss > tot_loss:
                best_model = net
                best_loss = tot_loss

    return net

def LISTA_test(net, Y, D, device, sigma):

    # convert the data into tensors
    D_t = torch.from_numpy(D.T)
    D_t = D_t.float().to(device)
    X_final = np.zeros((3000, 100))

    for i in range(3000):
        with torch.no_grad():
            # Compute the output
            net.eval()
            net.__setter__(visu_shrink(sigma[i], 30, 15))
            Y_t = Y.T[i].reshape(1, -1)
            Y_t = torch.from_numpy(Y_t)
            X_lista = net(Y_t.float())
            if len(Y.shape) <= 1:
                X_lista = X_lista.view(-1)
            X_final[i] = X_lista[-1].cpu().numpy().T.squeeze()

    return X_final

# ======================== Old version ========================
# def LISTA_test(net, Y, D, device, sigma):

#     # convert the data into tensors
#     Y_t = torch.from_numpy(Y.T)
#     if len(Y.shape) <= 1:
#         Y_t = Y_t.view(1, -1)
#     Y_t = Y_t.float().to(device)
#     D_t = torch.from_numpy(D.T)
#     D_t = D_t.float().to(device)
#     X_final = np.zeros((Y_t.shape[0], 100))
#     # ratio = 1
#     for i in range(len(Y_t)):
#         with torch.no_grad():
#             # Compute the output
#             net.eval()
#             net.__setter__(visu_shrink(sigma[i], 30, 15))
#             X_lista = net(Y_t[i].float())
#             print(X_lista.shape)
#             if len(Y.shape) <= 1:
#                 X_lista = X_lista.view(-1)
#             X_final[i] = X_lista[-1].cpu().numpy().T

#     return X_final