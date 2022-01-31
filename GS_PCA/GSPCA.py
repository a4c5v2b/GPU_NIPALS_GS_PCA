import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from pycuda import cumath
import numpy as np
from time import time
from nipals.kernels import multiply_transpose, normalize_vector, Norme2, multipy, update, get_eigenvalue, substract, slice_column, slice_M_right, slice_M_left


def GS_PCA_CPU(X, n, epsilon):
    R = np.copy(X)
    V = np.zeros((X.shape[0], n))
    Lambda = np.zeros((n, n))
    vectL = np.zeros(n)
    U = np.zeros((X.shape[1], n))
    for k in range(n):
        mu = 0
        V[:, k] = R[:, k]
        while True:
            # multiply transpose
            U[:, k] = R.T@V[:, k]  # @ is dot product/ matmul
            if k > 0:
                # NOTE :
                # multiply transpose + slicing numpy
                A = U[:, n-k:].T @ U[:, k]
                # multiply + gpuarray op

                U[:, k] = U[:, k] - U[:, n-k:]@A

            L2 = np.linalg.norm(U[:, k])

            if L2 != 0:
                # normalize
                U[:, k] = U[:, k]/L2
            # multiply
            V[:, k] = R@U[:, k]
            if k > 0:
                # multiply transpose
                B = V[:, :k].T @ V[:, k]
                # multiply + gpu op
                V[:, k] = V[:, k] - V[:, :k]@B

            # get eigen vector
            Lk = np.linalg.norm(V[:, k])

            if Lk != 0:
                # gpuarray op
                V[:, k] = V[:, k]/Lk
            if np.abs(Lk-mu) < epsilon:
                break
            mu = Lk
        # update 
        R = R - Lk*np.outer(V[:, k], U[:, k])
        Lambda[k, k] = Lk
        vectL[k] = Lk
    
    # Matrix Matrix Nxk @ kxk mult 
    T = V@Lambda
    P = U
    return T, P, R, Lambda, vectL

def GS_PCA_GPU(X, n, epsilon, maxiter=100):
    M = X.shape[0]
    N = X.shape[1]
    R = np.copy(X)
    R = R.astype(np.float32)
    V = np.zeros((X.shape[0], n)).astype(np.float32)
    U = np.zeros((X.shape[1], n)).astype(np.float32)
    Lambda = np.zeros((n, n)).astype(np.float32)
    # Gpu arrays alloc
    R_gpu = gpuarray.to_gpu(R)
    V_gpu = gpuarray.to_gpu(V)
    U_gpu = gpuarray.to_gpu(U)
    vectL = np.zeros(n)

    for k in range(n):
        mu = 0
        V_gpu[:, k] = R_gpu[:, k]
        for j in range(maxiter):
          # multiply transpose U[:, k] = R.T@V[:, k]
            Vk_gpu = gpuarray.empty((M,), dtype=np.float32)
            Uk_gpu = gpuarray.empty((N,), dtype=np.float32)

            slice_column(U_gpu, Uk_gpu, k, N, n)
            slice_column(V_gpu, Vk_gpu, k, N, n)

            U_gpu[:, k] = multiply_transpose(
                R_gpu, Vk_gpu, Uk_gpu, N, N)

            if k > 0:

                A_gpu = gpuarray.empty((k,), dtype=np.float32)

                U_gpu_right = gpuarray.empty((N, k), dtype=np.float32)

                # U_gpu_right = U[:,n-k:]
                U_gpu_right = slice_M_right(U_gpu, U_gpu_right, k, N, n)
                A_gpu = multiply_transpose(
                    U_gpu_right, U_gpu[:, k], A_gpu, N, k)

                # multiply + gpuarray op U[:, k] = U[:, k] - U[:, n-k:]@A
                temp_gpu = gpuarray.empty((M,), dtype=np.float32)
                temp_gpu = multipy(U_gpu_right, A_gpu, temp_gpu, M, k)

                out = np.zeros(N).astype(np.float32)
                out_gpu = gpuarray.to_gpu(out)

                kU = gpuarray.empty((N,), dtype=np.float32)

                slice_column(U_gpu, kU, k, N, n)
                dum = kU.get() - temp_gpu.get()

                U_gpu[:, k] = substract(out_gpu, kU, temp_gpu, M)

            # normalize   
            Uk_gpu = gpuarray.empty((N,), dtype=np.float32)

            slice_column(U_gpu, Uk_gpu, k, N, n)

            U_gpu[:, k] = normalize_vector(Uk_gpu, N)
            
            # multiply
            Vk_gpu = gpuarray.empty((N,), dtype=np.float32)
            Uk_gpu = gpuarray.empty((N,), dtype=np.float32)

            slice_column(U_gpu, Uk_gpu, k, N, n)
            slice_column(V_gpu, Vk_gpu, k, N, n)
            
            V_gpu[:, k] = multipy(R_gpu, Uk_gpu, Vk_gpu, M, M)

            if k > 0:
                # multiply transpose B = V[:, :k].T @ V[:, k]
                B_gpu = gpuarray.empty((k,), dtype=np.float32)

                Vk_gpu = gpuarray.empty((N,), dtype=np.float32)
                slice_column(V_gpu, Vk_gpu, k, N, n)

                # V_gpu_left = V_gpu[:, :k]
                V_gpu_left = gpuarray.empty((N, k), dtype=np.float32)
                V_gpu_left = slice_M_left(V_gpu, V_gpu_left, k, N, n)

                B_gpu = multiply_transpose(
                    V_gpu_left, Vk_gpu, B_gpu, N, k)

                # multiply + gpu op
                inter_gpu = gpuarray.empty((M,), dtype=np.float32)
                inter_gpu = multipy(V_gpu_left, B_gpu, inter_gpu, M, k)

                # PROBLEM
                out = np.zeros(N).astype(np.float32)
                out_gpu = gpuarray.to_gpu(out)
                # V[:, k] = V[:, k] - V[:, :k]@B
                V_gpu[:, k] = substract(out_gpu, Vk_gpu, inter_gpu, M)


            # get eigen value
            Vk_gpu = gpuarray.empty((N,), dtype=np.float32)
            slice_column(V_gpu, Vk_gpu, k, N, n)
            Lk = get_eigenvalue(Vk_gpu, M)
            # gpuarray op
            V_gpu[:, k] = normalize_vector(Vk_gpu, M)

            if np.abs(Lk-mu) < epsilon:
                break

            mu = Lk
        # update R = R - Lk*np.outer(V[:, k], U[:, k])
        Vk_gpu = gpuarray.empty((N,), dtype=np.float32)
        Uk_gpu = gpuarray.empty((N,), dtype=np.float32)

        slice_column(U_gpu, Uk_gpu, k, N, n)
        slice_column(V_gpu, Vk_gpu, k, N, n)

        R_gpu = update(R_gpu, Vk_gpu, Uk_gpu, M, N, Lk)

        # Lambda_gpu[k, k] = Lk
        vectL[k] = Lk
        Lambda[k,k]=Lk

    # Matrix Matrix Nxk @ kxk mult ??
    T = V_gpu.get()@Lambda
    P = U_gpu.get()    
    R = R_gpu.get()
    return T, P, R, Lambda, vectL



import torchvision
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
import numpy as np


train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])



train_set = datasets.MNIST('/MNIST/', train=True, download=True)

test_set = datasets.MNIST('/MNIST/', train=False, download=True)

train_images = train_set.train_data.unsqueeze(-1).numpy() / 255
train_labels = train_set.train_labels.numpy()
test_images = test_set.test_data.unsqueeze(-1).numpy() / 255
test_labels = test_set.test_labels.numpy()

class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

train_set = train_set.data.numpy()
test_set = test_set.data.numpy()

train_set = np.reshape(train_set,(train_set.shape[0],-1))/255
test_set = np.reshape(test_set,(test_set.shape[0],-1))/255
scaler = StandardScaler()
scaler.fit(train_set)

rng = np.random.RandomState(1)
X = np.dot(rng.rand(4, 2), rng.randn(2, 200)).T

#T, P, R, Lambda, vectL =  GS_PCA_CPU(X,2,1e-3)
T, P, R, Lambda, vectL =  GS_PCA_CPU(train_set[0:500,],500,1e-3)

projection = np.dot(T,P.T[:,:500])

import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from skcuda.linalg import PCA as cuPCA




pca = cuPCA(20)
X = train_set[0:500,].astype('double')
X_gpu = gpuarray.to_gpu(X)
T_gpu = pca.fit_transform(X_gpu)
dot_product = linalg.dot(T_gpu[:, 0].copy(), T_gpu[:, 1].copy())


# now get the variance of each eigenvector so we can see the percent explained by the first two
std_vec = np.std(T_gpu.get(), axis=0)
T = T_gpu.get()  # The principal components are not in descending order
T_2 = T[:,(-std_vec).argsort()[:20].tolist()]
std_vec = np.std(T_2, axis=0)
print("We explained " + str(100 * np.sum(std_vec[:20]) / np.sum(std_vec)) + "% of the variance with 500 principal components " )
explained_ratio = std_vec / np.sum(std_vec)
print(100 * explained_ratio[:20])



# var = []
# for i in range(T.shape[1]):
#     var.append(np.var(T[i]))
# var = np.array(var)

#T_g, P_g, R_g, Lambda_g, vectL_g =  GS_PCA_GPU(train_set[0:500,],500,1e-3)

# R = [[0.40348195], [0.38658295], [0.82931052]]
# V = [0.33452744, 0.33823673, 0.32723583]
# print("Rt_p: ", R)
# B = np.matmul(V, np.array(R)) / pow(np.linalg.norm(R), 2)
# print("B", B)
#
# (1, 3)

