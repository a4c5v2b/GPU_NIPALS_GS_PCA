# From https://scikit-cuda.readthedocs.io/en/latest/generated/skcuda.linalg.PCA.html


"""
Old version of skcuda
"""
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import skcuda.linalg as linalg
# from skcuda.linalg import PCA as cuPCA
# pca = cuPCA(n_components=4)
# X = np.random.rand(1000,100)
# #X = np.asfortranarray(X)
# X_gpu = gpuarray.GPUArray((1000,100), np.float64, order="C")
# X_gpu.set(X)
# T_gpu = pca.fit_transform(X_gpu)
# linalg.dot(T_gpu[:,0], T_gpu[:,1])


#From https://github.com/lebedov/scikit-cuda/blob/master/demos/pca_demo.py

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from skcuda.linalg import PCA as cuPCA
from matplotlib import pyplot as plt
from sklearn import datasets

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

iris = datasets.load_iris()
X_orig = iris.data
y = iris.target


# Transform to normalized Tensors
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST('./MNIST/', train=True, transform=transform, download=True)
# test_dataset = datasets.MNIST('./MNIST/', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
# test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

X_orig = next(iter(train_loader))[0].numpy()
y = train_dataset.targets.numpy()
X_orig = np.reshape(X_orig,(60000,-1))





pca = cuPCA(4)  # take all 4 principal components

demo_types = [np.float32, np.float64]  # we can use single or double precision
precisions = ['single', 'double']

print("Principal Component Analysis Demo!")
print("Compute 2 principal components of a 1000x4 IRIS data matrix")
print("Lets test if the first two resulting eigenvectors (principal components) are orthogonal,"
      " by dotting them and seeing if it is about zero, then we can see the amount of the origial"
      " variance explained by just two of the original 4 dimensions. Then we will plot the reults"
      " for the double precision experiment.\n\n\n")

for i in range(len(demo_types)):

    demo_type = demo_types[i]



    # 1000 samples of 4-dimensional data vectors
    X = X_orig.astype(demo_type)


    X_gpu = gpuarray.to_gpu(X)  # copy data to gpu

    T_gpu = pca.fit_transform(X_gpu)  # calculate the principal components

    # show that the resulting eigenvectors are orthogonal
    # Note that GPUArray.copy() is necessary to create a contiguous array
    # from the array slice, otherwise there will be undefined behavior
    dot_product = linalg.dot(T_gpu[:, 0].copy(), T_gpu[:, 1].copy())
    T = T_gpu.get()

    print("The dot product of the first two " + str(precisions[i]) +
          " precision eigenvectors is: " + str(dot_product))

    # now get the variance of the eigenvectors and create the ratio explained from the total
    std_vec = np.std(T, axis=0)
    print("We explained " + str(100 * np.sum(std_vec[:2]) / np.sum(std_vec)) +
          "% of the variance with 2 principal components in " +
          str(precisions[i]) + " precision\n\n")

    # Plot results for double precision
    if i == len(demo_types) - 1:
        # Different color means different IRIS class
        plt.scatter(T[:, 0], T[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=20)
        plt.show()







# import numpy as np
# import pycuda.gpuarray as gpuarray
# import pycuda.autoinit
# import skcuda.linalg as sklin
#
# a = np.random.randn(4, 4).astype(np.float32) #为了后续GPU上的计算顺利进行，矩阵数值设定为float32
# b = np.random.randn(4, 4).astype(np.float32)
#
# a_gpu = gpuarray.to_gpu(a)
# b_gpu = gpuarray.to_gpu(b)
#
# sklin.init()   # 使用skcuda线性代数库时要先调用内置函数进行初始化
# multi_gpu = sklin.dot(a_gpu, b_gpu)  # 矩阵乘法
# a_inv = sklin.inv(a_gpu)  # 矩阵求逆
# multi = multi_gpu.get()
# inv = a_inv.get()