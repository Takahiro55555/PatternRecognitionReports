# -*- coding: utf-8 -*-
# google colab を使う場合は，下の2行をコメントアウト
# from google.colab import files
# f= files.upload()

import numpy as np
import scipy.io
data = scipy.io.loadmat("data/digit.mat")
type(data) # dict
data.keys() # dict_keys(['__header__', '__version__', '__globals__', 'X', 'T'])

type(data["X"]) # numpy.ndarray
x = data["X"]
type(x) # numpy.ndarray
x.shape # (256, 500, 10)
[d, n, nc] = x.shape

z = x.reshape(d, n*nc)
z.shape # (256, 5000)

# 分散・共分散行列 V の計算
V = np.cov(z)
V.shape # (256, 256)

# 正定値対称行列 V の固有ベクトル・固有値の計算
[eigval, eigvec] = np.linalg.eig(V)
eigvec.shape # (256, 256)
eigval.shape # (256,)

# ここで固有ベクトルを固有値の大きい順に並べ替える．
index = np.argsort(eigval)[::-1]
eigvec = eigvec[:,index]
eigvec.shape # (256, 256)
e=eigvec[:,0:2]
e.shape # (256, 2)

X1 = x[:,:,0].T  # 数字1の500例．X1は 500x256 行列 
X1.shape # (500, 256)
C1 = X1.dot(e)  # 第1,2主成分方向の座標，500例．C1は 500x2 行列

X2 = x[:,:,1].T  # 数字2の500例．X2は 500x256 行列 
X2.shape # (500, 256)
C2 = X2.dot(e)  # 第1,2主成分方向の座標，500例．C2は 500x2 行列
C2.shape # (500, 2)

import matplotlib.pyplot as plt


fig = plt.figure()
# fig.patch.set_facecolor(’silver’) # 背景をシルバー

# plt.subplot(1, 5, 1)

# プロット
plt.scatter(C1[:,0],C1[:,1],s=10, c="red",label="digit 1")
plt.scatter(C2[:,0],C2[:,1],s=10, c="blue",label="digit 2")

# 凡例の表示
plt.legend()

# 描画した内容を画面表示
plt.show()


plt.subplot(1, 5, 2)
X3 = x[:,:,2].T  # 数字3の500例．X3は 500x256 行列
img = np.reshape(X3[0,:],(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)


plt.subplot(1, 5, 3)
e1=eigvec[:,0:1] # 第1主成分
e1.shape # (256,1)
img = np.reshape(e1,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)


plt.subplot(1, 5, 4)
X23 = x[:,22,4].T  # 数字5の23番の例．
img = np.reshape(X23,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)

plt.subplot(1, 5, 5)
s = np.zeros(256)
for i in range(10):
    a = X23.dot(eigvec[:,i]) # 第i主成分の重みを内積で求める．
    s = s + a*eigvec[:,i]
img = np.reshape(s,(16,16))
plt.imshow(img, cmap=plt.cm.gray_r)


# 描画した内容を画面表示
plt.show()

