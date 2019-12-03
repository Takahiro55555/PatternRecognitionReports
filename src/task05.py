# -*- coding: utf-8 -*-
# google colab を使う場合は，下の2行をコメントアウト
# from google.colab import files
# f= files.upload()

import random
import sys

import cv2
import numpy as np
from gen_qrcodes import gen_sequential_qr, gen_random_qr, RANDM_RANGE

SEED_01 = 20191204
SEED_02 = 19981124
IMG_NUM = 500
QR_SEQ_DIR = "qr_seq"
QR_RAND_DIR_1 = "qr_rand01"
RANDM_RANGE_1 = 1e20 + 1
QR_RAND_DIR_2 = "qr_rand02"
RANDM_RANGE_2 = 1e3

def main():
    resized_size = 16
    if len(sys.argv) == 2 and sys.argv[1] == "--generate":
        gen_sample_qrcodes()
    elif len(sys.argv) == 2 and sys.argv[1].isnumeric():
        resized_size = int(sys.argv[1])
    x = np.zeros((resized_size**2, IMG_NUM, 3))

    for i in range(IMG_NUM):
        f_name = "%s/%020d.png" % (QR_SEQ_DIR, i)
        x[:,i,0] = load_qr_image_array(f_name, resized_size)

    random.seed(SEED_01) #シード値を設定
    for i in range(IMG_NUM):
        f_name = "%s/%020d.png" % (QR_RAND_DIR_1, random.randrange(RANDM_RANGE_1))
        x[:,i,1] = load_qr_image_array(f_name, resized_size)

    random.seed(SEED_02) #シード値を設定
    for i in range(IMG_NUM):
        f_name = "%s/%020d.png" % (QR_RAND_DIR_2, random.randrange(RANDM_RANGE_2))
        x[:,i,2] = load_qr_image_array(f_name, resized_size)

    print(x.shape)

    x.shape # (256, IMG_NUM, 10)
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

    X1 = x[:,:,0].T  # IMG?NUM例．X1は IMG_NUMx256 行列 
    X1.shape # (IMG_NUM, 256)
    C1 = X1.dot(e)  # 第1,2主成分方向の座標，IMG?NUM例．C1は IMG_NUMx2 行列

    X2 = x[:,:,1].T  # IMG?NUM例．X2は IMG_NUMx256 行列 
    X2.shape # (IMG_NUM, 256)
    C2 = X2.dot(e)  # 第1,2主成分方向の座標，IMG?NUM例．C2は IMG_NUMx2 行列


    X3 = x[:,:,2].T  # IMG?NUM例．X2は IMG_NUMx256 行列 
    X3.shape # (IMG_NUM, 256)
    C3 = X3.dot(e)  # 第1,2主成分方向の座標，IMG?NUM例．C2は IMG_NUMx2 行列


    import matplotlib.pyplot as plt

    fig = plt.figure()
    # fig.patch.set_facecolor(’silver’) # 背景をシルバー
    fig = plt.figure()
    fig.suptitle("Original image and resized image")
    
    f_name = "%s/%020d.png" % (QR_SEQ_DIR, 0)
    img_original = cv2.imread(f_name)
    plt.subplot(1, 2, 1)
    plt.imshow(img_original, cmap=plt.cm.gray_r)
    plt.xlabel("Original")

    img_resized = load_qr_image(f_name, resized_size)
    plt.subplot(1, 2, 2)
    plt.imshow(img_resized, cmap=plt.cm.gray_r)
    plt.xlabel("Resized (%dx%d)" % (resized_size, resized_size))
    # グラフの保存
    plt.savefig("figs/qr_original_and_resized_%d.png" % resized_size)
    plt.show()


    plt.title("PCA: Sequential QR codes and random QR codes")

    # プロット
    plt.scatter(C1[:,0],C1[:,1],s=10, c="red",label="sequential (0 < value < %d)" % (IMG_NUM - 1))
    plt.scatter(C2[:,0],C2[:,1],s=10, c="blue",label="random1 (seed=%d)" % SEED_01)
    plt.scatter(C3[:,0],C3[:,1],s=10, c="green",label="random2 (seed=%d)" % SEED_02)
    plt.xlabel("0 <= random1 value <= {:5.0e}\n0 <= random2 value <= {:d}\nimage size:{}x{}".format(RANDM_RANGE_1 - 1, int(RANDM_RANGE_2 - 1), resized_size, resized_size))
    plt.tight_layout()
    # 凡例の表示
    plt.legend()

    # グラフの保存
    plt.savefig("figs/qr_pca_%d.png" % resized_size)

    # 描画した内容を画面表示
    plt.show()

def load_qr_image(f_name, resized_size):
    img = cv2.imread(f_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (resized_size, resized_size))
    return img_resized

def load_qr_image_array(f_name, resized_size):
    img_resized = load_qr_image(f_name, resized_size)
    img_array = img_resized.flatten()
    return img_array

def gen_sample_qrcodes():
    """
    qrコードを生成する関数
    """
    gen_sequential_qr(IMG_NUM, QR_SEQ_DIR)
    gen_random_qr(IMG_NUM, QR_RAND_DIR_1, SEED_01, RANDM_RANGE_1)
    gen_random_qr(IMG_NUM, QR_RAND_DIR_2, SEED_02, RANDM_RANGE_2)

if __name__ == "__main__":
    main()