# coding: utf-8
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np


'''
a = np.random.normal(
    loc   = 0,      # 平均
    scale = 1,      # 標準偏差
    size  = (10),# 出力配列のサイズ(タプルも可)
)

print(type(a))
print(a)
a[0] = 1.0
print(a)
plt.hist(a,bins=50)
plt.xlim(-10,10)
#plt.show()
'''

noise = np.random.normal(0, 1, (28, 28));

img = plt.imread('0.png')
print(noise.shape)
fig,ax = plt.subplots()
ax.imshow(noise, 'gray')
plt.show()
