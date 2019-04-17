# coding:utf-8
import sys, os
import numpy as np
sys.path.append(os.curdir)
from util import im2col

x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)#9,p;/


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        '''
        FN:フィルターの個数
        C:チャンネル
        '''

        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = int(1 + (H+2*self.pad-FH)/self.stride)
        out_w = int(1 + (W+2*self.pad-FW)/self.stride)

        col = im2col(x, FW, FH, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        #transposeで(N,C,H,W)の順に入れ替え
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col.reshape(-1, self.pool_h*self.pool_w)

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)


class SimpleConvNet:
    def __init__(self, input_dim=(1,28,28),conv_param={'filter_num':30, 'filter_size':5,
                    'pad':0, 'stride':1},
                    hidden_size=100,output_size=10, weight_init_std=0.01):

        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['filter_pad']
        filter_stride = conv_param['filter_stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
