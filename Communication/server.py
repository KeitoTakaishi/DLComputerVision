# coding: utf-8
import socket
import numpy as np
import cv2
import time


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#ソケットオブジェクト作成
s.bind(("", 6000))    # サーバー側PCのipと使用するポート
print("接続待機中")
s.listen(1)                     # 接続要求を待機
soc, addr = s.accept()          # 要求が来るまでブロック
print(str(addr)+"と接続完了")

#cam = cv2.VideoCapture(0)#カメラオブジェクト作成

while (True):

    #flag,img = cam.read()       #カメラから画像データを受け取る

    img = cv2.imread('17000.png', cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    img = img.tostring()        #numpy行列からバイトデータに変換

    soc.send(img)              # ソケットにデータを送信

    #time.sleep(0.5)            #フリーズするなら#を外す。

    k = cv2.waitKey(1)         #↖
    #if k== 13 :                #←　ENTERキーで終了
        #break                  #↙

#cam.releace()
