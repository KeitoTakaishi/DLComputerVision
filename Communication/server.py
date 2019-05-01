#coding: utf-8
import socket
import numpy as np
import cv2

img = cv2.imread('17000.png', cv2.IMREAD_GRAYSCALE)
#print(type(img))
#print(img.shape)
byteData = img.tostring()
#print(type(byteData));
#print(byteData)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 6000))
s.listen(1) 
print("接続待機中")
soc, addr = s.accept()
print(str(addr)+"と接続完了")

while True:
    soc.send(byteData)
    k = cv2.waitKey(1)
    if k== 13:
        break


'''
with closing(udp):
    udp.sendto(byteData, to_send_addr)
    udp.sendto(b'__end__', to_send_addr)
    udp.close()
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''
