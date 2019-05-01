import socket
import numpy as np
import cv2

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect(('192.168.0.55', 6000))
print("接続完了")

epoch = 0
while(1):

    #data = soc.recv(307200)
    data = soc.recv(784)
    data = np.fromstring(data,dtype=np.uint8)
    data = np.reshape(data,(28,28,1))#形状復元(これがないと一次元行列になってしまう。)　reshapeの第二引数の(480,640,3)は引数は送られてくる画像の形状


    #height =data.shape[0]
    #width = data.shape[1]
    #resized_img = cv2.resize(img,(width, height0))

    #cv2.imshow("img", resized_img)
    #cv2.namedWindow("img")
    #cv2.imshow("img",data);
    cv2.imwrite('Images\{}.jpeg'.format(epoch), data)
    epoch += 1

    k = cv2.waitKey(0)
    if k== 13 :
        cv2.destroyAllWindows()
        break
