from socket import socket, AF_INET, SOCK_DGRAM
import struct

HOST = ''
PORT = 5000

# ソケットを用意
s = socket(AF_INET, SOCK_DGRAM)
# バインドしておく
s.bind((HOST, PORT))

count = 0
while True:
    # 受信
    msg, address = s.recvfrom(8192)
    val = msg.decode()
    print(val)
    count = count + 1
    if count > 50:
        break
# ソケットを閉じておく
s.close()
