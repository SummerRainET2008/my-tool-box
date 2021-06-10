import socket

HOST = '192.168.1.192'                 # Symbolic name meaning all available interfaces
PORT = 50000               # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)
while 1:
    data = conn.recv(1024)
    if not data: break
    conn.send(data)
conn.close()
