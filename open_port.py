# -*- coding: utf-8 -*-
import socket

def open_port(host='0.0.0.0', port=29500):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)  # 최대 5개의 연결 대기
    print(f"Listening on {host}:{port}")
    
    while True:
        client_socket, address = server_socket.accept()
        print(f"Connection from {address}")
        client_socket.close()

if __name__ == "__main__":
    open_port()
