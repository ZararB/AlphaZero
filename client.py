import pickle as pkl 
import socket
from config import Config
import numpy as np 
from network import Network

config = Config()
network = Network(config, remote=True)
data = np.random.rand(8, 8, 18)
value, policy = network.inference(data)

print("Value: {}, Policy: {}".format(value, policy))


'''
HEADERSIZE = config.HEADERSIZE 


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ipAddress = input("Enter ip address of server: ")
port = input("Enter port number: ")
#s.connect((ipAddress, port))
s.connect((socket.gethostname(), 1234))

data = ['zarar', 'fafdsf', 'fasdfasd']
data = np.random.rand(1, 8, 8, 18)
msg = pkl.dumps(data)
flag = config.INFERENCE_FLAG

msg = bytes(f'{len(msg):<{HEADERSIZE}}' + f'{flag:<{config.FLAG_SIZE}}', 'utf-8') + data

s.send(msg)

header = s.recv(HEADERSIZE)
msglen = int(header)

data = s.recv(msglen)
data = pkl.loads(data)
print(data)

'''

