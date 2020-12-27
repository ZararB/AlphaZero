import pickle as pkl 
import socket
from network import Network
from config import Config 


config = Config()
network = Network(config)
print(network.model.summary())

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(1000)

training_step = 0 
inference_count = 0 

while True:

    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")    

    header = clientsocket.recv(config.HEADERSIZE)
    flag = clientsocket.recv(config.FLAGSIZE)
    datalen = int(header)
    flag = int(flag)

    data = clientsocket.recv(datalen)
    data = pkl.loads(data)

    if flag == config.INFERENCE_FLAG:
        
        print('Inference: {}'.format(inference_count))
        response = network.inference(data)
        inference_count += 1 
        response = pkl.dumps(response)
        response = bytes(f'{len(response):<{config.HEADERSIZE}}', 'utf-8') + response
        clientsocket.sendall(response)


    elif flag == config.UPDATE_FLAG:
        print('Update step {}'.format(training_step))
        batch = data
        network.update(batch)
        training_step += 1 
        

    if training_step % 10 == 0:
        network.model.save('models/{}.h5'.format(training_step))

    clientsocket.close()
    
    
    






    
    


