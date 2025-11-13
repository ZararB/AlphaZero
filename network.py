
from config import Config
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import socket
import pickle as pkl
import os 
K.clear_session()


class Network(object):

	def __init__(self, config, remote=False):

		self.config = config 
		self.num_actions = self.config.num_actions # change to len of moveDict
		self.batch_size = 1
		self.remote = remote


		if self.remote:

			self.ipaddress = input("Enter ip address of server: ")
			self.port = input("Enter port: ")

			self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			
		else:
			# Create models directory if it doesn't exist
			os.makedirs('models', exist_ok=True)
			
			model_files = [f for f in os.listdir('models/') if f.endswith('.h5')]
			model_files.sort()

			if model_files:
				model_filepath = 'models/' + model_files[-1]
				print('Loading model {}...'.format(model_filepath))
				self.model = load_model(model_filepath)
			
			else:
				self.model = self.build_model()

			# Warm up the model (TF 2.x uses eager execution, no need for session/graph)
			self.model.predict(np.random.randn(1,8,8,18), verbose=0)
		
			
	def build_model(self):
		#TODO Write for loop for adding blocks 

		inputs = Input(shape=(8, 8, 18))
		x = Conv2D(64, 3, activation='relu')(inputs)
		x = Conv2D(128, 3, activation='relu')(x)
		block_1_output =  MaxPooling2D(3)(x)

		x = Conv2D(128, 3, activation='relu', padding="same")(block_1_output)
		x = Conv2D(128, 3, activation='relu', padding="same")(x)
		block_2_output =  add([x, block_1_output])

		x = Conv2D(128, 3, activation='relu', padding="same")(block_2_output)
		x = Conv2D(128, 3, activation='relu', padding="same")(x)
		block_3_output =  add([x, block_2_output])

		x = Conv2D(128, 3, activation='relu', padding="same")(block_3_output)
		x = Conv2D(128, 3, activation='relu', padding="same")(x)
		block_4_output =  add([x, block_3_output])
		
		x = Conv2D(128, 3, activation='relu', padding="same")(block_4_output)
		x = Conv2D(128, 3, activation='relu', padding="same")(x)
		block_5_output =  add([x, block_4_output])

		x = Conv2D(128, 3, activation='relu', padding="same")(block_5_output)
		x = Conv2D(128, 3, activation='relu', padding="same")(x)
		block_6_output =  add([x, block_5_output])

		x = Conv2D(128, 3, activation='relu', padding="same")(block_6_output)
		x = Conv2D(128, 3, activation='relu', padding="same")(x)
		block_7_output =  add([x, block_6_output])

		base = Flatten()(x)
		
		value = Dense(512)(base)
		value = Dense(1, activation='tanh', name='value')(value)  # tanh for [-1, 1] range

		policy = Dense(self.num_actions, activation='softmax', name='policy')(base)
	
		model = Model(inputs=[inputs], outputs=[value, policy])
		#model.make_predict_function()
		model.compile(optimizer='adam',
		loss={
			'value':keras.losses.MeanSquaredError(),
			'policy':keras.losses.CategoricalCrossentropy(),
		},
		loss_weights=[1.,1.])
		
		#model.summary() 

		return model 


	def inference(self, image):


		if self.remote:

			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

				sock.connect((socket.gethostname(), 1234))
				data = pkl.dumps(image)
				data = bytes(f'{len(data):<{self.config.HEADERSIZE}}' + f'{self.config.INFERENCE_FLAG:<{self.config.FLAGSIZE}}', 'utf-8') + data
				sock.sendall(data)

				header = sock.recv(self.config.HEADERSIZE)
				datalen = int(header)
				response = sock.recv(datalen)
				value, policy = pkl.loads(response)
		  

		else:
			# TensorFlow 2.x uses eager execution - no need for session/graph context
			model_output = self.model.predict(np.array([image]), verbose=0)
			value  = np.array(model_output[0])
			policy = np.array(model_output[1])

		return (value, policy)  # Value, Policy


	def update(self, batch, training_step=0):

		if self.remote:

			data = pkl.dumps(batch)
			data = bytes(f'{len(data):<{self.config.HEADERSIZE}}' + f'{self.config.UPDATE_FLAG:<{self.config.FLAGSIZE}}', 'utf-8') + data
			self.s.send(data)

		else:
			# Batch training instead of single-sample training
			if len(batch) == 0:
				return
				
			images = np.array([img for img, _ in batch])
			target_values = np.array([val for _, (val, _) in batch])
			target_policies = np.array([pol for _, (_, pol) in batch])
			
			# Update learning rate based on schedule
			lr = self.get_learning_rate(training_step)
			K.set_value(self.model.optimizer.learning_rate, lr)
			
			self.model.fit(
				[images],
				{'value': target_values, 'policy': target_policies},
				verbose=0,
				epochs=1
			)
	
	def get_learning_rate(self, training_step):
		"""Get learning rate based on schedule."""
		schedule = self.config.learning_rate_schedule
		# Find the highest step that's <= training_step
		applicable_steps = [step for step in schedule.keys() if step <= training_step]
		if applicable_steps:
			return schedule[max(applicable_steps)]
		return schedule[0]  # Default to first learning rate



