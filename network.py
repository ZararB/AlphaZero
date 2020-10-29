from tensorflow.keras.models import Sequential
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
from config import Config



#TODO implement neural network 

class Network(object):

	def __init__(self, config=None):
		if config is None:
			config = Config()
		self.num_actions = config.num_actions # change to len of moveDict
		self.batch_size = 1
		
		# Naive network

		inputs = keras.Input(shape=(8, 8, 18))
		x = Conv2D(16, 3, activation='relu')(inputs)
		x = Conv2D(32, 3, activation='relu')(x)
		x = MaxPooling2D(2)(x)
		base = Flatten()(x)
		value = Dense(1, activation='sigmoid', name='value')(base)
		policy = Dense(self.num_actions, activation='softmax', name='policy')(base)
	

		self.model = keras.Model(inputs=[inputs], outputs=[value, policy])

		self.model.compile(optimizer='adam',
		loss={
			'value':keras.losses.MeanSquaredError(),
			'policy':keras.losses.CategoricalCrossentropy(),
		},
		loss_weights=[1.,1.])

		self.model.summary()



	def inference(self, image):
		model_output = self.model.predict(np.array([image]))
		value  = model_output[0]
		policy = model_output[1]
		return (value, policy)  # Value, Policy


	


	def get_weights(self):
		# Returns the weights of this network.
		return []
