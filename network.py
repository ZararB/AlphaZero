
from config import Config
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

K.clear_session()
class Network(object):

	def __init__(self, config=None, filepath=None):
		if config is None:
			config = Config()
		self.num_actions = config.num_actions # change to len of moveDict
		self.batch_size = 1

		# Naive network
		if filepath:
			self.model = load_model(filepath)
		else:
			self.model = self.build_model()

		self.model.predict(np.random.randn(1,8,8,18))
		self.session = K.get_session()
		self.graph = tf.get_default_graph()
		self.graph.finalize()
		
			
	def build_model(self):

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
		value = Dense(1, activation='sigmoid', name='value')(value)

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
		with self.session.as_default():
			with self.graph.as_default():
				model_output = self.model.predict(np.array([image]))
		value  = np.array(model_output[0])
		policy = np.array(model_output[1])
		return (value, policy)  # Value, Policy


	def update(self, batch):

		for image, (target_value, target_policy) in batch:
			image = np.array([image])
			target_value = np.array([target_value])
			
			self.model.fit(
				[image],
				{'value': target_value, 'policy':target_policy},
				verbose=0
			)

	

	def get_weights(self):
		# Returns the weights of this network.
		return []


