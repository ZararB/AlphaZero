from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout


class Network(object):

	def __init__(self, config):
		self.model = Sequential([

			]
			)

	def inference(self, image):
		
		model_output = self.model.predict(image)
    	return (model_output[0,0], model_output[0,1])  # Value, Policy

  	def get_weights(self):
	    # Returns the weights of this network.
	    return []

